"""Step recognition inference (TSM-ResNet18, 9 step classes) for single videos and sliding windows.

The step classifier provides ``S_t`` (task step) of the world state.
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import yaml
import argparse
from typing import Dict, List, Tuple

from ws_racr.perception.step_recognition.models import load_tsm_resnet18
from ws_racr.perception.step_recognition.transforms import VideoTransform


class StepRecognitionInference:
    """Step Recognition 推理器"""
    
    def __init__(self, checkpoint_path, config_path=None, device='cuda'):
        """
        初始化推理器
        
        Args:
            checkpoint_path: 模型檢查點路徑
            config_path: 配置檔案路徑（可選）
            device: 設備 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 載入檢查點
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 從檢查點中載入配置
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint.get('config', {})
        
        # 載入類別名稱
        self.class_names = [
            "place_platform",
            "insert_cylinder_under_platform",
            "tighten_small_screw_under_platform",
            "place_bracket_on_table",
            "align_pi_on_bracket",
            "insert_upper_big_screw",
            "align_bracket_to_platform",
            "insert_lower_big_screw",
            "idle"
        ]
        
        # 建立模型
        self.model = load_tsm_resnet18()(
            num_classes=len(self.class_names),
            n_segment=self.config['data']['clip_len'],
            pretrained=False,
            dropout=0.0  # 推理時不需要 dropout
        )
        
        # 載入權重
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 創建 transform
        self.transform = VideoTransform(
            input_size=tuple(self.config['data']['input_size']),
            mean=self.config['data']['mean'],
            std=self.config['data']['std'],
            augment=False  # 推理時不增強
        )
        
        print(f"✓ Loaded model from {checkpoint_path}")
        print(f"✓ Device: {self.device}")
        print(f"✓ Number of classes: {len(self.class_names)}")
    
    def predict_single_video(self, video_path) -> Dict:
        """
        對單一影片進行推理
        
        Args:
            video_path: 影片路徑
        
        Returns:
            result: {
                'pred_class_index': int,
                'pred_class_name': str,
                'probabilities': List[float]
            }
        """
        # 載入影片
        frames = self._load_video(video_path)
        
        if frames is None or len(frames) == 0:
            return {
                'pred_class_index': -1,
                'pred_class_name': 'unknown',
                'probabilities': [0.0] * len(self.class_names)
            }
        
        # 預處理
        frames = self.transform(frames, is_train=False)
        
        # 轉換為 tensor: (T, H, W, C) -> (1, C, T, H, W)
        clip = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float()
        clip = clip.to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(clip)  # (1, num_classes)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0)  # (num_classes,)
        
        # 獲取預測
        pred_idx = probabilities.argmax().item()
        pred_name = self.class_names[pred_idx]
        probs_list = probabilities.cpu().numpy().tolist()
        
        return {
            'pred_class_index': pred_idx,
            'pred_class_name': pred_name,
            'probabilities': probs_list
        }
    
    def predict_sliding_window(
        self, 
        video_path, 
        window_frames=None, 
        stride_frames=None
    ) -> List[Dict]:
        """
        對長影片進行滑動窗口推理
        
        Args:
            video_path: 影片路徑
            window_frames: 窗口大小（幀數）
            stride_frames: 滑動步長（幀數）
        
        Returns:
            results: List of {
                'start_frame': int,
                'end_frame': int,
                'pred_class_index': int,
                'pred_class_name': str,
                'confidence': float
            }
        """
        if window_frames is None:
            window_frames = self.config['data']['clip_len'] * self.config['data']['frame_interval']
        if stride_frames is None:
            stride_frames = window_frames // 2  # 50% 重疊
        
        # 載入完整影片
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if total_frames == 0:
            return []
        
        results = []
        
        # 滑動窗口
        start = 0
        while start + window_frames <= total_frames:
            end = start + window_frames
            
            # 載入窗口
            frames = self._load_video_segment(video_path, start, end)
            
            if frames is not None and len(frames) > 0:
                # 預處理
                frames = self.transform(frames, is_train=False)
                
                # 轉換為 tensor
                clip = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float()
                clip = clip.to(self.device)
                
                # 推理
                with torch.no_grad():
                    outputs = self.model(clip)
                    probabilities = torch.softmax(outputs, dim=1).squeeze(0)
                
                pred_idx = probabilities.argmax().item()
                pred_name = self.class_names[pred_idx]
                confidence = probabilities[pred_idx].item()
                
                results.append({
                    'start_frame': start,
                    'end_frame': end,
                    'pred_class_index': pred_idx,
                    'pred_class_name': pred_name,
                    'confidence': confidence
                })
            
            start += stride_frames
        
        return results
    
    def _load_video(self, video_path):
        """載入影片並採樣幀"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Warning: Cannot open video {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
        
        # 採樣幀索引
        clip_len = self.config['data']['clip_len']
        frame_interval = self.config['data']['frame_interval']
        required_frames = clip_len * frame_interval
        
        # 中心採樣
        if total_frames > required_frames:
            start_idx = (total_frames - required_frames) // 2
        else:
            start_idx = 0
        
        # 讀取幀
        frames = []
        for i in range(clip_len):
            frame_idx = start_idx + i * frame_interval
            frame_idx = min(frame_idx, total_frames - 1)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        
        return np.array(frames)
    
    def _load_video_segment(self, video_path, start_frame, end_frame):
        """載入影片片段"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        # 採樣幀索引
        clip_len = self.config['data']['clip_len']
        frame_interval = self.config['data']['frame_interval']
        
        segment_length = end_frame - start_frame
        
        # 均勻採樣
        frame_indices = np.linspace(start_frame, end_frame - 1, clip_len, dtype=int)
        
        # 讀取幀
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        
        return np.array(frames)


def main():
    parser = argparse.ArgumentParser(description='Step Recognition Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--sliding', action='store_true',
                        help='Use sliding window inference')
    parser.add_argument('--window', type=int, default=None,
                        help='Sliding window size (frames)')
    parser.add_argument('--stride', type=int, default=None,
                        help='Sliding window stride (frames)')
    
    args = parser.parse_args()
    
    # 創建推理器
    inferencer = StepRecognitionInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # 執行推理
    if args.sliding:
        print(f"\nRunning sliding window inference on {args.video}...")
        results = inferencer.predict_sliding_window(
            args.video,
            window_frames=args.window,
            stride_frames=args.stride
        )
        
        print(f"\nFound {len(results)} segments:")
        print("-" * 80)
        for i, result in enumerate(results):
            print(f"Segment {i+1}:")
            print(f"  Frames: {result['start_frame']} - {result['end_frame']}")
            print(f"  Prediction: {result['pred_class_name']}")
            print(f"  Confidence: {result['confidence']:.4f}")
    else:
        print(f"\nRunning inference on {args.video}...")
        result = inferencer.predict_single_video(args.video)
        
        print(f"\nPrediction: {result['pred_class_name']} (class {result['pred_class_index']})")
        print("\nClass Probabilities:")
        print("-" * 60)
        for i, (name, prob) in enumerate(zip(inferencer.class_names, result['probabilities'])):
            bar = '█' * int(prob * 50)
            print(f"{i} {name:40s} {prob:6.2%} {bar}")
        print("-" * 60)


if __name__ == "__main__":
    main()
