"""Real-time step recognition on a RealSense stream (16-frame clips, TSM-ResNet18).

Used by ``HRCSystem`` for the top-down D435i camera; produces the step class and confidence that
become ``S_t`` in the world state.
"""
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import torch.nn.functional as F
from collections import deque
import time
import argparse

from ws_racr.perception.step_recognition.models import load_tsm_resnet18
from ws_racr.perception.step_recognition.transforms import VideoTransform


class RealtimeStepRecognizer:
    """即時步驟識別器"""
    
    def __init__(self, checkpoint_path, device='cuda', clip_len=16, confidence_threshold=0.5):
        """
        Args:
            checkpoint_path: 模型檢查點路徑
            device: 'cuda' 或 'cpu'
            clip_len: 每次預測使用的幀數
            confidence_threshold: 信心度閾值（低於此值顯示為"不確定"）
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.clip_len = clip_len
        self.confidence_threshold = confidence_threshold
        
        # 載入模型
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 載入類別名稱（從 annotations/class_index.txt 讀取）
        class_index_file = Path(__file__).parent / 'data' / 'annotations' / 'class_index.txt'
        if class_index_file.exists():
            with open(class_index_file, 'r') as f:
                lines = f.read().strip().split('\n')
                self.class_names = [line.split()[1] for line in lines]
        else:
            # 備用：從 config 讀取或使用預設
            print("⚠️  找不到 class_index.txt，使用預設類別名稱")
            self.class_names = [
                'place_platform',
                'insert_cylinder_under_platform',
                'tighten_small_screw_under_platform',
                'place_bracket_on_table',
                'align_pi_on_bracket',
                'insert_upper_big_screw',
                'align_bracket_to_platform',
                'insert_lower_big_screw',
                'idle'
            ]
        
        self.num_classes = len(self.class_names)
        print(f"Classes ({self.num_classes}): {self.class_names}")
        
        # 建立模型
        self.model = load_tsm_resnet18()(
            num_classes=self.num_classes,
            n_segment=clip_len,
            pretrained=False
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
        
        # 數據轉換（驗證模式：只做 resize, center crop, normalize，不做增強）
        self.transform = VideoTransform(
            input_size=(224, 224),
            augment=False  # 推理時不做數據增強
        )
        
        # 幀緩衝區（使用 deque 實現滑動窗口）
        self.frame_buffer = deque(maxlen=clip_len * 2)  # 2x buffer for smooth sampling
        
        # 預測歷史（用於平滑預測結果）
        self.prediction_history = deque(maxlen=5)  # 保留最近 5 次預測
        
        # FPS 計算
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        
    def preprocess_frames(self, frames):
        """
        預處理幀序列
        Args:
            frames: list of BGR images (H, W, 3)
        Returns:
            tensor: (1, C, T, H, W)
        """
        # 轉換為 RGB 並轉為 numpy array (T, H, W, 3)
        frames_rgb = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
        
        # 應用轉換（不做增強）
        clip = self.transform(frames_rgb, is_train=False)  # numpy (T, H, W, 3)
        
        # 轉換維度順序: (T, H, W, C) -> (C, T, H, W)
        clip = np.transpose(clip, (3, 0, 1, 2))  # (C, T, H, W)
        
        # 轉換為 tensor
        clip = torch.from_numpy(clip).float()  # (C, T, H, W)
        
        # 增加 batch 維度
        clip = clip.unsqueeze(0)  # (1, C, T, H, W)
        
        return clip.to(self.device)
    
    def predict(self, frames, top_k=3):
        """
        預測當前幀序列的步驟（返回 top-k 候選）
        Args:
            frames: list of BGR images
            top_k: 返回前 k 個候選步驟
        Returns:
            dict: {
                'class_index': int,  # top-1 預測
                'class_name': str,
                'confidence': float,
                'probabilities': np.ndarray,
                'logits': np.ndarray,  # 原始 logits（用於融合）
                'top_k_predictions': [(step_id, step_name, prob), ...],  # top-k 候選
                'timestamp': float
            }
        """
        # 預處理
        clip = self.preprocess_frames(frames)
        
        # 推理
        with torch.no_grad():
            logits = self.model(clip)  # (1, num_classes)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # (num_classes,)
            logits_np = logits.cpu().numpy()[0]  # 保存原始 logits
        
        # 獲取 top-1 預測
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        # 獲取 top-k 預測
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        top_k_predictions = [
            (int(idx), self.class_names[idx], float(probs[idx]))
            for idx in top_k_indices
        ]
        
        return {
            'class_index': pred_idx,
            'class_name': self.class_names[pred_idx],
            'confidence': confidence,
            'probabilities': probs,
            'logits': logits_np,
            'top_k_predictions': top_k_predictions,
            'timestamp': time.time()
        }
    
    def get_smooth_prediction(self, top_k=3):
        """
        使用歷史預測進行平滑（支持 top-k）
        Returns:
            dict: 平滑後的預測結果
        """
        if len(self.prediction_history) == 0:
            return None
        
        # 平均所有歷史預測的機率
        avg_probs = np.mean([p['probabilities'] for p in self.prediction_history], axis=0)
        avg_logits = np.mean([p.get('logits', np.zeros(self.num_classes)) for p in self.prediction_history], axis=0)
        
        pred_idx = np.argmax(avg_probs)
        confidence = avg_probs[pred_idx]
        
        # 獲取 top-k 預測
        top_k_indices = np.argsort(avg_probs)[::-1][:top_k]
        top_k_predictions = [
            (int(idx), self.class_names[idx], float(avg_probs[idx]))
            for idx in top_k_indices
        ]
        
        return {
            'class_index': pred_idx,
            'class_name': self.class_names[pred_idx],
            'confidence': confidence,
            'probabilities': avg_probs,
            'logits': avg_logits,
            'top_k_predictions': top_k_predictions,
            'timestamp': time.time()
        }
    
    def draw_info(self, frame, prediction, fps):
        """
        在幀上繪製預測資訊
        Args:
            frame: BGR image
            prediction: 預測結果字典
            fps: 當前 FPS
        Returns:
            frame: 繪製後的幀
        """
        h, w = frame.shape[:2]
        
        if prediction is None:
            cv2.putText(frame, "Waiting for frames...", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            return frame
        
        # 當前步驟
        class_name = prediction['class_name']
        confidence = prediction['confidence']
        
        # 顏色：高信心度=綠色，低信心度=紅色
        if confidence >= self.confidence_threshold:
            color = (0, 255, 0)  # 綠色
            status = "Confident"
        else:
            color = (0, 165, 255)  # 橘色
            status = "Uncertain"
        
        # 繪製當前步驟
        cv2.putText(frame, f"Current Step: {class_name}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}% ({status})", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 繪製所有類別的機率條
        probs = prediction['probabilities']
        bar_height = 200  # 機率條區域高度
        bar_x = 20
        bar_y_start = h - bar_height + 20
        bar_width = w - 40
        bar_item_height = 20
        
        cv2.putText(frame, "Class Probabilities:", (bar_x, bar_y_start - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        for i, (class_name_i, prob) in enumerate(zip(self.class_names, probs)):
            y = bar_y_start + i * bar_item_height
            
            # 類別名稱
            text = f"{i}: {class_name_i[:25]}"  # 限制長度
            cv2.putText(frame, text, (bar_x, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 機率條背景
            cv2.rectangle(frame, (bar_x + 250, y + 2), (bar_x + bar_width, y + 18),
                         (50, 50, 50), -1)
            
            # 機率條（顏色根據機率值變化）
            filled_width = int((bar_width - 250) * prob)
            bar_color = (0, int(255 * prob), int(255 * (1 - prob)))  # 綠到紅
            cv2.rectangle(frame, (bar_x + 250, y + 2),
                         (bar_x + 250 + filled_width, y + 18),
                         bar_color, -1)
            
            # 機率值
            cv2.putText(frame, f"{prob*100:.1f}%", (bar_x + bar_width + 10, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_id=0, width=640, height=480, fps_target=30):
        """
        運行即時識別
        Args:
            camera_id: 相機 ID（0 為 D455, 1 為 D435i）
            width: 影像寬度
            height: 影像高度
            fps_target: 目標 FPS
        """
        # 配置 RealSense
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 獲取連接的設備
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("❌ 未找到 RealSense 相機！")
            return
        
        print(f"\n找到 {len(devices)} 個 RealSense 設備：")
        for i, dev in enumerate(devices):
            print(f"  [{i}] {dev.get_info(rs.camera_info.name)} (SN: {dev.get_info(rs.camera_info.serial_number)})")
        
        # 選擇設備
        if camera_id >= len(devices):
            print(f"⚠️  相機 ID {camera_id} 不存在，使用第一個設備")
            camera_id = 0
        
        selected_device = devices[camera_id]
        serial_number = selected_device.get_info(rs.camera_info.serial_number)
        camera_name = selected_device.get_info(rs.camera_info.name)
        
        print(f"\n使用相機: {camera_name} (SN: {serial_number})")
        
        # 啟用彩色流
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps_target)
        
        # 啟動管道
        print("啟動相機...")
        pipeline.start(config)
        
        print("\n" + "="*70)
        print("🎥 即時步驟識別系統已啟動！")
        print("="*70)
        print("控制：")
        print("  - 按 'q' 或 ESC 退出")
        print("  - 按 'r' 重置幀緩衝區")
        print("  - 按 's' 截圖")
        print("="*70 + "\n")
        
        frame_count = 0
        screenshot_count = 0
        
        try:
            while True:
                # 獲取幀
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # 轉換為 numpy 陣列
                color_image = np.asanyarray(color_frame.get_data())
                
                # 加入幀緩衝區
                self.frame_buffer.append(color_image.copy())
                
                # 當緩衝區有足夠的幀時進行預測
                prediction = None
                if len(self.frame_buffer) >= self.clip_len:
                    # 從緩衝區均勻採樣 clip_len 幀
                    indices = np.linspace(0, len(self.frame_buffer) - 1, self.clip_len, dtype=int)
                    sampled_frames = [self.frame_buffer[i] for i in indices]
                    
                    # 預測
                    prediction = self.predict(sampled_frames)
                    self.prediction_history.append(prediction)
                    
                    # 使用平滑預測
                    prediction = self.get_smooth_prediction()
                
                # 計算 FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
                self.fps_buffer.append(fps)
                avg_fps = np.mean(self.fps_buffer)
                self.last_time = current_time
                
                # 繪製資訊
                display_frame = self.draw_info(color_image, prediction, avg_fps)
                
                # 顯示
                cv2.imshow('Real-time Step Recognition', display_frame)
                
                # 處理按鍵
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' 或 ESC
                    print("\n退出...")
                    break
                elif key == ord('r'):  # 重置
                    print("重置幀緩衝區...")
                    self.frame_buffer.clear()
                    self.prediction_history.clear()
                elif key == ord('s'):  # 截圖
                    screenshot_path = f'screenshot_{screenshot_count:03d}.png'
                    cv2.imwrite(screenshot_path, display_frame)
                    print(f"截圖已保存: {screenshot_path}")
                    screenshot_count += 1
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n收到中斷信號，退出...")
        
        finally:
            # 停止管道
            pipeline.stop()
            cv2.destroyAllWindows()
            print(f"\n總共處理了 {frame_count} 幀")
            print("程式結束")


def main():
    parser = argparse.ArgumentParser(description='Real-time Step Recognition with RealSense')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/final_model.pth.tar',
                       help='模型檢查點路徑')
    parser.add_argument('--camera', type=int, default=0,
                       help='相機 ID (0=第一個設備, 1=第二個設備)')
    parser.add_argument('--width', type=int, default=1280,
                       help='影像寬度')
    parser.add_argument('--height', type=int, default=720,
                       help='影像高度')
    parser.add_argument('--fps', type=int, default=30,
                       help='目標 FPS')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='運算設備')
    parser.add_argument('--clip-len', type=int, default=16,
                       help='每次預測使用的幀數')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='信心度閾值')
    
    args = parser.parse_args()
    
    # 檢查檢查點是否存在
    if not Path(args.checkpoint).exists():
        print(f"❌ 找不到檢查點檔案: {args.checkpoint}")
        return
    
    # 創建識別器
    recognizer = RealtimeStepRecognizer(
        checkpoint_path=args.checkpoint,
        device=args.device,
        clip_len=args.clip_len,
        confidence_threshold=args.confidence
    )
    
    # 運行
    recognizer.run(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps_target=args.fps
    )


if __name__ == '__main__':
    main()
