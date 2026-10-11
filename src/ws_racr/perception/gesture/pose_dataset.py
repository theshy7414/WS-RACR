"""
Hand Gesture Dataset Module for Intent Classification
Extracts MediaPipe Hands features from images and creates dataset for training
"""

import logging
import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from torch.utils.data import Dataset
import torch

logger = logging.getLogger(__name__)


def imread_unicode(path):
    """OpenCV imread 在 Windows 無法讀含中文等非 ASCII 路徑。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class HandFeatureExtractor:
    """Extract hand gesture features from images using MediaPipe Hands"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,  # Only detect one hand (right hand)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe Hand has 21 landmarks
        # Key landmarks for gesture recognition:
        self.key_landmarks = {
            'WRIST': 0,
            'THUMB_CMC': 1, 'THUMB_MCP': 2, 'THUMB_IP': 3, 'THUMB_TIP': 4,
            'INDEX_MCP': 5, 'INDEX_PIP': 6, 'INDEX_DIP': 7, 'INDEX_TIP': 8,
            'MIDDLE_MCP': 9, 'MIDDLE_PIP': 10, 'MIDDLE_DIP': 11, 'MIDDLE_TIP': 12,
            'RING_MCP': 13, 'RING_PIP': 14, 'RING_DIP': 15, 'RING_TIP': 16,
            'PINKY_MCP': 17, 'PINKY_PIP': 18, 'PINKY_DIP': 19, 'PINKY_TIP': 20
        }
    
    def extract_keypoints(self, image_path):
        """
        Extract RIGHT hand keypoints from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            keypoints: np.ndarray of shape [21, 3] (x, y, z) or None if detection fails
        """
        image = imread_unicode(image_path)
        if image is None:
            return None
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Find RIGHT hand specifically
        # MediaPipe 的標籤是鏡像的(從攝影機視角):
        # "Right" = 圖片右側 = 使用者的左手
        # "Left" = 圖片左側 = 使用者的右手 
        right_hand_idx = None
        if results.multi_handedness:
            for idx, hand_info in enumerate(results.multi_handedness):
                label = hand_info.classification[0].label
                # 要抓使用者的右手,所以找 MediaPipe 標記為 "Left" 的
                if label == "Left":
                    right_hand_idx = idx
                    break
        
        # If no right hand detected, use the first hand as fallback
        if right_hand_idx is None:
            if len(results.multi_hand_landmarks) > 0:
                right_hand_idx = 0
            else:
                return None
        
        # Get right hand landmarks
        hand_landmarks = results.multi_hand_landmarks[right_hand_idx]
        
        # Extract landmarks (21 points with x, y, z)
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        return keypoints

    def extract_keypoints_from_array(self, image_bgr):
        """
        Extract RIGHT hand keypoints from a BGR numpy image (避免寫檔 IO)
        """
        if image_bgr is None:
            return None

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Process image
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        right_hand_idx = None
        if results.multi_handedness:
            for idx, hand_info in enumerate(results.multi_handedness):
                label = hand_info.classification[0].label
                if label == "Left":  # MediaPipe 鏡像，"Left" = 使用者右手
                    right_hand_idx = idx
                    break
        if right_hand_idx is None:
            if len(results.multi_hand_landmarks) > 0:
                right_hand_idx = 0
            else:
                return None
        
        hand_landmarks = results.multi_hand_landmarks[right_hand_idx]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        return keypoints
    
    def extract_features_from_hand(self, keypoints):
        """
        Extract normalized features from hand keypoints
        
        Args:
            keypoints: np.ndarray of shape [21, 3] (x, y, z)
            
        Returns:
            feature_vector: 1D numpy array containing normalized features
            Total features: 63 (coords) + 5 (extensions) = 68 dimensions
        """
        if keypoints is None:
            return None
        
        # Get wrist as reference point
        wrist = keypoints[0, :]  # WRIST
        
        # Compute hand size (distance from wrist to middle finger MCP)
        middle_mcp = keypoints[9, :]  # MIDDLE_MCP
        hand_size = np.linalg.norm(middle_mcp - wrist)
        
        if hand_size < 1e-6:  # Avoid division by zero
            return None
        
        # Normalize all keypoints relative to wrist and hand size
        normalized_kp = (keypoints - wrist) / hand_size
        
        # Flatten all normalized coordinates (21 points × 3 coords = 63)
        features = list(normalized_kp.flatten())
        
        # Add finger extension features (distance from fingertip to wrist vs MCP to wrist)
        finger_tips = [4, 8, 12, 16, 20]  # THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP
        finger_mcps = [2, 5, 9, 13, 17]   # MCP joints for each finger
        
        for tip_idx, mcp_idx in zip(finger_tips, finger_mcps):
            tip_dist = np.linalg.norm(keypoints[tip_idx] - wrist)
            mcp_dist = np.linalg.norm(keypoints[mcp_idx] - wrist)
            # Extension ratio: > 1 means extended, < 1 means bent
            extension = tip_dist / (mcp_dist + 1e-6)
            features.append(extension)
        
        # Total: 63 + 5 = 68 features
        return np.array(features, dtype=np.float32)
    
    def __del__(self):
        self.hands.close()


class HandGestureDataset(Dataset):
    """
    Dataset for hand gesture intent classification
    
    Classes based on actual dataset:
    0 = idle
    1 = stop
    2 = receive
    3 = pointing
    """
    
    def __init__(self, dataset_dir, feature_extractor=None, cache_features=True):
        """
        Args:
            dataset_dir: Path to dataset directory containing images
            feature_extractor: HandFeatureExtractor instance (creates new if None)
            cache_features: Whether to cache extracted features
        """
        self.dataset_dir = Path(dataset_dir)
        self.feature_extractor = feature_extractor or HandFeatureExtractor()
        self.cache_features = cache_features
        
        # Class mapping based on actual dataset
        self.class_names = ['idle', 'stop', 'receive', 'pointing']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load dataset
        self.samples = []
        self.features_cache = {} if cache_features else None
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all images and their labels"""
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            
            # Find all images for this class
            pattern = f"{class_name}_*.png"
            image_files = sorted(self.dataset_dir.glob(pattern))
            
            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))
        
        logger.info(f"Loaded {len(self.samples)} samples")
        for class_name in self.class_names:
            count = sum(1 for _, idx in self.samples if idx == self.class_to_idx[class_name])
            logger.info(f"  {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            feature_tensor: torch.Tensor of extracted features
            label: int (class index)
        """
        img_path, label = self.samples[idx]
        
        # Check cache
        if self.cache_features and img_path in self.features_cache:
            features = self.features_cache[img_path]
        else:
            # Create feature extractor per request to avoid multiprocessing issues
            if self.feature_extractor is None:
                self.feature_extractor = HandFeatureExtractor()
            
            # Extract features
            keypoints = self.feature_extractor.extract_keypoints(img_path)
            features = self.feature_extractor.extract_features_from_hand(keypoints)
            
            # Cache if enabled
            if self.cache_features and features is not None:
                self.features_cache[img_path] = features
        
        # Handle failed extraction
        if features is None:
            # Return zero features for failed detections
            # Feature dimension: 63 (coords) + 5 (extensions) = 68
            features = np.zeros(68, dtype=np.float32)
        
        return torch.from_numpy(features), label
    
    def get_class_name(self, idx):
        """Get class name from index"""
        return self.class_names[idx]


def create_datasets(dataset_dir, train_ratio=0.8, seed=42):
    """
    Create train and validation datasets with stratified split
    
    Args:
        dataset_dir: Path to dataset directory
        train_ratio: Ratio of training data
        seed: Random seed for reproducibility
        
    Returns:
        train_dataset, val_dataset
    """
    from torch.utils.data import Subset
    
    # Create full dataset
    full_dataset = HandGestureDataset(dataset_dir)
    
    # Stratified split
    np.random.seed(seed)
    
    train_indices = []
    val_indices = []
    
    for class_idx in range(len(full_dataset.class_names)):
        # Get all indices for this class
        class_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label == class_idx]
        
        # Shuffle
        np.random.shuffle(class_indices)
        
        # Split
        n_train = int(len(class_indices) * train_ratio)
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:])
    
    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    logger.info(f"\nDataset split:")
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test dataset loading
    dataset_dir = "datasets"
    
    print("Testing dataset loading...")
    dataset = HandGestureDataset(dataset_dir)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Classes: {dataset.class_names}")
    
    # Test first sample
    if len(dataset) > 0:
        features, label = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Features shape: {features.shape}")
        print(f"  Label: {label} ({dataset.get_class_name(label)})")
        print(f"  Feature stats: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}")
    
    # Test train/val split
    print("\n" + "="*50)
    train_ds, val_ds = create_datasets(dataset_dir, train_ratio=0.8)
