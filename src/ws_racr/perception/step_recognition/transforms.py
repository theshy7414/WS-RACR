#!/usr/bin/env python3
"""
Video Transforms for Data Augmentation
"""
import cv2
import numpy as np
import random


class VideoTransform:
    """
    影片數據增強
    
    Args:
        input_size: 輸入圖像大小 (height, width)
        mean: RGB 均值
        std: RGB 標準差
        augment: 是否應用增強（訓練時）
    """
    
    def __init__(
        self,
        input_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        augment=True
    ):
        self.input_size = input_size
        self.mean = np.array(mean).reshape(1, 1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 1, 3)
        self.augment = augment
        
    def __call__(self, frames, is_train=True):
        """
        應用變換
        
        Args:
            frames: numpy array of shape (T, H, W, 3), uint8, range [0, 255]
            is_train: 是否為訓練模式
        
        Returns:
            frames: numpy array of shape (T, H, W, 3), float32, normalized
        """
        # 1. Resize（短邊縮放到 256）
        frames = self._resize(frames, short_side=256)
        
        # 2. Random/Center crop to input_size
        if is_train and self.augment:
            frames = self._random_crop(frames, self.input_size)
        else:
            frames = self._center_crop(frames, self.input_size)
        
        # 3. Random horizontal flip (訓練時)
        if is_train and self.augment and random.random() < 0.5:
            frames = self._horizontal_flip(frames)
        
        # 4. Color jitter (訓練時)
        if is_train and self.augment:
            frames = self._color_jitter(frames)
        
        # 5. Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # 6. Standardize (ImageNet mean/std)
        frames = (frames - self.mean) / self.std
        
        return frames
    
    def _resize(self, frames, short_side=256):
        """
        短邊縮放
        
        Args:
            frames: (T, H, W, 3)
            short_side: 短邊目標大小
        
        Returns:
            resized_frames: (T, H', W', 3)
        """
        T, H, W, C = frames.shape
        
        # 計算縮放比例
        if H < W:
            new_H = short_side
            new_W = int(W * (short_side / H))
        else:
            new_W = short_side
            new_H = int(H * (short_side / W))
        
        # Resize 每一幀
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized)
        
        return np.array(resized_frames)
    
    def _random_crop(self, frames, size):
        """
        隨機裁剪
        
        Args:
            frames: (T, H, W, 3)
            size: (crop_h, crop_w)
        
        Returns:
            cropped_frames: (T, crop_h, crop_w, 3)
        """
        T, H, W, C = frames.shape
        crop_h, crop_w = size
        
        # 隨機起始點
        top = random.randint(0, H - crop_h) if H > crop_h else 0
        left = random.randint(0, W - crop_w) if W > crop_w else 0
        
        # 裁剪
        cropped = frames[:, top:top+crop_h, left:left+crop_w, :]
        
        return cropped
    
    def _center_crop(self, frames, size):
        """
        中心裁剪
        
        Args:
            frames: (T, H, W, 3)
            size: (crop_h, crop_w)
        
        Returns:
            cropped_frames: (T, crop_h, crop_w, 3)
        """
        T, H, W, C = frames.shape
        crop_h, crop_w = size
        
        # 中心起始點
        top = (H - crop_h) // 2 if H > crop_h else 0
        left = (W - crop_w) // 2 if W > crop_w else 0
        
        # 裁剪
        cropped = frames[:, top:top+crop_h, left:left+crop_w, :]
        
        return cropped
    
    def _horizontal_flip(self, frames):
        """
        水平翻轉
        
        Args:
            frames: (T, H, W, 3)
        
        Returns:
            flipped_frames: (T, H, W, 3)
        """
        return frames[:, :, ::-1, :].copy()
    
    def _color_jitter(self, frames, brightness=0.2, contrast=0.2, saturation=0.2):
        """
        顏色抖動
        
        Args:
            frames: (T, H, W, 3), uint8
            brightness: 亮度抖動範圍
            contrast: 對比度抖動範圍
            saturation: 飽和度抖動範圍
        
        Returns:
            jittered_frames: (T, H, W, 3), uint8
        """
        # 隨機調整參數
        brightness_factor = 1.0 + random.uniform(-brightness, brightness)
        contrast_factor = 1.0 + random.uniform(-contrast, contrast)
        saturation_factor = 1.0 + random.uniform(-saturation, saturation)
        
        jittered_frames = []
        
        for frame in frames:
            # 轉換到 HSV 進行飽和度調整
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # 亮度和對比度調整
            frame = frame.astype(np.float32)
            frame = frame * contrast_factor + (brightness_factor - 1.0) * 128
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            jittered_frames.append(frame)
        
        return np.array(jittered_frames)
