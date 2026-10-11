"""
Hand Gesture Intent Classification Model
MLP-based classifier for hand gesture intent recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HandGestureClassifier(nn.Module):
    """
    MLP model for hand gesture intent classification
    
    Classes:
    0 = idle
    1 = stop
    2 = receive  
    3 = pointing
    """
    
    def __init__(self, input_dim=68, num_classes=4):
        """
        Args:
            input_dim: Dimension of input features (default 68)
            num_classes: Number of output classes (default 4)
        """
        super(HandGestureClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        self.input_dim = input_dim
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class HandGestureModel:
    """
    Inference wrapper for hand gesture intent classification
    Handles feature extraction, prediction, and confidence thresholding
    """
    
    def __init__(self, model, feature_extractor, class_names, device='cpu', confidence_threshold=0.6):
        """
        Args:
            model: Trained HandGestureClassifier
            feature_extractor: HandFeatureExtractor instance
            class_names: List of class names
            device: torch device
            confidence_threshold: Minimum confidence to accept prediction (default 0.6)
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.class_names = class_names
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        self.model.to(device)
        self.model.eval()
    
    def predict_from_image(self, image_path):
        """
        Predict intent from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with keys:
                - class_idx: int (predicted class index)
                - class_name: str (predicted class name)
                - confidence: float (prediction confidence)
                - all_probs: dict mapping class names to probabilities
        """
        # Extract features
        keypoints = self.feature_extractor.extract_keypoints(image_path)
        features = self.feature_extractor.extract_features_from_hand(keypoints)
        
        if features is None:
            # Return idle with low confidence if detection fails
            return {
                'class_idx': 0,
                'class_name': 'idle',
                'confidence': 0.0,
                'all_probs': {name: 0.0 for name in self.class_names}
            }
        
        return self.predict_from_features(features)

    def predict_from_bgr(self, image_bgr):
        """
        Predict intent directly from BGR numpy image (避免寫檔 IO)
        """
        keypoints = self.feature_extractor.extract_keypoints_from_array(image_bgr)
        features = self.feature_extractor.extract_features_from_hand(keypoints)
        
        if features is None:
            return {
                'class_idx': 0,
                'class_name': 'idle',
                'confidence': 0.0,
                'all_probs': {name: 0.0 for name in self.class_names}
            }
        
        return self.predict_from_features(features)
    
    def predict_from_keypoints(self, keypoints):
        """
        Predict intent from keypoints
        
        Args:
            keypoints: np.ndarray of shape [21, 3] (hand landmarks)
            
        Returns:
            dict (same as predict_from_image)
        """
        features = self.feature_extractor.extract_features_from_hand(keypoints)
        
        if features is None:
            return {
                'class_idx': 0,
                'class_name': 'idle',
                'confidence': 0.0,
                'all_probs': {name: 0.0 for name in self.class_names}
            }
        
        return self.predict_from_features(features)
    
    def predict_from_features(self, features):
        """
        Predict intent from feature vector
        
        Args:
            features: np.ndarray of shape [feature_dim]
            
        Returns:
            dict (same as predict_from_image)
        """
        # Convert to tensor
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probs, dim=1)
            
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            # Force to idle if confidence too low
            predicted_idx = 0  # idle
            predicted_name = 'idle'
        else:
            predicted_name = self.class_names[predicted_idx]
        
        # Get all probabilities
        all_probs = {name: probs[0, i].item() for i, name in enumerate(self.class_names)}
        
        return {
            'class_idx': predicted_idx,
            'class_name': predicted_name,
            'confidence': confidence,
            'all_probs': all_probs
        }
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, feature_extractor, device='cpu', confidence_threshold=0.6):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            feature_extractor: HandFeatureExtractor instance
            device: torch device
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            HandGestureModel instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model
        model = HandGestureClassifier(
            input_dim=checkpoint['input_dim'],
            num_classes=checkpoint['num_classes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        class_names = checkpoint['class_names']
        
        return cls(model, feature_extractor, class_names, device, confidence_threshold)


if __name__ == "__main__":
    # Test model architecture
    print("Testing HandGestureClassifier...")
    
    model = HandGestureClassifier(input_dim=68, num_classes=4)
    print(f"Model: {model}")
    
    # Test forward pass
    batch_size = 4
    input_dim = 68
    x = torch.randn(batch_size, input_dim)
    
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test probabilities
    probs = F.softmax(output, dim=1)
    print(f"Probabilities:\n{probs}")
    print(f"Sum of probs: {probs.sum(dim=1)}")
