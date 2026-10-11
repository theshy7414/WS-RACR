"""
Temporal Smoothing and State Machine for Intent Recognition
Provides stable intent and safety states using multi-frame smoothing
"""

from collections import deque
import numpy as np


class IntentSmoothingStateMachine:
    """
    Temporal smoothing state machine for stable intent recognition
    
    Provides:
    - Multi-frame smoothing
    - Safety state (EMERGENCY_STOP detection)
    - Intent state (stable gesture recognition)
    """
    
    def __init__(self, 
                 window_size=10,
                 stop_threshold=0.7,
                 stop_min_count=5,
                 receive_threshold=0.7,
                 receive_min_count=5,
                 pointing_threshold=0.7,
                 pointing_min_count=5):
        """
        Args:
            window_size: Number of frames to keep in history
            stop_threshold: Confidence threshold for stop gesture
            stop_min_count: Minimum frames needed to trigger EMERGENCY_STOP
            receive_threshold: Confidence threshold for receive gesture
            receive_min_count: Minimum frames needed for receive state
            pointing_threshold: Confidence threshold for pointing gesture
            pointing_min_count: Minimum frames needed for pointing state
        """
        self.window_size = window_size
        self.stop_threshold = stop_threshold
        self.stop_min_count = stop_min_count
        self.receive_threshold = receive_threshold
        self.receive_min_count = receive_min_count
        self.pointing_threshold = pointing_threshold
        self.pointing_min_count = pointing_min_count
        
        # History buffer: stores (class_idx, confidence) tuples
        self.history = deque(maxlen=window_size)
        
        # Class name mapping (must match dataset)
        self.class_names = ['idle', 'stop', 'receive', 'pointing']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Current states
        self.safe_state = "OK"
        self.intent_state = "idle"
    
    def update(self, prediction):
        """
        Update state machine with new prediction
        
        Args:
            prediction: dict with keys:
                - class_idx: int
                - class_name: str
                - confidence: float
                
        Returns:
            dict with keys:
                - safe_state: "OK" | "EMERGENCY_STOP"
                - intent_state: "idle" | "stop" | "receive" | "pointing"
                - raw_class: str (current frame prediction)
                - raw_conf: float (current frame confidence)
        """
        # Add to history
        self.history.append((prediction['class_idx'], prediction['confidence']))
        
        # Update safety state (STOP detection)
        self._update_safe_state()
        
        # Update intent state
        self._update_intent_state()
        
        return {
            'safe_state': self.safe_state,
            'intent_state': self.intent_state,
            'raw_class': prediction['class_name'],
            'raw_conf': prediction['confidence']
        }
    
    def _update_safe_state(self):
        """Update safety state based on stop gesture detection"""
        stop_idx = self.class_to_idx['stop']
        
        # Count frames with stop gesture above threshold
        stop_count = sum(
            1 for class_idx, conf in self.history
            if class_idx == stop_idx and conf >= self.stop_threshold
        )
        
        # Update safety state
        if stop_count >= self.stop_min_count:
            self.safe_state = "EMERGENCY_STOP"
        else:
            self.safe_state = "OK"
    
    def _update_intent_state(self):
        """Update intent state based on gesture detection"""
        receive_idx = self.class_to_idx['receive']
        pointing_idx = self.class_to_idx['pointing']
        
        # Count frames for each gesture
        receive_count = sum(
            1 for class_idx, conf in self.history
            if class_idx == receive_idx and conf >= self.receive_threshold
        )
        
        pointing_count = sum(
            1 for class_idx, conf in self.history
            if class_idx == pointing_idx and conf >= self.pointing_threshold
        )
        
        # Priority: receive > pointing > idle
        # (receive is more important for handover tasks)
        if receive_count >= self.receive_min_count:
            self.intent_state = "receive"
        elif pointing_count >= self.pointing_min_count:
            self.intent_state = "pointing"
        else:
            self.intent_state = "idle"
    
    def reset(self):
        """Reset state machine"""
        self.history.clear()
        self.safe_state = "OK"
        self.intent_state = "idle"
    
    def get_history_summary(self):
        """Get summary of current history buffer"""
        if not self.history:
            return {
                'history_size': 0,
                'class_counts': {},
                'avg_confidence': 0.0
            }
        
        class_counts = {}
        for class_idx, conf in self.history:
            name = self.class_names[class_idx]
            class_counts[name] = class_counts.get(name, 0) + 1
        
        avg_conf = np.mean([conf for _, conf in self.history])
        
        return {
            'history_size': len(self.history),
            'class_counts': class_counts,
            'avg_confidence': float(avg_conf)
        }


if __name__ == "__main__":
    # Test state machine
    print("Testing IntentSmoothingStateMachine...")
    
    sm = IntentSmoothingStateMachine(
        window_size=10,
        stop_threshold=0.7,
        stop_min_count=5
    )
    
    print("\n1. Testing STOP detection:")
    for i in range(6):
        pred = {'class_idx': 1, 'class_name': 'stop', 'confidence': 0.8}
        result = sm.update(pred)
        print(f"  Frame {i+1}: safe_state={result['safe_state']}, intent_state={result['intent_state']}")
    
    print("\n2. Reset and test receive detection:")
    sm.reset()
    for i in range(6):
        pred = {'class_idx': 2, 'class_name': 'receive', 'confidence': 0.75}
        result = sm.update(pred)
        print(f"  Frame {i+1}: safe_state={result['safe_state']}, intent_state={result['intent_state']}")
    
    print("\n3. Test mixed signals:")
    sm.reset()
    predictions = [
        {'class_idx': 0, 'class_name': 'idle', 'confidence': 0.9},
        {'class_idx': 0, 'class_name': 'idle', 'confidence': 0.85},
        {'class_idx': 3, 'class_name': 'pointing', 'confidence': 0.8},
        {'class_idx': 3, 'class_name': 'pointing', 'confidence': 0.75},
        {'class_idx': 3, 'class_name': 'pointing', 'confidence': 0.8},
        {'class_idx': 3, 'class_name': 'pointing', 'confidence': 0.85},
        {'class_idx': 3, 'class_name': 'pointing', 'confidence': 0.8},
        {'class_idx': 0, 'class_name': 'idle', 'confidence': 0.9},
    ]
    
    for i, pred in enumerate(predictions):
        result = sm.update(pred)
        print(f"  Frame {i+1} ({pred['class_name']}): intent_state={result['intent_state']}")
    
    print("\n4. History summary:")
    summary = sm.get_history_summary()
    print(f"  {summary}")
