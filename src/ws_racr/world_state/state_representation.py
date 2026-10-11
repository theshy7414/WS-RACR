"""
World State Representation
Defines the unified world state JSON schema
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class SafetyState(Enum):
    """Safety state enumeration"""
    OK = "OK"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    WARNING = "WARNING"


class InteractionState(Enum):
    """High-level human-robot interaction state, derived from gesture + robot gripper.

    This is a semantic interpretation layer above raw gesture classification,
    designed for paper narrative: LLM reasons over interaction_state, not raw intent.
    """
    OBSERVING = "observing"                      # idle, no active collaboration
    HUMAN_REQUESTING_TOOL = "human_requesting_tool"   # receive gesture + gripper empty
    HUMAN_READY_FOR_HANDOVER = "human_ready_for_handover"  # receive gesture + gripper holding
    HUMAN_POINTING_TARGET = "human_pointing_target"   # pointing gesture active
    HUMAN_REQUESTING_STOP = "human_requesting_stop"   # stop gesture


class IntentState(Enum):
    """Human intent state"""
    IDLE = "idle"
    RECEIVE = "receive"
    POINTING = "pointing"
    UNKNOWN = "unknown"


class VisibilityState(Enum):
    """Object visibility state (區分真實消失 vs 暫時遮擋)"""
    VISIBLE = "visible"        # 當前可見
    OCCLUDED = "occluded"      # 暫時遮擋（預期仍存在）
    MISSING = "missing"        # 真實消失（已被拿走/移出場景）


class RobotState(Enum):
    """Robot execution state"""
    IDLE = "idle"
    MOVING = "moving"
    GRASPING = "grasping"
    RELEASING = "releasing"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class ObjectState:
    """State of a single object"""
    object_type: str  # e.g., "cylinder", "platform", etc.
    object_id: str    # e.g., "cylinder_1"
    
    # Raw detection data
    location: str     # e.g., "in_tray", "in_hand", "on_platform", "missing" (raw inference)
    position: Optional[List[float]] = None  # [x, y, z] in robot base frame (raw)
    confidence: float = 0.0
    last_seen: Optional[float] = None  # timestamp
    
    # Stabilized data (hysteresis + occlusion handling)
    location_stable: Optional[str] = None  # Stable location after hysteresis
    visibility: str = VisibilityState.VISIBLE.value  # visible/occluded/missing
    location_confidence: float = 0.0  # Confidence in location inference
    
    # Occlusion tracking
    occluded_by: Optional[str] = None  # object_id of likely occluder
    occluded_by_confidence: float = 0.0  # 對遮擋物判斷的信心度
    
    # Semantic Location (Robot IO Truth) ===
    semantic_location: str = "unknown"  # 'in_gripper' / 'on_table' / 'unknown'
    forced_by_io: bool = False  # True = 被 Robot IO 強制設定（優先級最高）
    last_vision_position: Optional[List[float]] = None  # 最後一次視覺檢測的位置（用於 freeze）
    
    # Source tracking
    source: Optional[str] = None  # "top_view", "front_view", "fused", "robot_io"
    
    # Hysteresis tracking (internal)
    _location_history: List[str] = field(default_factory=list)  # 最近N次location
    _visibility_counter: int = 0  # 連續不可見計數


@dataclass
class AssemblyProgress:
    """Assembly progress tracking"""
    cylinders_inserted: int = 0
    cylinders_screwed: int = 0
    bracket_ready: bool = False
    pi_ready: bool = False
    screws_small_used: int = 0
    screws_big_upper_used: int = 0
    screws_big_lower_used: int = 0
    current_phase: int = 0  # 0=not started, 1/2/3


@dataclass
class HumanState:
    """Human operator state"""
    # Raw detection
    intent: str = IntentState.IDLE.value
    safety: str = SafetyState.OK.value
    confidence: float = 0.0
    gesture_raw: Optional[str] = None
    pointing_target: Optional[str] = None
    pointing_position: Optional[List[float]] = None   # [x, y, z] metres, world-frame
    pointing_pixel: Optional[List[int]] = None        # [cx, cy] top-camera pixel from pointing detection

    # Stabilized state
    intent_stable: Optional[str] = None
    safety_stable: Optional[str] = None
    handover_ready: bool = False  # Explicit flag (不只靠邊存在)
    
    # Timing
    last_intent_change_time: Optional[float] = None
    pointing_target_confidence: float = 0.0


@dataclass
class RobotStatus:
    """Robot status"""
    state: str = RobotState.IDLE.value
    current_action: Optional[str] = None
    last_action: Optional[str] = None
    action_success: Optional[bool] = None
    
    # Truth from IO (TM5 DI signals)
    holding_object_id: Optional[str] = None  # 由 DI + grasp outcome 決定
    gripper_state: Optional[str] = None  # "open" or "closed" (from DI)
    last_skill_result: Optional[str] = None  # "success", "failed", "timeout"
    failure_code: Optional[str] = None  # Specific failure reason


@dataclass
class WorldState:
    """
    Complete world state representation
    Aggregates all perception and state information
    """
    # Timestamp
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    # Step recognition
    step_class: str = "idle"
    step_confidence: float = 0.0
    phase: int = 0
    
    # Object states
    objects: Dict[str, ObjectState] = field(default_factory=dict)
    
    # Assembly progress
    assembly_progress: AssemblyProgress = field(default_factory=AssemblyProgress)
    
    # Human state
    human: HumanState = field(default_factory=HumanState)
    
    # Robot state
    robot: RobotStatus = field(default_factory=RobotStatus)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json_compatible(self) -> Dict:
        """Convert to JSON-compatible dictionary (處理 NumPy 類型)"""
        import numpy as np
        
        def convert_value(obj):
            """遞迴轉換 NumPy 類型為 Python 原生類型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_value(item) for item in obj]
            else:
                return obj
        
        data = self.to_dict()
        # Convert NumPy types to native Python types
        data = convert_value(data)
        # Convert timestamp to ISO format for readability
        data['timestamp_iso'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return data
    
    def get_object(self, object_id: str) -> Optional[ObjectState]:
        """Get object by ID"""
        return self.objects.get(object_id)
    
    def update_object(self, obj: ObjectState):
        """Update or add object"""
        self.objects[obj.object_id] = obj
    
    def get_objects_by_type(self, object_type: str) -> List[ObjectState]:
        """Get all objects of a specific type"""
        return [obj for obj in self.objects.values() if obj.object_type == object_type]
    
    def get_objects_at_location(self, location: str) -> List[ObjectState]:
        """Get all objects at a specific location"""
        return [obj for obj in self.objects.values() if obj.location == location]
    
    @property
    def interaction_state(self) -> str:
        """Computed semantic interaction state.

        Derived from human.intent (gesture layer) combined with robot gripper state.
        This is what LLM reasoning should consume — not raw gesture labels.

        Returns:
            InteractionState value string
        """
        intent = self.human.intent
        gripper = self.robot.gripper_state  # "open", "closed", or None

        if intent == IntentState.RECEIVE.value:
            if gripper == "closed":
                return InteractionState.HUMAN_READY_FOR_HANDOVER.value
            else:
                return InteractionState.HUMAN_REQUESTING_TOOL.value
        elif intent == IntentState.POINTING.value:
            return InteractionState.HUMAN_POINTING_TARGET.value
        elif intent == "stop":
            return InteractionState.HUMAN_REQUESTING_STOP.value
        else:
            return InteractionState.OBSERVING.value

    @property
    def safety_status(self) -> str:
        """Explicit safety status field for paper Layer 1 output contract.

        Maps human.safety (raw perception) → SafetyState enum value.
        Consumers should use this field rather than reading human.safety directly.

        Returns:
            SafetyState value string
        """
        return self.human.safety

    def is_safe(self) -> bool:
        """Check if system is in safe state"""
        return self.human.safety == SafetyState.OK.value

    def requires_emergency_stop(self) -> bool:
        """Check if emergency stop is required"""
        return self.human.safety == SafetyState.EMERGENCY_STOP.value
    
    def get_phase_name(self) -> str:
        """Get human-readable phase name"""
        phase_names = {
            0: "Not Started",
            1: "Phase 1: Assemble Platform Base",
            2: "Phase 2: Attach Bracket",
            3: "Phase 3: Attach Pi Component"
        }
        return phase_names.get(self.phase, "Unknown Phase")
    
    def summary(self) -> str:
        """Get human-readable summary"""
        return f"""
World State Summary:
  Time: {datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
  Phase: {self.get_phase_name()}
  Step: {self.step_class} (conf: {self.step_confidence:.2%})
  
  Objects: {len(self.objects)} detected
  Assembly Progress:
    - Cylinders: {self.assembly_progress.cylinders_inserted}/4 inserted, {self.assembly_progress.cylinders_screwed}/4 screwed
    - Bracket: {'Ready' if self.assembly_progress.bracket_ready else 'Not ready'}
    - Pi: {'Ready' if self.assembly_progress.pi_ready else 'Not ready'}
  
  Human:
    - Intent: {self.human.intent}
    - Interaction: {self.interaction_state}
    - Safety: {self.safety_status}
    {'- ⚠️  EMERGENCY STOP REQUIRED!' if self.requires_emergency_stop() else ''}
  
  Robot:
    - State: {self.robot.state}
    - Current Action: {self.robot.current_action or 'None'}
"""


def create_empty_world_state() -> WorldState:
    """Create an empty world state"""
    return WorldState()


def create_world_state_from_dict(data: Dict) -> WorldState:
    """Create WorldState from dictionary"""
    # Extract nested objects
    objects = {}
    if 'objects' in data:
        for obj_id, obj_data in data['objects'].items():
            objects[obj_id] = ObjectState(**obj_data)
    
    assembly = AssemblyProgress()
    if 'assembly_progress' in data:
        assembly = AssemblyProgress(**data['assembly_progress'])
    
    human = HumanState()
    if 'human' in data:
        human = HumanState(**data['human'])
    
    robot = RobotStatus()
    if 'robot' in data:
        robot = RobotStatus(**data['robot'])
    
    return WorldState(
        timestamp=data.get('timestamp', datetime.now().timestamp()),
        step_class=data.get('step_class', 'idle'),
        step_confidence=data.get('step_confidence', 0.0),
        phase=data.get('phase', 0),
        objects=objects,
        assembly_progress=assembly,
        human=human,
        robot=robot,
        metadata=data.get('metadata', {})
    )
