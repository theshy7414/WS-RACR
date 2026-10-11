"""Data records shared by the memory stores (enums, step summaries, episodes, semantic rules)."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum
logger = logging.getLogger(__name__)
SPATIAL_CALC_AVAILABLE = True


class HumanStateEnum(str, Enum):
    """人類狀態枚舉"""
    WORKING = "working"
    HANDOVER_READY = "handover_ready"
    STOP = "stop"
    IDLE = "idle"
    UNKNOWN = "unknown"


class SafetyStateEnum(str, Enum):
    """安全狀態枚舉"""
    SAFE = "safe"
    WARNING = "warning"
    STOP = "stop"


class OutcomeEnum(str, Enum):
    """執行結果枚舉"""
    SUCCESS = "success"
    REJECTED = "rejected"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class EventTypeEnum(str, Enum):
    """事件類型枚舉"""
    HANDOVER = "handover"
    ASSIST = "assist"
    WAIT = "wait"
    STOP_RESPONSE = "stop_response"
    FETCH = "fetch"
    REPOSITION = "reposition"


@dataclass
class StepSummary:
    """
    步驟摘要 - Information Preprocessor 輸出
    
    用於記錄當前觀察到的狀態（寫入導向）
    """
    time: datetime
    human_state: HumanStateEnum
    current_step: str
    objects_involved: List[str]
    handover_intent: bool
    safety_state: SafetyStateEnum
    robot_last_action: Optional[str]
    outcome: OutcomeEnum
    
    # 額外資訊
    detected_gesture: Optional[str] = None
    human_position: Optional[List[float]] = None
    robot_position: Optional[List[float]] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {
            'time': self.time.isoformat(),
            'human_state': self.human_state.value,
            'current_step': self.current_step,
            'objects_involved': self.objects_involved,
            'handover_intent': self.handover_intent,
            'safety_state': self.safety_state.value,
            'robot_last_action': self.robot_last_action,
            'outcome': self.outcome.value,
            'detected_gesture': self.detected_gesture,
            'human_position': self.human_position,
            'robot_position': self.robot_position,
            'confidence': self.confidence
        }
    
    def to_text(self) -> str:
        """轉換為文字描述（用於 LLM）"""
        text = f"[{self.time.strftime('%H:%M:%S')}] "
        text += f"Human: {self.human_state.value}, "
        text += f"Step: {self.current_step}, "
        text += f"Objects: {', '.join(self.objects_involved)}, "
        text += f"Safety: {self.safety_state.value}"
        
        if self.robot_last_action:
            text += f", Robot action: {self.robot_last_action} ({self.outcome.value})"
        
        if self.detected_gesture:
            text += f", Gesture: {self.detected_gesture}"
        
        return text


@dataclass
class MemoryQuery:
    """
    記憶查詢 - Information Preprocessor 輸出
    
    用於檢索相關記憶（檢索導向）
    """
    query_text: str
    query_type: str  # "episodic", "semantic", "spatial"
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'query_text': self.query_text,
            'query_type': self.query_type,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class EpisodicMemory:
    """
    情節記憶 - 單次人機互動事件
    
    不同於 RoboMemory 的任務級別，這裡專注於互動事件級別
    """
    event_id: str
    event_type: EventTypeEnum
    
    # Context
    step: str
    tool: Optional[str]
    zone: Optional[str]
    human_state: HumanStateEnum
    
    # Action & Response
    robot_action: str
    human_response: str  # "accepted", "rejected", "ignored", "interrupted"
    outcome: OutcomeEnum
    
    # Timing
    timestamp: datetime
    duration_seconds: float
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'step': self.step,
            'tool': self.tool,
            'zone': self.zone,
            'human_state': self.human_state.value,
            'robot_action': self.robot_action,
            'human_response': self.human_response,
            'outcome': self.outcome.value,
            'timestamp': self.timestamp.isoformat(),
            'duration_seconds': self.duration_seconds,
            'metadata': self.metadata
        }


@dataclass
class StepTransitionMemory:
    """Structured record of a single assembly step transition.

    Stored whenever the perceived step changes.  Designed for retrieval-
    augmented reasoning: LLM can query "what happened last time the
    assembly moved into step X?" and receive perception-grounded context.

    Fields map directly to WorldState sub-fields so every value here has
    a clear perception source.
    """
    from_step: int              # phase index before transition
    to_step: int                # phase index after transition
    from_step_class: str        # TSM label e.g. "insert_cylinder_under_platform"
    to_step_class: str          # TSM label at new step
    objects_present: List[str]  # object_type strings visible at transition moment
    interaction_state: str      # WorldState.interaction_state at transition
    robot_last_action: Optional[str]  # last robot action before the step changed
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'from_step': self.from_step,
            'to_step': self.to_step,
            'from_step_class': self.from_step_class,
            'to_step_class': self.to_step_class,
            'objects_present': self.objects_present,
            'interaction_state': self.interaction_state,
            'robot_last_action': self.robot_last_action,
            'timestamp': self.timestamp.isoformat(),
        }

    def to_text(self) -> str:
        """Human-readable form for LLM context injection."""
        objs = ', '.join(self.objects_present) if self.objects_present else 'none'
        action = self.robot_last_action or 'none'
        return (
            f"[{self.timestamp.strftime('%H:%M:%S')}] "
            f"Step transition: {self.from_step_class} -> {self.to_step_class} "
            f"(phase {self.from_step}->{self.to_step}), "
            f"objects: {objs}, "
            f"interaction: {self.interaction_state}, "
            f"robot last action: {action}"
        )


@dataclass
class SemanticRule:
    """
    語義規則 - 從情節記憶歸納的一般化協作規則
    """
    rule_id: str
    rule_text: str
    confidence: float
    supporting_episodes: List[str]  # event_id list
    counter_episodes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'rule_id': self.rule_id,
            'rule_text': self.rule_text,
            'confidence': self.confidence,
            'supporting_episodes': self.supporting_episodes,
            'counter_episodes': self.counter_episodes,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }
