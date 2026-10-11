"""World state layer (thesis Ch. 3, "World state representation").

The world state is the shared context between perception, reasoning and
execution:

    W_t = {S_t, O_t, H_t, R_t}

* ``S_t`` task step / assembly progress
* ``O_t`` objects on the workbench (pose, visibility, stability)
* ``H_t`` human interaction state (gesture intent, safety, pointing target)
* ``R_t`` robot execution state (gripper, held object, last skill result)

:class:`WorldStateBuilder` fuses the outputs of the perception modules into a
stabilised :class:`WorldState`.
"""
from ws_racr.world_state.state_representation import (
    AssemblyProgress,
    HumanState,
    IntentState,
    ObjectState,
    RobotState,
    RobotStatus,
    SafetyState,
    WorldState,
    create_empty_world_state,
    create_world_state_from_dict,
)
from ws_racr.world_state.builder import AssemblyProgressTracker, WorldStateBuilder
from ws_racr.world_state.spatial_relations import SpatialRelation, SpatialRelationCalculator

__all__ = [
    "AssemblyProgress",
    "AssemblyProgressTracker",
    "HumanState",
    "IntentState",
    "ObjectState",
    "RobotState",
    "RobotStatus",
    "SafetyState",
    "SpatialRelation",
    "SpatialRelationCalculator",
    "WorldState",
    "WorldStateBuilder",
    "create_empty_world_state",
    "create_world_state_from_dict",
]
