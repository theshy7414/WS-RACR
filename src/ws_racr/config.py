"""Runtime configuration: environment variables, ``.env`` loading, default paths.

Everything that used to be hard-coded (database credentials, model paths, robot
parameters, execution backend) is read from environment variables here. A ``.env``
file in the current working directory or the repository root is loaded
automatically when ``python-dotenv`` is installed; see ``.env.example``.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Paths and .env loading
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    """Repository root when running from a checkout, otherwise the current directory."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return Path.cwd()


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional
        return
    for env_file in (repo_root() / ".env", Path.cwd() / ".env"):
        if env_file.exists():
            load_dotenv(env_file, override=False)


_load_dotenv()


def get_yolo_weights_path() -> Path:
    """Path of the YOLOv8-seg workpiece weights (``WS_RACR_YOLO_WEIGHTS``, default ``models/workpiece.pt``)."""
    return Path(os.getenv("WS_RACR_YOLO_WEIGHTS", str(repo_root() / "models" / "workpiece.pt")))


def get_data_dir() -> Path:
    """Directory for calibration files and runtime data (``WS_RACR_DATA_DIR``, default ``data/``)."""
    return Path(os.getenv("WS_RACR_DATA_DIR", str(repo_root() / "data")))


def get_step_checkpoint_path() -> Path:
    """Path of the TSM-ResNet18 step-recognition checkpoint (``WS_RACR_STEP_CHECKPOINT``)."""
    return Path(os.getenv("WS_RACR_STEP_CHECKPOINT", str(repo_root() / "models" / "step_recognition.pth.tar")))


def get_gesture_checkpoint_path() -> Path:
    """Path of the gesture classifier checkpoint (``WS_RACR_GESTURE_CHECKPOINT``)."""
    return Path(os.getenv("WS_RACR_GESTURE_CHECKPOINT", str(repo_root() / "models" / "gesture.pth")))


def get_sop_yaml_path() -> Path:
    """Bundled assembly SOP definition."""
    return Path(__file__).resolve().parent / "database" / "assembly_sop.yaml"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PI = math.pi
ORIGIN = [0.291, -0.225, 0.613]  # robot home position (m)
ANGLE = [PI, 0.0, PI / 2]  # home orientation (rad)

#: True when launched through ROS 2 (``ros2 run`` / ``colcon`` installs).
FROM_ROS = (
    any("--ros-args" in arg for arg in sys.argv)
    or any("install/lib" in arg or "build/lib" in arg for arg in sys.argv)
    or "AMENT_PREFIX_PATH" in os.environ
)

DEFAULT_ROBOT_ORIENTATION = {"rx": 180.0, "ry": 0.0, "rz": -90.0}
DEFAULT_ROBOT_VELOCITY = 100
DEFAULT_ROBOT_BLEND = 100
DEFAULT_VELOCITY_RANGE = (10, 100)
DEFAULT_VELOCITY_STEP = 5

ROS_SPIN_TIMEOUT = 0.1
ROS_SHUTDOWN_TIMEOUT = 2.0

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def get_openai_client():
    """Return an OpenAI client; requires ``OPENAI_API_KEY``."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the 'llm' extra: pip install -e .[llm]") from exc
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)


def get_model_name() -> str:
    """Chat model name (``OPENAI_MODEL``, default ``gpt-4o-mini``)."""
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Robot / perception / execution
# ---------------------------------------------------------------------------


def get_robot_config() -> Dict[str, Any]:
    return {
        "name": os.getenv("ROBOT_NAME", "TM_Robot"),
        "max_velocity": float(os.getenv("MAX_VELOCITY", 0.8)),
        "max_acceleration": float(os.getenv("MAX_ACCELERATION", 0.5)),
        "safe_height": float(os.getenv("SAFE_HEIGHT", 300.0)),
    }


def get_detection_config() -> Dict[str, Any]:
    return {
        "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", 0.3)),
        "detection_timeout": float(os.getenv("DETECTION_TIMEOUT", 5.0)),
    }


def get_execution_config() -> Dict[str, Any]:
    return {
        "execution_timeout": float(os.getenv("EXECUTION_TIMEOUT", 30.0)),
        "retry_attempts": int(os.getenv("RETRY_ATTEMPTS", 3)),
        "backend": os.getenv("EXECUTION_BACKEND", "tm_setpos"),  # moveit / tm_setpos / tmscript
    }


def get_backend_config() -> Dict[str, Any]:
    """Execution backend selection and per-backend parameters.

    ``EXECUTION_BACKEND`` is one of ``moveit`` (MoveIt2 planning with collision
    avoidance), ``tm_setpos`` (TM SetPositions service, default) or ``tmscript``
    (batched TMScript).
    """
    return {
        "type": os.getenv("EXECUTION_BACKEND", "tm_setpos"),
        "moveit": {
            "group_name": os.getenv("MOVEIT_GROUP_NAME", "tm_arm"),
            "base_frame": os.getenv("MOVEIT_BASE_FRAME", "base_link"),
            "ee_frame": os.getenv("MOVEIT_EE_FRAME", "tool0"),
            "planner": os.getenv("MOVEIT_PLANNER", "RRTConnect"),
            "planning_time": float(os.getenv("MOVEIT_PLANNING_TIME", 5.0)),
            "velocity_scaling": float(os.getenv("MOVEIT_VELOCITY_SCALING", 0.5)),
            "acceleration_scaling": float(os.getenv("MOVEIT_ACCELERATION_SCALING", 0.5)),
        },
        "tm_setpos": {
            "default_velocity": int(os.getenv("TM_DEFAULT_VELOCITY", 100)),
            "default_acc_time": int(os.getenv("TM_DEFAULT_ACC_TIME", 200)),
            "movement_timeout": float(os.getenv("TM_MOVEMENT_TIMEOUT", 30.0)),
        },
        "tmscript": {
            "default_velocity": int(os.getenv("TMSCRIPT_DEFAULT_VELOCITY", 100)),
            "default_acc_time": int(os.getenv("TMSCRIPT_DEFAULT_ACC_TIME", 200)),
            "default_blend": int(os.getenv("TMSCRIPT_DEFAULT_BLEND", 0)),
            "script_timeout": float(os.getenv("TMSCRIPT_TIMEOUT", 60.0)),
        },
    }


def get_database_config() -> Dict[str, Any]:
    """PostgreSQL connection settings: ``DATABASE_URL`` or ``DB_HOST/PORT/NAME/USER/PASSWORD``."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        name = os.getenv("DB_NAME", "ragdb")
        user = os.getenv("DB_USER", "your_user")
        password = os.getenv("DB_PASSWORD", "your_password")
        db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    return {
        "url": db_url,
        "vector_index_lists": int(os.getenv("VECTOR_INDEX_LISTS", 100)),
    }


def get_gripper_config() -> Dict[str, Any]:
    """Gripper digital-IO configuration."""
    return {
        "module": os.getenv("GRIPPER_MODULE", "EndModule"),
        "io_type": os.getenv("GRIPPER_TYPE", "DO"),
        "pin": int(os.getenv("GRIPPER_PIN", 0)),
        "open_delay": float(os.getenv("GRIPPER_OPEN_DELAY", 1.0)),
        "close_delay": float(os.getenv("GRIPPER_CLOSE_DELAY", 1.5)),
    }


def get_movement_config() -> Dict[str, Any]:
    return {
        "robot_name": os.getenv("ROBOT_NAME", "TM5-900"),
        "default_velocity": int(os.getenv("DEFAULT_VELOCITY", 100)),
        "default_acc_time": float(os.getenv("DEFAULT_ACC_TIME", 0.2)),
        "safe_height": float(os.getenv("SAFE_HEIGHT", 0.2)),
        "approach_height": float(os.getenv("APPROACH_HEIGHT", 0.05)),
        "max_velocity": float(os.getenv("MAX_VELOCITY", 800.0)),
        "max_acceleration": float(os.getenv("MAX_ACCELERATION", 500.0)),
    }
