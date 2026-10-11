# Setup

WS-RACR can be used at two levels:

1. **Decision logic only** - the Layer 1/2/3 policy, world-state types, unit tests and the experiments in
   `evaluation/`. Needs Python and a few packages; no GPU, camera or API key.
2. **Full system** - cameras, GPU, ROS 2, PostgreSQL + pgvector, an OpenAI key and a Techman TM5-900 arm.

## 1. Quick start

Requires Python >= 3.10.

```bash
git clone https://github.com/theshy7414/WS-RACR.git
cd WS-RACR
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"             # numpy, pyyaml, python-dotenv, pytest, pyflakes, sqlalchemy

pytest                              # unit tests
python -m evaluation.run_all        # EXP1-EXP4, checks the headline numbers
```

See [`evaluation.md`](evaluation.md) for the experiments.

## 2. Full system

### Platform

Ubuntu 22.04, ROS 2 Humble, Python 3.10, CUDA 12.4 (YOLOv8, TSM-ResNet18 and MediaPipe on GPU),
two Intel RealSense cameras (D435i top-down, D455 front-facing), a Techman TM5-900 arm with gripper.

### Python environment

```bash
conda env create -f environment.yml          # run from the repository root
conda activate ws_racr
```

or, with pip, pick the extras you need: `pip install -e ".[perception,llm,db,voice]"`.

### Models and robot driver

- **Model files.** Provide the YOLOv8-seg weights, the TSM-ResNet18 step-recognition checkpoint and the
  gesture checkpoint, and point the `WS_RACR_*` variables below at them (defaults are under `models/`).
- **Robot driver.** Install the official Techman ROS 2 driver ([tmr_ros2](https://github.com/TechmanRobotInc/tmr_ros2))
  together with the URDF and MoveIt2 configuration for the TM5-900. The execution backends import their ROS 2
  dependencies lazily, so the decision logic works without ROS 2.

### Configuration

Everything machine-specific is read from environment variables or a `.env` file in the repository root
(never commit it):

```bash
cp .env.example .env     # then fill in OPENAI_API_KEY and the DB_* values
```

| Variable | Meaning | Default |
|---|---|---|
| `OPENAI_API_KEY`, `OPENAI_MODEL` | LLM planner and constrained command fallback | model `gpt-4o-mini` |
| `DATABASE_URL` or `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` | PostgreSQL | `localhost:5432/ragdb`, placeholder user/password |
| `WS_RACR_YOLO_WEIGHTS` | YOLOv8-seg weights | `models/workpiece.pt` |
| `WS_RACR_STEP_CHECKPOINT` | TSM-ResNet18 step-recognition checkpoint | `models/step_recognition.pth.tar` |
| `WS_RACR_GESTURE_CHECKPOINT` | gesture classifier checkpoint | `models/gesture.pth` |
| `WS_RACR_DATA_DIR` | calibration and runtime data | `data/` (calibration in `data/calibration/cal_info`) |
| `D435I_SERIAL` | pin the top-down camera (otherwise auto-detected by model name) | unset |
| `EXECUTION_BACKEND` | `tm_setpos`, `moveit` or `tmscript` | `tm_setpos` |

### Database (PostgreSQL + pgvector)

Plan RAG stores completed plans with 384-dimensional embeddings
(`paraphrase-multilingual-MiniLM-L12-v2`). Install PostgreSQL with the pgvector extension and create a database, then:

```bash
python -m ws_racr.database.init_db                   # extension + tasks table + indexes
python -m ws_racr.database.import_knowledge --all    # SOP + key-knowledge base
```

Alembic migrations are in `src/ws_racr/database/alembic/`.

### Run

```bash
# perception only, no robot
python -m ws_racr.app.main --perception-only

# with the robot
python -m ws_racr.app.main --enable-robot

# optional: explicit database URL, log level
python -m ws_racr.app.main --db-url postgresql+psycopg2://user:pass@localhost:5432/ragdb --log-level DEBUG
```

The package also installs the console scripts `hrc_system` and `ws-racr`. As a ROS 2 package
(`package.xml`, ament_python):

```bash
colcon build --packages-select ws_racr
source install/setup.bash
ros2 run ws_racr hrc_system --enable-robot
```

### Hand-eye calibration and tools

`tools/calibration/` contains the dual-camera hand-eye calibration scripts (they use ROS 2, the TM driver
and a RealSense). Results are written under `WS_RACR_DATA_DIR/calibration/`. `tools/check_realsense.py`
lists connected cameras.

### Training

`training/gesture/` holds the training and validation scripts for the gesture classifier.
