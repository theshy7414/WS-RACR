"""Command-line entry point of the HRC system (``ws-racr`` / ``python -m ws_racr.app.main``).

Start-up sequence: cameras -> perception modules -> database and memory -> robot controller
(only with ``--enable-robot``) -> workspace clutter check -> episode start -> main loop.
"""

import argparse
import logging

from ws_racr.app.system import HRCSystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Human-robot collaborative assembly system (WS-RACR)")
    parser.add_argument("--db-url", type=str, default=None,
                        help="SQLAlchemy database URL (default: DATABASE_URL / DB_* environment variables)")
    parser.add_argument("--enable-robot", action="store_true",
                        help="enable robot control (requires the TM5-900 and ROS 2)")
    parser.add_argument("--perception-only", action="store_true",
                        help="perception only; never drives the robot (overrides --enable-robot)")
    parser.add_argument("--log-level", default="INFO",
                        help="logging level (DEBUG, INFO, WARNING, ...)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    system = HRCSystem(
        db_url=args.db_url,
        enable_robot=args.enable_robot and not args.perception_only,
    )

    system.init_cameras()
    system.init_perception_modules()
    system.init_database()

    if system.enable_robot:
        system.init_robot_controller()

    # Last init step: one-off workspace clutter check (not per frame).
    system.run_startup_clutter_check()

    system.start_episode()
    system.run()


if __name__ == "__main__":
    main()
