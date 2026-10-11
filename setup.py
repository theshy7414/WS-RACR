"""Package metadata lives in ``pyproject.toml``.

This file only adds the ament_python resource-index entries so that ``colcon build`` and
``ros2 run ws_racr hrc_system`` work. Plain ``pip install -e .`` ignores them.
"""
from setuptools import setup

package_name = "ws_racr"

setup(
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
)
