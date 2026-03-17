"""iOS automatic jigsaw puzzle solver."""

from .connector import DeviceConnector
from .screenshot import Screenshot
from .gesture import Gesture
from .planner import MotionPlanner
from .automation import run_automation

__all__ = [
    "DeviceConnector",
    "Screenshot",
    "Gesture",
    "MotionPlanner",
    "run_automation",
]
