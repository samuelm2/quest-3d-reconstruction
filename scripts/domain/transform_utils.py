from enum import Enum
from dataclasses import dataclass
import numpy as np


class CoordinateSystem(Enum):
    UNITY = "Unity"
    OPEN3D = "Open3D"
    OPENGL = "OpenGL"


@dataclass
class Transform:
    coordinate_system: CoordinateSystem

    position: np.ndarray # shape=(3,), (x, y, z)
    rotation: np.ndarray # shape=(4,), (x, y, z, w)