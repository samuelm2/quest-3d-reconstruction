from dataclasses import dataclass
from pathlib import Path
import numpy as np

from domain.models.transforms import Transforms, CoordinateSystem


@dataclass
class CameraDataset:
    image_relative_paths: np.ndarray

    timestamps: np.ndarray

    fx: np.ndarray
    fy: np.ndarray
    cx: np.ndarray
    cy: np.ndarray

    transforms: Transforms

    widths: np.ndarray
    heights: np.ndarray


    def __post_init__(self):
        length = len(self.timestamps)
        assert all(len(a) == length for _, a in self.to_dict() if isinstance(a, np.ndarray))


    def find_nearest_index(self, timestamp: int) -> int:
        i = np.searchsorted(self.timestamps, timestamp, side='left')
        if i == len(self.timestamps):
            return i - 1
        elif i == 0:
            return 0
        elif abs(self.timestamps[i] - timestamp) < abs(self.timestamps[i - 1] - timestamp):
            return i
        else:
            return i - 1
        

    def get_intrinsic_matrices(self) -> np.ndarray:
        N = len(self.fx)
        intrinsic_matrices = np.zeros((N, 3, 3), dtype=np.float32)

        intrinsic_matrices[:, 0, 0] = self.fx
        intrinsic_matrices[:, 1, 1] = self.fy
        intrinsic_matrices[:, 2, 2] = 1.0

        intrinsic_matrices[:, 0, 2] = self.cx
        intrinsic_matrices[:, 1, 2] = self.cy

        return intrinsic_matrices


    def to_dict(self) -> dict:
        d = {
            "image_relative_paths": self.image_relative_paths,
            "timestamps": self.timestamps,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "coordinate_system": self.transforms.coordinate_system.name,
            "positions": self.transforms.positions,
            "rotations": self.transforms.rotations,
            "widths": self.widths,
            "heights": self.heights,
        }

        return d


    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            **self.to_dict()
        )

    
    @staticmethod
    def parse_transforms(data):
        data['transforms'] = Transforms(
            coordinate_system=CoordinateSystem[data.pop('coordinate_system').item()],
            positions=data.pop('positions'),
            rotations=data.pop('rotations')
        )

    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


    @classmethod
    def load(cls, path: Path):
        data = dict(np.load(path, allow_pickle=False))
        cls.parse_transforms(data)

        return cls.from_dict(data=data)
    

@dataclass
class DepthDataset(CameraDataset):
    nears: np.ndarray
    fars: np.ndarray


    def to_dict(self) -> dict:
        d = super().to_dict()
        d['nears'] = self.nears
        d['fars'] = self.fars

        return d


    @classmethod
    def from_dict(cls, data):
        return cls(**data)


    @classmethod
    def load(cls, path: Path):
        data = dict(np.load(path, allow_pickle=False))
        cls.parse_transforms(data)

        return cls.from_dict(data=data)
