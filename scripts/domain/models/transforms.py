from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R


class CoordinateSystem(Enum):
    """
    Enum representing different coordinate systems used in 3D graphics and computer vision.
    
    - UNITY:
        - World: Y-up, left-handed
        - Camera: X-right, Y-up, Z-forward
        - Used in Unity3D engine
    - OPENGL:
        - World: Y-up, right-handed
        - Camera: X-left, Y-up, Z-forward
        - Used in OpenGL/Open3D
    - NERFSTUDIO:
        - World: Z-up, right-handed
        - Camera: X-right, Y-up, Z-backward
        - Used in NerfStudio
    - COLMAP:
        - World: Y-down, right-handed
        - Camera: X-right, Y-down, Z-forward
        - Used in COLMAP
    """
    UNITY = "Unity"
    OPENGL = "OpenGL"
    NERFSTUDIO = "NerfStudio"
    COLMAP = "COLMAP"


class ExtrinsicMode(Enum):
    CameraToWorld = "camera_to_world"
    WorldToCamera = "world_to_camera"



@dataclass
class Transforms:
    coordinate_system: CoordinateSystem

    positions: np.ndarray # shape=(N, 3), axis1=(x, y, z)
    rotations: np.ndarray # shape=(N, 4), axis1=(x, y, z, w)


    @property
    def extrinsics_wc(self) -> np.ndarray:
        return self.to_extrinsic_matrices(mode=ExtrinsicMode.WorldToCamera)


    @property
    def extrinsics_cw(self) -> np.ndarray:
        return self.to_extrinsic_matrices(mode=ExtrinsicMode.CameraToWorld)


    def convert_coordinate_system(
        self, 
        target_coordinate_system: CoordinateSystem,
        is_camera: bool = False
    ) -> 'Transforms':
        if self.coordinate_system == target_coordinate_system:
            return self
        
        positions = self.positions.copy()
        rotation_matrices = R.from_quat(self.rotations).as_matrix()  # shape (N, 3, 3)

        # Convert positions and rotations into the UNITY coordinate system
        if self.coordinate_system != CoordinateSystem.UNITY:
            if self.coordinate_system == CoordinateSystem.OPENGL:
                positions[:, 0] *= -1
                rotation_matrices[:, :, 0] = -rotation_matrices[:, :, 0]
                rotation_matrices[:, 0, :] = -rotation_matrices[:, 0, :]

            elif self.coordinate_system == CoordinateSystem.NERFSTUDIO:
                positions[:, [1, 2]] = positions[:, [2, 1]]
                rotation_matrices[:, 1:3, 1:3] = rotation_matrices[:, 1:3, 1:3].T
                if is_camera:
                    rotation_matrices = rotation_matrices @ R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0])).as_matrix()

            elif self.coordinate_system == CoordinateSystem.COLMAP:
                positions[:, 1] *= -1
                rotation_matrices[:, :, 1] = -rotation_matrices[:, :, 1]
                rotation_matrices[:, 1, :] = -rotation_matrices[:, 1, :]

            else:
                raise ValueError(f"Unsupported coordinate system: {self.coordinate_system}")
            
        # Convert positions and rotations into the target coordinate system
        if target_coordinate_system == CoordinateSystem.UNITY:
            pass
        elif target_coordinate_system == CoordinateSystem.OPENGL:
            positions[:, 0] *= -1
            rotation_matrices[:, :, 0] = -rotation_matrices[:, :, 0]
            rotation_matrices[:, 0, :] = -rotation_matrices[:, 0, :]
        elif target_coordinate_system == CoordinateSystem.NERFSTUDIO:
            positions[:, [1, 2]] = positions[:, [2, 1]]
            rotation_matrices[:, 1:3, 1:3] = rotation_matrices[:, 1:3, 1:3].T
            if is_camera:
                rotation_matrices = rotation_matrices @ R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0])).as_matrix()
        elif target_coordinate_system == CoordinateSystem.COLMAP:
            positions[:, 1] *= -1
            rotation_matrices[:, :, 1] = -rotation_matrices[:, :, 1]
            rotation_matrices[:, 1, :] = -rotation_matrices[:, 1, :]
        else:
            raise ValueError(f"Unsupported target coordinate system: {target_coordinate_system}")
        
        return Transforms(
            coordinate_system=target_coordinate_system,
            positions=positions,
            rotations=R.from_matrix(rotation_matrices).as_quat()
        )
        

    def to_extrinsic_matrices(self, mode: ExtrinsicMode = ExtrinsicMode.WorldToCamera) -> np.ndarray:
        N = len(self.positions)

        R_cw = R.from_quat(self.rotations).as_matrix()  # (N, 3, 3)

        extrinsic_matrices = np.zeros((N, 4, 4), dtype=np.float32)
        extrinsic_matrices[:, :3, :3] = R_cw
        extrinsic_matrices[:, :3, 3] = self.positions
        extrinsic_matrices[:, 3, 3] = 1.0

        if mode == ExtrinsicMode.WorldToCamera:
            return np.linalg.inv(extrinsic_matrices)
        elif mode == ExtrinsicMode.CameraToWorld:
            return extrinsic_matrices
        else:
            raise ValueError(f"Unsupported extrinsic mode: {mode}")


    def compose_transform(
        self,
        local_position: np.ndarray, # shape=(3), axis1=(x, y, z)
        local_rotation: np.ndarray, # shape=(4), axis1=(x, y, z, w)
    ) -> 'Transforms':
        parent_rotations = R.from_quat(self.rotations)
        rotated_local_positions = parent_rotations.apply(local_position)

        world_positions = self.positions + rotated_local_positions
        world_rotations = parent_rotations * R.from_quat(local_rotation)

        return Transforms(
            coordinate_system=self.coordinate_system,
            positions=world_positions,
            rotations=world_rotations.as_quat()
        )
    

    def to_dict(self) -> dict:
        d = {
            "coordinate_system": self.coordinate_system,
            "positions": self.positions,
            "rotations": self.rotations,
        }

        return d
    

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            **self.to_dict()
        )


    @classmethod
    def from_dict(cls, data):
        return cls(**data)
    

    @classmethod
    def load(cls, path: Path):
        data = dict(np.load(path, allow_pickle=False))
        return cls.from_dict(data=data)