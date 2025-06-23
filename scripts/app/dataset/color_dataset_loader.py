from typing import Optional
import numpy as np

from domain.models.side import Side
from domain.models.transforms import Transforms, CoordinateSystem
from domain.models.camera_dataset import CameraDataset
from infra.io.project_io_manager import ProjectIOManager


class ColorDatasetLoader:
    def __init__(
        self,
        project_io_manager: ProjectIOManager,
        side: Side
    ):
        self.project_io_manager = project_io_manager
        self.side = side
        self.dataset: Optional[CameraDataset] = None

    
    def load_dataset(self) -> CameraDataset:
        if self.dataset is not None:
            print(f"[Info] Color dataset already loaded. Returning cached dataset...")
            return self.dataset

        if self.project_io_manager.color_dataset_cache_exists(side=self.side):
            print(f"[Info] Color dataset cache found. Loading cached dataset...")

            try:
                return self.project_io_manager.load_color_dataset_cache(side=self.side)
            except Exception as e:
                print(f"[Error] Color dataset cache is corrupted or invalid. Rebuilding cache from the original source...\n{e}")

        else:
            print(f"[Info] Color dataset not found. Rebuilding cache from the original source...")

        characteristics = self.project_io_manager.get_camera_characteristics(side=self.side)
        rgb_image_repo = self.project_io_manager.get_rgb_repo(side=self.side)
        hmd_pose_interp = self.project_io_manager.get_hmd_pose_interpolator()

        image_relative_paths = []
        timestamps = []
        hmd_positions = []
        hmd_rotations = []

        for color_map_abs_path in rgb_image_repo.paths:
            stem = color_map_abs_path.stem
            color_map_relative_path = rgb_image_repo.get_relaive_path(file_stem=stem)

            timestamp = int(stem)

            result = hmd_pose_interp.interpolate_pose(timestamp=timestamp)

            if result is None:
                continue

            hmd_position, hmd_rotation = result

            image_relative_paths.append(color_map_relative_path)
            timestamps.append(timestamp)
            hmd_positions.append(hmd_position)
            hmd_rotations.append(hmd_rotation)

        N = len(image_relative_paths)

        hmd_transforms = Transforms(
            coordinate_system=CoordinateSystem.UNITY,
            positions=np.array(hmd_positions),
            rotations=np.array(hmd_rotations),
        )
        camera_transforms = hmd_transforms.compose_transform(
            characteristics.transl,
            characteristics.rot_quat,
        )

        fxs = np.full(N, characteristics.fx, dtype=np.float32)
        fys = np.full(N, characteristics.fy, dtype=np.float32)
        cxs = np.full(N, characteristics.cx, dtype=np.float32)
        cys = np.full(N, characteristics.cy, dtype=np.float32)
        widths = np.full(N, characteristics.width, dtype=np.int16)
        heights = np.full(N, characteristics.height, dtype=np.int16)

        self.dataset = CameraDataset(
            image_relative_paths=image_relative_paths,
            timestamps=timestamps,
            transforms=camera_transforms,
            fx=fxs, fy=fys,
            cx=cxs, cy=cys,
            widths=widths,
            heights=heights
        )

        self.project_io_manager.save_color_dataset_cache(side=self.side, dataset=self.dataset)
        print(f"[Info] Color dataset loaded successfully. Dataset size: {len(self.dataset.image_relative_paths)}")

        return self.dataset
            