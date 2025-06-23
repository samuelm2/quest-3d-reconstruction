import json
from pathlib import Path
from typing import Optional, Union
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import open3d as o3d

import config.path_config as path_config
from domain.models.camera_dataset import DepthDataset, CameraDataset, RGBDDataset
from domain.models.camera_characteristics import CameraCharacteristics
from domain.models.side import Side
from infra.io.yuv_repository import YUVRepository
from infra.io.image_repository import ImageRepository
from infra.io.depth_repository import DepthRepository
from infra.helper.pose_interpolator import PoseInterpolator


class ProjectIOManager:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

        self.left_yuv_repo = YUVRepository(
            yuv_dir=project_dir / path_config.LEFT_CAMERA_YUV_IMAGE_DIR,
            format_json=project_dir / path_config.LEFT_CAMERA_IMAGE_FORMAT_JSON
        )
        self.right_yuv_repo = YUVRepository(
            yuv_dir=project_dir / path_config.RIGHT_CAMERA_YUV_IMAGE_DIR,
            format_json=project_dir / path_config.RIGHT_CAMERA_IMAGE_FORMAT_JSON
        )

        self.left_rgb_repo = ImageRepository(
            project_root=project_dir,
            image_subdir=path_config.LEFT_CAMERA_RGB_IMAGE_DIR,
        )
        self.right_rgb_repo = ImageRepository(
            project_root=project_dir,
            image_subdir=path_config.RIGHT_CAMERA_RGB_IMAGE_DIR,
        )

        self.left_depth_grey_repo = ImageRepository(
            project_root=project_dir,
            image_subdir=path_config.LEFT_DEPTH_GRAY_IMAGE_DIR,
        )
        self.right_depth_grey_repo = ImageRepository(
            project_root=project_dir,
            image_subdir=path_config.RIGHT_DEPTH_GRAY_IMAGE_DIR,
        )

        self.left_depth_repo = DepthRepository(
            project_root=project_dir,
            depth_dir=project_dir / path_config.LEFT_DEPTH_RAW_DATA_DIR
        )
        self.right_depth_repo = DepthRepository(
            project_root=project_dir,
            depth_dir=project_dir / path_config.RIGHT_DEPTH_RAW_DATA_DIR
        )


    def get_yuv_repo(self, side: Side) -> YUVRepository:
        if side == Side.LEFT:
            return self.left_yuv_repo
        else:
            return self.right_yuv_repo


    def get_rgb_repo(self, side: Side) -> ImageRepository:
        if side == Side.LEFT:
            return self.left_rgb_repo
        else:
            return self.right_rgb_repo
        
    
    def get_depth_repo(self, side: Side) -> DepthRepository:
        if side == Side.LEFT:
            return self.left_depth_repo
        else:
            return self.right_depth_repo
        

    def get_depth_grey_repo(self, side: Side) -> ImageRepository:
        if side == Side.LEFT:
            return self.left_depth_grey_repo
        else:
            return self.right_depth_grey_repo
        

    def get_hmd_pose_interpolator(self) -> PoseInterpolator:
        return PoseInterpolator(
            pose_csv_path=self.project_dir / path_config.HMD_POSE_CSV
        )


    def get_camera_characteristics(self, side: Side) -> CameraCharacteristics:
        if side == Side.LEFT:
            characteristics_json_path = self.project_dir / path_config.LEFT_CAMERA_CHARACTERISTICS_JSON
        else:
            characteristics_json_path = self.project_dir / path_config.RIGHT_CAMERA_CHARACTERISTICS_JSON

        with open(characteristics_json_path, "r", encoding="utf-8") as f:
            camera_characteristics = json.load(f)

        array_size = camera_characteristics["sensor"]["activeArraySize"]
        width = array_size["right"] - array_size["left"]
        height = array_size["bottom"] - array_size["top"]

        intrinsics = camera_characteristics["intrinsics"]

        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]

        camera_pose = camera_characteristics["pose"]

        transl = camera_pose["translation"]
        transl[2] *= -1
        if len(transl) < 3:
            transl = np.array((0, 0, 0))

        rot_quat = camera_pose["rotation"]
        if len(rot_quat) >=4:
            qx = -rot_quat[0]
            qy = -rot_quat[1]
            qz = rot_quat[2]
            qw = rot_quat[3]

            rot = R.from_quat((qx, qy, qz, qw)).inv()
            # Apply a 180-degree rotation to align the Android camera pose with the HMD world coordinate system.
            rot *= R.from_euler('x', np.pi)

            rot_quat = rot.as_quat()
            
        else:
            rot_quat = np.array((0, 0, 0, 1))

        return CameraCharacteristics(
            width=width,
            height=height,
            fx=fx, fy=fy,
            cx=cx, cy=cy,
            transl=transl,
            rot_quat=rot_quat,
        )
    

    def load_depth_descriptor(self, side: Side) -> pd.DataFrame:
        if side == Side.LEFT:
            descriptor_csv_path = self.project_dir / path_config.LEFT_DEPTH_DESCRIPTOR_CSV
        else:
            descriptor_csv_path = self.project_dir / path_config.RIGHT_DEPTH_DESCRIPTOR_CSV

        if not descriptor_csv_path.exists():
            raise FileNotFoundError(f"Depth descriptor CSV for {side.value} side does not exist.")

        return pd.read_csv(descriptor_csv_path)


    def load_depth_map_by_index(
        self,
        side: Side,
        index: int,
        dataset: DepthDataset = None,
        validate_depth: bool = False
    ) -> Optional[np.ndarray]:
        depth_map = self.load_depth_map(
            side=side,
            timestamp=dataset.timestamps[index],
            width=dataset.widths[index],
            height=dataset.heights[index],
            near=dataset.nears[index],
            far=dataset.fars[index]
        )

        if validate_depth:
            is_valid = True
            is_valid = is_valid and (depth_map != 0).any()
            is_valid = is_valid and not np.isnan(depth_map).any()
            is_valid = is_valid and (depth_map >= 0).all()

            if not is_valid:
                return None
        
        return depth_map

    
    def load_depth_map(
        self, 
        side: Side,
        timestamp: int,
        width: int,
        height: int,
        near: float,
        far: float,
    ) -> np.ndarray:
        if side == Side.LEFT:
            return self.left_depth_repo.load(
                timestamp=timestamp,
                width=width, height=height,
                near=near, far=far
            )
        else:
            return self.right_depth_repo.load(
                timestamp=timestamp,
                width=width, height=height,
                near=near, far=far
            )
        

    def load_color_map_by_index(
        self,
        side: Side,
        index: int,
        dataset: CameraDataset
    ) -> np.ndarray:
        timestamp = dataset.timestamps[index]
        
        if side == Side.LEFT:
            return self.left_rgb_repo.load(
                file_stem=str(timestamp)
            )
        else:
            return self.right_rgb_repo.load(
                file_stem=str(timestamp)
            )
        

    def depth_dataset_cache_exists(self, side: Side) -> bool:
        if side == Side.LEFT:
            cache_path = self.project_dir / path_config.LEFT_DEPTH_DATASET_CACHE
        else:
            cache_path = self.project_dir / path_config.RIGHT_DEPTH_DATASET_CACHE

        return cache_path.exists() and cache_path.is_file()
    

    def load_depth_dataset_cache(self, side: Side) -> DepthDataset:
        if not self.depth_dataset_cache_exists(side):
            raise FileNotFoundError(f"Depth dataset cache for {side.value} side does not exist.")

        if side == Side.LEFT:
            cache_path = self.project_dir / path_config.LEFT_DEPTH_DATASET_CACHE
        else:
            cache_path = self.project_dir / path_config.RIGHT_DEPTH_DATASET_CACHE

        return DepthDataset.load(cache_path)
    

    def save_depth_dataset_cache(self, side: Side, dataset: DepthDataset):
        if side == Side.LEFT:
            cache_path = self.project_dir / path_config.LEFT_DEPTH_DATASET_CACHE
        else:
            cache_path = self.project_dir / path_config.RIGHT_DEPTH_DATASET_CACHE

        dataset.save(cache_path)
        print(f"[Info] Depth dataset cache saved to {cache_path}.")


    def color_dataset_cache_exists(self, side: Side) -> bool:
        if side == Side.LEFT:
            cache_path = self.project_dir / path_config.LEFT_COLOR_DATASET_CACHE
        else:
            cache_path = self.project_dir / path_config.RIGHT_COLOR_DATASET_CACHE

        return cache_path.exists() and cache_path.is_file()
    

    def load_color_dataset_cache(self, side: Side) -> CameraDataset:
        if not self.color_dataset_cache_exists(side):
            raise FileNotFoundError(f"Color dataset cache for {side.value} side does not exist.")

        if side == Side.LEFT:
            cache_path = self.project_dir / path_config.LEFT_COLOR_DATASET_CACHE
        else:
            cache_path = self.project_dir / path_config.RIGHT_COLOR_DATASET_CACHE

        return CameraDataset.load(cache_path)


    def save_color_dataset_cache(self, side: Side, dataset: CameraDataset):
        if side == Side.LEFT:
            cache_path = self.project_dir / path_config.LEFT_COLOR_DATASET_CACHE
        else:
            cache_path = self.project_dir / path_config.RIGHT_COLOR_DATASET_CACHE

        dataset.save(cache_path)
        print(f"[Info] Color dataset cache saved to {cache_path}.")


    def rgbd_dataset_cache_exists(self, side: Side) -> bool:
        if side == Side.LEFT:
            cache_path = self.project_dir / path_config.LEFT_RGBD_DATASET_CACHE
        else:
            cache_path = self.project_dir / path_config.RIGHT_RGBD_DATASET_CACHE

        return cache_path.exists() and cache_path.is_file()
    

    def load_rgbd_dataset_cache(self, side: Side) -> RGBDDataset:
        if not self.rgbd_dataset_cache_exists(side):
            raise FileNotFoundError(f"Color dataset cache for {side.value} side does not exist.")

        if side == Side.LEFT:
            cache_path = self.project_dir / path_config.LEFT_RGBD_DATASET_CACHE
        else:
            cache_path = self.project_dir / path_config.RIGHT_RGBD_DATASET_CACHE

        return RGBDDataset.load(cache_path)


    def save_rgbd_dataset_cache(self, side: Side, dataset: RGBDDataset):
        if side == Side.LEFT:
            cache_path = self.project_dir / path_config.LEFT_RGBD_DATASET_CACHE
        else:
            cache_path = self.project_dir / path_config.RIGHT_RGBD_DATASET_CACHE

        dataset.save(cache_path)
        print(f"[Info] RGBD dataset cache saved to {cache_path}.")


    def colorless_vbg_tensor_exists(self) -> bool:
        colorless_vbg_path = self.project_dir / path_config.COLORLESS_VBG_TENSOR
        return colorless_vbg_path.exists() and colorless_vbg_path.is_file()


    def load_colorless_voxel_grid_tensor(self) -> o3d.t.geometry.VoxelBlockGrid:
        colorless_vbg_path = self.project_dir / path_config.COLORLESS_VBG_TENSOR
        if not colorless_vbg_path.exists():
            raise FileNotFoundError(f"Colorless voxel box grid file does not exist: {colorless_vbg_path}")

        voxel_grid = o3d.t.geometry.VoxelBlockGrid.load(
            str(colorless_vbg_path)
        )
        return voxel_grid
    

    def save_colorless_voxel_grid_tensor(self, voxel_grid: o3d.t.geometry.VoxelBlockGrid):
        colorless_vbg_path = self.project_dir / path_config.COLORLESS_VBG_TENSOR

        colorless_vbg_path.parent.mkdir(parents=True, exist_ok=True)

        voxel_grid.save(str(colorless_vbg_path))
        print(f"[Info] Colorless voxel grid saved to {colorless_vbg_path}.")


    def colorless_pcd_exists(self) -> bool:
        colorless_pcd_path = self.project_dir / path_config.COLORLESS_PCD
        return colorless_pcd_path.exists() and colorless_pcd_path.is_file()
    

    def load_colorless_point_cloud(self) -> o3d.geometry.PointCloud:
        colorless_pcd_path = self.project_dir / path_config.COLORLESS_PCD
        if not colorless_pcd_path.exists():
            raise FileNotFoundError(f"Colorless point cloud file does not exist: {colorless_pcd_path}")

        pcd = o3d.io.read_point_cloud(str(colorless_pcd_path))
        return pcd
    

    def save_colorless_point_cloud_tensor(self, point_cloud: Union[o3d.geometry.PointCloud, o3d.t.geometry.PointCloud]):
        colorless_pcd_path = self.project_dir / path_config.COLORLESS_PCD
        colorless_pcd_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(point_cloud, o3d.t.geometry.PointCloud):
            point_cloud = point_cloud.to_legacy()
        o3d.io.write_point_cloud(str(colorless_pcd_path), point_cloud)

        print(f"[Info] Colorless point cloud saved to {colorless_pcd_path}.")