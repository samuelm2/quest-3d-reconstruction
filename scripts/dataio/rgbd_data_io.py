from typing import Optional
import numpy as np
import open3d as o3d
from dataio.depth_data_io import DepthDataIO
from dataio.image_data_io import ImageDataIO
from config.project_path_config import RGBDPathConfig
from models.camera_dataset import CameraDataset
from models.side import Side


class RGBDDataIO:
    def __init__(self,
        image_data_io: ImageDataIO,
        depth_data_io: DepthDataIO,
        rgbd_path_config: RGBDPathConfig
    ):
        self.image_data_io = image_data_io
        self.depth_data_io = depth_data_io
        self.rgbd_path_config = rgbd_path_config

    
    def load_color_aligned_depth(self, side: Side, timestamp: int) -> np.ndarray:
        color_aligned_depth_path = self.rgbd_path_config.get_color_aligned_depth_path(side=side, timestamp=timestamp)
        
        if not color_aligned_depth_path.exists():
            raise FileNotFoundError(f"Color-aligned depth file not found: {color_aligned_depth_path}")
        
        return np.load(color_aligned_depth_path)
    

    def save_color_aligned_depth(self, depth_map: np.ndarray, side: Side, timestamp: int):
        color_aligned_depth_path = self.rgbd_path_config.get_color_aligned_depth_path(side=side, timestamp=timestamp)
        color_aligned_depth_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(color_aligned_depth_path, depth_map)


    def load_colorless_vbg(self) -> Optional[o3d.t.geometry.VoxelBlockGrid]:
        colorless_vbg_path = self.rgbd_path_config.get_colorless_vbg_path()

        if not colorless_vbg_path.exists():
            return None
        
        return o3d.t.geometry.VoxelBlockGrid.load(str(colorless_vbg_path))
    

    def save_colorless_vbg(self, vbg: o3d.t.geometry.VoxelBlockGrid):
        colorless_vbg_path = self.rgbd_path_config.get_colorless_vbg_path()
        colorless_vbg_path.parent.mkdir(parents=True, exist_ok=True)

        vbg.save(str(colorless_vbg_path))


    def load_color_vbg(self) -> Optional[o3d.t.geometry.VoxelBlockGrid]:
        color_vbg_path = self.rgbd_path_config.get_color_vbg_path()

        if not color_vbg_path.exists():
            return None
        
        return o3d.t.geometry.VoxelBlockGrid.load(str(color_vbg_path))
    

    def save_color_vbg(self, vbg: o3d.t.geometry.VoxelBlockGrid):
        color_vbg_path = self.rgbd_path_config.get_color_vbg_path()
        color_vbg_path.parent.mkdir(parents=True, exist_ok=True)

        vbg.save(str(color_vbg_path))


    def load_color_pcd(self, device: o3d.core.Device) -> Optional[o3d.t.geometry.PointCloud]:
        color_pcd_path = self.rgbd_path_config.get_color_pcd_path()

        if not color_pcd_path.exists():
            return None
        
        pcd_legacy = o3d.io.read_point_cloud(
            filename=str(color_pcd_path),
            format='auto',
            remove_nan_points=True,
            remove_infinite_points=True,            
        )
        
        return o3d.t.geometry.PointCloud.from_legacy(
            pcd_legacy=pcd_legacy,
            device=device
        )
    

    def save_color_pcd(self, pcd: o3d.t.geometry.PointCloud):
        color_pcd_path = self.rgbd_path_config.get_color_pcd_path()
        color_pcd_path.parent.mkdir(parents=True, exist_ok=True)

        o3d.io.write_point_cloud(
            filename=str(color_pcd_path),
            pointcloud=pcd.to_legacy(),
            format='auto',
            write_ascii=False,
            compressed=True
        )