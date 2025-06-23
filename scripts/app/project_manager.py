from pathlib import Path
import numpy as np
import open3d as o3d

from domain.models.side import Side
from infra.io.project_io_manager import ProjectIOManager
from app.dataset import DepthDatasetLoader, ColorDatasetLoader
from app.pipeline import (
    convert_yuv_directory_to_png, 
    convert_depth_directory_to_linear, 
    generate_colorless_point_cloud_tensor,
    generate_colorless_point_cloud_legacy,
)


class ProjectManager:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.io_manager = ProjectIOManager(project_dir)

        self.left_depth_dataset_loader = DepthDatasetLoader(
            project_io_manager=self.io_manager,
            side=Side.LEFT
        )
        self.right_depth_dataset_loader = DepthDatasetLoader(
            project_io_manager=self.io_manager,
            side=Side.RIGHT
        )
        self.left_color_dataset_loader = ColorDatasetLoader(
            project_io_manager=self.io_manager,
            side=Side.LEFT
        )
        self.right_color_dataset_loader = ColorDatasetLoader(
            project_io_manager=self.io_manager,
            side=Side.RIGHT
        )


    
    def load_depth_dataset(self, side: Side):
        print(f"[Info] Loading depth dataset for {side} camera...")
        if side == Side.LEFT:
            return self.left_depth_dataset_loader.load_dataset()
        else:
            return self.right_depth_dataset_loader.load_dataset()
        

    def load_color_dataset(self, side: Side):
        print(f"[Info] Loading color dataset for {side} camera...")
        if side == Side.LEFT:
            return self.left_color_dataset_loader.load_dataset()
        else:
            return self.right_color_dataset_loader.load_dataset()


    def convert_yuv_to_rgb(
        self,
        apply_filter: bool = False,
        blur_threshold: float = 50.0,
        exposure_threshold_low: float = 0.1,
        exposure_threshold_high: float = 0.1
    ):
        for side in Side:
            print(f"[Info] Converting {side} camera images...")

            yuv_repo = self.io_manager.get_yuv_repo(side)
            rgb_repo = self.io_manager.get_rgb_repo(side)

            convert_yuv_directory_to_png(
                yuv_repo=yuv_repo,
                rgb_repo=rgb_repo,
                apply_filter=apply_filter,
                blur_threshold=blur_threshold,
                exposure_threshold_low=exposure_threshold_low,
                exposure_threshold_high=exposure_threshold_high
            )

    
    def convert_depth_to_linear_map(self, clip_near: float = 0.1, clip_far: float = 100.0):
        for side in Side:
            print(f"[Info] Converting {side} camera depth images to linear format...")

            dataset = self.load_depth_dataset(side=side)
            convert_depth_directory_to_linear(
                project_io_manager=self.io_manager,
                side=side,
                dataset=dataset,
                clip_near=clip_near,
                clip_far=clip_far
            )


    def generate_point_cloud_tensor(
        self, 
        no_cache: bool = False,
        color: bool = False,
        block_resolution: int = 16,
        voxel_size: float = 0.01,
        block_count: int = 50_000,
        depth_max: float = 2,
        weight_threshold: float = 3.0,
        trunc_voxel_multiplier: float = 8,
        down_voxel_size: float = 0.02,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
        device: str = "CUDA:0",
    ) -> o3d.geometry.PointCloud:
        print(f"[Info] Generating colorless point cloud (tensor) with the following parameters:")
        print(f"  - Block resolution: {block_resolution}")
        print(f"  - Voxel size: {voxel_size} m")
        print(f"  - Block count: {block_count}")
        print(f"  - Depth max: {depth_max} m")
        print(f"  - Trunc voxel multiplier: {trunc_voxel_multiplier}")
        print(f"  - Down voxel size: {down_voxel_size} m")
        print(f"  - Device: {device}")

        depth_datasets = {
            Side.LEFT: self.load_depth_dataset(Side.LEFT),
            Side.RIGHT: self.load_depth_dataset(Side.RIGHT)
        }

        pcd = generate_colorless_point_cloud_tensor(
            project_io_manager=self.io_manager,
            depth_datasets=depth_datasets,
            no_cache=no_cache,
            block_resolution=block_resolution,
            voxel_size=voxel_size,
            block_count=block_count,
            depth_max=depth_max,
            weight_threshold=weight_threshold,
            trunc_voxel_multiplier=trunc_voxel_multiplier,
            down_voxel_size=down_voxel_size,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
            device=device
        )

        print("[Info] Colorless point cloud generation completed.")

        self.io_manager.save_colorless_point_cloud_tensor(pcd)
        print("[Info] Colorless point cloud saved successfully.")

        pcd.paint_uniform_color(np.array([0.5, 0.5, 0.5]))

        return pcd.to_legacy()
    

    def generate_point_cloud_legacy(
        self, 
        color: bool = False,
        voxel_length: float = 0.01,
        sdf_trunc: float = 0.05,
        depth_trunc: float = 2.0,
        down_voxel_size: float = 0.02,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
    )-> o3d.geometry.PointCloud:
        print(f"[Info] Generating colorless point cloud (legacy) with the following parameters:")
        print(f"  - Voxel length: {voxel_length} m")
        print(f"  - SDF trunc: {sdf_trunc}")
        print(f"  - Depth trunc: {depth_trunc} m")
        print(f"  - Down voxel size: {down_voxel_size} m")

        depth_datasets = {
            Side.LEFT: self.load_depth_dataset(Side.LEFT),
            Side.RIGHT: self.load_depth_dataset(Side.RIGHT)
        }

        pcd = generate_colorless_point_cloud_legacy(
            project_io_manager=self.io_manager,
            depth_datasets=depth_datasets,
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            depth_trunc=depth_trunc,
            down_voxel_size=down_voxel_size,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )

        print("[Info] Colorless point cloud generation completed.")

        self.io_manager.save_colorless_point_cloud_tensor(pcd)
        print("[Info] Colorless point cloud saved successfully.")

        pcd.paint_uniform_color(np.array([0.5, 0.5, 0.5]))

        return pcd