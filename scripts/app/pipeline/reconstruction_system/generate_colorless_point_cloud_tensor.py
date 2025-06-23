import numpy as np
import open3d as o3d
from tqdm import tqdm

from domain.models.camera_dataset import DepthDataset
from domain.models.side import Side
from domain.models.transforms import CoordinateSystem
from infra.io.project_io_manager import ProjectIOManager


def load_or_generate_colorless_vbg(
    project_io_manager: ProjectIOManager,
    depth_datasets: dict[Side, DepthDataset],
    no_cache: bool = False,
    block_resolution: int = 16,
    voxel_size: float = 0.01,
    block_count: int = 10_000,
    depth_max: float = 2.0,
    trunc_voxel_multiplier: float = 8.0,
    device: o3d.core.Device = o3d.core.Device("CUDA:0"),
) -> o3d.t.geometry.VoxelBlockGrid:
    if not no_cache and project_io_manager.colorless_vbg_tensor_exists():
        print("[Info] Colorless voxel block grid already exists. Loading from cache...")

        try:
            vbg = project_io_manager.load_colorless_voxel_grid_tensor()
            print("[Info] Colorless voxel block grid loaded successfully from cache.")
            return vbg
        except Exception as e:
            print(f"[Error] Failed to load colorless voxel block grid from cache. Rebuilding from depth datasets...\n{e}")
    else:
        print("[Info] Colorless voxel block grid does not exist. Integrating depth datasets into voxel block grid...")

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3d.core.Dtype.Float32, o3d.core.Dtype.Float32, o3d.core.Dtype.Float32),
        attr_channels=((1,),(1,),(3,)),
        block_resolution=block_resolution,
        voxel_size=voxel_size,
        block_count=block_count,
        device=device
    )

    for side in Side:
        depth_dataset = depth_datasets[side]
        if not depth_dataset:
            print(f"[Warning] No depth dataset found for side {side.name}. Skipping integration for this side.")
            continue

        N = len(depth_dataset.timestamps)

        print(f"[Info] Integrating {N} depth maps into the voxel block grid...")

        transforms = depth_dataset.transforms
        transforms = transforms.convert_coordinate_system(
            target_coordinate_system=CoordinateSystem.OPENGL,
            is_camera=True
        )

        extrinsics = transforms.extrinsics_wc

        for index in tqdm(range(N), desc="Integrating depth maps"):
            fx = depth_dataset.fx[index]
            fy = depth_dataset.fy[index]
            cx = depth_dataset.cx[index]
            cy = depth_dataset.cy[index]
            width = depth_dataset.widths[index]
            height = depth_dataset.heights[index]

            cx = width - cx
            cy = height - cy

            intrinsic = o3d.core.Tensor(
                [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ],
                dtype=o3d.core.Dtype.Float64,
            )
            extrinsic = o3d.core.Tensor(
                extrinsics[index],
                dtype=o3d.core.Dtype.Float64,
            )

            depth_np = project_io_manager.load_depth_map_by_index(
                side=side,
                index=index,
                dataset=depth_dataset,
                validate_depth=True
            )

            if depth_np is None:
                continue

            depth = o3d.t.geometry.Image(
                tensor=o3d.core.Tensor(
                    np.flipud(np.fliplr(depth_np)),
                    dtype=o3d.core.Dtype.Float32,
                    device=device
                )
            )

            block_coords = vbg.compute_unique_block_coordinates(
                depth=depth,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                depth_scale=1.0,
                depth_max=depth_max,
                trunc_voxel_multiplier=trunc_voxel_multiplier,
            )

            vbg.integrate(
                block_coords=block_coords,
                depth=depth,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                depth_scale=1.0,
                depth_max=depth_max,
                trunc_voxel_multiplier=trunc_voxel_multiplier,
            )

    print("[Info] Colorless voxel block grid integration completed. Saving to cache...")
    project_io_manager.save_colorless_voxel_grid_tensor(vbg)
    print("[Info] Colorless voxel block grid saved successfully.")
    
    return vbg


def generate_colorless_point_cloud_tensor(
    project_io_manager: ProjectIOManager,
    depth_datasets: dict[Side, DepthDataset],
    no_cache: bool = False,
    block_resolution: int = 16,
    voxel_size: float = 0.01,
    block_count: int = 10_000,
    depth_max: float = 2.0,
    weight_threshold: float = 3.0,
    trunc_voxel_multiplier: float = 8.0,
    down_voxel_size: float = 0.02,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    device: str = "CUDA:0",
) -> o3d.t.geometry.PointCloud:
    device = o3d.core.Device(device)

    vbg = load_or_generate_colorless_vbg(
        project_io_manager=project_io_manager,
        depth_datasets=depth_datasets,
        no_cache=no_cache,
        block_resolution=block_resolution,
        voxel_size=voxel_size,
        block_count=block_count,
        depth_max=depth_max,
        trunc_voxel_multiplier=trunc_voxel_multiplier,
        device=device
    )

    print("[Info] Extracting point cloud from voxel block grid...")
    pcd = vbg.extract_point_cloud(
        weight_threshold=weight_threshold
    )

    N = len(pcd.point.positions)
    print(f"[Info] Extracted {N} points from voxel block grid.")

    print("[Info] Removing non-finite points from voxel block grid...")
    pcd, _ = pcd.remove_non_finite_points()

    print("[Info] Downsampling point cloud using voxel grid...")
    pcd = pcd.voxel_down_sample(voxel_size=down_voxel_size)

    print("[Info] Removing statistical outliers from point cloud...")
    pcd, _ = pcd.remove_statistical_outliers(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    return pcd