import numpy as np
import open3d as o3d
from tqdm import tqdm

from domain.models.camera_dataset import DepthDataset
from domain.models.side import Side
from domain.models.transforms import CoordinateSystem
from infra.io.project_io_manager import ProjectIOManager


def generate_colorless_volume(
    project_io_manager: ProjectIOManager,
    depth_datasets: dict[Side, DepthDataset],
    voxel_length: float = 0.01,
    sdf_trunc: float = 0.05,
    depth_trunc: float = 2.0,
)-> o3d.geometry.PointCloud:
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
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

            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
            extrinsic = extrinsics[index]

            depth_np = project_io_manager.load_depth_map_by_index(
                side=side,
                index=index,
                dataset=depth_dataset,
                validate_depth=True
            )

            if depth_np is None:
                continue

            depth_np = np.flipud(np.fliplr(depth_np))
            depth_np[depth_np > depth_trunc] = 0

            depth = o3d.geometry.Image(np.ascontiguousarray(depth_np))
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8)),
                depth,
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=True
            )

            volume.integrate(rgbd, intrinsic, extrinsic)

    print("[Info] Colorless voxel block grid integration completed.")
    
    return volume


def generate_colorless_point_cloud_legacy(
    project_io_manager: ProjectIOManager,
    depth_datasets: dict[Side, DepthDataset],
    voxel_length: float = 0.01,
    sdf_trunc: float = 0.05,
    depth_trunc:float = 2.0,
    down_voxel_size: float = 0.02,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    volume = generate_colorless_volume(
        project_io_manager=project_io_manager,
        depth_datasets=depth_datasets,
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc,
    )

    print("[Info] Extracting point cloud from voxel block grid...")
    pcd = volume.extract_point_cloud()

    N = len(pcd.points)
    print(f"[Info] Extracted {N} points from voxel block grid.")

    print("[Info] Removing non-finite points from voxel block grid...")
    pcd = pcd.remove_non_finite_points()

    print("[Info] Downsampling point cloud using voxel grid...")
    pcd = pcd.voxel_down_sample(voxel_size=down_voxel_size)

    print("[Info] Removing statistical outliers from point cloud...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    print("[Info] Colorless point cloud generation completed.")

    return pcd