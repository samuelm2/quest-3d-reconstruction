from pathlib import Path
import argparse
import open3d as o3d

from app.project_manager import ProjectManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir", "-p",
        type=Path,
        required=True,
        help="Path to the project directory"
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable caching and force regeneration of the point cloud"
    )
    parser.add_argument(
        "--color",
        action='store_true',
        help="Sample volume color from the color dataset"
    )
    parser.add_argument(
        "--use_legacy",
        action="store_true",
        help="Use tensor-based TSDF volume integration"
    )
    parser.add_argument(
        "--visualize",
        action='store_true',
        help="Visualize the TSDF volume"
    )

    tensor_group = parser.add_argument_group("Tensor Option")
    tensor_group.add_argument(
        "--tensor_block_resolution",
        type=int,
        default=16,
        help="Resolution of the voxel block grid (default: 16)"
    )
    tensor_group.add_argument(
        "--tensor_voxel_size",
        type=float,
        default=0.01,
        help="Length of the voxel in meters for TSDF volume (default: 0.01 m)"
    )
    tensor_group.add_argument(
        "--tensor_block_count",
        type=int,
        default=10_000,
        help="Number of blocks in the voxel block grid (default: 10,000)"
    )
    tensor_group.add_argument(
        "--tensor_depth_max",
        type=float,
        default=2.0,
        help="Maximum depth value for TSDF volume (default: 2.0 m)"
    )
    tensor_group.add_argument(
        "--tensor_weight_threshold",
        type=float,
        default=3.0,
        help="Weight threshold for point cloud extraction (default: 3.0)"
    )
    tensor_group.add_argument(
        "--tensor_trunc_voxel_multiplier",
        type=float,
        default=5.0,
        help="Multiplier for trunk voxel size (default: 5.0)"
    )
    tensor_group.add_argument(
        "--tensor_nb_neighbors",
        type=int,
        default=20,
        help="Number of neighbors for point cloud smoothing (default: 20)"
    )
    tensor_group.add_argument(
        "--tensor_std_ratio",
        type=float,
        default=2.0,
        help="Standard deviation ratio for point cloud smoothing (default: 2.0)"
    )
    tensor_group.add_argument(
        "--tensor_down_voxel_size",
        type=float,
        default=0.02,
        help="Voxel size for downsampling. Controls how much to reduce point density (default: 0.02)"
    )
    tensor_group.add_argument(
        "--tensor_device",
        type=str,
        default="CUDA:0",
        help="Device to use for processing (default: 'CUDA:0'). Use 'CPU:0' for CPU processing"
    )

    legacy_group = parser.add_argument_group("Legacy Option")
    legacy_group.add_argument(
        "--legacy_voxel_length",
        type=float,
        default=0.01,
        help="Length of the voxel in meters for TSDF volume (default: 0.01 m)"
    )
    legacy_group.add_argument(
        "--legacy_sdf_trunc",
        type=float,
        default=0.05,
        help="Truncation value for signed distance function (default: 0.05 m)"
    )
    legacy_group.add_argument(
        "--legacy_down_voxel_size",
        type=float,
        default=0.02,
        help="Voxel size for downsampling the point cloud (default: 0.02 m)"
    )
    legacy_group.add_argument(
        "--legacy_nb_neighbors",
        type=int,
        default=20,
        help="Number of neighbors for point cloud smoothing (default: 20)"
    )
    legacy_group.add_argument(
        "--legacy_std_ratio",
        type=float,
        default=2.0,
        help="Standard deviation ratio for point cloud smoothing (default: 2.0)"
    )

    return parser.parse_args()


def main(args):
    project_dir = args.project_dir
    print(f"[Info] Project path: {project_dir}")

    project_manager = ProjectManager(project_dir=project_dir)

    if args.use_legacy:
        pcd = project_manager.generate_point_cloud_legacy(
            color=args.color,
            voxel_length=args.legacy_voxel_length,
            sdf_trunc=args.legacy_sdf_trunc,
            down_voxel_size=args.legacy_down_voxel_size,
            nb_neighbors=args.legacy_nb_neighbors,
            std_ratio=args.legacy_std_ratio,
        )
    else:
        pcd = project_manager.generate_point_cloud_tensor(
            no_cache=args.no_cache,
            color=args.color,
            block_resolution=args.tensor_block_resolution,
            voxel_size=args.tensor_voxel_size,
            block_count=args.tensor_block_count,
            depth_max=args.tensor_depth_max,
            weight_threshold=args.tensor_weight_threshold,
            trunc_voxel_multiplier=args.tensor_trunc_voxel_multiplier,
            nb_neighbors=args.tensor_nb_neighbors,
            std_ratio=args.tensor_std_ratio,
            down_voxel_size=args.tensor_down_voxel_size,
            device=args.tensor_device
        )

    if args.visualize:
        print("[Info] Visualizing the generated point cloud...")
        o3d.visualization.draw_geometries([pcd], window_name="Generated Point Cloud")


if __name__ == "__main__":
    args = parse_args()
    main(args)