import open3d as o3d

from config.reconstruction_config import ReconstructionConfig
from dataio.data_io import DataIO
from models.camera_dataset import DepthDataset
from models.side import Side
from models.transforms import CoordinateSystem
from processing.reconstruction.confidence_estimation.estimate_depth_confidences import estimate_depth_confidences
from processing.reconstruction.depth_optimization.depth_pose_optimizer import DepthPoseOptimizer
from processing.reconstruction.utils.log_utils import log_step
from processing.reconstruction.utils.o3d_utils import integrate


def reconstruct_scene(data_io: DataIO):
    # TODO: Inject as an argument
    config = ReconstructionConfig()

    if config.estimate_depth_confidences:
        log_step("Estimate depth confidences")
        estimate_depth_confidences(
            depth_data_io=data_io.depth,
            config=config.confidence_estimation
        )

    if config.optimize_depth_pose:
        optimizer = DepthPoseOptimizer(
            depth_data_io=data_io.depth,
            recon_data_io=data_io.reconstruction,
            config=config
        )
        depth_dataset_map = optimizer()
    else: 
        depth_dataset_map: dict[Side, DepthDataset] = {}
        for side in Side:
            dataset = data_io.depth.load_depth_dataset(
                side=side,
                use_cache=config.fragment_generation.use_dataset_cache
            )
            dataset.transforms = dataset.transforms.convert_coordinate_system(
                target_coordinate_system=CoordinateSystem.OPEN3D,
                is_camera=True
            )
            depth_dataset_map[side] = dataset

    print("[Info] Visualizing the generated point cloud...")
    pcds = []
    for side, dataset in depth_dataset_map.items():
        vgb = integrate(
            dataset, 
            data_io.depth, 
            side, 
            True, 0.05,
            0.01, 16, 50_000, 1.5, 8.0, 
            o3d.core.Device("CUDA:0")
        )
        pcds.append(vgb.extract_point_cloud())

    legacy_pcds = [pcd.to_legacy() for pcd in pcds]

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries(legacy_pcds + [axis], window_name="Generated Point Cloud") # type: ignore