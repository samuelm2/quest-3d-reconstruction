from typing import Optional
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

    # Depth confidence estimation
    if config.estimate_depth_confidences:
        log_step("Estimate depth confidences")
        estimate_depth_confidences(
            depth_data_io=data_io.depth,
            config=config.confidence_estimation
        )

    # Depth pose optimization
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

    # TSDF integration
    vbg: Optional[o3d.t.geometry.VoxelBlockGrid] = None
    if config.use_colorless_vbg_cache:
        vbg = data_io.rgbd.load_colorless_vbg()

    if vbg is None:
        log_step("Integrate depth maps")
        integration_config = config.depth_integration

        for side, dataset in depth_dataset_map.items():
            vbg = integrate(
                dataset=dataset, 
                depth_data_io=data_io.depth, 
                side=side, 
                use_confidence_filtered_depth=integration_config.use_confidence_filtered_depth,
                confidence_threshold=integration_config.confidence_threshold,
                voxel_size=integration_config.voxel_size,
                block_resolution=integration_config.block_resolution,
                block_count=integration_config.block_count,
                depth_max=integration_config.depth_max,
                trunc_voxel_multiplier=integration_config.trunc_voxel_multiplier,
                device=integration_config.device,
                show_progress=True,
                desc=f"[{side.name}] Integrating depth maps ...",
                vbg_opt=vbg
            )

    if vbg is None:
        print("[Error] Failed to generate VoxelBlockGrid. Please check the integration parameters and input data.")
        return

    data_io.rgbd.save_colorless_vbg(vbg=vbg)

    print("[Info] Visualizing the generated point cloud...")
    pcds = []
    pcds.append(vbg.extract_point_cloud())

    legacy_pcds = [pcd.to_legacy() for pcd in pcds]

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries(legacy_pcds + [axis], window_name="Generated Point Cloud") # type: ignore