import itertools
from typing import Optional, cast
import numpy as np
import open3d as o3d
from config.reconstruction_config import FragmentPoseRefinementConfig
from dataio.depth_data_io import DepthDataIO
from dataio.reconstruction_data_io import ReconstructionDataIO
from models.camera_dataset import DepthDataset
from models.side import Side
from processing.reconstruction.o3d_utils import integrate
from utils.paralell_utils import parallel_map


def integrate_fragment_point_cloud(
    depth_data_io: DepthDataIO,
    frag_dataset: DepthDataset,
    side: Side,
    config: FragmentPoseRefinementConfig
) -> tuple[Side, o3d.t.geometry.PointCloud]:
    vbg = integrate(
        dataset=frag_dataset,
        depth_data_io=depth_data_io,
        side=side,
        voxel_size=config.voxel_size,
        block_resolution=config.block_resolution,
        block_count=config.block_count,
        depth_max=config.depth_max,
        trunc_voxel_multiplier=config.trunc_voxel_multiplier,
        device=config.device,
        show_progress=False,
        desc=None
    )

    return side, vbg.extract_point_cloud()


def integrate_and_save_fragment_point_clouds(
    depth_data_io: DepthDataIO,
    recon_data_io: ReconstructionDataIO,
    fragment_dataset_map: dict[Side, list[DepthDataset]],
    config: FragmentPoseRefinementConfig
):
    arg_list = []

    for side, frag_datasets in fragment_dataset_map.items():
        arg_list.extend([
            (depth_data_io, frag_dataset, side, config)
            for frag_dataset in frag_datasets
        ])

    results: list[Optional[tuple[Side, o3d.t.geometry.VoxelBlockGrid]]] = parallel_map(
        func=integrate_fragment_point_cloud,
        args_list=arg_list,
        max_workers=None,
        use_multiprocessing=config.use_multi_threading,
        context="spawn",
        default_on_error=None,
        show_progress=True,
        desc="[Info] Integrating Fragments..."
    )

    if any(result is None for result in results):
        raise Exception("Failed to integrate fragment point clouds.")
    
    frag_vbg_list = cast(list[tuple[Side, o3d.t.geometry.PointCloud]], results)

    indices_map: dict[Side, int] = {}
    for side, pcd in frag_vbg_list:
        if side not in indices_map:
            index = 0
        else:
            index = indices_map[side] + 1

        indices_map[side] = index

        recon_data_io.save_fragment_pcd(pcd=pcd, side=side, index=index)


def compute_pcd_pair_edge(
    recon_data_io: ReconstructionDataIO,
    node_side_index_list: list[tuple[Side, int]],
    source_node_index:int,
    target_node_index:int,
    config: FragmentPoseRefinementConfig,
    uncertain: bool,
) -> Optional[o3d.pipelines.registration.PoseGraphEdge]:
    source_side, source_index = node_side_index_list[source_node_index]
    target_side, target_index = node_side_index_list[target_node_index]

    source_pcd = recon_data_io.load_fragment_pcd(side=source_side, index=source_index)
    target_pcd = recon_data_io.load_fragment_pcd(side=target_side, index=target_index)

    icp_result = o3d.t.pipelines.registration.multi_scale_icp(
        source=source_pcd,
        target=target_pcd,
        voxel_sizes=o3d.utility.DoubleVector(config.icp_voxel_sizes),
        criteria_list=config.icp_criteria_list,
        max_correspondence_distances=o3d.utility.DoubleVector(config.max_corr_dists),
        init_source_to_target=np.eye(4),
        estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    if not icp_result.converged \
        or icp_result.fitness < config.icp_fitness_threshold \
        or icp_result.inlier_rmse > config.icp_inlier_rmse_threshold:
        raise Exception(f"[Error] Converged: {icp_result.converged}, Fitness: {icp_result.fitness}, Inlier RMSE: {icp_result.inlier_rmse}")
    
    info = o3d.t.pipelines.registration.get_information_matrix(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=config.max_corr_dists[-1],
        transformation=icp_result.transformation
    )

    edge = o3d.pipelines.registration.PoseGraphEdge(
        source_node_id=source_node_index,
        target_node_id=target_node_index,
        transformation=icp_result.transformation,
        information=info.cpu().numpy(),
        uncertain=uncertain,
        confidence=1.0
    )
    
    return edge


def build_pose_graph_for_scene(
    recon_data_io: ReconstructionDataIO,
    fragment_counts: dict[Side, int],
    config: FragmentPoseRefinementConfig
) -> o3d.pipelines.registration.PoseGraph:
    pose_graph = o3d.pipelines.registration.PoseGraph()

    # register nodes
    node_side_index_list: list[tuple[Side, int]] = []
    side_index_to_node_map: dict[tuple[Side, int], int] = {}

    node_index = 0
    for side, fragment_count in fragment_counts.items():
        node_side_index_list.extend([
            (side, index)
            for index in range(fragment_count)
        ])
        
        for index in range(fragment_count):
            side_index_to_node_map[(side, index)] = node_index
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(
                    pose=np.eye(4)
                )
            )
            node_index += 1

    args_list = []
    
    # odometry
    for side, fragment_count in fragment_counts.items():
        args_list.extend([
            (
                recon_data_io,
                node_side_index_list,
                side_index_to_node_map[(side, source_index)],
                side_index_to_node_map[(side, source_index + 1)],
                config,
                False,
            )
            for source_index in range(fragment_count - 1)
        ])

    # loop closure
    N = len(node_side_index_list)
    args_list.extend([
        (
            recon_data_io,
            node_side_index_list,
            source_index,
            target_index,
            config,
            True,
        )
        for source_index, target_index in itertools.combinations(range(N), 2)
    ])

    edges = parallel_map(
        compute_pcd_pair_edge,
        args_list=args_list,
        max_workers=None,
        use_multiprocessing=config.use_multi_threading,
        context="spawn",
        default_on_error=None,
        show_progress=True,
        desc="[Info] Computing PoseGraph edges..."
    )
    valid_edges = [edge for edge in edges if edge is not None]
    print(f"[Info] Valid edges: {len(valid_edges)} / {len(edges)}")

    pose_graph.edges.extend(valid_edges)

    return pose_graph


def refine_fragment_poses(
    depth_data_io: DepthDataIO,
    recon_data_io: ReconstructionDataIO,
    fragment_dataset_map: dict[Side, list[DepthDataset]],
    config: FragmentPoseRefinementConfig
):
    integrate_and_save_fragment_point_clouds(
        depth_data_io=depth_data_io,
        recon_data_io=recon_data_io,
        fragment_dataset_map=fragment_dataset_map,
        config=config
    )

    fragment_counts = {}
    for side, fragment_datasets in fragment_dataset_map.items():
        fragment_counts[side] = len(fragment_datasets)
    
    pose_graph = build_pose_graph_for_scene(
        recon_data_io=recon_data_io,
        fragment_counts=fragment_counts,
        config=config
    )