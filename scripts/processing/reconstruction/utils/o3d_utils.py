from typing import Optional, cast
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from dataio.depth_data_io import DepthDataIO
from models.camera_dataset import DepthDataset
from models.side import Side
from models.transforms import CoordinateSystem, Transforms


def compute_o3d_intrinsic_matrices(dataset: DepthDataset) -> np.ndarray:
    widths = dataset.widths
    intrinsic_matrices = dataset.get_intrinsic_matrices()
    intrinsic_matrices[:, 0, 2] = widths - intrinsic_matrices[:, 0, 2]

    return intrinsic_matrices


def convert_transforms_to_pose_graph(transforms: Transforms) -> o3d.pipelines.registration.PoseGraph:
    pose_graph = o3d.pipelines.registration.PoseGraph()

    extrinsics_cw = transforms.extrinsics_cw
    N = len(extrinsics_cw)

    for i in range(N):
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                pose=extrinsics_cw[i]
            )
        )
    
    return pose_graph


def convert_pose_graph_to_transforms(pose_graph: o3d.pipelines.registration.PoseGraph) -> Transforms:
    pose = np.array([node.pose for node in pose_graph.nodes])

    return Transforms(
        coordinate_system=CoordinateSystem.OPEN3D,
        positions=pose[:, :3, 3],
        rotations=R.from_matrix(pose[:, :3, :3]).as_quat(),
    )


def load_depth_map(
    depth_data_io: DepthDataIO,
    side: Side,
    index: int,
    dataset: DepthDataset,
    device: o3d.core.Device,
) -> Optional[o3d.t.geometry.Image]:
    depth_np = depth_data_io.load_depth_map(
        side=side,
        timestamp=dataset.timestamps[index],
        width=dataset.widths[index],
        height=dataset.heights[index],
        near=dataset.nears[index],
        far=dataset.fars[index],
    )

    if depth_np is None:
        return None

    return o3d.t.geometry.Image(
        tensor=o3d.core.Tensor(
            depth_np,
            dtype=o3d.core.Dtype.Float32,
            device=device
        )
    )


def integrate(
    dataset: DepthDataset,
    depth_data_io: DepthDataIO,
    side: Side,
    voxel_size: float,
    block_resolution: int,
    block_count: int,
    depth_max: float,
    trunc_voxel_multiplier: float,
    device: o3d.core.Device,
    show_progress: bool = False,
    desc: Optional[str] = None,
    vbg_opt: Optional[o3d.t.geometry.VoxelBlockGrid] = None,
) -> o3d.t.geometry.VoxelBlockGrid:
    if vbg_opt is None:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight'),
            attr_dtypes=(o3d.core.float32, o3d.core.float32),
            attr_channels=((1), (1)),
            voxel_size=voxel_size,
            block_resolution=block_resolution,
            block_count=block_count,
            device=device,
        )
    else:
        vbg = vbg_opt

    N = len(dataset.timestamps)

    extrinsic_wc = dataset.transforms.extrinsics_wc
    intrinsic_matrices = compute_o3d_intrinsic_matrices(dataset=dataset)

    def integrate(index: int):
        depth_map = load_depth_map(
            depth_data_io=depth_data_io,
            side=side,
            index=index,
            dataset=dataset,
            device=device
        )

        if depth_map is None:
            return

        intrinsic = o3d.core.Tensor(
            intrinsic_matrices[index],
            dtype=o3d.core.Dtype.Float64
        )
        extrinsic = o3d.core.Tensor(
            extrinsic_wc[index],
            dtype=o3d.core.Dtype.Float64
        )

        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth=depth_map,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            depth_scale=1.0,
            depth_max=float(depth_max),
            trunc_voxel_multiplier=float(trunc_voxel_multiplier),
        )

        vbg.integrate(
            block_coords=frustum_block_coords,
            depth=depth_map,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            depth_scale=1.0,
            depth_max=float(depth_max),
            trunc_voxel_multiplier=float(trunc_voxel_multiplier),
        )

    if show_progress:
        for index in tqdm(range(N), desc=desc):
            integrate(index)
    else:
        for index in range(N):
            integrate(index)

    return vbg