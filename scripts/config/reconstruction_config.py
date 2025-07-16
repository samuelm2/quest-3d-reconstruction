import open3d as o3d

from dataclasses import dataclass


@dataclass
class FragmentGenerationConfig:
    fragment_size: int = 100

    depth_max: float = 3.0

    odometry_loop_interval: int = 10
    overlap_ratio_threshold: float = 0.1
    loop_yaw_info_density_threshold: float = 0.3

    dist_threshold: float = 0.07
    edge_prune_threshold: float = 0.25

    device: o3d.core.Device = o3d.core.Device("CUDA:0")

    use_dataset_cache: bool = True
    use_multi_threading: bool = True


@dataclass
class FragmentPoseRefinementConfig:
    pass


@dataclass
class ReconstructionConfig:
    fragment_generation: FragmentGenerationConfig = FragmentGenerationConfig()
    fragment_pose_refinement: FragmentPoseRefinementConfig = FragmentPoseRefinementConfig()

    use_fragment_dataset_cache: bool = True
