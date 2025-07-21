import open3d as o3d

from dataclasses import dataclass, field


@dataclass
class DepthConfidenceEstimationConfig:
    target_frame_range: int = 10
    depth_max: float = 3.0
    error_threshold: float = 0.05

    skip_if_output_dir_exists: bool = True

    use_dataset_cache: bool = True
    use_multi_threading: bool = True



@dataclass
class FragmentGenerationConfig:
    fragment_size: int = 100

    use_confidence_filtered_depth: bool = True
    confidence_threshold: float = 0.05
    depth_max: float = 3.0

    odometry_loop_interval: int = 10
    overlap_ratio_threshold: float = 0.1
    loop_yaw_info_density_threshold: float = 0.3

    dist_threshold: float = 0.07
    edge_prune_threshold: float = 0.25

    device: o3d.core.Device = o3d.core.Device("CUDA:0")

    use_dataset_cache: bool = True
    use_multi_threading: bool = False


@dataclass
class FragmentPoseRefinementConfig:
    use_confidence_filtered_depth: bool = True
    confidence_threshold: float = 0.05

    voxel_size: float = 0.01
    block_resolution: int = 16
    block_count: int = 50_000
    depth_max: float = 1.5
    trunc_voxel_multiplier: float = 8.0

    use_pre_filtering: bool = True
    pre_filter_every_k_points: float = 30
    pre_filter_max_corr_dist: float = 0.1
    pre_filter_inlier_rmse_threshold: float = 0.05
    pre_filter_fitness_threshold: float = 0.2

    icp_voxel_sizes: list[float] = field(default_factory=lambda: [0.05, 0.025, 0.0125])
    max_corr_dists: list[float] = field(default_factory=lambda: [0.1, 0.05, 0.025])
    max_iterations: list[int] = field(default_factory=lambda: [50, 31, 14])
    relative_fitnesses: list[float] = field(default_factory=lambda: [1e-6, 1e-6, 1e-6])
    relative_rmses: list[float] = field(default_factory=lambda: [1e-6, 1e-6, 1e-6])

    icp_fitness_threshold: float = 0.2
    icp_inlier_rmse_threshold: float = 0.05

    dist_threshold: float = 0.07
    edge_prune_threshold: float = 0.25

    device: o3d.core.Device = o3d.core.Device("CUDA:0")

    use_multi_threading: bool = False

    @property
    def icp_criteria_list(self):
        return [
            o3d.t.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations[i],
                relative_fitness=self.relative_fitnesses[i],
                relative_rmse=self.relative_rmses[i]
            )
            for i in range(len(self.icp_voxel_sizes))
        ]


@dataclass
class IntegrationConfig:
    use_confidence_filtered_depth: bool = True
    confidence_threshold: float = 0.05
    voxel_size: float = 0.01
    block_resolution: int = 16
    block_count: int = 50_000
    depth_max: float = 1.5
    trunc_voxel_multiplier: float = 8.0
    device: o3d.core.Device = o3d.core.Device("CUDA:0")


@dataclass
class ReconstructionConfig:
    confidence_estimation: DepthConfidenceEstimationConfig = DepthConfidenceEstimationConfig()
    fragment_generation: FragmentGenerationConfig = FragmentGenerationConfig()
    fragment_pose_refinement: FragmentPoseRefinementConfig = FragmentPoseRefinementConfig()
    depth_integration: IntegrationConfig = IntegrationConfig()

    estimate_depth_confidences: bool = True

    optimize_depth_pose: bool = True
    use_fragment_dataset_cache: bool = True
    use_optimized_dataset_cache: bool = True

    use_colorless_vbg_cache: bool = True

    generate_color_aligned_depth_maps: bool = True