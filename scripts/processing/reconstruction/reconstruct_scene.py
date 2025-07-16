import open3d as o3d
from tqdm import tqdm

from config.reconstruction_config import ReconstructionConfig
from dataio.depth_data_io import DepthDataIO
from dataio.reconstruction_data_io import ReconstructionDataIO
from processing.reconstruction.make_fragments import make_fragment_datasets
from processing.reconstruction.o3d_utils import integrate
from processing.reconstruction.refine_fragment_poses import refine_fragment_poses


def log_step(title: str):
    print("\n" + "="*40)
    print(f">>> [Step] {title}")
    print("="*40)


def load_or_make_fragment_dataset(depth_data_io: DepthDataIO, recon_data_io: ReconstructionDataIO, config: ReconstructionConfig):
    frag_dataset_loaded = False
    if config.use_fragment_dataset_cache:
        frag_dataset_map = recon_data_io.load_fragment_datasets()

        if len(frag_dataset_map) > 0 and any([len(frag_datasets) > 0 for frag_datasets in frag_dataset_map.values()]):
            print("[Info] Fragment datasets loaded from cache.")
            frag_dataset_loaded = True
    
    if not frag_dataset_loaded:
        log_step("Make Fragments")
        frag_dataset_map = make_fragment_datasets(depth_data_io=depth_data_io, config=config.fragment_generation)

        print("[Info] Saving fragment datasets to cache...")
        for side, frag_datasets in frag_dataset_map.items():
            for i, frag_dataset in enumerate(tqdm(frag_datasets, desc=f"[{side.name}] Saving fragment datasets...")):
                recon_data_io.save_fragment_dataset(dataset=frag_dataset, side=side, index=i)

    return frag_dataset_map


def reconstruct_scene(depth_data_io: DepthDataIO, recon_data_io: ReconstructionDataIO):
    # TODO: Inject as an argument
    config = ReconstructionConfig()

    frag_dataset_map = load_or_make_fragment_dataset(depth_data_io, recon_data_io, config)
    refine_fragment_poses(depth_data_io, recon_data_io, frag_dataset_map, config.fragment_pose_refinement)

    # print("[Info] Visualizing the generated point cloud...")
    # fragments = []
    # for side, frag_datasets in frag_dataset_map.items():
    #     for frag_dataset in frag_datasets:
    #         vgb = integrate(frag_dataset, depth_data_io, side, 0.01, 16, 50_000, 1.5, 8.0, o3d.core.Device("CUDA:0"))
    #         fragments.append(vgb.extract_point_cloud())
# 
    # legacy_fragments = [f.to_legacy() for f in fragments]
# 
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries(legacy_fragments + [axis], window_name="Generated Point Cloud")