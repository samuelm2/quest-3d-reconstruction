import open3d as o3d

from dataio.depth_data_io import DepthDataIO
from processing.reconstruction.make_fragments import make_fragments


def log_step(title: str):
    print("\n" + "="*40)
    print(f">>> [Step] {title}")
    print("="*40)


def reconstruct_scene(depth_data_io: DepthDataIO):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

    log_step("Make Fragments")
    fragments = make_fragments(depth_data_io=depth_data_io, use_cache=True)

    print("[Info] Visualizing the generated point cloud...")
    legacy_fragments = [f.to_legacy() for f in fragments]

    o3d.visualization.draw_geometries(legacy_fragments + [axis], window_name="Generated Point Cloud")