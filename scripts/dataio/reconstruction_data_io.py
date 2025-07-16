from typing import Optional
import open3d as o3d

from config.project_path_config import ReconstructionPathConfig
from models.camera_dataset import DepthDataset
from models.side import Side


class ReconstructionDataIO:
    def __init__(self, reconstruction_path_config: ReconstructionPathConfig):
        self.reconstruction_path_config = reconstruction_path_config


    def load_fragment_datasets(self) -> dict[Side, list[DepthDataset]]:
        fragment_path_map = self.reconstruction_path_config.get_fragment_dataset_paths()

        fragment_datasets: dict[Side, list[DepthDataset]] = {}

        for side, paths in fragment_path_map.items():
            fragment_datasets[side] = [DepthDataset.load(path) for path in paths]

        return fragment_datasets    

    
    def save_fragment_dataset(self, dataset: DepthDataset, side: Side, index: int):
        path = self.reconstruction_path_config.get_fragment_dataset_path(side=side, index=index)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save(path)

    
    def load_fragment_pcd(self, side: Side, index: int) -> o3d.t.geometry.PointCloud:
        path = self.reconstruction_path_config.get_fragment_pcd_path(side=side, index=index)
        return o3d.t.io.read_point_cloud(str(path))
    

    def save_fragment_pcd(self, pcd: o3d.t.geometry.PointCloud, side: Side, index: int):
        path = self.reconstruction_path_config.get_fragment_pcd_path(side=side, index=index)
        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.t.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=True)