from pathlib import Path

from config.project_path_config import ProjectPathConfig
from dataio.depth_data_io import DepthDataIO
from dataio.image_data_io import ImageDataIO
from dataio.reconstruction_data_io import ReconstructionDataIO
from dataio.rgbd_data_io import RGBDDataIO


class DataIO:
    def __init__(self, project_dir: Path):
        self.path_config = ProjectPathConfig(project_dir=project_dir)
        self.color = ImageDataIO(image_path_config=self.path_config.image)
        self.depth = DepthDataIO(depth_path_config=self.path_config.depth)
        self.rgbd = RGBDDataIO(image_data_io=self.color, depth_data_io=self.depth, rgbd_path_config=self.path_config.rgbd)
        self.reconstruction = ReconstructionDataIO(reconstruction_path_config=self.path_config.reconstruction)