from pathlib import Path

from config.project_path_config import ProjectPathConfig
from dataio.image_data_io import ImageDataIO


class DataIO:
    def __init__(self, project_dir: Path):
        self.path_config = ProjectPathConfig(project_dir=project_dir)
        self.image = ImageDataIO(image_path_config=self.path_config.image)