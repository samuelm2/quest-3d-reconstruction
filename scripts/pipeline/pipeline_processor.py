from pathlib import Path
from config.pipeline_configs import PipelineConfigs
from dataio.data_io import DataIO
from processing.yuv_conversion.convert_yuv_dir import convert_yuv_directory


class PipelineProcessor:
    def __init__(self, project_dir: Path, config_yml_path: Path):
        self.data_io = DataIO(project_dir=project_dir)
        self.pipeline_configs = PipelineConfigs.parse_config_yml(config_yml_path)


    def convert_yuv_to_rgb(self):
        convert_yuv_directory(image_io=self.data_io.image, config=self.pipeline_configs.yuv_to_rgb)