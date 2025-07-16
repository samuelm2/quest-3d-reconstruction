from pathlib import Path
from config.pipeline_configs import PipelineConfigs
from dataio.data_io import DataIO
from processing.depth_conversion.convert_depth_to_linear import convert_depth_directory
from processing.reconstruction.reconstruct_scene import reconstruct_scene
from processing.yuv_conversion.convert_yuv_dir import convert_yuv_directory


class PipelineProcessor:
    def __init__(self, project_dir: Path, config_yml_path: Path):
        self.data_io = DataIO(project_dir=project_dir)
        self.pipeline_configs = PipelineConfigs.parse_config_yml(config_yml_path)


    def convert_yuv_to_rgb(self):
        convert_yuv_directory(image_io=self.data_io.image, config=self.pipeline_configs.yuv_to_rgb)

    
    def convert_depth_to_linear(self):
        convert_depth_directory(depth_data_io=self.data_io.depth, depth_to_linear_config=self.pipeline_configs.depth_to_linear)


    def reconstruct_scene(self):
        reconstruct_scene(depth_data_io=self.data_io.depth, recon_data_io=self.data_io.reconstruction)