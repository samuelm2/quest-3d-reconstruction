import yaml
from pathlib import Path
from dataclasses import dataclass

from config.depth_to_linear_config import Depth2LinearConfig
from config.yuv_to_rgb_config import Yuv2RgbConfig


@dataclass
class PipelineConfigs:
    yuv_to_rgb: Yuv2RgbConfig
    depth_to_linear: Depth2LinearConfig


    @classmethod
    def parse_config_yml(cls, yml_path: Path) -> 'PipelineConfigs':
        with open(yml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        yuv_to_rgb = Yuv2RgbConfig.parse(config_dict['yuv_to_rgb'])
        depth_to_linear = Depth2LinearConfig.parse(config_dict['depth_to_linear'])

        return PipelineConfigs(
            yuv_to_rgb=yuv_to_rgb,
            depth_to_linear=depth_to_linear,
        )