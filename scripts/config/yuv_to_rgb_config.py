from dataclasses import dataclass


@dataclass
class Yuv2RgbConfig:
    blur_filter: bool
    blur_threshold: float

    exposure_filter: bool
    exposure_threshold_low: float
    exposure_threshold_high: float

    tone_mapping: bool
    tone_mapping_method: str
    clahe_clip_limit: float
    clahe_tile_grid_size: int
    gamma_correction: float


    @classmethod
    def parse(cls, yuv_to_rgb_config_dict: dict) -> 'Yuv2RgbConfig':
        return cls(
            blur_filter=yuv_to_rgb_config_dict['blur_filter'],
            blur_threshold=yuv_to_rgb_config_dict['blur_threshold'],
            exposure_filter=yuv_to_rgb_config_dict['exposure_filter'],
            exposure_threshold_low=yuv_to_rgb_config_dict['exposure_threshold_low'],
            exposure_threshold_high=yuv_to_rgb_config_dict['exposure_threshold_high'],
            tone_mapping=yuv_to_rgb_config_dict.get('tone_mapping', False),
            tone_mapping_method=yuv_to_rgb_config_dict.get('tone_mapping_method', 'clahe'),
            clahe_clip_limit=yuv_to_rgb_config_dict.get('clahe_clip_limit', 2.0),
            clahe_tile_grid_size=yuv_to_rgb_config_dict.get('clahe_tile_grid_size', 8),
            gamma_correction=yuv_to_rgb_config_dict.get('gamma_correction', 1.2),
        )