from pathlib import Path
from enum import Enum

import config.path_config as path_config
from infra.io.yuv_repository import YUVRepository
from infra.io.rgb_repository import RGBRepository


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


class ProjectManager:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

        self.left_yuv = YUVRepository(
            yuv_dir=project_dir / path_config.LEFT_CAMERA_YUV_IMAGE_DIR,
            format_json=project_dir / path_config.LEFT_CAMERA_IMAGE_FORMAT_JSON
        )
        self.right_yuv = YUVRepository(
            yuv_dir=project_dir / path_config.RIGHT_CAMERA_YUV_IMAGE_DIR,
            format_json=project_dir / path_config.RIGHT_CAMERA_IMAGE_FORMAT_JSON
        )

        self.left_rgb = RGBRepository(project_dir / path_config.LEFT_CAMERA_RGB_IMAGE_DIR)
        self.right_rgb = RGBRepository(project_dir / path_config.RIGHT_CAMERA_RGB_IMAGE_DIR)


    def get_yuv_repo(self, side: Side) -> YUVRepository:
        if side == Side.LEFT:
            return self.left_yuv
        else:
            return self.right_yuv


    def get_rgb_repo(self, side: Side) -> RGBRepository:
        if side == Side.LEFT:
            return self.left_rgb
        else:
            return self.right_rgb
