from pathlib import Path

from models.side import Side


YUV_DIR_MAP = {
    Side.LEFT: 'left_camera_raw',
    Side.RIGHT: 'right_camera_raw'
}

RGB_DIR_MAP = {
    Side.LEFT: 'left_camera_rgb',
    Side.RIGHT: 'right_camera_tgb'
}

CAMERA_CHARACTERISTICS_JSON_MAP = {
    Side.LEFT: 'left_camera_characteristics.json',
    Side.RIGHT: 'right_camera_characteristics.json'    
}

CAMERA_FORMAT_INFO_JSON_MAP = {
    Side.LEFT: 'left_camera_image_format.json',
    Side.RIGHT: 'right_camera_image_format.json'
}

DEPTH_DIR_MAP = {
    Side.LEFT: 'left_depth',
    Side.RIGHT: 'right_depth'
}

DEPTH_DESCRIPTOR_CSV_MAP = {
    Side.LEFT: 'left_depth_descriptors.csv',
    Side.RIGHT: 'left_depth_descriptors.csv',
}

LINEAR_DEPTH_DIR_MAP = {
    Side.LEFT: 'left_depth_linear',
    Side.RIGHT: 'right_depth_linear'
}

DEPTH_DATASET_NPZ_MAP = {
    Side.LEFT: 'dataset/left_depth_dataset.npz',
    Side.RIGHT: 'dataset/right_depth_dataset.npz',
}

class ImagePathConfig:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir


    def get_yuv_dir(self, side: Side) -> Path:
        return self.project_dir / YUV_DIR_MAP[side]
    

    def get_yuv_image_paths(self, side: Side) -> list[Path]:
        yuv_dir = self.get_yuv_dir(side)
        return sorted(yuv_dir.glob("*.yuv"))
    

    def get_rgb_dir(self, side: Side) -> Path:
        return self.project_dir / RGB_DIR_MAP[side]
    

    def get_rgb_image_paths(self, side: Side) -> list[Path]:
        rgb_dir = self.get_rgb_dir(side)
        return sorted(rgb_dir.glob("*.png"))
    

    def get_camera_characteristic_json_path(self, side: Side) -> Path:
        return self.project_dir / CAMERA_CHARACTERISTICS_JSON_MAP[side]
    

    def get_camera_format_format_json_path(self, side: Side) -> Path:
        return self.project_dir / CAMERA_FORMAT_INFO_JSON_MAP[side]


class DepthPathConfig:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir


    def get_depth_dir(self, side: Side) -> Path:
        return self.project_dir / DEPTH_DIR_MAP[side]
    

    def get_depth_map_paths(self, side: Side) -> list[Path]:
        depth_dir = self.get_depth_dir(side)
        return sorted(depth_dir.glob("*.raw"))


    def get_depth_map_filename(self, timestamp: int) -> str:
        return f"{timestamp}.raw"


    def get_depth_map_path(self, side: Side, timestamp: int) -> Path:
        depth_dir = self.get_depth_dir(side=side)
        return depth_dir / self.get_depth_map_filename(timestamp=timestamp)


    def get_depth_descriptor_path(self, side: Side) -> Path:
        return self.project_dir / DEPTH_DESCRIPTOR_CSV_MAP[side]
    

    def get_depth_dataset_path(self, side: Side) -> Path:
        return self.project_dir / DEPTH_DATASET_NPZ_MAP[side]
    

    def get_linear_depth_dir(self, side: Side) -> Path:
        return self.project_dir / LINEAR_DEPTH_DIR_MAP[side]
    

    def get_relative_path(self, path: Path) -> Path:
        return path.relative_to(self.project_dir)
    

class ProjectPathConfig:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.image = ImagePathConfig(project_dir=project_dir)
        self.depth = DepthPathConfig(project_dir=project_dir)
