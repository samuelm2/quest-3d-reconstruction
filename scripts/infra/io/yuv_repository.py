import json
from pathlib import Path
import numpy as np

from domain.image_format_info import ImageFormatInfo, ImagePlaneInfo, BaseTime


def load_image_format_info(format_json: Path) -> ImageFormatInfo:
    with open(format_json) as f:
        format_info_dict = json.load(f)
    
    width = format_info_dict["width"]
    height = format_info_dict["height"]

    format = format_info_dict["format"]

    planes = [
        ImagePlaneInfo(
            buffer_size=plane["bufferSize"],
            row_stride=plane["rowStride"],
            pixel_stride=plane["pixelStride"]
        ) for plane in format_info_dict["planes"]
    ]

    base_time_dict = format_info_dict["baseTime"]
    base_time = BaseTime(
        mono_time_ns=base_time_dict["baseMonoTimeNs"],
        unix_time_ns=base_time_dict["baseUnixTimeMs"]
    )


    return ImageFormatInfo(
        width=width,
        height=height,
        format=format,
        planes=planes,
        base_time=base_time
    )


class YUVRepository:
    def __init__(
        self,
        yuv_dir: Path,
        format_json: Path
    ):
        self.yuv_dir = yuv_dir
        self.format_json = format_json
        self._image_format_info = None


    @property
    def image_format_info(self) -> ImageFormatInfo:
        if self._image_format_info is None:
            self._image_format_info = load_image_format_info(self.format_json)
        
        return self._image_format_info


    @property
    def paths(self) -> list[Path]:
        return sorted(self.yuv_dir.glob("*.yuv"))
    

    def load(self, path: Path) -> np.ndarray:
        return np.fromfile(path, dtype=np.uint8)