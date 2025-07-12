import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from config.project_path_config import ImagePathConfig
from models.camera_characteristics import CameraCharacteristics
from models.image_format_info import BaseTime, ImageFormatInfo, ImagePlaneInfo
from models.side import Side


class ImageDataIO:
    def __init__(self, image_path_config: ImagePathConfig):
        self.image_path_config = image_path_config

    
    def get_yuv_timestamps(self, side: Side) -> list[int]:
        yuv_files = self.image_path_config.get_yuv_image_paths(side=side)
        return [
            int(yuv_file.stem)
            for yuv_file
            in yuv_files
        ]

    
    def get_rgb_timestamps(self, side: Side) -> list[int]:
        rgb_files = self.image_path_config.get_rgb_image_paths(side=side)
        return [
            int(rgb_file.stem)
            for rgb_file
            in rgb_files
        ]

    
    def load_yuv(self, side: Side, timestamp: int) -> np.ndarray:
        yuv_dir = self.image_path_config.get_yuv_dir(side=side)
        file_path = yuv_dir / f'{timestamp}.yuv'
        return np.fromfile(file_path, dtype=np.uint8)
    

    def load_rgb(self, side: Side, timestamp: int) -> np.ndarray:
        rgb_dir = self.image_path_config.get_rgb_dir(side=side)
        file_path = rgb_dir / f'{timestamp}.png'
        return cv2.imread(file_path)
    

    def save_rgb(self, rgb: np.ndarray, side: Side, timestamp: int):
        rgb_dir = self.image_path_config.get_rgb_dir(side=side)
        rgb_dir.mkdir(parents=True, exist_ok=True)

        file_path = rgb_dir / f'{timestamp}.png'

        cv2.imwrite(str(file_path), rgb)


    def save_bgr(self, bgr: np.ndarray, side: Side, timestamp: int):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.save_rgb(rgb=rgb, side=side, timestamp=timestamp)


    def load_image_format_info(self, side: Side) -> ImageFormatInfo:
        format_json_path = self.image_path_config.get_camera_format_format_json_path(side)

        with open(format_json_path) as f:
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


    def load_camera_characteristics(self, side: Side) -> CameraCharacteristics:
        characteristics_json_path = self.image_path_config.get_camera_characteristic_json_path(side)

        with open(characteristics_json_path, "r", encoding="utf-8") as f:
            camera_characteristics = json.load(f)

        array_size = camera_characteristics["sensor"]["activeArraySize"]
        width = array_size["right"] - array_size["left"]
        height = array_size["bottom"] - array_size["top"]

        intrinsics = camera_characteristics["intrinsics"]

        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]

        camera_pose = camera_characteristics["pose"]

        transl = camera_pose["translation"]
        transl[2] *= -1
        if len(transl) < 3:
            transl = np.array((0, 0, 0))

        rot_quat = camera_pose["rotation"]
        if len(rot_quat) >=4:
            qx = -rot_quat[0]
            qy = -rot_quat[1]
            qz = rot_quat[2]
            qw = rot_quat[3]

            rot = R.from_quat((qx, qy, qz, qw)).inv()
            # Apply a 180-degree rotation to align the Android camera pose with the HMD world coordinate system.
            rot *= R.from_euler('x', np.pi)

            rot_quat = rot.as_quat()
            
        else:
            rot_quat = np.array((0, 0, 0, 1))

        return CameraCharacteristics(
            width=width,
            height=height,
            fx=fx, fy=fy,
            cx=cx, cy=cy,
            transl=transl,
            rot_quat=rot_quat,
        )