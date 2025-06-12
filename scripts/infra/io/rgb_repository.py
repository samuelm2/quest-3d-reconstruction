from pathlib import Path
import numpy as np
import cv2


class RGBRepository:
    def __init__(self, rgb_dir: Path):
        self.rgb_dir = rgb_dir


    @property
    def paths(self) -> list[Path]:
        return sorted(self.rgb_dir.glob("*.png"))
    

    def save(self, file_stem: str, bgr_img: np.ndarray):
        self.rgb_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.rgb_dir / f"{file_stem}.png"
        cv2.imwrite(str(output_path), bgr_img)