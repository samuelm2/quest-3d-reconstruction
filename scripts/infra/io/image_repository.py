from pathlib import Path
import numpy as np
import cv2


class ImageRepository:
    def __init__(
        self, project_root: Path, 
        image_subdir: str
    ):
        self.project_root = project_root
        self.image_dir = project_root / image_subdir


    @property
    def paths(self) -> list[Path]:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory {self.image_dir} does not exist.")
        
        return sorted(self.image_dir.glob("*.png"))


    def get_relaive_path(self, file_stem: str) -> str:
        depth_map_path = self.image_dir / f"{file_stem}.raw"
        return depth_map_path.relative_to(self.project_root)
    

    def load(self, file_stem: str) -> np.ndarray:
        image_path = self.image_dir / f"{file_stem}.png"
        img_bgr = cv2.imread(image_path)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        

    def save(self, file_stem: str, image: np.ndarray):
        self.image_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.image_dir / f"{file_stem}.png"
        cv2.imwrite(str(output_path), image)