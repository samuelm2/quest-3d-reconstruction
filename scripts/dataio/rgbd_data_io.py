import numpy as np
from dataio.depth_data_io import DepthDataIO
from dataio.image_data_io import ImageDataIO
from config.project_path_config import RGBDPathConfig
from models.camera_dataset import CameraDataset
from models.side import Side


class RGBDDataIO:
    def __init__(self,
        image_data_io: ImageDataIO,
        depth_data_io: DepthDataIO,
        rgbd_path_config: RGBDPathConfig
    ):
        self.image_data_io = image_data_io
        self.depth_data_io = depth_data_io
        self.rgbd_path_config = rgbd_path_config

    
    def load_color_aligned_depth(self, side: Side, timestamp: int) -> np.ndarray:
        color_aligned_depth_path = self.rgbd_path_config.get_color_aligned_depth_path(side, timestamp)
        
        if not color_aligned_depth_path.exists():
            raise FileNotFoundError(f"Color-aligned depth file not found: {color_aligned_depth_path}")
        
        return np.load(color_aligned_depth_path)
    

    def save_color_aligned_depth(self, side: Side, timestamp: int, depth_data: np.ndarray) -> None:
        color_aligned_depth_path = self.rgbd_path_config.get_color_aligned_depth_path(side, timestamp)        
        color_aligned_depth_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(color_aligned_depth_path, depth_data)

    
    def load_rgbd_dataset(self, side: Side, use_cache: bool = True) -> CameraDataset:
        rgbd_dataset_path = self.rgbd_path_config.get_RGBD_dataset_path(side=side)

        if use_cache and rgbd_dataset_path.exists():
            try:
                print(f"[Info] Loading cached RGBD dataset for {side.name} from {rgbd_dataset_path} ...")
                return CameraDataset.load(rgbd_dataset_path)
            except Exception as e:
                print(f"[Error] RGBD dataset cache is corrupted or invalid. Rebuilding cache from the original source...\n{e}")

        else:
            print(f"[Info] RGBD dataset not found. Rebuilding cache from the original source...")

        rgbd_dataset = self.build_rgbd_dataset(side=side, use_cache=use_cache)
        rgbd_dataset.save(rgbd_dataset_path)

        return rgbd_dataset


    def build_rgbd_dataset(
        self, side: Side,
        use_cache: bool = True,
        interval_ms: int = 100,
    ) -> CameraDataset:
        depth_dataset = self.depth_data_io.load_depth_dataset(side=side, use_cache=use_cache)
        color_dataset = self.image_data_io.load_color_dataset(side=side, use_cache=use_cache)

        if len(depth_dataset.timestamps) == 0 or len(color_dataset.timestamps) == 0:
            raise ValueError(f"No data available for side {side.name}. Ensure that both depth and color datasets are loaded.")
        
        interval_indices = self.split_dual_timestamps(
            base_timestamps=depth_dataset.timestamps,
            other_timestamps=color_dataset.timestamps,
            interval_ms=interval_ms
        )

        pass



    @staticmethod
    def split_dual_timestamps(
        base_timestamps: np.ndarray, 
        other_timestamps: np.ndarray,
        interval_ms: int
    ) -> list[
    tuple[
        tuple[int, int], 
        tuple[int, int]
    ]
]:
        start_time = base_timestamps[0]
        end_time = base_timestamps[-1]

        bin_edges = np.arange(start_time, end_time + interval_ms, interval_ms)

        interval_indices = []

        for i in range(len(bin_edges) - 1):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]

            start_idx_base = np.searchsorted(base_timestamps, bin_start, side='left')
            end_idx_base = np.searchsorted(base_timestamps, bin_end, side='right')

            if start_idx_base == end_idx_base:
                continue

            start_idx_other = np.searchsorted(other_timestamps, bin_start, side='left')
            end_idx_other = np.searchsorted(other_timestamps, bin_end, side='right')

            if start_idx_other == end_idx_other:
                continue

            interval_indices.append(((start_idx_base, end_idx_base), (start_idx_other, end_idx_other)))

        return interval_indices