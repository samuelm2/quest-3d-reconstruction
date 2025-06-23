from tqdm import tqdm

from domain.models.camera_dataset import DepthDataset
from infra.io.project_io_manager import ProjectIOManager


def convert_depth_directory_to_linear(
    project_io_manager: ProjectIOManager,
    side: str,
    dataset: DepthDataset,
    clip_near = 0.1,
    clip_far = 100.0,
):
    depth_grey_repo = project_io_manager.get_depth_grey_repo(side)

    num_frames = len(dataset.timestamps)

    for i in tqdm(range(num_frames), total=num_frames, desc="Converting depth images"):
        timestamp = dataset.timestamps[i]

        depth_map = project_io_manager.load_depth_map_by_index(
            side=side,
            index=i,
            dataset=dataset,
            validate_depth=True,
        )

        if depth_map is None:
            continue

        linear_depth_map = (depth_map - clip_near) / (clip_far - clip_near) * 255.0
        depth_grey_repo.save(file_stem=timestamp, image=linear_depth_map)

    print(f"[Info] Converted depth images for {side} camera to linear format.")