from pathlib import Path
from typing import Callable, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import numpy as np
from tqdm import tqdm

from domain.utils.image_utils import convert_yuv420_888_to_bgr, is_valid_image
from infra.io.yuv_repository import YUVRepository
from infra.io.image_repository import ImageRepository


def process_file(
    yuv_file: Path,
    yuv_repo: YUVRepository,
    rgb_repo: ImageRepository,
    is_valid_image: Union[Callable[[np.ndarray], bool], None] = None,
) -> bool:
    try:
        raw_data = yuv_repo.load(yuv_file)
        format_info = yuv_repo.image_format_info

        bgr_img = convert_yuv420_888_to_bgr(raw_data, format_info)

        if is_valid_image:
            if not is_valid_image(bgr_img):
                return False

        file_stem = yuv_file.stem
        rgb_repo.save(file_stem=file_stem, image=bgr_img)

        return True

    except Exception:
        raise RuntimeError(f"Failed in {yuv_file}:\n{traceback.format_exc()}")


def convert_yuv_directory_to_png(
    yuv_repo: YUVRepository,
    rgb_repo: ImageRepository,
    apply_filter: bool = False,
    blur_threshold: float = 50.0,
    exposure_threshold_low: float = 0.1,
    exposure_threshold_high: float = 0.1
):
    filter = None

    if apply_filter:
        def filter(bgr_img): return is_valid_image(
            bgr_img,
            blur_threshold=blur_threshold,
            exposure_threshold_low=exposure_threshold_low,
            exposure_threshold_high=exposure_threshold_high
        )

    yuv_files = yuv_repo.paths

    excluded_count = 0
    processed_count = 0
    exception_count = 0

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, yuv_file, yuv_repo, rgb_repo, filter)
            for yuv_file in yuv_files
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting YUV to PNG"):
            try:
                result = future.result()
                if result:
                    processed_count += 1
                else:
                    excluded_count += 1

            except Exception as e:
                print(f"[Exception] Worker failed: {e}")
                exception_count += 1
                continue            


    print(f"[Info] {processed_count} images written to {rgb_repo.image_dir}")

    if is_valid_image:
        print(f"[Info] {excluded_count} images were excluded by filtering.")

    if exception_count > 0:
        print(f"[Error] {exception_count} files failed due to exceptions.")