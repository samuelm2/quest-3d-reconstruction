import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from typing import Union, Callable
from pathlib import Path
import numpy as np
from tqdm import tqdm

from domain.image_utils import convert_yuv420_888_to_bgr, is_valid_image
from infra.io.project_manager import Side, ProjectManager
from infra.io.yuv_repository import YUVRepository
from infra.io.rgb_repository import RGBRepository


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir", "-p",
        type=Path,
        required=True,
        help="Path to the project directory containing QRC data."
    )
    parser.add_argument(
        "--filter", action="store_true", default=False,
        help="Enable image quality filtering."
    )
    parser.add_argument(
        "--blur_threshold", type=float, default=50.0,
        help="Blur threshold (Laplacian variance). Lower means more blur. Default: 50.0"
    )
    parser.add_argument(
        "--exposure_threshold_low", type=float, default=0.1,
        help="Cumulative histogram threshold to detect underexposure. Default: 0.1"
    )
    parser.add_argument(
        "--exposure_threshold_high", type=float, default=0.1,
        help="Cumulative histogram threshold to detect overexposure. Default: 0.1"
    )
    args = parser.parse_args()

    if not args.project_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.project_dir}")

    return args


def process_file(
    yuv_file: Path,
    yuv_repo: YUVRepository,
    rgb_repo: RGBRepository,
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
        rgb_repo.save(file_stem=file_stem, bgr_img=bgr_img)

        return True

    except Exception:
        raise RuntimeError(f"Failed in {yuv_file}:\n{traceback.format_exc()}")


def convert_yuv_directory_to_png(
    yuv_repo: YUVRepository,
    rgb_repo: RGBRepository,
    is_valid_image: Union[Callable[[np.ndarray], bool], None] = None,
):
    yuv_files = yuv_repo.paths

    excluded_count = 0
    processed_count = 0
    exception_count = 0

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, yuv_file, yuv_repo, rgb_repo, is_valid_image)
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


    print(f"[Info] {processed_count} images written to {rgb_repo.rgb_dir}")

    if is_valid_image:
        print(f"[Info] {excluded_count} images were excluded by filtering.")

    if exception_count > 0:
        print(f"[Error] {exception_count} files failed due to exceptions.")


def main(args):
    project_manager = ProjectManager(project_dir=args.project_dir)

    for side in Side:
        print(f"[Info] Converting {side} camera images...")

        filter = None
        if args.filter:
            def filter(bgr_img): return is_valid_image(
                bgr_img,
                blur_threshold=args.blur_threshold,
                exposure_threshold_low=args.exposure_threshold_low,
                exposure_threshold_high=args.exposure_threshold_high
            )

        yuv_repo = project_manager.get_yuv_repo(side=side)
        rgb_repo = project_manager.get_rgb_repo(side=side)

        convert_yuv_directory_to_png(
            yuv_repo=yuv_repo,
            rgb_repo=rgb_repo,
            is_valid_image=filter,
        )


if __name__ == "__main__":
    args = parse_args()

    print(f"[Info] Project Directory: {args.project_dir}")

    main(args)