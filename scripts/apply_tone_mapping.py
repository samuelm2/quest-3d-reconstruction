"""
Script to apply tone mapping to existing RGB images.

This is useful for improving dynamic range in scenes with bright windows
and dark interiors. Run this BEFORE color optimization to improve results.

Usage:
    python apply_tone_mapping.py --project_dir /path/to/project --method clahe
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import cv2
import numpy as np
from tqdm import tqdm

from dataio.data_io import DataIO
from models.side import Side
from utils.image_utils import apply_tone_mapping


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply tone mapping to existing RGB images to improve dynamic range."
    )
    parser.add_argument(
        "--project_dir", "-p",
        type=Path,
        required=True,
        help="Path to the project directory containing RGB images."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="clahe",
        choices=["clahe", "gamma", "clahe+gamma"],
        help="Tone mapping method to use."
    )
    parser.add_argument(
        "--clip_limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit (higher = more contrast). Default: 2.0"
    )
    parser.add_argument(
        "--tile_grid_size",
        type=int,
        default=8,
        help="CLAHE tile grid size. Default: 8 (creates 8x8 grid)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.2,
        help="Gamma correction value (>1 brightens). Default: 1.2"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup original images before processing."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers. Default: 8"
    )
    
    args = parser.parse_args()
    
    if not args.project_dir.is_dir():
        parser.error(f"Project directory does not exist: {args.project_dir}")
    
    return args


def process_image(
    side: Side,
    timestamp: int,
    data_io: DataIO,
    method: str,
    clip_limit: float,
    tile_grid_size: int,
    gamma: float,
    backup: bool
) -> tuple[bool, str]:
    """Process a single image with tone mapping."""
    try:
        # Load RGB image
        rgb_path = data_io.path_config.image.get_rgb_file_path(side=side, timestamp=timestamp)
        
        if not rgb_path.exists():
            return False, f"File not found: {rgb_path}"
        
        # Backup if requested
        if backup:
            backup_path = rgb_path.with_suffix('.png.backup')
            if not backup_path.exists():
                import shutil
                shutil.copy2(rgb_path, backup_path)
        
        # Load image
        img_bgr = cv2.imread(str(rgb_path))
        if img_bgr is None:
            return False, f"Failed to load: {rgb_path}"
        
        # Apply tone mapping
        img_tone_mapped = apply_tone_mapping(
            img_bgr,
            method=method,
            clip_limit=clip_limit,
            tile_grid_size=(tile_grid_size, tile_grid_size),
            gamma=gamma
        )
        
        # Save back
        cv2.imwrite(str(rgb_path), img_tone_mapped)
        
        return True, ""
    
    except Exception as e:
        return False, f"{side.name}/{timestamp}: {traceback.format_exc()}"


def main(args):
    print(f"[Info] Project directory: {args.project_dir}")
    print(f"[Info] Tone mapping method: {args.method}")
    print(f"[Info] Parameters:")
    if args.method in ["clahe", "clahe+gamma"]:
        print(f"  - CLAHE clip limit: {args.clip_limit}")
        print(f"  - CLAHE tile grid size: {args.tile_grid_size}x{args.tile_grid_size}")
    if args.method in ["gamma", "clahe+gamma"]:
        print(f"  - Gamma: {args.gamma}")
    
    if args.backup:
        print(f"[Info] Original images will be backed up with .backup extension")
    
    data_io = DataIO(project_dir=args.project_dir)
    
    # Collect all images to process
    tasks = []
    for side in Side:
        rgb_dir = data_io.path_config.image.get_rgb_dir(side=side)
        if not rgb_dir.exists():
            print(f"[Warning] RGB directory not found for {side.name}: {rgb_dir}")
            continue
        
        # Get all PNG files
        rgb_files = sorted(rgb_dir.glob("*.png"))
        # Filter out backup files
        rgb_files = [f for f in rgb_files if not f.name.endswith('.backup')]
        
        for rgb_file in rgb_files:
            # Extract timestamp from filename
            try:
                timestamp = int(rgb_file.stem)
                tasks.append((side, timestamp))
            except ValueError:
                print(f"[Warning] Skipping file with invalid timestamp: {rgb_file}")
                continue
    
    if len(tasks) == 0:
        print("[Error] No RGB images found to process!")
        return
    
    print(f"[Info] Processing {len(tasks)} images...")
    
    # Process images in parallel
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_image,
                side, timestamp, data_io,
                args.method, args.clip_limit, args.tile_grid_size, args.gamma, args.backup
            )
            for side, timestamp in tasks
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Applying tone mapping"):
            try:
                success, error_msg = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    if error_msg:
                        print(f"[Error] {error_msg}")
            except Exception as e:
                error_count += 1
                print(f"[Exception] Worker failed: {e}")
    
    print(f"\n[Info] Tone mapping completed!")
    print(f"  - Successfully processed: {success_count} images")
    if error_count > 0:
        print(f"  - Errors: {error_count} images")
    
    print(f"\n[Next Steps]")
    print(f"  1. Re-run color optimization (skip YUV and reconstruction):")
    print(f"     python e2e_quest_to_colmap.py \\")
    print(f"       --project_dir {args.project_dir} \\")
    print(f"       --output_dir <output_path> \\")
    print(f"       --skip_yuv_conversion \\")
    print(f"       --skip_reconstruction \\")
    print(f"       --use_colored_pointcloud")


if __name__ == "__main__":
    args = parse_args()
    main(args)







