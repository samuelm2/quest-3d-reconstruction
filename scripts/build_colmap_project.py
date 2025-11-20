import argparse
from pathlib import Path
import shutil
import numpy as np
from tqdm import tqdm

from dataio.data_io import DataIO
from models.camera_dataset import CameraDataset
from models.side import Side
from models.transforms import CoordinateSystem, Transforms
from third_party.colmap.read_and_write_model import Camera, Image, Point3D, write_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export camera and image data to COLMAP format.")
    
    parser.add_argument(
        "--project_dir", "-p",
        type=Path,
        required=True,
        help="Path to the project directory containing QRC data."
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        required=True,
        help="Path to the output directory where COLMAP model files will be saved."
    )
    parser.add_argument(
        "--use_colored_pointcloud",
        action="store_true",
        help="Include colored 3D point cloud if available."
    )
    parser.add_argument(
        "--use_optimized_color_dataset",
        action="store_true",
        help="Use optimized color datasets if available."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Sampling interval for image export. Use every N-th image."
    )

    args = parser.parse_args()

    if not args.project_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.project_dir}")

    if not args.output_dir.exists():
        print(f"[Info] Output directory does not exist. Creating: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    return args


def load_dataset_map(data_io: DataIO, use_optimized_color_dataset: bool = True) -> dict[Side, CameraDataset]:
    dataset_map: dict[Side, CameraDataset] = {}

    if use_optimized_color_dataset:
        for side in Side:
            dataset = data_io.color.load_optimized_color_dataset(side=side)
            if dataset is not None:
                dataset_map[side] = dataset

        if len(dataset_map) == 0:
            print("[Warning] Optimized color datasets not found. Falling back to original color datasets.")

    if len(dataset_map) == 0:
        for side in Side:
            dataset = data_io.color.load_color_dataset(side=side)
            dataset_map[side] = dataset

    return dataset_map


def read_cameras_and_images(
    data_io: DataIO, 
    dataset_map: dict[Side, CameraDataset], 
    input_dir: Path,
    interval: int = 1
) -> tuple[dict[int, Camera], dict[int, Image]]:
    cameras = {}
    images = {}

    camera_id = 0
    image_id = 0

    for side, dataset in dataset_map.items():
        print(f"[{side.name}] Exporting images and camera data ...")

        dataset = dataset[::interval]

        transforms = dataset.transforms.convert_coordinate_system(
            target_coordinate_system=CoordinateSystem.COLMAP,
            is_camera=True
        )
        positions = transforms.positions_cw
        rotations = transforms.rotations_cw[:, [3, 0, 1, 2]]  # (w, x, y, z)

        camera = Camera(
            id=camera_id,
            model="PINHOLE",
            width=dataset.widths[0],
            height=dataset.heights[0],
            params=np.array([
                dataset.fx[0],
                dataset.fy[0],
                dataset.cx[0],
                dataset.cy[0]
            ])
        )
        cameras[camera_id] = camera

        for i in tqdm(range(len(dataset)), desc=f"[{side.name}] Copying images", unit="img"):
            try:
                timestamp = dataset.timestamps[i]
                dst_filename = f"{side.name}_{timestamp}.png"

                src_path = data_io.path_config.image.get_rgb_file_path(side=side, timestamp=timestamp)
                dst_path = input_dir / dst_filename

                shutil.copy2(src=src_path, dst=dst_path)

                image = Image(
                    id=image_id,
                    qvec=rotations[i],
                    tvec=positions[i],
                    camera_id=camera_id,
                    name=dst_filename,
                    xys=np.empty((0, 2)),
                    point3D_ids=np.empty((0,)),
                )

                images[image_id] = image
                image_id += 1

            except FileNotFoundError:
                print(f"[Error] RGB image not found at path: {src_path}")
                continue
            except Exception as e:
                print(f"[Error] Unexpected error while copying: {e}")
                continue

        camera_id += 1

    return cameras, images


def read_points_3d(data_io: DataIO) -> dict[int, Point3D]:
    print("[Info] Reading colored point cloud ...")

    pcd = data_io.reconstruction.load_colored_pcd()
    if pcd is None:
        raise Exception("[Error] Colored point cloud not found. Please ensure it has been generated before export.")
    
    print("[Info] Finished reading colored point cloud.")

    positions = pcd.point.positions.numpy()
    colors = pcd.point.colors.numpy()

    positions = Transforms(
        coordinate_system=CoordinateSystem.OPEN3D,
        positions=positions,
        rotations=np.empty(())
    ).convert_coordinate_system(
        target_coordinate_system=CoordinateSystem.COLMAP,
        is_camera=False,
        skip_rotation=True
    ).positions

    point3D_id = 0
    points3D: dict[int, Point3D] = {}

    for position, color in tqdm(zip(positions, colors), desc="[Info] Creating 3D points", unit="pt", total=len(positions)):
        point3D = Point3D(
            id=point3D_id,
            xyz=position,
            rgb=color,
            error=0.0,
            image_ids=np.array([], dtype=np.int64),
            point2D_idxs=np.array([], dtype=np.int64),
        )
        points3D[point3D_id] = point3D
        point3D_id += 1

    return points3D


def main(args):
    data_io = DataIO(project_dir=args.project_dir)
    dataset_map = load_dataset_map(
        data_io=data_io,
        use_optimized_color_dataset=args.use_optimized_color_dataset
    )

    print(f"[Info] Output COLMAP project will be saved to: {args.output_dir}")

    model_dir = args.output_dir / "sparse/0"
    input_dir = args.output_dir / "images"

    model_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    cameras, images = read_cameras_and_images(
        data_io=data_io,
        dataset_map=dataset_map,
        input_dir=input_dir,
        interval=args.interval
    )

    if args.use_colored_pointcloud:
        points3d = read_points_3d(data_io=data_io)
    else:
        points3d = {}

    write_model(
        cameras=cameras,
        images=images,
        points3D=points3d,
        path=model_dir,
        ext=".bin"
    )


if __name__ == "__main__":
    args = parse_args()
    print(f"[Info] Project directory: {args.project_dir}")
    main(args)
