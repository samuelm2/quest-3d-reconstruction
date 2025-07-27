# Meta Quest 3D Reconstruction

<p align="center">
  <img src="docs/overview.png" alt="QuestRealityCapture" width="480"/>
</p>

**Reconstruct 3D scenes from image and depth data captured using [Quest Reality Capture (QRC)](https://github.com/t-34400/QuestRealityCapture/).**

---

## 🧭 Overview

This project provides a complete pipeline for generating 3D reconstructions using passthrough images and depth data captured on Meta Quest devices. The system supports both Open3D-based volumetric reconstruction and COLMAP-based SfM workflows.

---

## 🚀 Setup

### Environment Setup (with conda)

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) to manage environments.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate mq3drecon
```

---

## 🔧 Processing Pipeline

### Step 1: Convert Passthrough Images to RGB

```bash
python scripts/convert_yuv_to_rgb.py \
  --project_dir path/to/your/project \
  --config config/pipeline_config.yml
```

This generates:

* `left_camera_rgb/`
* `right_camera_rgb/`

**Note:** After conversion, manually remove any unnecessary or corrupted images.

---

### Step 2: Reconstruct 3D Scene

```bash
python scripts/reconstruct_scene.py \
  --project_dir path/to/your/project \
  --config config/pipeline_config.yml
```

This produces:

* TSDF-based **voxel grid** (colorless)
* Textured **mesh model**

Depending on your YAML config (`reconstruction:` section), the following additional outputs may be generated:

| Option                                       | Output                                                                   |
| -------------------------------------------- | ------------------------------------------------------------------------ |
| `optimize_depth_pose: true`                  | Optimized **depth dataset**                                              |
| `optimize_color_pose: true`                  | Optimized **color dataset**                                              |
| `sample_point_cloud_from_colored_mesh: true` | Colored **point cloud**                                                  |
| `render_color_aligned_depth: true`           | Depth images aligned to RGB frames |
| `color_aligned_depth_rendering.only_use_optimized_dataset: true`           | Only aligned for optimized color dataset                                 |

---

### Step 3: Export COLMAP Project (Optional)

```bash
python scripts/build_colmap_project.py \
  --project_dir path/to/your/project \
  --output_dir path/to/output/colmap_project \
  --use_colored_pointcloud \
  --use_optimized_color_dataset \
  --interval 1
```

**Options:**

* `--use_colored_pointcloud`: Include colored point cloud if available.
* `--use_optimized_color_dataset`: Use optimized color dataset.
* `--interval`: Export every N-th frame.

---

### \[Optional] Convert Raw Depth to Linear Depth Map

```bash
python scripts/convert_depth_to_linear_map.py \
  --project_dir path/to/your/project \
  --config config/pipeline_config.yml
```

This step is **standalone** and not required for other scripts.

---

## 🛠️ Custom Data Processing

You can write your own scripts by importing the unified `DataIO` interface:

```python
from dataio.data_io import DataIO
from models.side import Side
from models.transforms import CoordinateSystem


data_io = DataIO(project_dir=args.project_dir)

# Load depth maps
dataset = data_io.depth.load_depth_dataset(Side.LEFT)
depth_map = data_io.depth.load_depth_map_by_index(Side.LEFT, dataset, index=0)

# Load RGB frames
color_dataset = data_io.color.load_color_dataset(Side.LEFT)
timestamp = color_dataset.timestamps[0]
rgb = data_io.color.load_rgb(Side.LEFT, timestamp)

color_dataset.transforms = color_dataset.transforms.convert_coordinate_system(
    target_coordinate_system=CoordinateSystem.OPEN3D,
    is_camera=True
)
```

Explore:

* `scripts/dataio/` for loadable datasets
* `scripts/models/` for internal data structures

---

## 📁 Directory Structure (after full pipeline)

```text
your_project/
├── left_camera_rgb/
├── right_camera_rgb/
├── reconstruction/
│   ├── tsdf/
│   ├── mesh/
│   ├── point_cloud/
│   └── aligned_depth/
├── colmap_export/
├── config/
│   └── pipeline_config.yml
```

---

## 📢 NOTICE (v1.1.0+)

As of **Quest Reality Capture v1.1.0**, camera poses are now stored as **raw values** directly from the Android Camera2 API.
If you're using older logs (v1.0.x), apply the following transformation:

* **Translation**: `(x, y, z)` → `(x, y, -z)`
* **Rotation** (quaternion): `(x, y, z, w)` → `(-x, -y, z, w)`

---

## 🧩 Third-Party Code

This project includes components from [COLMAP](https://github.com/colmap/colmap), licensed under the 3-clause BSD License. See [`scripts/third_party/colmap/COPYING.txt`](./scripts/third_party/colmap/COPYING.txt) for details.

---

## 📝 License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for full text.

---

## 📌 TODO

* [ ] Implement carving to remove free-space artifacts
* [ ] Add Nerfstudio export instructions
