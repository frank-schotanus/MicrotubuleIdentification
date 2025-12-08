# Microtubule Detection

PyTorch pipeline for detecting microtubules in low-magnification cryo-EM images (2D MRC files).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Format

**MRC images**: Single 2D grayscale `.mrc` files.

**Annotations**: Tab-separated file with point coordinates:

```
image_name	x_coord	y_coord
24dec20a_00035gr_00051sq_v01_00003hl	486	520
24dec20a_00035gr_00051sq_v01_00003hl	918	474
```

Each row marks one microtubule. Multiple rows per image = multiple microtubules.

## Usage

### Train

```bash
python train.py \
  --mrc_dir /path/to/mrc/files \
  --annotation_file /path/to/annotations.txt \
  --output_dir output/exp1 \
  --model_type unet \
  --batch_size 4 \
  --max_epochs 100 \
  --augment
```

Saves checkpoints to `output/exp1/best_model.pth`.

### Inference

Single image:

```bash
python inference.py \
  --mrc_path /path/to/image.mrc \
  --model_path output/exp1/best_model.pth \
  --model_type unet
```

Batch:

```bash
python inference.py \
  --mrc_dir /path/to/mrc/files \
  --model_path output/exp1/best_model.pth \
  --output_file predictions.txt \
  --model_type unet
```

### Evaluate

```bash
python evaluate.py \
  --mrc_dir /path/to/mrc/files \
  --annotation_file /path/to/annotations.txt \
  --model_path output/exp1/best_model.pth \
  --distance_threshold 10.0
```

Reports precision, recall, and F1.

## Options

| Flag                 | Description                               |
| -------------------- | ----------------------------------------- |
| `--model_type`       | `unet` (default) or `simple`              |
| `--normalization`    | `zscore`, `minmax`, or `percentile`       |
| `--heatmap_sigma`    | Gaussian sigma for targets (default: 3.0) |
| `--detection_method` | `fixed` or `adaptive`                     |
| `--augment`          | Enable flips/rotations                    |

See `python train.py --help` for full list.
