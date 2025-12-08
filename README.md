# Microtubule Detection

Detects microtubules in low-magnification cryo-EM images using a U-Net model trained on point annotations.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Verify installation:

```bash
python test_installation.py
```

## Data Format

**Images**: 2D grayscale MRC files (`.mrc`).

**Annotations**: Tab-separated file where each row is one microtubule location:

```
image_name	x_coord	y_coord
24dec20a_00035gr_00051sq_00003hl	486	520
24dec20a_00035gr_00051sq_00003hl	918	474
```

## Training

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

Outputs:

- `best_model.pth` — model with lowest validation loss
- `history.json` — training/validation loss per epoch
- `split.json` — train/val/test image names

## Inference

Single image:

```bash
python inference.py \
  --mrc_path /path/to/image.mrc \
  --model_path output/exp1/best_model.pth \
  --model_type unet
```

Directory of images:

```bash
python inference.py \
  --mrc_dir /path/to/mrc/files \
  --model_path output/exp1/best_model.pth \
  --output_file predictions.txt \
  --model_type unet
```

## Evaluation

```bash
python evaluate.py \
  --mrc_dir /path/to/mrc/files \
  --annotation_file /path/to/annotations.txt \
  --model_path output/exp1/best_model.pth \
  --distance_threshold 10.0
```

Computes precision, recall, and F1 by matching predictions to ground truth within the distance threshold.

## Parameters

| Flag                 | Default  | Description                          |
| -------------------- | -------- | ------------------------------------ |
| `--model_type`       | `unet`   | `unet` or `simple`                   |
| `--normalization`    | `zscore` | `zscore`, `minmax`, or `percentile`  |
| `--heatmap_sigma`    | `3.0`    | Gaussian sigma for target heatmaps   |
| `--detection_method` | `fixed`  | `fixed` or `adaptive` peak detection |
| `--threshold`        | `0.5`    | Detection threshold (fixed method)   |
| `--min_distance`     | `5`      | Minimum pixels between detections    |
| `--augment`          | off      | Enable random flips and rotations    |
| `--patience`         | `10`     | Early stopping patience              |

Run `python train.py --help` or `python inference.py --help` for all options.


