# Microtubule Detection in Cryo-EM Images

A PyTorch-based machine learning pipeline for automatically detecting microtubules in low-magnification cryo-EM images stored as 2D MRC files.

## Overview

This project provides a complete end-to-end solution for:

- Loading and preprocessing 2D MRC cryo-EM images
- Training deep learning models to detect microtubules from point annotations
- Running inference on new images to predict microtubule locations
- Evaluating model performance with standard detection metrics

**Important Note**: Microtubules are long, thin, cylindrical structures, not circular objects. This implementation treats annotations as point labels along elongated structures and does NOT assume circular positive regions around each point.

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full list

### Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Format

### MRC Images

- Single 2D grayscale images (low magnification cryo-EM)
- Stored as `.mrc` files
- Readable with the `mrcfile` library

### Annotation File

Tab-separated text file with format:

```
image_name   x_coord   y_coord
24dec20a_a_00035gr_00051sq_v01_00003hl_v01_00002ex   486     520
24dec20a_a_00035gr_00051sq_v01_00003hl_v01_00002ex   918     474
```

- Each row = one microtubule instance
- Multiple rows per image = multiple microtubules
- `(x_coord, y_coord)` = single point near microtubule center
- Images with zero microtubules may be absent from file

## Quick Start

### 1. Training

```bash
python train.py \
  --mrc_dir /path/to/mrc/files \
  --annotation_file /path/to/annotations.txt \
  --output_dir output/experiment1 \
  --model_type unet \
  --batch_size 4 \
  --max_epochs 100 \
  --lr 1e-4 \
  --augment
```

**Key Arguments**:

- `--mrc_dir`: Directory containing MRC files
- `--annotation_file`: Path to tab-separated annotation file
- `--output_dir`: Where to save checkpoints and logs
- `--model_type`: Model architecture (`unet` or `simple`)
- `--target_type`: Target representation (`heatmap` or `distance`)
- `--normalization`: Image normalization (`zscore`, `minmax`, `percentile`)
- `--heatmap_sigma`: Gaussian sigma for heatmap targets (default: 3.0)
- `--augment`: Enable data augmentation (flips, rotations)
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--max_epochs`: Maximum training epochs
- `--patience`: Early stopping patience

**Output**:

- `output_dir/best_model.pth`: Best model checkpoint (lowest validation loss)
- `output_dir/latest_model.pth`: Most recent checkpoint
- `output_dir/history.json`: Training history (losses per epoch)
- `output_dir/split.json`: Train/val/test split information
- `output_dir/config.json`: Training configuration

### 2. Inference

**Single Image**:

```bash
python inference.py \
  --mrc_path /path/to/image.mrc \
  --model_path output/experiment1/best_model.pth \
  --model_type unet
```

**Batch Prediction**:

```bash
python inference.py \
  --mrc_dir /path/to/mrc/files \
  --model_path output/experiment1/best_model.pth \
  --output_file predictions.txt \
  --model_type unet \
  --detection_method adaptive \
  --min_distance 5
```

**Key Arguments**:

- `--mrc_path`: Single MRC file to process
- `--mrc_dir`: Directory for batch processing
- `--model_path`: Path to trained model checkpoint
- `--output_file`: Where to save predictions (batch mode)
- `--detection_method`: Peak detection method (`fixed` or `adaptive`)
- `--threshold`: Detection threshold for `fixed` method (default: 0.5)
- `--min_distance`: Minimum pixel distance between detections (default: 5)

**Output Format** (predictions.txt):

```
image_name   x_coord   y_coord
image1       123.45   678.90
image1       234.56   789.01
image2       345.67   890.12
```

### 3. Evaluation

```bash
python evaluate.py \
  --mrc_dir /path/to/mrc/files \
  --annotation_file /path/to/annotations.txt \
  --model_path output/experiment1/best_model.pth \
  --output_file evaluation_results.json \
  --distance_threshold 10.0 \
  --split_file output/experiment1/split.json
```

**Key Arguments**:

- `--distance_threshold`: Max pixel distance for matching predictions to ground truth
- `--split_file`: Optional JSON file to evaluate only on test set
- `--output_file`: Where to save evaluation results

**Metrics Computed**:

- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: Harmonic mean of precision and recall
- True Positives (TP): Predictions within threshold of GT
- False Positives (FP): Predictions not matching any GT
- False Negatives (FN): GT points not matched by predictions

## API Usage

### Python API for Inference

```python
from inference import predict_microtubules

# Predict microtubules in a single image
coordinates = predict_microtubules(
    mrc_path="path/to/image.mrc",
    model_path="output/experiment1/best_model.pth",
    model_type="unet",
    normalization="zscore",
    detection_method="adaptive",
    min_distance=5
)

# coordinates is a list of (x, y) tuples
for x, y in coordinates:
    print(f"Microtubule at ({x:.2f}, {y:.2f})")
```

### Python API for Evaluation

```python
from evaluate import evaluate_model

# Evaluate model on test set
results = evaluate_model(
    mrc_dir="path/to/mrc/files",
    annotation_file="path/to/annotations.txt",
    model_path="output/experiment1/best_model.pth",
    distance_threshold=10.0,
    model_type="unet"
)

print(f"Precision: {results['overall']['precision']:.4f}")
print(f"Recall: {results['overall']['recall']:.4f}")
print(f"F1 Score: {results['overall']['f1']:.4f}")
```

## Project Structure

```
MicrotubuleIdentification/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # MRC loading, annotation parsing, preprocessing
│   ├── dataset.py          # PyTorch Dataset class and augmentation
│   └── models.py           # Neural network architectures
├── train.py                # Training script
├── inference.py            # Inference script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Model Architectures

### U-Net (default)

- Standard U-Net with encoder-decoder structure
- Skip connections preserve spatial details
- Outputs dense heatmap predictions
- Best for accurate localization

### SimpleConvNet

- Lighter fully-convolutional network
- Faster training and inference
- Good baseline for quick experiments

## Target Representations

### Heatmap (default)

- Gaussian peaks at each microtubule center
- Sigma parameter controls spread (default: 3.0 pixels)
- Suitable for regression with MSE loss
- **Note**: Gaussian spread does NOT imply circular object shape

### Distance Transform

- Distance to nearest microtubule point
- Normalized to [0, 1] range
- Alternative representation for training

## Customization

### Modifying Training Targets

Edit `src/data_loader.py` to implement custom target generation:

```python
def create_custom_target(image_shape, coordinates, **params):
    # Your custom target logic here
    return target_array
```

Then modify `src/dataset.py` to use it:

```python
elif self.target_type == 'custom':
    target = create_custom_target(image_shape, coords, **custom_params)
```

### Custom Post-processing

Edit `inference.py` to implement custom peak detection:

```python
def custom_peak_detection(heatmap, **params):
    # Your custom detection logic here
    return list_of_coordinates
```

### Adding New Models

Add to `src/models.py`:

```python
class CustomModel(nn.Module):
    def __init__(self, ...):
        # Your model architecture
        pass

    def forward(self, x):
        # Forward pass
        return output
```

Then update `create_model()` function to include your model.

## TODO / Future Enhancements

- [ ] Implement advanced data augmentation (elastic deformations, intensity)
- [ ] Add attention mechanisms to models
- [ ] Implement orientation prediction for microtubules
- [ ] Add visualization tools for predictions
- [ ] Support for multi-scale detection
- [ ] Integration with tracking algorithms
- [ ] Support for 3D MRC files
- [ ] Implement focal loss for class imbalance
- [ ] Add TensorBoard logging
- [ ] Hyperparameter optimization tools

## Troubleshooting

### Out of Memory

- Reduce `--batch_size`
- Use `--model_type simple` for lighter model
- Reduce `--init_features` for U-Net

### Poor Detection Performance

- Adjust `--heatmap_sigma` (try 2.0-5.0)
- Try `--detection_method adaptive`
- Tune `--distance_threshold` in evaluation
- Enable `--augment` for more training data variety
- Try different `--normalization` methods

### Training Not Converging

- Adjust learning rate (`--lr`)
- Increase `--max_epochs`
- Check data quality and annotations
- Try different model architectures

## Citation

If you use this code in your research, please cite:

```
[Your citation information here]
```

## License

[Your license information here]

## Contact

[Your contact information here]
