# Quick Start Guide for Microtubule Detection

## Initial Setup

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install required packages
pip install -r requirements.txt
```

### 2. Verify Your Data

First, inspect your dataset to ensure everything is correct:

```bash
python inspect_data.py \
  --mrc_dir /path/to/your/mrc/files \
  --annotation_file /path/to/your/annotations.txt \
  --output_file data_report.json
```

This will show you:

- Number of images and annotations
- Distribution of microtubules per image
- Image dimensions and data types
- Whether coordinates are valid

## Training Workflow

### Step 1: Configure Training

Edit `config/default.txt` or create your own config file with your paths and hyperparameters:

```
--mrc_dir=/actual/path/to/mrc/files
--annotation_file=/actual/path/to/annotations.txt
--output_dir=output/my_experiment
--model_type=unet
--batch_size=4
--max_epochs=100
--augment
```

### Step 2: Start Training

```bash
# Using config file
python train.py @config/default.txt

# Or with command-line arguments
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

Training outputs:

- `output/experiment1/best_model.pth` - Best model checkpoint
- `output/experiment1/latest_model.pth` - Latest checkpoint
- `output/experiment1/history.json` - Training curves
- `output/experiment1/split.json` - Train/val/test split
- `output/experiment1/config.json` - Training configuration

### Step 3: Monitor Training

View training progress:

```bash
python visualize.py \
  --mode history \
  --history_file output/experiment1/history.json \
  --output training_curves.png
```

## Inference

### Single Image Prediction

```bash
python inference.py \
  --mrc_path /path/to/test_image.mrc \
  --model_path output/experiment1/best_model.pth \
  --model_type unet
```

### Batch Prediction

```bash
python inference.py \
  --mrc_dir /path/to/test/images \
  --model_path output/experiment1/best_model.pth \
  --output_file predictions.txt \
  --model_type unet \
  --detection_method adaptive
```

### Visualize Predictions

```bash
python visualize.py \
  --mode prediction \
  --mrc_path /path/to/test_image.mrc \
  --model_path output/experiment1/best_model.pth \
  --annotation_file /path/to/annotations.txt \
  --output prediction_viz.png
```

## Evaluation

### Evaluate on Test Set

```bash
python evaluate.py \
  --mrc_dir /path/to/mrc/files \
  --annotation_file /path/to/annotations.txt \
  --model_path output/experiment1/best_model.pth \
  --output_file evaluation_results.json \
  --distance_threshold 10.0 \
  --split_file output/experiment1/split.json
```

This computes:

- Precision: How many predictions are correct
- Recall: How many ground truth microtubules were found
- F1 Score: Overall detection quality
- Per-image detailed results

## Hyperparameter Tuning Tips

### If model is underfitting (poor training loss):

- Increase model capacity: `--init_features 64`
- Train longer: `--max_epochs 200`
- Reduce regularization if any

### If model is overfitting (good training, poor validation):

- Enable augmentation: `--augment`
- Reduce model size: `--init_features 16`
- Add more training data

### If detection is missing microtubules (low recall):

- Use adaptive detection: `--detection_method adaptive`
- Adjust heatmap sigma: `--heatmap_sigma 5.0` (larger)
- Lower detection threshold in inference

### If too many false positives (low precision):

- Increase detection threshold
- Reduce heatmap sigma: `--heatmap_sigma 2.0`
- Use fixed threshold method with higher value

### If training is too slow:

- Use simpler model: `--model_type simple`
- Reduce batch size: `--batch_size 2`
- Reduce features: `--init_features 16`

### If running out of memory:

- Reduce batch size: `--batch_size 1`
- Use simpler model: `--model_type simple`
- Reduce features: `--init_features 16`

## Common Issues and Solutions

### Issue: "Import mrcfile could not be resolved"

**Solution**: Install dependencies: `pip install -r requirements.txt`

### Issue: "CUDA out of memory"

**Solution**: Reduce `--batch_size` or use `--device cpu`

### Issue: Poor detection performance

**Solutions**:

1. Check data quality with `inspect_data.py`
2. Verify annotations are correct
3. Try different normalization methods
4. Adjust heatmap_sigma parameter
5. Enable data augmentation

### Issue: Training loss not decreasing

**Solutions**:

1. Check learning rate (try 1e-3 or 1e-5)
2. Verify data is loading correctly
3. Check that targets are being generated properly
4. Increase model capacity

### Issue: Predictions are all zeros or all ones

**Solutions**:

1. Adjust detection threshold
2. Check model output range
3. Verify normalization is consistent between train/inference
4. Try adaptive detection method

## File Structure After Training

```
MicrotubuleIdentification/
├── output/
│   └── experiment1/
│       ├── best_model.pth          # Best model weights
│       ├── latest_model.pth        # Latest checkpoint
│       ├── history.json            # Training history
│       ├── split.json              # Dataset split
│       └── config.json             # Training config
├── predictions.txt                 # Inference results
├── evaluation_results.json         # Evaluation metrics
└── data_report.json               # Data inspection report
```

## Next Steps

1. **Run examples**: Execute `python examples.py` after updating paths
2. **Inspect your data**: Use `inspect_data.py` to understand your dataset
3. **Start with small experiment**: Train on subset with `--max_epochs 10`
4. **Iterate**: Adjust hyperparameters based on results
5. **Evaluate thoroughly**: Use multiple distance thresholds
6. **Visualize results**: Check predictions visually with `visualize.py`

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Verify file paths are correct
3. Ensure data format matches expected format
4. Review the README.md for detailed documentation
5. Check the code comments for implementation details
