# Getting Started Checklist

Follow this checklist to get up and running with microtubule detection.

## ☐ Step 1: Environment Setup

### 1.1 Clone/Download the Project

```bash
cd "/Users/frankschotanus/Downloads/School/Fall 25/Computer Vision/MicrotubuleIdentification"
```

### 1.2 Install Dependencies

```bash
# Using the helper script
./run.sh setup

# Or manually
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### 1.3 Verify Installation

```bash
python test_installation.py
```

**Expected output**: All checkmarks (✓) indicating successful installation.

---

## ☐ Step 2: Prepare Your Data

### 2.1 Organize Your Data Structure

```
your_data/
├── mrc_files/
│   ├── image1.mrc
│   ├── image2.mrc
│   └── ...
└── annotations.txt
```

### 2.2 Verify Annotation Format

Your `annotations.txt` should be tab-separated:

```
image_name    x_coord    y_coord
image1        486        520
image1        918        474
image2        123        456
```

### 2.3 Inspect Your Data

```bash
python inspect_data.py \
  --mrc_dir /path/to/mrc_files \
  --annotation_file /path/to/annotations.txt \
  --output_file data_report.json
```

**What to check:**

- ✓ All MRC files are found
- ✓ Coordinates are within image bounds
- ✓ Images have consistent dimensions (or note if they vary)
- ✓ Distribution of microtubules per image looks reasonable

---

## ☐ Step 3: First Training Run

### 3.1 Create Your Configuration

Copy and edit `config/default.txt`:

```bash
cp config/default.txt config/my_experiment.txt
# Edit the file with your actual paths
```

### 3.2 Run a Quick Test Training

Start with a small experiment (10 epochs) to verify everything works:

```bash
python train.py \
  --mrc_dir /path/to/mrc_files \
  --annotation_file /path/to/annotations.txt \
  --output_dir output/test_run \
  --model_type simple \
  --batch_size 2 \
  --max_epochs 10 \
  --lr 1e-4
```

**Expected output:**

- Progress bars showing training
- Training and validation loss decreasing
- Checkpoints saved to `output/test_run/`

### 3.3 Check Training Results

```bash
# View training curves
python visualize.py \
  --mode history \
  --history_file output/test_run/history.json \
  --output training_curves.png
```

---

## ☐ Step 4: Run Inference

### 4.1 Test on a Single Image

```bash
python inference.py \
  --mrc_path /path/to/test_image.mrc \
  --model_path output/test_run/best_model.pth \
  --model_type simple
```

**Expected output:**

- List of detected (x, y) coordinates
- Number of detected microtubules

### 4.2 Visualize Predictions

```bash
python visualize.py \
  --mode prediction \
  --mrc_path /path/to/test_image.mrc \
  --model_path output/test_run/best_model.pth \
  --annotation_file /path/to/annotations.txt \
  --model_type simple \
  --output prediction_viz.png
```

**What to check:**

- Are predictions (red X) near ground truth (green circles)?
- Too many false positives or false negatives?
- Adjust detection threshold if needed

---

## ☐ Step 5: Evaluate Performance

### 5.1 Run Evaluation

```bash
python evaluate.py \
  --mrc_dir /path/to/mrc_files \
  --annotation_file /path/to/annotations.txt \
  --model_path output/test_run/best_model.pth \
  --model_type simple \
  --distance_threshold 10.0 \
  --split_file output/test_run/split.json \
  --output_file evaluation_results.json
```

**Expected output:**

- Precision, Recall, F1 scores
- True positives, false positives, false negatives

### 5.2 Interpret Results

- **High precision, low recall**: Model is conservative (missing microtubules)
  - Solution: Lower detection threshold or increase heatmap sigma
- **Low precision, high recall**: Model is too aggressive (false positives)
  - Solution: Raise detection threshold or decrease heatmap sigma
- **Both low**: Model needs more training or better architecture
  - Solution: Train longer, use U-Net, enable augmentation

---

## ☐ Step 6: Full Training Run

### 6.1 Configure Full Training

Edit your config file with optimized settings:

```
--model_type=unet
--init_features=32
--batch_size=4
--max_epochs=100
--augment
--patience=15
```

### 6.2 Start Full Training

```bash
python train.py @config/my_experiment.txt
# Or use the helper script:
./run.sh train @config/my_experiment.txt
```

**Tips:**

- Training will take longer (hours to days depending on data size)
- Monitor validation loss - should decrease and plateau
- Early stopping will trigger if no improvement

### 6.3 Compare Multiple Runs

Try different configurations:

- Different models (unet vs simple)
- Different heatmap_sigma values (2.0, 3.0, 5.0)
- With/without augmentation
- Different learning rates

---

## ☐ Step 7: Production Inference

### 7.1 Batch Processing

```bash
python inference.py \
  --mrc_dir /path/to/new/images \
  --model_path output/best_experiment/best_model.pth \
  --output_file predictions.txt \
  --model_type unet \
  --detection_method adaptive \
  --min_distance 5
```

### 7.2 Process Results

The `predictions.txt` file contains:

```
image_name    x_coord    y_coord
new_image1    123.45    678.90
new_image1    234.56    789.01
```

You can now:

- Import into your analysis pipeline
- Visualize results
- Compare with expert annotations
- Use for downstream tracking or analysis

---

## Troubleshooting Common Issues

### Issue: Import errors

```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue: CUDA out of memory

```bash
# Solution: Reduce batch size
python train.py ... --batch_size 1 --model_type simple
```

### Issue: No detections or all detections

```bash
# Solution: Adjust detection method and threshold
python inference.py ... --detection_method adaptive
# or
python inference.py ... --detection_method fixed --threshold 0.3
```

### Issue: Poor validation loss

```bash
# Solutions to try:
# 1. Check data quality with inspect_data.py
# 2. Enable augmentation: --augment
# 3. Adjust learning rate: --lr 1e-3 or --lr 1e-5
# 4. Increase model capacity: --init_features 64
# 5. Change normalization: --normalization percentile
```

---

## Next Steps After Getting Started

1. **Experiment with hyperparameters**

   - Systematic grid search or manual tuning
   - Document what works best for your data

2. **Analyze failure cases**

   - Which images have poor detection?
   - Are there specific patterns the model misses?

3. **Collect more data**

   - If performance plateaus, more annotated data helps
   - Focus on challenging cases

4. **Customize the pipeline**

   - Modify target generation in `src/data_loader.py`
   - Add custom augmentation in `src/dataset.py`
   - Implement new models in `src/models.py`
   - Adjust peak detection in `inference.py`

5. **Extend functionality**
   - Add orientation prediction
   - Implement tracking across images
   - Create web interface for annotations
   - Export to specific formats for your tools

---

## Quick Reference Commands

```bash
# Setup
./run.sh setup

# Inspect data
./run.sh inspect --mrc-dir DATA --annotation-file ANNOT

# Train
./run.sh train --mrc-dir DATA --annotation-file ANNOT

# Inference (single)
./run.sh inference --mrc-path IMAGE.mrc --model-path MODEL.pth

# Inference (batch)
./run.sh inference --mrc-dir DATA --model-path MODEL.pth

# Evaluate
./run.sh evaluate --mrc-dir DATA --annotation-file ANNOT --model-path MODEL.pth

# Visualize training
./run.sh visualize --mode history --history-file output/*/history.json

# Visualize predictions
./run.sh visualize --mode prediction --mrc-path IMAGE.mrc --model-path MODEL.pth
```

---

## Support and Documentation

- **Full Documentation**: See `README.md`
- **Quick Start**: See `QUICKSTART.md`
- **Project Summary**: See `PROJECT_SUMMARY.md`
- **Code Examples**: See `examples.py`
- **This Checklist**: Keep for reference!

---

## Checklist Summary

- ☐ Environment setup complete
- ☐ Dependencies installed and verified
- ☐ Data prepared and inspected
- ☐ First training run successful
- ☐ Inference tested on sample image
- ☐ Results visualized
- ☐ Model evaluated
- ☐ Full training run completed
- ☐ Production inference working
- ☐ Ready for real work!

Good luck with your microtubule detection project! 🔬
