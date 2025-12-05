# 🔬 Microtubule Detection in Cryo-EM Images

## Complete Machine Learning Pipeline - Project Overview

**Status**: ✅ Production-Ready  
**Language**: Python 3.8+  
**Framework**: PyTorch  
**Purpose**: Automated detection of microtubules in 2D cryo-EM MRC images using point annotations

---

## 📁 Project Structure

```
MicrotubuleIdentification/
│
├── 📚 Documentation
│   ├── README.md              # Complete project documentation
│   ├── QUICKSTART.md          # Step-by-step getting started
│   ├── PROJECT_SUMMARY.md     # Technical overview
│   ├── CHECKLIST.md          # Getting started checklist
│   └── requirements.txt       # Python dependencies
│
├── 🔧 Core Modules (src/)
│   ├── __init__.py
│   ├── data_loader.py        # MRC loading, annotations, preprocessing
│   ├── dataset.py            # PyTorch Dataset + augmentation
│   └── models.py             # Neural network architectures
│
├── 🚀 Scripts
│   ├── train.py              # Training pipeline
│   ├── inference.py          # Prediction pipeline
│   ├── evaluate.py           # Evaluation metrics
│   ├── visualize.py          # Visualization tools
│   ├── inspect_data.py       # Dataset inspection
│   ├── examples.py           # Demo/tutorial code
│   ├── test_installation.py  # Installation verification
│   └── run.sh               # Helper script (executable)
│
├── ⚙️ Configuration
│   └── config/
│       └── default.txt       # Example training config
│
└── 🔒 Version Control
    └── .gitignore            # Git ignore rules
```

---

## ✨ Key Features

### Data Handling

- ✅ Load 2D MRC cryo-EM images
- ✅ Parse tab-separated point annotations
- ✅ Multiple normalization methods (z-score, min-max, percentile)
- ✅ Handle images with 0, 1, or many microtubules
- ✅ Automatic train/val/test splitting

### Training

- ✅ U-Net and SimpleConvNet architectures
- ✅ Gaussian heatmap or distance transform targets
- ✅ Data augmentation (flips, rotations)
- ✅ Early stopping with patience
- ✅ Checkpoint management (best + latest)
- ✅ Training history logging
- ✅ Multi-device support (CUDA, MPS, CPU)

### Inference

- ✅ Single image and batch prediction
- ✅ Adaptive and fixed threshold detection
- ✅ Non-maximum suppression
- ✅ API and command-line interfaces

### Evaluation

- ✅ Precision, Recall, F1 metrics
- ✅ Distance-based matching
- ✅ Per-image and overall statistics
- ✅ JSON export for analysis

### Visualization

- ✅ Predictions overlaid on images
- ✅ Ground truth comparison
- ✅ Training history plots
- ✅ Heatmap visualization

---

## 🎯 Critical Design Principles

### 1. Respects Microtubule Geometry

**Microtubules are NOT circular objects!**

- Long, thin, cylindrical structures
- Point annotations represent centers, not circular regions
- Gaussian heatmaps indicate localization uncertainty, not object shape
- Distance-based evaluation appropriate for elongated structures

### 2. Modular Architecture

- Clear separation of concerns
- Easy to swap components
- Well-documented extension points
- Reusable utilities

### 3. Production-Ready

- Comprehensive error handling
- Progress indicators
- Reproducible experiments (random seeds)
- Flexible configuration
- Well-tested components

---

## 🚦 Quick Start (5 Minutes)

```bash
# 1. Install dependencies
./run.sh setup

# 2. Test installation
python test_installation.py

# 3. Inspect your data
python inspect_data.py \
  --mrc_dir /path/to/mrc \
  --annotation_file /path/to/annotations.txt

# 4. Train a model (quick test)
python train.py \
  --mrc_dir /path/to/mrc \
  --annotation_file /path/to/annotations.txt \
  --output_dir output/test \
  --model_type simple \
  --max_epochs 10

# 5. Run inference
python inference.py \
  --mrc_path /path/to/test.mrc \
  --model_path output/test/best_model.pth \
  --model_type simple

# 6. Evaluate
python evaluate.py \
  --mrc_dir /path/to/mrc \
  --annotation_file /path/to/annotations.txt \
  --model_path output/test/best_model.pth \
  --model_type simple
```

---

## 📊 Input/Output Formats

### Input: Annotation File (Tab-Separated)

```
image_name                                          x_coord    y_coord
24dec20a_a_00035gr_00051sq_v01_00003hl_v01_00002ex 486        520
24dec20a_a_00035gr_00051sq_v01_00003hl_v01_00002ex 918        474
24dec20a_a_00035gr_00051sq_v01_00003hl_v01_00002ex 610        197
```

### Input: MRC Files

- 2D grayscale images
- `.mrc` or `.MRC` extension
- Standard MRC format (readable by mrcfile library)

### Output: Predictions (Same Format as Input)

```
image_name    x_coord    y_coord
test_image    123.45     678.90
test_image    234.56     789.01
```

### Output: Evaluation Results (JSON)

```json
{
  "overall": {
    "precision": 0.85,
    "recall": 0.78,
    "f1": 0.81,
    "tp": 120,
    "fp": 21,
    "fn": 34
  },
  "per_image": { ... }
}
```

---

## 🔄 Typical Workflow

```
┌─────────────────┐
│  1. DATA PREP   │  Organize MRC files + annotations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. INSPECT     │  Verify data quality & statistics
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. TRAIN       │  Train model with validation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. EVALUATE    │  Compute metrics on test set
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. TUNE        │  Adjust hyperparameters
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6. DEPLOY      │  Run inference on new images
└─────────────────┘
```

---

## 🛠️ Customization Points

### Easy Modifications

1. **Change model architecture**: Edit `src/models.py`
2. **Adjust target generation**: Modify `src/data_loader.py`
3. **Add new augmentations**: Extend `src/dataset.py`
4. **Customize peak detection**: Update `inference.py`
5. **Change loss function**: Modify `train.py`

### Extension Ideas

- Implement orientation prediction
- Add multi-scale detection
- Create custom architectures for elongated structures
- Integrate tracking across image sequences
- Build web interface for annotation/prediction
- Add 3D support for volumetric MRC files

---

## 📈 Hyperparameter Tuning Guide

### Model Size

```bash
# Smaller (faster, less memory)
--model_type simple --init_features 16

# Balanced (recommended)
--model_type unet --init_features 32

# Larger (more capacity)
--model_type unet --init_features 64
```

### Target Representation

```bash
# Tighter localization
--heatmap_sigma 2.0

# Standard (recommended)
--heatmap_sigma 3.0

# Broader (more tolerant)
--heatmap_sigma 5.0
```

### Detection Threshold

```bash
# More detections (higher recall, lower precision)
--detection_method adaptive
# or
--detection_method fixed --threshold 0.3

# Fewer detections (lower recall, higher precision)
--detection_method fixed --threshold 0.7
```

---

## 📚 Documentation Hierarchy

1. **START HERE**: `CHECKLIST.md` - Step-by-step getting started
2. **QUICK GUIDE**: `QUICKSTART.md` - Commands and workflows
3. **FULL DOCS**: `README.md` - Complete documentation
4. **TECHNICAL**: `PROJECT_SUMMARY.md` - Architecture details
5. **CODE**: `examples.py` - Working code examples

---

## 🧪 Testing & Validation

### Installation Test

```bash
python test_installation.py
```

Verifies all dependencies and basic functionality.

### Data Validation

```bash
python inspect_data.py --mrc-dir ... --annotation-file ...
```

Checks data quality, consistency, and coordinate validity.

### Model Test

```bash
# Quick 10-epoch test run
python train.py ... --max_epochs 10
```

Verifies training pipeline works before long runs.

---

## 💡 Best Practices

### Data Preparation

- ✓ Inspect data first with `inspect_data.py`
- ✓ Verify all coordinates are within bounds
- ✓ Check for consistent image dimensions
- ✓ Understand distribution of microtubules per image

### Training

- ✓ Start with small test run (10 epochs)
- ✓ Enable data augmentation for better generalization
- ✓ Monitor both training and validation loss
- ✓ Save configuration files for reproducibility
- ✓ Try multiple random seeds

### Inference

- ✓ Use adaptive detection for robustness
- ✓ Visualize predictions to check quality
- ✓ Adjust detection threshold based on precision/recall needs
- ✓ Process in batches for efficiency

### Evaluation

- ✓ Always evaluate on held-out test set
- ✓ Try multiple distance thresholds (5, 10, 20 pixels)
- ✓ Examine per-image results for patterns
- ✓ Visualize failure cases

---

## 🐛 Common Issues & Solutions

| Issue                | Solution                                     |
| -------------------- | -------------------------------------------- |
| Import errors        | `pip install -r requirements.txt`            |
| CUDA out of memory   | Reduce `--batch_size` or use `--device cpu`  |
| No detections        | Use `--detection_method adaptive`            |
| Poor validation loss | Enable `--augment`, adjust learning rate     |
| Slow training        | Use `--model_type simple` or reduce features |
| All predictions same | Check normalization consistency              |

---

## 📦 Dependencies Summary

**Core ML**: PyTorch 2.0+, torchvision  
**Scientific**: NumPy, SciPy, scikit-learn, pandas  
**Imaging**: mrcfile (MRC format), Pillow  
**Visualization**: Matplotlib  
**Utilities**: tqdm (progress bars)

See `requirements.txt` for complete list with versions.

---

## 🎓 Learning Resources

### Understanding the Code

1. Start with `examples.py` - hands-on demonstrations
2. Read docstrings in each module
3. Follow the workflow in `QUICKSTART.md`
4. Experiment with `test_installation.py` outputs

### Understanding the Method

- Point-based detection vs. segmentation
- Gaussian heatmaps for localization
- U-Net architecture for dense prediction
- Non-maximum suppression for peak detection

---

## 📝 TODO List for Users

- [ ] Update paths in `config/default.txt`
- [ ] Update paths in `examples.py` if using it
- [ ] Run `test_installation.py` to verify setup
- [ ] Run `inspect_data.py` on your data
- [ ] Start first training run
- [ ] Evaluate on test set
- [ ] Tune hyperparameters
- [ ] Document best configuration for your data

---

## 🤝 Contributing & Extending

This is a complete, modular pipeline designed for:

- **Research**: Experiment with new architectures and methods
- **Production**: Deploy for automated analysis
- **Education**: Learn ML pipeline development
- **Extension**: Build domain-specific tools

Feel free to:

- Modify models and training strategies
- Add new features and utilities
- Adapt to related detection tasks
- Create custom visualizations
- Build interfaces and tools

---

## 📞 Support

For questions or issues:

1. Check documentation (README.md, QUICKSTART.md)
2. Review code comments and docstrings
3. Run `test_installation.py` for diagnostics
4. Inspect your data with `inspect_data.py`
5. Try the examples in `examples.py`

---

## 🎯 Success Criteria

You're ready when you can:

- ✅ Install and run all scripts
- ✅ Load and inspect your data
- ✅ Train a model to completion
- ✅ Run inference on new images
- ✅ Evaluate performance with metrics
- ✅ Visualize results
- ✅ Understand the pipeline flow
- ✅ Modify components for your needs

---

**Built with care for the cryo-EM community. Happy detecting! 🔬**

_Remember: Microtubules are long and thin, not circular!_
