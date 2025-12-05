"""
Example/demo script showing basic usage of the microtubule detection pipeline.

This script demonstrates:
1. Loading and inspecting data
2. Creating a small training run
3. Running inference
4. Evaluating results
"""

import os
from pathlib import Path

# Example paths - MODIFY THESE to match your data
MRC_DIR = "/path/to/your/mrc/files"
ANNOTATION_FILE = "/path/to/your/annotations.txt"
OUTPUT_DIR = "output/demo"


def example_load_data():
    """
    Example: Load and inspect data.
    """
    print("="*60)
    print("EXAMPLE 1: Loading and Inspecting Data")
    print("="*60)
    
    from src.data_loader import load_annotations, get_image_list, load_mrc_image
    
    # Load annotations
    print("\n1. Loading annotations...")
    annotations = load_annotations(ANNOTATION_FILE)
    print(f"   Found annotations for {len(annotations)} images")
    
    # Show example annotations
    example_image = list(annotations.keys())[0]
    example_coords = annotations[example_image]
    print(f"\n2. Example image: {example_image}")
    print(f"   Number of microtubules: {len(example_coords)}")
    print(f"   First few coordinates: {example_coords[:3]}")
    
    # Get image list
    print("\n3. Scanning MRC directory...")
    image_names = get_image_list(MRC_DIR, annotations, include_unlabeled=True)
    print(f"   Found {len(image_names)} MRC files")
    
    # Load an example image
    print(f"\n4. Loading example MRC image...")
    example_mrc_path = Path(MRC_DIR) / f"{example_image}.mrc"
    image = load_mrc_image(str(example_mrc_path))
    print(f"   Image shape: {image.shape}")
    print(f"   Image dtype: {image.dtype}")
    print(f"   Value range: [{image.min():.2f}, {image.max():.2f}]")
    
    print("\n" + "="*60 + "\n")


def example_create_dataset():
    """
    Example: Create PyTorch dataset.
    """
    print("="*60)
    print("EXAMPLE 2: Creating PyTorch Dataset")
    print("="*60)
    
    from src.data_loader import load_annotations, split_dataset, get_image_list
    from src.dataset import MicrotubuleDataset
    
    # Load data
    annotations = load_annotations(ANNOTATION_FILE)
    image_names = get_image_list(MRC_DIR, annotations)
    
    # Split dataset
    print("\n1. Splitting dataset...")
    train_names, val_names, test_names = split_dataset(
        image_names, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15
    )
    print(f"   Train: {len(train_names)}")
    print(f"   Val: {len(val_names)}")
    print(f"   Test: {len(test_names)}")
    
    # Create dataset
    print("\n2. Creating training dataset...")
    train_dataset = MicrotubuleDataset(
        mrc_dir=MRC_DIR,
        image_names=train_names[:5],  # Just first 5 for demo
        annotations=annotations,
        target_type='heatmap',
        normalization='zscore',
        heatmap_sigma=3.0
    )
    print(f"   Dataset size: {len(train_dataset)}")
    
    # Get a sample
    print("\n3. Loading a sample...")
    sample = train_dataset[0]
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Target shape: {sample['target'].shape}")
    print(f"   Number of coordinates: {len(sample['coords'])}")
    print(f"   Image name: {sample['name']}")
    
    print("\n" + "="*60 + "\n")


def example_train_model():
    """
    Example: Train a model (simplified).
    """
    print("="*60)
    print("EXAMPLE 3: Training a Model")
    print("="*60)
    
    import torch
    from src.models import create_model
    
    # Create model
    print("\n1. Creating model...")
    model = create_model(
        model_type='simple',  # Use simpler model for demo
        in_channels=1,
        out_channels=1,
        base_features=16  # Smaller for demo
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")
    
    # Example forward pass
    print("\n2. Testing forward pass...")
    dummy_input = torch.randn(1, 1, 256, 256)  # Batch of 1, 256x256 image
    output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n3. To train the full model, run:")
    print(f"   python train.py \\")
    print(f"     --mrc_dir {MRC_DIR} \\")
    print(f"     --annotation_file {ANNOTATION_FILE} \\")
    print(f"     --output_dir {OUTPUT_DIR} \\")
    print(f"     --model_type simple \\")
    print(f"     --batch_size 4 \\")
    print(f"     --max_epochs 50")
    
    print("\n" + "="*60 + "\n")


def example_inference():
    """
    Example: Run inference (requires trained model).
    """
    print("="*60)
    print("EXAMPLE 4: Running Inference")
    print("="*60)
    
    model_path = Path(OUTPUT_DIR) / "best_model.pth"
    
    if not model_path.exists():
        print(f"\nNo trained model found at {model_path}")
        print("Please train a model first using example_train_model()")
        print("\n" + "="*60 + "\n")
        return
    
    from inference import predict_microtubules
    from src.data_loader import load_annotations, get_image_list
    
    # Get an example image
    annotations = load_annotations(ANNOTATION_FILE)
    image_names = get_image_list(MRC_DIR, annotations)
    example_image = image_names[0]
    example_path = Path(MRC_DIR) / f"{example_image}.mrc"
    
    print(f"\n1. Running inference on: {example_image}")
    
    # Predict
    coordinates = predict_microtubules(
        mrc_path=str(example_path),
        model_path=str(model_path),
        model_type='simple',
        detection_method='adaptive'
    )
    
    print(f"\n2. Detected {len(coordinates)} microtubules:")
    for i, (x, y) in enumerate(coordinates[:5], 1):  # Show first 5
        print(f"   {i}. ({x:.2f}, {y:.2f})")
    
    # Compare to ground truth
    gt_coords = annotations.get(example_image, [])
    print(f"\n3. Ground truth has {len(gt_coords)} microtubules")
    
    print("\n4. To run batch inference:")
    print(f"   python inference.py \\")
    print(f"     --mrc_dir {MRC_DIR} \\")
    print(f"     --model_path {model_path} \\")
    print(f"     --output_file predictions.txt")
    
    print("\n" + "="*60 + "\n")


def example_evaluation():
    """
    Example: Evaluate model performance.
    """
    print("="*60)
    print("EXAMPLE 5: Evaluating Model")
    print("="*60)
    
    model_path = Path(OUTPUT_DIR) / "best_model.pth"
    
    if not model_path.exists():
        print(f"\nNo trained model found at {model_path}")
        print("Please train a model first")
        print("\n" + "="*60 + "\n")
        return
    
    print("\n1. To evaluate the model, run:")
    print(f"   python evaluate.py \\")
    print(f"     --mrc_dir {MRC_DIR} \\")
    print(f"     --annotation_file {ANNOTATION_FILE} \\")
    print(f"     --model_path {model_path} \\")
    print(f"     --output_file evaluation_results.json \\")
    print(f"     --distance_threshold 10.0")
    
    print("\n2. This will compute:")
    print("   - Precision (accuracy of predictions)")
    print("   - Recall (fraction of GT microtubules detected)")
    print("   - F1 Score (harmonic mean)")
    print("   - Per-image detailed results")
    
    print("\n" + "="*60 + "\n")


def main():
    """
    Run all examples.
    """
    print("\n" + "="*60)
    print("MICROTUBULE DETECTION PIPELINE - DEMO")
    print("="*60)
    print("\nBefore running, please update the paths at the top of this file:")
    print(f"  MRC_DIR = {MRC_DIR}")
    print(f"  ANNOTATION_FILE = {ANNOTATION_FILE}")
    print(f"  OUTPUT_DIR = {OUTPUT_DIR}")
    print("\n" + "="*60 + "\n")
    
    # Check if paths are still defaults
    if MRC_DIR == "/path/to/your/mrc/files":
        print("⚠️  Please update the paths in this file before running!")
        print("   Edit the variables at the top of examples.py")
        return
    
    # Run examples
    try:
        example_load_data()
        example_create_dataset()
        example_train_model()
        example_inference()
        example_evaluation()
        
        print("="*60)
        print("DEMO COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("1. Review the code in each example function")
        print("2. Run full training with train.py")
        print("3. Evaluate with evaluate.py")
        print("4. Visualize results with visualize.py")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please check that your data paths are correct")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
