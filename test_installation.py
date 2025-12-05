"""
Test script to verify installation and basic functionality.
Run this after installing dependencies to check if everything works.
"""

import sys
import importlib

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

def check_torch_device():
    """Check available PyTorch devices."""
    try:
        import torch
        print("\nPyTorch Device Information:")
        print(f"  PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("  ✗ CUDA not available")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  ✓ MPS available (Apple Silicon GPU)")
        else:
            print("  ✗ MPS not available")
        
        print("  ✓ CPU available")
        return True
    except Exception as e:
        print(f"  Error checking devices: {e}")
        return False

def test_basic_imports():
    """Test basic module imports from the project."""
    try:
        print("\nTesting project modules:")
        from src import data_loader
        print("  ✓ src.data_loader")
        
        from src import dataset
        print("  ✓ src.dataset")
        
        from src import models
        print("  ✓ src.models")
        
        return True
    except Exception as e:
        print(f"  ✗ Error importing project modules: {e}")
        return False

def test_model_creation():
    """Test creating a simple model."""
    try:
        print("\nTesting model creation:")
        import torch
        from src.models import create_model
        
        model = create_model('simple', in_channels=1, out_channels=1, base_features=8)
        dummy_input = torch.randn(1, 1, 64, 64)
        output = model(dummy_input)
        
        assert output.shape == (1, 1, 64, 64), f"Unexpected output shape: {output.shape}"
        print("  ✓ Model creation and forward pass successful")
        print(f"    Input shape: {dummy_input.shape}")
        print(f"    Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Error testing model: {e}")
        return False

def test_data_utilities():
    """Test basic data utilities."""
    try:
        print("\nTesting data utilities:")
        import numpy as np
        from src.data_loader import normalize_image, create_heatmap_target
        
        # Test normalization
        dummy_image = np.random.randn(100, 100).astype(np.float32)
        normalized = normalize_image(dummy_image, method='zscore')
        assert normalized.shape == dummy_image.shape
        print("  ✓ Image normalization")
        
        # Test heatmap creation
        coords = [(50, 50), (25, 75)]
        heatmap = create_heatmap_target((100, 100), coords, sigma=3.0)
        assert heatmap.shape == (100, 100)
        assert heatmap.max() > 0
        print("  ✓ Heatmap target creation")
        
        return True
    except Exception as e:
        print(f"  ✗ Error testing data utilities: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Microtubule Detection - Installation Test")
    print("="*60)
    
    print("\nChecking dependencies...")
    all_ok = True
    
    # Check required packages
    required = {
        'torch': 'PyTorch',
        'torchvision': 'torchvision',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'mrcfile': 'mrcfile',
        'sklearn': 'scikit-learn',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'PIL': 'Pillow'
    }
    
    for module, package in required.items():
        if not check_import(module, package):
            all_ok = False
    
    # Check PyTorch devices
    if not check_torch_device():
        all_ok = False
    
    # Test project modules
    if not test_basic_imports():
        all_ok = False
    
    # Test model creation
    if not test_model_creation():
        all_ok = False
    
    # Test data utilities
    if not test_data_utilities():
        all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ All tests passed!")
        print("\nYou're ready to use the microtubule detection pipeline.")
        print("\nNext steps:")
        print("  1. Inspect your data: python inspect_data.py --mrc-dir ... --annotation-file ...")
        print("  2. Train a model: python train.py --mrc-dir ... --annotation-file ...")
        print("  3. Run inference: python inference.py --mrc-path ... --model-path ...")
    else:
        print("✗ Some tests failed.")
        print("\nPlease check the error messages above and:")
        print("  1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  2. Verify you're using Python 3.8+")
        print("  3. Check that PyTorch is properly installed")
    print("="*60)
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
