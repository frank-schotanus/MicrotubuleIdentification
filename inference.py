"""
Inference utilities for microtubule detection.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import label, maximum_filter
from scipy.spatial.distance import cdist

from src.data_loader import load_mrc_image, normalize_image
from src.models import create_model


def detect_peaks(heatmap: np.ndarray, 
                 threshold: float = 0.5,
                 min_distance: int = 5) -> List[Tuple[float, float]]:
    """
    Detect peaks in a heatmap to extract microtubule coordinates.
    
    This uses local maximum detection with non-maximum suppression.
    
    Args:
        heatmap: 2D numpy array (H, W) with detection scores
        threshold: Minimum value to consider as a detection
        min_distance: Minimum distance between detections (in pixels)
    
    Returns:
        List of (x, y) coordinate tuples for detected microtubules
    """
    # Threshold the heatmap
    binary = heatmap > threshold
    
    if not binary.any():
        return []
    
    # Find local maxima
    # A pixel is a local maximum if it's the maximum in its neighborhood
    local_max = (heatmap == maximum_filter(heatmap, size=min_distance))
    
    # Combine thresholding and local maximum
    peaks = binary & local_max
    
    # Get coordinates of peaks
    y_coords, x_coords = np.where(peaks)
    
    # Convert to list of tuples (x, y)
    coordinates = [(float(x), float(y)) for x, y in zip(x_coords, y_coords)]
    
    return coordinates


def detect_peaks_adaptive(heatmap: np.ndarray,
                          percentile_threshold: float = 95.0,
                          min_distance: int = 5,
                          min_detections: int = 1) -> List[Tuple[float, float]]:
    """
    Detect peaks using adaptive thresholding based on percentiles.
    
    This is more robust when the absolute heatmap values vary across images.
    
    Args:
        heatmap: 2D numpy array (H, W) with detection scores
        percentile_threshold: Percentile value for adaptive threshold (0-100)
        min_distance: Minimum distance between detections
        min_detections: Minimum number of detections to return
    
    Returns:
        List of (x, y) coordinate tuples
    """
    # Compute adaptive threshold
    threshold = np.percentile(heatmap, percentile_threshold)
    
    # Ensure threshold is reasonable
    threshold = max(threshold, 0.1)
    
    coordinates = detect_peaks(heatmap, threshold=threshold, min_distance=min_distance)
    
    # If we got too few detections, try lowering the threshold
    if len(coordinates) < min_detections and percentile_threshold > 50:
        coordinates = detect_peaks_adaptive(
            heatmap, 
            percentile_threshold=percentile_threshold - 10,
            min_distance=min_distance,
            min_detections=min_detections
        )
    
    return coordinates


def predict_microtubules(mrc_path: str,
                         model_path: str,
                         model_type: str = 'unet',
                         normalization: str = 'zscore',
                         detection_method: str = 'adaptive',
                         threshold: float = 0.5,
                         min_distance: int = 5,
                         device: Optional[torch.device] = None) -> List[Tuple[float, float]]:
    """
    Predict microtubule locations in a single MRC image.
    
    Args:
        mrc_path: Path to MRC file
        model_path: Path to trained model checkpoint
        model_type: Type of model architecture ('unet' or 'simple')
        normalization: Image normalization method
        detection_method: Peak detection method ('fixed' or 'adaptive')
        threshold: Detection threshold (for 'fixed' method)
        min_distance: Minimum distance between detections
        device: Device to run inference on (auto-detected if None)
    
    Returns:
        List of (x, y) coordinate tuples for predicted microtubules
    """
    # Auto-detect device if not provided
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    # Load and preprocess image
    image = load_mrc_image(mrc_path)
    image = normalize_image(image, method=normalization)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    image_tensor = image_tensor.to(device)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model (infer parameters from checkpoint if needed)
    model = create_model(model_type=model_type, in_channels=1, out_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)  # Convert to probabilities
        heatmap = output.squeeze().cpu().numpy()  # (H, W)
    
    # Detect peaks
    if detection_method == 'adaptive':
        coordinates = detect_peaks_adaptive(heatmap, min_distance=min_distance)
    else:
        coordinates = detect_peaks(heatmap, threshold=threshold, min_distance=min_distance)
    
    return coordinates


def batch_predict(mrc_dir: str,
                  model_path: str,
                  output_file: str,
                  image_names: Optional[List[str]] = None,
                  **kwargs) -> None:
    """
    Run inference on multiple MRC images and save predictions.
    
    Args:
        mrc_dir: Directory containing MRC files
        model_path: Path to trained model checkpoint
        output_file: Path to save predictions (tab-separated format)
        image_names: List of image names to process (None = all images)
        **kwargs: Additional arguments for predict_microtubules
    """
    mrc_dir = Path(mrc_dir)
    
    # Get list of images to process
    if image_names is None:
        mrc_files = list(mrc_dir.glob("*.mrc")) + list(mrc_dir.glob("*.MRC"))
        image_names = [f.stem for f in mrc_files]
    
    # Predict for each image
    all_predictions = []
    
    print(f"Processing {len(image_names)} images...")
    for image_name in image_names:
        mrc_path = mrc_dir / f"{image_name}.mrc"
        if not mrc_path.exists():
            mrc_path = mrc_dir / f"{image_name}.MRC"
        
        if not mrc_path.exists():
            print(f"Warning: MRC file not found for {image_name}")
            continue
        
        try:
            coordinates = predict_microtubules(str(mrc_path), model_path, **kwargs)
            
            for x, y in coordinates:
                all_predictions.append({
                    'image_name': image_name,
                    'x_coord': x,
                    'y_coord': y
                })
        
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue
    
    # Save predictions to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for pred in all_predictions:
            f.write(f"{pred['image_name']}\t{pred['x_coord']:.2f}\t{pred['y_coord']:.2f}\n")
    
    print(f"Saved {len(all_predictions)} predictions to {output_path}")


def main():
    """
    Main inference function (command-line interface).
    """
    parser = argparse.ArgumentParser(description='Predict microtubule locations in MRC images')
    
    # Input/output arguments
    parser.add_argument('--mrc_path', type=str,
                        help='Path to single MRC file (for single prediction)')
    parser.add_argument('--mrc_dir', type=str,
                        help='Directory containing MRC files (for batch prediction)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_file', type=str, default='predictions.txt',
                        help='Output file for predictions')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'simple'],
                        help='Model architecture')
    parser.add_argument('--normalization', type=str, default='zscore',
                        choices=['zscore', 'minmax', 'percentile'],
                        help='Image normalization method')
    
    # Detection arguments
    parser.add_argument('--detection_method', type=str, default='adaptive',
                        choices=['fixed', 'adaptive'],
                        help='Peak detection method')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold (for fixed method)')
    parser.add_argument('--min_distance', type=int, default=5,
                        help='Minimum distance between detections')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, mps, or cpu)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Single image or batch prediction
    if args.mrc_path:
        print(f"Predicting microtubules in {args.mrc_path}...")
        coordinates = predict_microtubules(
            mrc_path=args.mrc_path,
            model_path=args.model_path,
            model_type=args.model_type,
            normalization=args.normalization,
            detection_method=args.detection_method,
            threshold=args.threshold,
            min_distance=args.min_distance,
            device=device
        )
        
        print(f"Found {len(coordinates)} microtubules:")
        for i, (x, y) in enumerate(coordinates, 1):
            print(f"  {i}. ({x:.2f}, {y:.2f})")
    
    elif args.mrc_dir:
        print(f"Running batch prediction on {args.mrc_dir}...")
        batch_predict(
            mrc_dir=args.mrc_dir,
            model_path=args.model_path,
            output_file=args.output_file,
            model_type=args.model_type,
            normalization=args.normalization,
            detection_method=args.detection_method,
            threshold=args.threshold,
            min_distance=args.min_distance,
            device=device
        )
    
    else:
        parser.error("Must provide either --mrc_path or --mrc_dir")


if __name__ == '__main__':
    main()
