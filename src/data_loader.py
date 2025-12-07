"""
Data loading and preprocessing utilities for MRC cryo-EM images
and microtubule point annotations.
"""

import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import mrcfile
from pathlib import Path


def load_annotations(annotation_file: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Load microtubule annotations from a tab-separated text file.
    
    Args:
        annotation_file: Path to the annotation file with format:
                        image_name\tx_coord\ty_coord
    
    Returns:
        Dictionary mapping image_name -> list of (x, y) coordinate tuples
    """
    # Read the tab-separated file
    df = pd.read_csv(annotation_file, sep='\t', header=None, 
                     names=['image_name', 'x_coord', 'y_coord'])
    
    # Ensure coordinates are numeric (convert from strings if needed)
    df['x_coord'] = pd.to_numeric(df['x_coord'], errors='coerce')
    df['y_coord'] = pd.to_numeric(df['y_coord'], errors='coerce')
    
    # Drop any rows where coordinates couldn't be converted
    df = df.dropna(subset=['x_coord', 'y_coord'])
    
    # Group by image_name and collect coordinates
    annotations = {}
    for image_name, group in df.groupby('image_name'):
        # Explicitly convert to Python float to ensure numeric types
        coords = [(float(x), float(y)) for x, y in zip(group['x_coord'].values, group['y_coord'].values)]
        annotations[image_name] = coords
    
    return annotations


def load_mrc_image(mrc_path: str) -> np.ndarray:
    """
    Load a 2D MRC file and return as a numpy array.
    
    Args:
        mrc_path: Path to the MRC file
    
    Returns:
        2D numpy array containing the image data
    """
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        # Read the data - should be 2D for these low-mag images
        data = mrc.data.copy()
        
        # Ensure it's 2D
        if data.ndim > 2:
            # If there's an extra dimension of size 1, squeeze it
            data = np.squeeze(data)
        
        assert data.ndim == 2, f"Expected 2D image, got shape {data.shape}"
        
    return data


def normalize_image(image: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize image intensities.
    
    Args:
        image: 2D numpy array
        method: Normalization method ('zscore', 'minmax', or 'percentile')
    
    Returns:
        Normalized image array
    """
    if method == 'zscore':
        # Z-score normalization
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            normalized = (image - mean) / std
        else:
            normalized = image - mean
    
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(image)
    
    elif method == 'percentile':
        # Robust normalization using percentiles (clips outliers)
        p1, p99 = np.percentile(image, [1, 99])
        clipped = np.clip(image, p1, p99)
        if p99 > p1:
            normalized = (clipped - p1) / (p99 - p1)
        else:
            normalized = np.zeros_like(image)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.float32)


def get_image_list(mrc_dir: str, annotations: Dict[str, List[Tuple[float, float]]],
                   include_unlabeled: bool = True) -> List[str]:
    """
    Get list of image names to use, considering both available MRC files
    and annotation availability.
    
    Args:
        mrc_dir: Directory containing MRC files
        annotations: Dictionary of annotations (from load_annotations)
        include_unlabeled: Whether to include images with no annotations
    
    Returns:
        List of image names (without .mrc extension)
    """
    mrc_dir = Path(mrc_dir)
    
    # Get all MRC files in directory
    mrc_files = list(mrc_dir.glob("*.mrc")) + list(mrc_dir.glob("*.MRC"))
    image_names = [f.stem for f in mrc_files]
    
    if not include_unlabeled:
        # Only keep images that have annotations
        image_names = [name for name in image_names if name in annotations]
    
    return sorted(image_names)


def create_heatmap_target(image_shape: Tuple[int, int], 
                          coordinates: List[Tuple[float, float]],
                          sigma: float = 3.0) -> np.ndarray:
    """
    Create a Gaussian heatmap target for point-based labels.
    
    This is suitable for microtubule detection where each label is a point
    along a long, thin structure. The Gaussian blob represents uncertainty
    in the exact location, but does NOT imply the object is circular.
    
    Args:
        image_shape: (height, width) of the image
        coordinates: List of (x, y) coordinate tuples
        sigma: Standard deviation of Gaussian (controls spread around each point)
    
    Returns:
        2D numpy array with Gaussian peaks at each coordinate
    """
    height, width = image_shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Create coordinate grids
    y_grid, x_grid = np.ogrid[0:height, 0:width]
    
    for x, y in coordinates:
        # Create Gaussian centered at (x, y)
        # Note: x is column index, y is row index
        gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
        
        # Take maximum to handle overlapping Gaussians
        heatmap = np.maximum(heatmap, gaussian)
    
    return heatmap


def create_distance_transform_target(image_shape: Tuple[int, int],
                                     coordinates: List[Tuple[float, float]],
                                     max_distance: float = 50.0) -> np.ndarray:
    """
    Create a distance transform target (distance to nearest microtubule point).
    
    This can be useful for training networks to predict proximity to microtubules
    without assuming circular shape.
    
    Args:
        image_shape: (height, width) of the image
        coordinates: List of (x, y) coordinate tuples
        max_distance: Maximum distance to compute (distances beyond this are clipped)
    
    Returns:
        2D numpy array with distance values (normalized to [0, 1])
    """
    from scipy.ndimage import distance_transform_edt
    
    height, width = image_shape
    
    if len(coordinates) == 0:
        # No microtubules - return all ones (maximum distance)
        return np.ones((height, width), dtype=np.float32)
    
    # Create binary mask with points at microtubule locations
    mask = np.zeros((height, width), dtype=bool)
    for x, y in coordinates:
        # Round to nearest pixel
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            mask[yi, xi] = True
    
    # Compute distance transform (distance to nearest True pixel)
    distances = distance_transform_edt(~mask)
    
    # Clip and normalize
    distances = np.clip(distances, 0, max_distance)
    normalized_distances = 1.0 - (distances / max_distance)
    
    return normalized_distances.astype(np.float32)


def split_dataset(image_names: List[str], 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split image names into train/val/test sets.
    
    Args:
        image_names: List of image names
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_names, val_names, test_names)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Shuffle with fixed seed
    rng = np.random.RandomState(random_seed)
    shuffled_names = image_names.copy()
    rng.shuffle(shuffled_names)
    
    n_total = len(shuffled_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_names = shuffled_names[:n_train]
    val_names = shuffled_names[n_train:n_train + n_val]
    test_names = shuffled_names[n_train + n_val:]
    
    return train_names, val_names, test_names
