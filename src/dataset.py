"""
PyTorch Dataset class for microtubule detection in cryo-EM images.
"""

import os
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from .data_loader import (
    load_mrc_image, 
    normalize_image, 
    create_heatmap_target,
    create_distance_transform_target
)


class MicrotubuleDataset(Dataset):
    """
    PyTorch Dataset for microtubule detection from MRC images with point annotations.
    
    Each sample consists of:
    - An MRC image (as a tensor)
    - A target representation (heatmap or other format)
    - The list of ground truth coordinates
    - The image name
    """
    
    def __init__(self,
                 mrc_dir: str,
                 image_names: List[str],
                 annotations: Dict[str, List[Tuple[float, float]]],
                 target_type: str = 'heatmap',
                 normalization: str = 'zscore',
                 heatmap_sigma: float = 3.0,
                 distance_max: float = 50.0,
                 transform: Optional[Callable] = None,
                 resize_to: Optional[Tuple[int, int]] = None):
        """
        Initialize the dataset.
        
        Args:
            mrc_dir: Directory containing MRC files
            image_names: List of image names (without .mrc extension) to include
            annotations: Dictionary mapping image_name -> list of (x, y) coordinates
            target_type: Type of target to generate ('heatmap', 'distance', 'points')
            normalization: Image normalization method ('zscore', 'minmax', 'percentile')
            heatmap_sigma: Sigma for Gaussian heatmaps (used if target_type='heatmap')
            distance_max: Max distance for distance transform (used if target_type='distance')
            transform: Optional transform to apply to images (for data augmentation)
            resize_to: Optional (height, width) to resize all images to. If None, images keep original size.
        """
        self.mrc_dir = Path(mrc_dir)
        self.image_names = image_names
        self.annotations = annotations
        self.target_type = target_type
        self.normalization = normalization
        self.heatmap_sigma = heatmap_sigma
        self.distance_max = distance_max
        self.transform = transform
        self.resize_to = resize_to
        
        # Verify that MRC files exist
        self._verify_files()
    
    def _verify_files(self):
        """Verify that all MRC files exist."""
        missing_files = []
        for name in self.image_names:
            mrc_path = self.mrc_dir / f"{name}.mrc"
            if not mrc_path.exists():
                # Try uppercase extension
                mrc_path = self.mrc_dir / f"{name}.MRC"
                if not mrc_path.exists():
                    missing_files.append(name)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} MRC files not found")
            if len(missing_files) <= 5:
                print(f"Missing files: {missing_files}")
    
    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - 'image': Tensor of shape (1, H, W) - single channel image
            - 'target': Tensor target (format depends on target_type)
            - 'coords': List of ground truth (x, y) tuples
            - 'name': Image name string
            - 'image_shape': Original (H, W) shape
        """
        image_name = self.image_names[idx]
        
        # Load MRC image
        mrc_path = self.mrc_dir / f"{image_name}.mrc"
        if not mrc_path.exists():
            mrc_path = self.mrc_dir / f"{image_name}.MRC"
        
        image = load_mrc_image(str(mrc_path))
        original_shape = image.shape  # (H, W)
        
        # Normalize image
        image = normalize_image(image, method=self.normalization)
        
        # Get annotations for this image (may be empty list)
        coords = self.annotations.get(image_name, []).copy()  # Copy to avoid modifying original
        
        # Ensure coordinates are numeric (convert to float explicitly)
        coords = [(float(x), float(y)) for x, y in coords]
        
        # Resize image and adjust coordinates if needed
        if self.resize_to is not None:
            from scipy.ndimage import zoom
            target_h, target_w = self.resize_to
            orig_h, orig_w = original_shape
            
            # Skip resizing if image is already the target size
            if (orig_h, orig_w) != (target_h, target_w):
                # Calculate scaling factors
                scale_h = target_h / orig_h
                scale_w = target_w / orig_w
                
                # Resize image (ensure float32 output)
                image = zoom(image, (scale_h, scale_w), order=1).astype(np.float32)
                
                # Scale coordinates
                coords = [(x * scale_w, y * scale_h) for x, y in coords]
            
            image_shape = self.resize_to
        else:
            image_shape = original_shape
        
        # Ensure image is always float32 and has exactly the expected shape
        image = image.astype(np.float32)
        
        # Verify the image shape matches expectations
        if self.resize_to is not None:
            expected_shape = tuple(self.resize_to)
            if image.shape != expected_shape:
                raise RuntimeError(
                    f"Image {image_name} has unexpected shape after resize: "
                    f"got {image.shape}, expected {expected_shape}. "
                    f"Original shape was {original_shape}."
                )
        
        # Create target based on target_type
        if self.target_type == 'heatmap':
            target = create_heatmap_target(image_shape, coords, sigma=self.heatmap_sigma)
        elif self.target_type == 'distance':
            target = create_distance_transform_target(image_shape, coords, 
                                                      max_distance=self.distance_max)
        elif self.target_type == 'points':
            # Return coordinates as-is (for custom training loops)
            target = coords
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")
        
        # Apply transforms if provided (for data augmentation)
        # Note: Transform should handle both image and target appropriately
        if self.transform is not None:
            image, target = self.transform(image, target, coords)
        
        # Convert to tensors with explicit float32 dtype
        image_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).float()  # Add channel dim: (1, H, W)
        
        if self.target_type != 'points':
            target_tensor = torch.from_numpy(target.astype(np.float32)).unsqueeze(0).float()  # (1, H, W)
        else:
            target_tensor = coords  # Keep as list for 'points' mode
        
        return {
            'image': image_tensor,
            'target': target_tensor,
            'coords': coords,
            'name': image_name,
            'image_shape': image_shape,
            'original_shape': original_shape
        }


class SimpleAugmentation:
    """
    Simple data augmentation for cryo-EM images and point targets.
    
    Applies random flips and rotations (90 degree increments) to maintain
    compatibility with point coordinates.
    """
    
    def __init__(self, 
                 p_hflip: float = 0.5,
                 p_vflip: float = 0.5,
                 p_rot90: float = 0.5):
        """
        Args:
            p_hflip: Probability of horizontal flip
            p_vflip: Probability of vertical flip
            p_rot90: Probability of 90-degree rotation
        """
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90
    
    def __call__(self, image: np.ndarray, target: np.ndarray, 
                 coords: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to image and target.
        
        Args:
            image: 2D numpy array (H, W)
            target: 2D numpy array (H, W)
            coords: List of (x, y) coordinates (updated in place)
        
        Returns:
            Augmented (image, target) tuple
        """
        height, width = image.shape
        
        # Horizontal flip
        if np.random.random() < self.p_hflip:
            image = np.fliplr(image).copy()
            target = np.fliplr(target).copy()
            # Update coordinates: x' = width - 1 - x
            coords[:] = [(width - 1 - x, y) for x, y in coords]
        
        # Vertical flip
        if np.random.random() < self.p_vflip:
            image = np.flipud(image).copy()
            target = np.flipud(target).copy()
            # Update coordinates: y' = height - 1 - y
            coords[:] = [(x, height - 1 - y) for x, y in coords]
        
        # 90-degree rotation
        if np.random.random() < self.p_rot90:
            k = np.random.randint(1, 4)  # Rotate 90, 180, or 270 degrees
            image = np.rot90(image, k=k).copy()
            target = np.rot90(target, k=k).copy()
            
            # Update coordinates based on rotation
            for _ in range(k):
                # 90-degree counter-clockwise: (x, y) -> (y, width - 1 - x)
                coords[:] = [(y, width - 1 - x) for x, y in coords]
                # After rotation, dimensions swap
                height, width = width, height
        
        return image, target


# TODO: Consider implementing more sophisticated augmentation strategies:
# - Random cropping (would need to update coordinates accordingly)
# - Intensity augmentation (brightness, contrast)
# - Gaussian noise addition
# - Elastic deformations (more complex, need to warp coordinates)
