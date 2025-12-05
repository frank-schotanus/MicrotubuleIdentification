"""
Visualization utilities for microtubule detection results.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.data_loader import load_mrc_image, normalize_image, load_annotations
from inference import predict_microtubules


def visualize_predictions(image: np.ndarray,
                         pred_coords: List[Tuple[float, float]],
                         gt_coords: Optional[List[Tuple[float, float]]] = None,
                         heatmap: Optional[np.ndarray] = None,
                         save_path: Optional[str] = None,
                         title: str = "Microtubule Detection"):
    """
    Visualize microtubule detection results.
    
    Args:
        image: 2D numpy array (H, W) - the original image
        pred_coords: List of predicted (x, y) coordinates
        gt_coords: Optional list of ground truth (x, y) coordinates
        heatmap: Optional prediction heatmap to overlay
        save_path: Optional path to save figure
        title: Figure title
    """
    # Create figure
    n_subplots = 2 if heatmap is not None else 1
    fig, axes = plt.subplots(1, n_subplots, figsize=(8 * n_subplots, 8))
    
    if n_subplots == 1:
        axes = [axes]
    
    # Plot 1: Image with detections
    axes[0].imshow(image, cmap='gray')
    
    # Plot ground truth (if available)
    if gt_coords:
        gt_x = [coord[0] for coord in gt_coords]
        gt_y = [coord[1] for coord in gt_coords]
        axes[0].scatter(gt_x, gt_y, c='green', s=100, marker='o',
                       facecolors='none', edgecolors='green', linewidths=2,
                       label=f'Ground Truth (n={len(gt_coords)})')
    
    # Plot predictions
    if pred_coords:
        pred_x = [coord[0] for coord in pred_coords]
        pred_y = [coord[1] for coord in pred_coords]
        axes[0].scatter(pred_x, pred_y, c='red', s=80, marker='x',
                       linewidths=2, label=f'Predictions (n={len(pred_coords)})')
    
    axes[0].set_title(f'{title}\nDetections')
    axes[0].legend()
    axes[0].axis('off')
    
    # Plot 2: Heatmap overlay (if available)
    if heatmap is not None:
        axes[1].imshow(image, cmap='gray', alpha=0.5)
        axes[1].imshow(heatmap, cmap='hot', alpha=0.5)
        axes[1].set_title(f'{title}\nPrediction Heatmap')
        axes[1].axis('off')
        
        # Add colorbar
        plt.colorbar(axes[1].images[1], ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_comparison(image: np.ndarray,
                        pred_coords: List[Tuple[float, float]],
                        gt_coords: List[Tuple[float, float]],
                        matches: List[Tuple[int, int]],
                        save_path: Optional[str] = None,
                        title: str = "Detection Comparison"):
    """
    Visualize comparison between predictions and ground truth with matches.
    
    Args:
        image: 2D numpy array
        pred_coords: List of predicted coordinates
        gt_coords: List of ground truth coordinates
        matches: List of (pred_idx, gt_idx) tuples for matched pairs
        save_path: Optional path to save figure
        title: Figure title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    ax.imshow(image, cmap='gray')
    
    # Create sets of matched indices
    matched_pred_idx = {m[0] for m in matches}
    matched_gt_idx = {m[1] for m in matches}
    
    # Plot ground truth
    for i, (x, y) in enumerate(gt_coords):
        if i in matched_gt_idx:
            # Matched GT (green circle)
            circle = patches.Circle((x, y), radius=10, facecolor='none',
                                   edgecolor='green', linewidth=2)
            ax.add_patch(circle)
        else:
            # Missed GT (yellow circle)
            circle = patches.Circle((x, y), radius=10, facecolor='none',
                                   edgecolor='yellow', linewidth=2)
            ax.add_patch(circle)
    
    # Plot predictions
    for i, (x, y) in enumerate(pred_coords):
        if i in matched_pred_idx:
            # Matched prediction (green X)
            ax.plot(x, y, 'gx', markersize=12, markeredgewidth=2)
        else:
            # False positive (red X)
            ax.plot(x, y, 'rx', markersize=12, markeredgewidth=2)
    
    # Draw lines connecting matches
    for pred_idx, gt_idx in matches:
        px, py = pred_coords[pred_idx]
        gx, gy = gt_coords[gt_idx]
        ax.plot([px, gx], [py, gy], 'g-', alpha=0.3, linewidth=1)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='green', markersize=10, linewidth=2,
               label='True Positive (GT)'),
        Line2D([0], [0], marker='x', color='g', markersize=10,
               markeredgewidth=2, label='True Positive (Pred)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='yellow', markersize=10, linewidth=2,
               label='False Negative (missed)'),
        Line2D([0], [0], marker='x', color='r', markersize=10,
               markeredgewidth=2, label='False Positive')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_summary_plot(history_file: str,
                       save_path: Optional[str] = None):
    """
    Create training history summary plot.
    
    Args:
        history_file: Path to history.json file
        save_path: Optional path to save figure
    """
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax.plot(best_epoch, best_val_loss, 'g*', markersize=15,
            label=f'Best (Epoch {best_epoch})')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """
    Command-line interface for visualization.
    """
    parser = argparse.ArgumentParser(description='Visualize microtubule detection results')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['prediction', 'history'],
                       help='Visualization mode')
    
    # For prediction mode
    parser.add_argument('--mrc_path', type=str,
                       help='Path to MRC file')
    parser.add_argument('--model_path', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--annotation_file', type=str,
                       help='Optional path to annotations for ground truth')
    parser.add_argument('--model_type', type=str, default='unet',
                       help='Model architecture')
    
    # For history mode
    parser.add_argument('--history_file', type=str,
                       help='Path to history.json file')
    
    # Output
    parser.add_argument('--output', type=str,
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    if args.mode == 'prediction':
        if not args.mrc_path or not args.model_path:
            parser.error("--mrc_path and --model_path required for prediction mode")
        
        # Load image
        image = load_mrc_image(args.mrc_path)
        image_name = Path(args.mrc_path).stem
        
        # Get predictions
        pred_coords = predict_microtubules(args.mrc_path, args.model_path,
                                          model_type=args.model_type)
        
        # Load ground truth if available
        gt_coords = None
        if args.annotation_file:
            annotations = load_annotations(args.annotation_file)
            gt_coords = annotations.get(image_name, [])
        
        # Normalize image for display
        image_display = normalize_image(image, method='percentile')
        
        # Visualize
        visualize_predictions(image_display, pred_coords, gt_coords,
                            save_path=args.output,
                            title=f"Image: {image_name}")
    
    elif args.mode == 'history':
        if not args.history_file:
            parser.error("--history_file required for history mode")
        
        create_summary_plot(args.history_file, save_path=args.output)


if __name__ == '__main__':
    main()
