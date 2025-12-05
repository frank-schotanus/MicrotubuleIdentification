"""
Evaluation utilities for microtubule detection.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
from scipy.spatial.distance import cdist

from src.data_loader import load_annotations
from inference import predict_microtubules, batch_predict


def match_predictions(pred_coords: List[Tuple[float, float]],
                     gt_coords: List[Tuple[float, float]],
                     distance_threshold: float = 10.0) -> Tuple[int, int, int]:
    """
    Match predicted coordinates to ground truth coordinates.
    
    A prediction is considered a true positive if it's within distance_threshold
    of a ground truth point. Uses greedy matching (each GT can match only once).
    
    Args:
        pred_coords: List of predicted (x, y) coordinates
        gt_coords: List of ground truth (x, y) coordinates
        distance_threshold: Maximum distance for a match (in pixels)
    
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    if len(pred_coords) == 0 and len(gt_coords) == 0:
        return 0, 0, 0
    
    if len(pred_coords) == 0:
        return 0, 0, len(gt_coords)
    
    if len(gt_coords) == 0:
        return 0, len(pred_coords), 0
    
    # Convert to arrays for distance computation
    pred_array = np.array(pred_coords)
    gt_array = np.array(gt_coords)
    
    # Compute pairwise distances
    distances = cdist(pred_array, gt_array, metric='euclidean')
    
    # Greedy matching: for each GT, find closest prediction within threshold
    matched_preds = set()
    matched_gts = set()
    
    for gt_idx in range(len(gt_coords)):
        # Find closest prediction to this GT
        min_dist_idx = np.argmin(distances[:, gt_idx])
        min_dist = distances[min_dist_idx, gt_idx]
        
        if min_dist <= distance_threshold and min_dist_idx not in matched_preds:
            matched_preds.add(min_dist_idx)
            matched_gts.add(gt_idx)
    
    true_positives = len(matched_preds)
    false_positives = len(pred_coords) - true_positives
    false_negatives = len(gt_coords) - len(matched_gts)
    
    return true_positives, false_positives, false_negatives


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Compute detection metrics from TP, FP, FN counts.
    
    Args:
        tp: Number of true positives
        fp: Number of false positives
        fn: Number of false negatives
    
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def evaluate_predictions(pred_coords_dict: Dict[str, List[Tuple[float, float]]],
                        gt_coords_dict: Dict[str, List[Tuple[float, float]]],
                        distance_threshold: float = 10.0) -> Dict:
    """
    Evaluate predictions across multiple images.
    
    Args:
        pred_coords_dict: Dictionary mapping image_name -> list of predicted coords
        gt_coords_dict: Dictionary mapping image_name -> list of ground truth coords
        distance_threshold: Maximum distance for a match
    
    Returns:
        Dictionary with overall metrics and per-image results
    """
    # Get all image names (union of predictions and ground truth)
    all_images = set(pred_coords_dict.keys()) | set(gt_coords_dict.keys())
    
    # Accumulate counts
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    per_image_results = {}
    
    for image_name in all_images:
        pred_coords = pred_coords_dict.get(image_name, [])
        gt_coords = gt_coords_dict.get(image_name, [])
        
        tp, fp, fn = match_predictions(pred_coords, gt_coords, distance_threshold)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        per_image_results[image_name] = compute_metrics(tp, fp, fn)
    
    # Compute overall metrics
    overall_metrics = compute_metrics(total_tp, total_fp, total_fn)
    
    return {
        'overall': overall_metrics,
        'per_image': per_image_results,
        'num_images': len(all_images),
        'distance_threshold': distance_threshold
    }


def evaluate_model(mrc_dir: str,
                  annotation_file: str,
                  model_path: str,
                  output_file: Optional[str] = None,
                  distance_threshold: float = 10.0,
                  image_names: Optional[List[str]] = None,
                  **inference_kwargs) -> Dict:
    """
    Evaluate a trained model on a set of images.
    
    Args:
        mrc_dir: Directory containing MRC files
        annotation_file: Path to ground truth annotations
        model_path: Path to trained model checkpoint
        output_file: Optional path to save evaluation results
        distance_threshold: Maximum distance for matching predictions to GT
        image_names: List of image names to evaluate (None = all annotated images)
        **inference_kwargs: Additional arguments for predict_microtubules
    
    Returns:
        Dictionary with evaluation results
    """
    print("Loading ground truth annotations...")
    gt_annotations = load_annotations(annotation_file)
    
    # Determine which images to evaluate
    if image_names is None:
        image_names = list(gt_annotations.keys())
    
    print(f"Evaluating on {len(image_names)} images...")
    
    # Generate predictions
    pred_annotations = {}
    
    for image_name in image_names:
        mrc_path = Path(mrc_dir) / f"{image_name}.mrc"
        if not mrc_path.exists():
            mrc_path = Path(mrc_dir) / f"{image_name}.MRC"
        
        if not mrc_path.exists():
            print(f"Warning: MRC file not found for {image_name}")
            continue
        
        try:
            coords = predict_microtubules(str(mrc_path), model_path, **inference_kwargs)
            pred_annotations[image_name] = coords
        except Exception as e:
            print(f"Error predicting {image_name}: {e}")
            pred_annotations[image_name] = []
    
    # Evaluate predictions
    print("Computing metrics...")
    results = evaluate_predictions(pred_annotations, gt_annotations, distance_threshold)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of images: {results['num_images']}")
    print(f"Distance threshold: {results['distance_threshold']} pixels")
    print()
    print("Overall Metrics:")
    print(f"  Precision: {results['overall']['precision']:.4f}")
    print(f"  Recall:    {results['overall']['recall']:.4f}")
    print(f"  F1 Score:  {results['overall']['f1']:.4f}")
    print()
    print(f"  True Positives:  {results['overall']['tp']}")
    print(f"  False Positives: {results['overall']['fp']}")
    print(f"  False Negatives: {results['overall']['fn']}")
    print("="*60)
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    return results


def precision_recall_curve(pred_coords_dict: Dict[str, List[Tuple[float, float]]],
                           gt_coords_dict: Dict[str, List[Tuple[float, float]]],
                           distance_thresholds: List[float]) -> Dict:
    """
    Compute precision-recall curve for different distance thresholds.
    
    Args:
        pred_coords_dict: Dictionary of predictions
        gt_coords_dict: Dictionary of ground truth
        distance_thresholds: List of distance thresholds to evaluate
    
    Returns:
        Dictionary with precision/recall values at each threshold
    """
    results = []
    
    for threshold in distance_thresholds:
        eval_result = evaluate_predictions(pred_coords_dict, gt_coords_dict, threshold)
        results.append({
            'threshold': threshold,
            'precision': eval_result['overall']['precision'],
            'recall': eval_result['overall']['recall'],
            'f1': eval_result['overall']['f1']
        })
    
    return {
        'thresholds': distance_thresholds,
        'results': results
    }


def main():
    """
    Main evaluation function (command-line interface).
    """
    parser = argparse.ArgumentParser(description='Evaluate microtubule detection model')
    
    # Required arguments
    parser.add_argument('--mrc_dir', type=str, required=True,
                        help='Directory containing MRC files')
    parser.add_argument('--annotation_file', type=str, required=True,
                        help='Path to ground truth annotation file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    
    # Output arguments
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                        help='Path to save evaluation results')
    
    # Evaluation arguments
    parser.add_argument('--distance_threshold', type=float, default=10.0,
                        help='Maximum distance for matching (in pixels)')
    parser.add_argument('--split_file', type=str,
                        help='Path to split.json to evaluate only on test set')
    
    # Model arguments (for inference)
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'simple'],
                        help='Model architecture')
    parser.add_argument('--normalization', type=str, default='zscore',
                        choices=['zscore', 'minmax', 'percentile'],
                        help='Image normalization method')
    parser.add_argument('--detection_method', type=str, default='adaptive',
                        choices=['fixed', 'adaptive'],
                        help='Peak detection method')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--min_distance', type=int, default=5,
                        help='Minimum distance between detections')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, mps, or cpu)')
    
    args = parser.parse_args()
    
    # Determine which images to evaluate
    image_names = None
    if args.split_file:
        with open(args.split_file, 'r') as f:
            split_data = json.load(f)
        image_names = split_data.get('test', [])
        print(f"Evaluating on {len(image_names)} test images from split file")
    
    # Run evaluation
    evaluate_model(
        mrc_dir=args.mrc_dir,
        annotation_file=args.annotation_file,
        model_path=args.model_path,
        output_file=args.output_file,
        distance_threshold=args.distance_threshold,
        image_names=image_names,
        model_type=args.model_type,
        normalization=args.normalization,
        detection_method=args.detection_method,
        threshold=args.threshold,
        min_distance=args.min_distance,
        device=args.device
    )


if __name__ == '__main__':
    main()
