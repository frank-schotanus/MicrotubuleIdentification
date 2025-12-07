"""
Utility script to inspect and analyze the dataset.
"""

import argparse
from pathlib import Path
from collections import Counter
import json

import numpy as np

from src.data_loader import load_annotations, get_image_list, load_mrc_image


def analyze_annotations(annotation_file: str) -> dict:
    """
    Analyze the annotation file and return statistics.
    
    Args:
        annotation_file: Path to annotation file
    
    Returns:
        Dictionary with annotation statistics
    """
    annotations = load_annotations(annotation_file)
    
    # Count microtubules per image
    counts = [len(coords) for coords in annotations.values()]
    
    stats = {
        'num_images': len(annotations),
        'total_microtubules': sum(counts),
        'avg_microtubules_per_image': np.mean(counts),
        'median_microtubules_per_image': np.median(counts),
        'min_microtubules': min(counts) if counts else 0,
        'max_microtubules': max(counts) if counts else 0,
        'std_microtubules': np.std(counts)
    }
    
    # Distribution histogram
    count_distribution = Counter(counts)
    stats['distribution'] = dict(sorted(count_distribution.items()))
    
    return stats


def analyze_images(mrc_dir: str, annotations: dict, sample_size: int = 10) -> dict:
    """
    Analyze MRC images and return statistics.
    
    Args:
        mrc_dir: Directory containing MRC files
        annotations: Dictionary of annotations
        sample_size: Number of images to sample for analysis
    
    Returns:
        Dictionary with image statistics
    """
    image_names = get_image_list(mrc_dir, annotations, include_unlabeled=True)
    
    # Sample images for analysis
    sample_names = image_names[:min(sample_size, len(image_names))]
    
    shapes = []
    dtypes = []
    value_ranges = []
    
    for name in sample_names:
        try:
            mrc_path = Path(mrc_dir) / f"{name}.mrc"
            if not mrc_path.exists():
                mrc_path = Path(mrc_dir) / f"{name}.MRC"
            
            image = load_mrc_image(str(mrc_path))
            shapes.append(image.shape)
            dtypes.append(str(image.dtype))
            value_ranges.append((float(image.min()), float(image.max())))
        except Exception as e:
            print(f"Warning: Could not load {name}: {e}")
    
    # Check if all images have the same shape
    unique_shapes = set(shapes)
    
    stats = {
        'num_mrc_files': len(image_names),
        'num_sampled': len(shapes),
        'unique_shapes': [list(s) for s in unique_shapes],
        'all_same_shape': len(unique_shapes) == 1,
        'dtypes': list(set(dtypes)),
        'value_ranges': {
            'min': min(vr[0] for vr in value_ranges) if value_ranges else None,
            'max': max(vr[1] for vr in value_ranges) if value_ranges else None
        }
    }
    
    return stats


def check_coordinate_validity(mrc_dir: str, annotations: dict) -> dict:
    """
    Check if coordinates are valid (within image bounds).
    
    Args:
        mrc_dir: Directory containing MRC files
        annotations: Dictionary of annotations
    
    Returns:
        Dictionary with validation results
    """
    invalid_coords = []
    
    for image_name, coords in annotations.items():
        try:
            mrc_path = Path(mrc_dir) / f"{image_name}.mrc"
            if not mrc_path.exists():
                mrc_path = Path(mrc_dir) / f"{image_name}.MRC"
            
            if not mrc_path.exists():
                invalid_coords.append({
                    'image': image_name,
                    'issue': 'MRC file not found'
                })
                continue
            
            image = load_mrc_image(str(mrc_path))
            height, width = image.shape
            
            for x, y in coords:
                # Ensure coordinates are numeric
                try:
                    x_val = float(x)
                    y_val = float(y)
                except (ValueError, TypeError):
                    invalid_coords.append({
                        'image': image_name,
                        'coord': (x, y),
                        'issue': f'Coordinate is not numeric: ({x}, {y})'
                    })
                    continue
                
                if not (0 <= x_val < width and 0 <= y_val < height):
                    invalid_coords.append({
                        'image': image_name,
                        'coord': (x_val, y_val),
                        'image_size': (width, height),
                        'issue': 'Coordinate out of bounds'
                    })
        
        except Exception as e:
            invalid_coords.append({
                'image': image_name,
                'issue': f'Error loading image: {e}'
            })
    
    return {
        'num_checked': len(annotations),
        'num_invalid': len(invalid_coords),
        'invalid_details': invalid_coords[:10]  # Show first 10
    }


def print_report(stats: dict, title: str):
    """Pretty print statistics report."""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


def main():
    """
    Main data inspection function.
    """
    parser = argparse.ArgumentParser(description='Inspect and analyze microtubule dataset')
    
    parser.add_argument('--mrc_dir', type=str, required=True,
                       help='Directory containing MRC files')
    parser.add_argument('--annotation_file', type=str, required=True,
                       help='Path to annotation file')
    parser.add_argument('--sample_size', type=int, default=10,
                       help='Number of images to sample for analysis')
    parser.add_argument('--output_file', type=str,
                       help='Optional path to save report as JSON')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("DATASET INSPECTION TOOL")
    print("="*60)
    print(f"\nMRC Directory: {args.mrc_dir}")
    print(f"Annotation File: {args.annotation_file}")
    
    # Analyze annotations
    print("\n[1/3] Analyzing annotations...")
    annotation_stats = analyze_annotations(args.annotation_file)
    print_report(annotation_stats, "ANNOTATION STATISTICS")
    
    # Analyze images
    print("\n[2/3] Analyzing images...")
    annotations = load_annotations(args.annotation_file)
    image_stats = analyze_images(args.mrc_dir, annotations, args.sample_size)
    print_report(image_stats, "IMAGE STATISTICS")
    
    # Check coordinate validity
    print("\n[3/3] Checking coordinate validity...")
    validity_stats = check_coordinate_validity(args.mrc_dir, annotations)
    print_report(validity_stats, "COORDINATE VALIDATION")
    
    # Overall summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Found {annotation_stats['num_images']} annotated images")
    print(f"✓ Total {annotation_stats['total_microtubules']} microtubules")
    print(f"✓ Average {annotation_stats['avg_microtubules_per_image']:.1f} microtubules per image")
    print(f"✓ Found {image_stats['num_mrc_files']} MRC files")
    
    if image_stats['all_same_shape']:
        print(f"✓ All images have consistent shape: {image_stats['unique_shapes'][0]}")
    else:
        print(f"⚠ Images have different shapes: {image_stats['unique_shapes']}")
    
    if validity_stats['num_invalid'] == 0:
        print("✓ All coordinates are valid")
    else:
        print(f"⚠ Found {validity_stats['num_invalid']} invalid coordinates")
    
    print("="*60 + "\n")
    
    # Save report
    if args.output_file:
        report = {
            'annotations': annotation_stats,
            'images': image_stats,
            'validation': validity_stats
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {args.output_file}")


if __name__ == '__main__':
    main()
