#!/usr/bin/env python3
"""
Script to extract images from analysis JSON files.

This script processes misclassified_samples.json and low_confidence_samples.json
from the analysis_epoch_20 directory and copies the corresponding images to
separate output directories for further analysis.
"""

import json
import shutil
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse


def load_json_file(json_path: Path) -> List[Dict[str, Any]]:
    """Load and parse a JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_path_to_unix(windows_path: str, base_dir: Path) -> Path:
    """Convert Windows-style path to Unix path relative to base directory."""
    # Convert backslashes to forward slashes and remove any drive letters
    unix_path = windows_path.replace('\\', '/')
    
    # Remove any drive letters (e.g., "C:")
    if ':' in unix_path:
        unix_path = unix_path.split(':', 1)[1]
    
    # Remove leading slash if present
    if unix_path.startswith('/'):
        unix_path = unix_path[1:]
    
    return base_dir / unix_path


def extract_images(samples: List[Dict[str, Any]], 
                  output_dir: Path, 
                  base_dir: Path,
                  sample_type: str) -> None:
    """Extract images from sample data to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    missing_count = 0
    
    print(f"Processing {len(samples)} {sample_type} samples...")
    
    for i, sample in enumerate(samples):
        # Convert Windows path to Unix path
        image_path = convert_path_to_unix(sample['path'], base_dir)
        
        if not image_path.exists():
            print(f"Warning: Image not found - {image_path}")
            missing_count += 1
            continue
        
        # Create output filename with additional metadata
        true_label = sample['true_label']
        pred_label = sample['pred_label']
        confidence = sample['confidence']
        
        # Extract original filename without extension
        original_name = image_path.stem
        extension = image_path.suffix
        
        # Create descriptive filename
        output_filename = f"{original_name}_true{true_label}_pred{pred_label}_conf{confidence:.3f}{extension}"
        output_path = output_dir / output_filename
        
        # Copy the image
        try:
            shutil.copy2(image_path, output_path)
            copied_count += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples...")
                
        except Exception as e:
            print(f"Error copying {image_path}: {e}")
            missing_count += 1
    
    print(f"Completed {sample_type}:")
    print(f"  - Successfully copied: {copied_count} images")
    print(f"  - Missing/failed: {missing_count} images")


def main():
    parser = argparse.ArgumentParser(description="Extract images from analysis JSON files")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(),
                       help="Base directory for the project (default: current directory)")
    parser.add_argument("--analysis-dir", type=Path, default="analysis_epoch_20",
                       help="Directory containing the analysis JSON files")
    parser.add_argument("--output-dir", type=Path, default="extracted_analysis_images",
                       help="Output directory for extracted images")
    
    args = parser.parse_args()
    
    # Set up paths
    base_dir = args.base_dir
    analysis_dir = base_dir / args.analysis_dir
    output_base_dir = base_dir / args.output_dir
    
    # JSON file paths
    misclassified_json = analysis_dir / "misclassified_samples.json"
    low_confidence_json = analysis_dir / "low_confidence_samples.json"
    
    # Output directories
    misclassified_output_dir = output_base_dir / "misclassified"
    low_confidence_output_dir = output_base_dir / "low_confidence"
    
    print(f"Base directory: {base_dir}")
    print(f"Analysis directory: {analysis_dir}")
    print(f"Output directory: {output_base_dir}")
    print()
    
    # Process misclassified samples
    try:
        misclassified_samples = load_json_file(misclassified_json)
        extract_images(misclassified_samples, misclassified_output_dir, 
                      base_dir, "misclassified")
        print()
    except Exception as e:
        print(f"Error processing misclassified samples: {e}")
    
    # Process low confidence samples
    try:
        low_confidence_samples = load_json_file(low_confidence_json)
        extract_images(low_confidence_samples, low_confidence_output_dir, 
                      base_dir, "low_confidence")
        print()
    except Exception as e:
        print(f"Error processing low confidence samples: {e}")
    
    print("Image extraction completed!")
    print(f"Check the output directory: {output_base_dir}")


if __name__ == "__main__":
    main()