#!/usr/bin/env python3
"""
Full Rotation Preprocessing Pipeline

This script processes rotation data batches for OCR training:
1. Converts COCO annotations to OBB format
2. Crops oriented bounding boxes from images (with batch info in filename)

Directory layout expected:
rotation/
└── batches/
    ├── task_batch_name/
    │   ├── images/
    │   │   ├── default/*.png
    │   │   └── boxes/ (created by script)
    │   └── annotations/
    │       ├── instances_default.json
    │       └── instances_updated.json (created by script)
    └── ...
"""

import cv2
import os
import math
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm
import json
import shutil
import random
import datetime as dt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Configuration
coco_cache = {}

def setup_paths():
    """Setup and verify directory paths"""
    print("Working dir :", Path.cwd())
    
    # Show the absolute target
    BATCHES_DIR = Path("../data/rotation/batches")
    print("Batch dir   :", BATCHES_DIR)
    
    # Does it exist?
    print("Exists?     :", BATCHES_DIR.exists())
    if BATCHES_DIR.exists():
        print("Contents    :", list(BATCHES_DIR.iterdir())[:5])  # peek first 5 entries
    
    return BATCHES_DIR

def check_batch_structure(batches_dir: Path) -> None:
    """Check and report on batch structure without renaming"""
    print("\nChecking batch structure:")
    batch_dirs = [p for p in batches_dir.iterdir() if p.is_dir()]
    
    valid_batches = []
    for batch_dir in batch_dirs:
        if (batch_dir / 'images' / 'default').exists() and (batch_dir / 'annotations' / 'instances_default.json').exists():
            valid_batches.append(batch_dir)
            print(f"✓ Valid batch: {batch_dir.name}")
        else:
            print(f"✗ Invalid batch: {batch_dir.name} - missing images/default or annotations")
    
    print(f"\nFound {len(valid_batches)} valid batches out of {len(batch_dirs)} total directories")
    return valid_batches

def load_coco(json_path: Path) -> Dict[str, Any]:
    """Load COCO JSON with caching"""
    if json_path in coco_cache:
        return coco_cache[json_path]
    
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    coco_cache[json_path] = coco
    return coco

def create_obb_tuple(anns):
    """Convert COCO bbox to OBB format"""
    bbox = anns.get("bbox", "No bbox found")
    if len(bbox) == 4:    
        x, y, w, h = anns["bbox"]
        cx = x + (w/2)
        cy = y + (h/2)
        angle = anns["attributes"].get("rotation", 0.0)
        obb_list = [cx, cy, w, h, angle]
        anns["bbox"] = obb_list
    else: 
        print("Weirdle after every element is on 5 tuples it starts to iterate again")

def process_single_batch(batch_path: Path):
    """Process a single batch for OBB conversion"""
    json_path = batch_path / "annotations" / "instances_default.json"
    if not json_path.exists():
        return
    
    coco = load_coco(json_path)
    
    for anns in coco['annotations']:
        create_obb_tuple(anns)
        
    output_path = batch_path / "annotations" / "instances_updated.json"
    with open(output_path, 'w') as f:
        json.dump(coco, f)
    
    print(f"Processed: {batch_path.name}")

def convert_all_batches(batches_dir: Path):
    """Convert all batches to OBB format in parallel"""
    batch_paths = [p for p in batches_dir.iterdir() if p.is_dir() and 
                   (p / 'images' / 'default').exists() and 
                   (p / 'annotations' / 'instances_default.json').exists()]
    
    with ThreadPoolExecutor(max_workers=min(len(batch_paths), mp.cpu_count())) as executor:
        futures = [executor.submit(process_single_batch, p) for p in batch_paths]
        
        for future in tqdm(futures, desc="Converting batches"):
            future.result()

def crop_oriented_bbox(img, cx, cy, w, h, theta):
    """Crop oriented bounding box from image"""
    # Step 1: Rotate the entire image around the bbox center
    M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    # Step 2: Crop the now-aligned rectangle
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)
    
    # Ensure bounds are within image
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    
    cropped = rotated[y1:y2, x1:x2]
    return cropped

def process_image_group(args):
    """Process a single image and all its annotations"""
    image_path, file_name, annotations, dest_dir = args
    
    img_path = image_path / file_name
    img_arr = cv2.imread(str(img_path))
    if img_arr is None:
        return f"Could not load {img_path}"
    
    file_number = file_name.replace('.png', "")
    crops_made = 0
    
    # Process all annotations for this image
    for ann in annotations:
        try:
            cx, cy, w, h, theta = ann["bbox"]
            rotated_box = crop_oriented_bbox(img_arr, cx, cy, w, h, theta)
            
            # Filename format: filename_annotationid.png
            output_path = dest_dir / f"{file_number}_{ann['id']}.png"
            cv2.imwrite(str(output_path), rotated_box)
            crops_made += 1
        except Exception as e:
            continue
    
    return f"Processed {file_name}: {crops_made} crops"

def crop_boxes_from_batch(batch_path: Path):
    """Process box cropping for a single batch with live progress"""
    if not ((batch_path / "annotations" / "instances_updated.json").exists() and 
            (batch_path / "images" / "default").exists()):
        return f"Skipped {batch_path.name} - missing files"
    
    image_path = batch_path / "images" / "default"
    coco = load_coco(batch_path / "annotations" / "instances_updated.json")
    
    DEST_IMG_DIR = batch_path / "images" / "boxes"
    DEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Group annotations by image_id
    image_groups = {}
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        if image_id not in image_groups:
            image_groups[image_id] = []
        image_groups[image_id].append(ann)
    
    # Prepare arguments for parallel processing
    process_args = []
    for image_id, annotations in image_groups.items():
        img_meta = next((img for img in coco["images"] if img["id"] == image_id), None)
        if not img_meta:
            continue
        
        file_name = img_meta.get('file_name')
        process_args.append((image_path, file_name, annotations, DEST_IMG_DIR))
    
    # Process images in parallel with live progress
    max_workers = min(len(process_args), mp.cpu_count())
    processed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_image_group, args): args for args in process_args}
        
        # Process with live progress
        with tqdm(total=len(process_args), desc=f"Processing {batch_path.name}", 
                 unit="images", leave=False) as pbar:
            for future in as_completed(futures):
                result = future.result()
                processed_count += 1
                pbar.update(1)
                pbar.set_postfix({"completed": processed_count})
    
    return f"✓ {batch_path.name}: {processed_count} images, {len(coco['annotations'])} crops"

def crop_all_boxes_with_progress(batches_dir: Path):
    """Process all batches with nested progress bars"""
    batch_paths = [p for p in batches_dir.iterdir() if p.is_dir() and 
                   (p / 'images' / 'default').exists() and 
                   (p / 'annotations' / 'instances_updated.json').exists()]
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor for better progress tracking
    max_batch_workers = min(len(batch_paths), 4)
    
    with ThreadPoolExecutor(max_workers=max_batch_workers) as executor:
        # Submit all batch tasks
        futures = {executor.submit(crop_boxes_from_batch, p): p for p in batch_paths}
        
        # Process with main progress bar
        with tqdm(total=len(batch_paths), desc="Processing batches", 
                 unit="batch", position=0) as main_pbar:
            for future in as_completed(futures):
                batch_path = futures[future]
                result = future.result()
                main_pbar.update(1)
                main_pbar.set_postfix({"current": batch_path.name})
                print(f"  {result}")


def main():
    """Main pipeline execution"""
    print("Starting Full Rotation Preprocessing Pipeline...")
    
    # Setup paths
    batches_dir = setup_paths()
    
    # Step 1: Check batch structure
    print("\n1. Checking batch structure...")
    valid_batches = check_batch_structure(batches_dir)
    
    if not valid_batches:
        print("No valid batches found! Exiting.")
        return
    
    # Step 2: Convert to OBB format
    print("\n2. Converting to OBB format...")
    convert_all_batches(batches_dir)
    
    # Step 3: Crop oriented boxes
    print("\n3. Cropping oriented boxes...")
    crop_all_boxes_with_progress(batches_dir)
    
    print("\n✓ Pipeline completed successfully!")
    print("\nCropped images are saved in each batch's 'images/boxes/' directory.")
    print("Filename format: {image_name}_{annotation_id}.png")

if __name__ == "__main__":
    main()