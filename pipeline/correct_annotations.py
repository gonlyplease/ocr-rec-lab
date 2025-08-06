# %% [markdown]
#  notebook provides a production-ready pipeline for:
# 
# 1. Converting COCO axis-aligned bboxes to oriented bounding boxes (OBBs).
# 2. Cropping image patches based on OBBs and trimming borders.
# 3. Predicting object orientation using a trained ResNet18 model.
# 4. Updating COCO annotations with predicted rotations.
# 5. Running end-to-end over all batches in a specified directory.
# 
# The flow follows modular functions, dynamic batch discovery, progress bars, and clean logging.

# %%


# %%
# Imports & Configuration
import json
import logging
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Dict, Any
from tqdm.auto import tqdm

# Base directory (adjust if needed)
BASE_DIR = Path().resolve().parent
DATA_DIR = BASE_DIR / "data" / "rotation" / "batches"
CHECKPOINT_PATH = BASE_DIR / "pipeline" / "checkpoints" / "best_model.pth"
DEBUG_IMAGES_DIR = BASE_DIR / "pipeline" / "debug_imgs"
RESULTS_CSV = BASE_DIR / "pipeline" / "results.csv"

# Classes and device
CLASS_NAMES = [0, 90, 180, 270]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 300

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rotation_pipeline")

# Transform
from torchvision.transforms import InterpolationMode
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# %%
# I/O Functions
def load_coco(path: Path) -> dict:
    logger.info(f"Loading COCO JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def save_coco(coco: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving updated COCO to {path}")
    path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

# %%
# Model Loading
def load_model(ckpt_path: Path) -> nn.Module:
    logger.info(f"Loading model from {ckpt_path}")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval()

# %%
# OBB Utilities
def create_obb(ann: Dict[str, Any]):
    x, y, w, h = ann["bbox"]
    cx, cy = x + w/2, y + h/2
    angle = ann.get("attributes", {}).get("rotation", 0.0)
    ann["bbox"] = [cx, cy, w, h, angle]

def crop_oriented_bbox(img, cx, cy, w, h, theta):
    """Crop oriented bounding box from image - from extract_crops.py"""
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

# %%
# Image & Rotation Helpers
def crop_obb_trim(img: np.ndarray, cx, cy, w, h, angle, pad=0) -> np.ndarray:
    theta = np.deg2rad(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    corners = np.float32([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]) @ R.T + np.array([cx, cy])
    xs, ys = corners[:,0], corners[:,1]
    x0, x1 = max(int(np.floor(xs.min()))-pad, 0), min(int(np.ceil(xs.max()))+pad, img.shape[1]-1)
    y0, y1 = max(int(np.floor(ys.min()))-pad, 0), min(int(np.ceil(ys.max()))+pad, img.shape[0]-1)
    roi = img[y0:y1+1, x0:x1+1]
    mask = cv2.fillPoly(np.zeros(roi.shape[:2], np.uint8), [np.round(corners - [x0, y0]).astype(np.int32)], 255)
    masked = cv2.bitwise_and(roi, roi, mask=mask)
    ys_nz, xs_nz = np.where(mask>0)
    return masked[ys_nz.min():ys_nz.max()+1, xs_nz.min():xs_nz.max()+1]

def predict_rotation(model: nn.Module, patch: np.ndarray) -> float:
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
    return float(CLASS_NAMES[torch.argmax(logits, dim=1).item()])


def bbox_to_points(x, y, w, h):
    """Convert bbox [x,y,w,h] to 4 corner points."""
    return [
        [x, y],             # top-left
        [x + w, y],         # top-right
        [x + w, y + h],     # bottom-right
        [x, y + h]          # bottom-left
    ]


def roll_points(points, correction_angle):
    """Roll points based on rotation angle (rounded to nearest 90 degrees)."""
    # Normalize angle to 0-360 range and round to nearest 90 degrees
    angle = correction_angle % 360
    angle = round(angle / 90) * 90
    angle = angle % 360
    
    if angle == 0:
        return points
    elif angle == 90:
        # Roll forward by 1: TL->TR, TR->BR, BR->BL, BL->TL
        return [points[3], points[0], points[1], points[2]]
    elif angle == 180:
        # Roll forward by 2: TL->BR, TR->BL, BR->TL, BL->TR
        return [points[2], points[3], points[0], points[1]]
    elif angle == 270:
        # Roll forward by 3: TL->BL, TR->TL, BR->TR, BL->BR
        return [points[1], points[2], points[3], points[0]]


def points_to_bbox(points):
    """Convert 4 points back to [x,y,w,h] format by finding axis-aligned bounding box."""
    # Extract all x and y coordinates
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    
    # Find min/max to create axis-aligned bounding box
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Convert to [x, y, w, h] format
    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min
    
    return x, y, w, h


def rotate_axis_aligned_bbox(x, y, w, h, correction_angle, img_width, img_height):
    """
    Update bbox for rotation correction by swapping dimensions when needed.
    
    Args:
        x, y: Top-left corner of original bbox
        w, h: Width and height of original bbox
        correction_angle: Rotation correction in degrees 
        img_width, img_height: Image dimensions for boundary checking
    
    Returns:
        new_x, new_y, new_w, new_h: New axis-aligned bbox
    """
    # Normalize angle to 0-360 range and round to nearest 90 degrees
    angle = correction_angle % 360
    angle = round(angle / 90) * 90
    angle = angle % 360
    
    # Center of the original bbox
    cx = x + w/2
    cy = y + h/2
    
    if angle == 0:
        new_x, new_y, new_w, new_h = x, y, w, h
    elif angle == 90:
        # 90° rotation: width becomes height, height becomes width
        new_w = h
        new_h = w
        new_x = cx - new_w/2
        new_y = cy - new_h/2
    elif angle == 180:
        # 180° rotation: dimensions stay the same, just recentered
        new_w = w
        new_h = h
        new_x = cx - new_w/2
        new_y = cy - new_h/2
    elif angle == 270:
        # 270° rotation: width becomes height, height becomes width
        new_w = h
        new_h = w
        new_x = cx - new_w/2
        new_y = cy - new_h/2
    
    # Ensure bbox stays within image bounds
    new_x = max(0, min(new_x, img_width - new_w))
    new_y = max(0, min(new_y, img_height - new_h))
    
    return new_x, new_y, new_w, new_h


def update_annotations_with_predictions(coco_data: dict, predictions: pd.DataFrame) -> dict:
    """Update COCO annotations with corrected polygon points based on rotation predictions."""
    coco_corrected = deepcopy(coco_data)
    pred_dict = predictions.set_index('id')[['pred', 'original_angle']].to_dict('index')
    
    for ann in coco_corrected["annotations"]:
        if ann["id"] in pred_dict:
            predicted_rotation = pred_dict[ann["id"]]['pred']
            original_angle = pred_dict[ann["id"]]['original_angle']
            
            if predicted_rotation != 0:  # Only update if correction is needed
                # Get original bbox [x, y, w, h]
                x, y, w, h = ann["bbox"]
                
                # Apply rotation transformation to bbox
                # For -270° (equivalent to +90°): w becomes h, h becomes w
                # For -180°: dimensions stay same, position changes
                # For -90° (equivalent to +270°): w becomes h, h becomes w
                rotation_normalized = (-predicted_rotation) % 360
                
                if rotation_normalized == 90 or rotation_normalized == 270:
                    # Width and height swap for 90° and 270° rotations
                    new_w, new_h = h, w
                    # Adjust position - center stays roughly the same
                    cx, cy = x + w/2, y + h/2
                    new_x = cx - new_w/2
                    new_y = cy - new_h/2
                else:
                    # For 0° and 180°, dimensions stay the same
                    new_x, new_y, new_w, new_h = x, y, w, h
                
                ann["bbox"] = [new_x, new_y, new_w, new_h]
                
                # Update segmentation if present
                if "segmentation" in ann and ann["segmentation"] and len(ann["segmentation"]) > 0:
                    seg_points = ann["segmentation"][0]
                    if len(seg_points) >= 8:  # At least 4 points * 2 coordinates
                        # Convert to point pairs (taking first 4 points if more exist)
                        seg_point_pairs = [[seg_points[i], seg_points[i+1]] for i in range(0, 8, 2)]
                        
                        # Roll the segmentation points the same way
                        rolled_seg_points = roll_points(seg_point_pairs, -predicted_rotation)
                        
                        # Convert back to flat list (keep any additional points unchanged)
                        new_seg_points = [coord for point in rolled_seg_points for coord in point]
                        if len(seg_points) > 8:
                            new_seg_points.extend(seg_points[8:])  # Keep additional points
                        ann["segmentation"][0] = new_seg_points
                
                # Update rotation attribute - final rotation after applying correction
                # Since we rolled the polygon by -predicted_rotation, 
                # final rotation = original_rotation - predicted_rotation
                final_rotation = (original_angle - predicted_rotation) % 360
                if "attributes" not in ann:
                    ann["attributes"] = {}
                ann["attributes"]["rotation"] = final_rotation
            else:
                # No correction needed, keep original rotation
                if "attributes" not in ann:
                    ann["attributes"] = {}
                ann["attributes"]["rotation"] = original_angle
    
    return coco_corrected

# %%
# Batch Processing
def process_batch(batch_dir: Path, model: nn.Module, debug: bool=False, save_corrected: bool=True) -> pd.DataFrame:
    img_dir = batch_dir / "images" / "default"
    coco_default = load_coco(batch_dir / "annotations" / "instances_default.json")
    
    # Convert to OBB format for oriented cropping
    coco_obb = deepcopy(coco_default)
    for ann in coco_obb["annotations"]:
        create_obb(ann)

    records = []
    cache = {}
    for ann in tqdm(coco_obb["annotations"], desc=batch_dir.name):
        # Get OBB format [cx, cy, w, h, angle]
        cx, cy, w, h, angle = ann["bbox"]
        
        img_info = next(img for img in coco_default["images"] if img["id"]==ann["image_id"])
        path = img_dir / img_info["file_name"]
        if not path.exists():
            logger.error(f"Missing file {path}")
            continue
        # Load image into cache
        if path in cache:
            img = cache[path]
        else:
            img = cv2.imread(str(path))
            if img is None:
                logger.error(f"Failed to read {path}")
                continue
            cache[path] = img

        # Extract oriented crop (this makes it horizontal)
        crop = crop_oriented_bbox(img, cx, cy, w, h, angle)
        if crop.size == 0:
            logger.warning(f"Empty crop for annotation {ann['id']}")
            continue
            
        # Predict the correct orientation on the horizontal crop
        pred_rotation = predict_rotation(model, crop)
        
        records.append({
            "id": ann["id"], 
            "pred": pred_rotation, 
            "file": path.name,
            "original_angle": angle
        })
        
        if debug:
            # Create separate folders for original and corrected crops
            orig_dir = DEBUG_IMAGES_DIR / "orig"
            corrected_dir = DEBUG_IMAGES_DIR / "corrected"
            orig_dir.mkdir(parents=True, exist_ok=True)
            corrected_dir.mkdir(parents=True, exist_ok=True)
            
            # Save original oriented crop (should be horizontal)
            cv2.imwrite(str(orig_dir / f"{batch_dir.name}_{ann['id']}.png"), crop)
            
            # Save corrected crop (rotated in opposite direction of prediction)
            if pred_rotation != 0:
                corrected_crop = cv2.rotate(crop, {
                    90: cv2.ROTATE_90_CLOCKWISE,      # Opposite of 90° CCW
                    180: cv2.ROTATE_180,              # 180° is same both ways
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE  # Opposite of 270° (90° CW)
                }.get(pred_rotation, cv2.ROTATE_180))
            else:
                corrected_crop = crop
            cv2.imwrite(str(corrected_dir / f"{batch_dir.name}_{ann['id']}.png"), corrected_crop)
    
    from pandas import DataFrame
    df_results = DataFrame(records)
    
    # Save corrected annotations if requested
    if not df_results.empty:
        corrected_coco = update_annotations_with_predictions(coco_default, df_results)
        corrected_path = batch_dir / "annotations" / "instances_corrected.json"
        save_coco(corrected_coco, corrected_path)
        logger.info(f"Saved corrected annotations to {corrected_path}")
    
    return df_results

# %%
# Run All Batches

# Configuration
SAVE_CORRECTIONS = False  # Set to False for analysis-only mode
DEBUG_MODE = True

torch_model = load_model(CHECKPOINT_PATH)
all_results = []

# Find batch directories (skip zip files)
batch_dirs = [d for d in sorted(DATA_DIR.iterdir()) if d.is_dir()]
logger.info(f"Found {len(batch_dirs)} batch directories to process")

for batch in batch_dirs:
    logger.info(f"Processing batch: {batch.name}")
    df = process_batch(batch, torch_model, debug=DEBUG_MODE, save_corrected=SAVE_CORRECTIONS)
    df["batch"] = batch.name
    all_results.append(df)



