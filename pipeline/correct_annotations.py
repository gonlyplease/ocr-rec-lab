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


def rotate_axis_aligned_bbox(x, y, w, h, correction_angle, img_width, img_height):
    """
    Rotate an axis-aligned bbox by correction_angle and return the new axis-aligned bbox.
    
    Args:
        x, y: Top-left corner of original bbox
        w, h: Width and height of original bbox
        correction_angle: Rotation angle in degrees (positive = counter-clockwise)
        img_width, img_height: Image dimensions for boundary checking
    
    Returns:
        new_x, new_y, new_w, new_h: New axis-aligned bbox that contains the rotated rectangle
    """
    # Convert to radians (negative because OpenCV uses clockwise rotation)
    theta = np.deg2rad(-correction_angle)
    
    # Center point of the bbox
    cx = x + w/2
    cy = y + h/2
    
    # Four corners of the original bbox
    corners = np.array([
        [x, y],           # top-left
        [x + w, y],       # top-right  
        [x + w, y + h],   # bottom-right
        [x, y + h]        # bottom-left
    ])
    
    # Rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta], 
        [sin_theta, cos_theta]
    ])
    
    # Translate corners to origin (center at 0,0)
    centered_corners = corners - np.array([cx, cy])
    
    # Apply rotation
    rotated_corners = np.dot(centered_corners, rotation_matrix.T)
    
    # Translate back to original position
    final_corners = rotated_corners + np.array([cx, cy])
    
    # Find the bounding box of rotated corners
    min_x = np.min(final_corners[:, 0])
    max_x = np.max(final_corners[:, 0])
    min_y = np.min(final_corners[:, 1])
    max_y = np.max(final_corners[:, 1])
    
    # Calculate new bbox parameters
    new_x = min_x
    new_y = min_y
    new_w = max_x - min_x
    new_h = max_y - min_y
    
    # Clamp to image boundaries
    # First clamp the position
    if new_x < 0:
        new_w += new_x  # Reduce width by the amount we're shifting
        new_x = 0
    if new_y < 0:
        new_h += new_y  # Reduce height by the amount we're shifting
        new_y = 0
    
    # Then clamp the size
    if new_x + new_w > img_width:
        new_w = img_width - new_x
    if new_y + new_h > img_height:
        new_h = img_height - new_y
    
    # Ensure we don't have negative dimensions
    new_w = max(0, new_w)
    new_h = max(0, new_h)
    
    return new_x, new_y, new_w, new_h


# Annotation Correction
def rotate_axis_aligned_bbox(x, y, w, h, correction_angle, img_width, img_height):
    """Rotate an axis-aligned bbox and recalculate the new top-left corner."""
    # Convert correction angle to radians
    theta = np.deg2rad(correction_angle)
    
    # Current center point
    cx = x + w/2
    cy = y + h/2
    
    # Define the four corners of the original bbox
    corners = np.array([
        [x, y],           # top-left
        [x + w, y],       # top-right  
        [x + w, y + h],   # bottom-right
        [x, y + h]        # bottom-left
    ])
    
    # Rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta], 
                               [sin_theta, cos_theta]])
    
    # Rotate corners around center point
    centered_corners = corners - np.array([cx, cy])
    rotated_corners = centered_corners @ rotation_matrix.T
    final_corners = rotated_corners + np.array([cx, cy])
    
    # Find new axis-aligned bounding box
    min_x = np.min(final_corners[:, 0])
    min_y = np.min(final_corners[:, 1])
    max_x = np.max(final_corners[:, 0])
    max_y = np.max(final_corners[:, 1])
    
    # Clamp to image boundaries
    new_x = max(0, min_x)
    new_y = max(0, min_y)
    new_w = min(img_width - new_x, max_x - min_x)
    new_h = min(img_height - new_y, max_y - min_y)
    
    return new_x, new_y, new_w, new_h

def update_annotations_with_predictions(coco_data: dict, predictions: pd.DataFrame) -> dict:
    """Update COCO annotations with corrected rotations and adjusted bounding boxes."""
    coco_corrected = deepcopy(coco_data)
    pred_dict = predictions.set_index('id')[['orig', 'pred']].to_dict('index')
    
    # Create image lookup for dimensions
    img_lookup = {img['id']: img for img in coco_corrected['images']}
    
    for ann in coco_corrected["annotations"]:
        if ann["id"] in pred_dict:
            orig_rotation = pred_dict[ann["id"]]['orig']
            predicted_rotation = pred_dict[ann["id"]]['pred']
            
            # Calculate correction needed
            correction_angle = predicted_rotation - orig_rotation
            
            if correction_angle != 0:  # Only update if correction is needed
                # Get image dimensions
                img_info = img_lookup[ann['image_id']]
                img_width = img_info['width']
                img_height = img_info['height']
                
                # Original bbox format: [x, y, w, h]
                x, y, w, h = ann["bbox"]
                
                # Calculate new bbox after rotation correction
                new_x, new_y, new_w, new_h = rotate_axis_aligned_bbox(
                    x, y, w, h, correction_angle, img_width, img_height
                )
                
                # Update both bbox coordinates AND rotation attribute
                ann["bbox"] = [new_x, new_y, new_w, new_h]
                
                # Update the rotation attribute to the corrected value
                if "attributes" not in ann:
                    ann["attributes"] = {}
                ann["attributes"]["rotation"] = predicted_rotation
    
    return coco_corrected

# %%
# Batch Processing
def process_batch(batch_dir: Path, model: nn.Module, debug: bool=False, save_corrected: bool=True) -> pd.DataFrame:
    img_dir = batch_dir / "images" / "default"
    coco_default = load_coco(batch_dir / "annotations" / "instances_default.json")
    coco_obb = deepcopy(coco_default)
    for ann in coco_obb["annotations"]:
        create_obb(ann)

    records = []
    cache = {}
    for ann in tqdm(coco_obb["annotations"], desc=batch_dir.name):
        cx, cy, w, h, orig = ann["bbox"]
        if orig not in CLASS_NAMES:
            continue
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

        patch = crop_obb_trim(img, cx, cy, w, h, orig)
        pred = predict_rotation(model, patch)
        records.append({"id": ann["id"], "orig": orig, "pred": pred, "file": path.name})
        if debug:
            # Create separate folders for original and corrected crops
            orig_dir = DEBUG_IMAGES_DIR / "orig"
            corrected_dir = DEBUG_IMAGES_DIR / "corrected"
            orig_dir.mkdir(parents=True, exist_ok=True)
            corrected_dir.mkdir(parents=True, exist_ok=True)
            
            # Save original crop
            cv2.imwrite(str(orig_dir / f"{batch_dir.name}_{ann['id']}.png"), patch)
            
            # Save corrected crop (always save, even if same as original)
            corrected_patch = crop_obb_trim(img, cx, cy, w, h, pred)
            cv2.imwrite(str(corrected_dir / f"{batch_dir.name}_{ann['id']}.png"), corrected_patch)
    
    from pandas import DataFrame
    df_results = DataFrame(records)
    
    # Save corrected annotations if requested
    if save_corrected and not df_results.empty:
        corrected_coco = update_annotations_with_predictions(coco_default, df_results)
        corrected_path = batch_dir / "annotations" / "instances_corrected.json"
        save_coco(corrected_coco, corrected_path)
        logger.info(f"Saved corrected annotations to {corrected_path}")
    
    return df_results

# %%
# Run All Batches

# Configuration
SAVE_CORRECTIONS = True  # Set to False for analysis-only mode
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



