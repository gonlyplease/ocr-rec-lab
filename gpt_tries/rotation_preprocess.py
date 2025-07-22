#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reusable first-step preprocessing utilities for rotation-aware CAD-crop detection.

Sub-commands
------------
rename      : Apply batch-folder naming convention.
inspect     : Summarise a COCO json (images, cats, bbox stats, rotation stats …).
visualize   : Draw rotated bboxes on one or many sample images for QC.

Usage examples
--------------
python rotation_preprocess.py rename   --root /rotation/batches
python rotation_preprocess.py inspect  --json /rotation/batches/annotations/instances_default.json
python rotation_preprocess.py visualize --json ... --images /rotation/batches/images/default --sample 5
"""

import argparse, json, math, random, shutil, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import cv2  # OpenCV ≥4.5 for drawing / display
    import numpy as np  # only needed by `visualize`
except ImportError:
    pass  # not required for rename / inspect


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _date_stamp(path: Path) -> str:
    """File modification date → YYYYMMDD string."""
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d")


def _rotate_rect(x, y, w, h, angle_deg):
    """Return the 4 corner points of a rotated bbox."""
    cx, cy = x + w / 2.0, y + h / 2.0
    rad = math.radians(angle_deg % 360)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    corners = []
    for dx, dy in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]:
        rx = cx + dx * cos_a - dy * sin_a
        ry = cy + dx * sin_a + dy * cos_a
        corners.append((int(rx), int(ry)))
    return corners


# ------------------------------------------------------------
# Sub-command: rename
# ------------------------------------------------------------
def rename_batches(root: Path):
    day_counters = defaultdict(int)
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        stamp = (
            _date_stamp(p / "annotations" / "instances_default.json")
            if (p / "annotations" / "instances_default.json").exists()
            else _date_stamp(p)
        )
        day_counters[stamp] += 1
        new_name = f"batch_{stamp}_{day_counters[stamp]:04d}"
        new_path = p.with_name(new_name)
        if new_path.exists():
            print(f"[WARN] {new_path} already exists, skipping.")
            continue
        print(f"Renaming {p.name} → {new_name}")
        shutil.move(str(p), str(new_path))


# ------------------------------------------------------------
# Sub-command: inspect
# ------------------------------------------------------------
def inspect_json(json_path: Path):
    coco = json.loads(json_path.read_text())
    img_cnt = len(coco["images"])
    cat_cnt = len(coco["categories"])
    ann_cnt = len(coco["annotations"])

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    cat_hist = Counter(a["category_id"] for a in coco["annotations"])
    rot_values = [a["attributes"].get("rotation", 0) for a in coco["annotations"]]

    print(f"Images      : {img_cnt}")
    print(f"Categories  : {cat_cnt}")
    print(f"Annotations : {ann_cnt}\n")

    print("Category histogram:")
    for cid, n in sorted(cat_hist.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {cats.get(cid, cid):20s} {n:>6}")

    if rot_values:
        print("\nRotation (deg) statistics:")
        print(f"  min / max : {min(rot_values):.2f} / {max(rot_values):.2f}")
        mean = sum(rot_values) / len(rot_values)
        print(f"  mean      : {mean:.2f}")
        med = sorted(rot_values)[len(rot_values) // 2]
        print(f"  median    : {med:.2f}")
        zero = sum(1 for r in rot_values if abs(r) < 1e-3)
        print(f"  ≈0° count : {zero} ({zero / len(rot_values):.1%})")


# ------------------------------------------------------------
# Sub-command: visualize
# ------------------------------------------------------------
def visualize(
    json_path: Path, images_root: Path, sample_n: int, specific_id: int | None
):
    if not cv2:
        sys.exit("opencv-python is required for visualization.")
    coco = json.loads(json_path.read_text())
    id2img = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    img_ids = (
        [specific_id] if specific_id else random.sample(list(id2img.keys()), sample_n)
    )
    for iid in img_ids:
        im_meta = id2img[iid]
        img_p = images_root / im_meta["file_name"]
        img = cv2.imread(str(img_p))
        if img is None:
            print(f"[WARN] couldn’t open {img_p}")
            continue

        for a in anns_by_img[iid]:
            x, y, w, h = a["bbox"]
            rot = a["attributes"].get("rotation", 0)
            pts = _rotate_rect(x, y, w, h, rot)
            cv2.polylines(img, [np.array(pts)], True, 255, 2)
            cv2.putText(
                img, f"{rot:.1f}°", pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1
            )

        out_p = img_p.with_name(f"{img_p.stem}_viz.png")
        cv2.imwrite(str(out_p), img)
        print(f"saved → {out_p}")


# ------------------------------------------------------------
# CLI glue
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Pre-processing helpers for rotated-bbox CAD dataset."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_ren = sub.add_parser("rename", help="rename batch folders")
    sp_ren.add_argument("--root", type=Path, default=Path("rotation/batches"))

    sp_ins = sub.add_parser("inspect", help="dataset stats")
    sp_ins.add_argument("--json", type=Path, required=True)

    sp_vis = sub.add_parser("visualize", help="draw a few sample images")
    sp_vis.add_argument("--json", type=Path, required=True)
    sp_vis.add_argument("--images", type=Path, required=True)
    sp_vis.add_argument(
        "--sample",
        type=int,
        default=3,
        help="number of random images (ignored if --id)",
    )
    sp_vis.add_argument(
        "--id", type=int, default=None, help="visualize a specific image-id"
    )

    args = ap.parse_args()
    if args.cmd == "rename":
        rename_batches(args.root)
    elif args.cmd == "inspect":
        inspect_json(args.json)
    elif args.cmd == "visualize":
        visualize(args.json, args.images, args.sample, args.id)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
