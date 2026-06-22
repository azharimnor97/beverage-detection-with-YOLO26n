"""
Image Data Augmentation Script for Computer Vision / Object Detection
======================================================================
Applies a simplified set of 6 augmentation techniques to every image in an input folder
and saves the results to an output folder.

Augmentations applied per image:
  1. Horizontal flip
  2. Vertical flip
  3. High brightness    (+60)
  4. Low brightness     (-60)
  5. Gaussian blur      (radius 2)
  6. Grayscale

Dependencies:
  pip install Pillow numpy
"""

import os
import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# ── Configuration ────────────────────────────────────────────────────────────

INPUT_DIR  = "../data/raw_images/sample_images"   # folder containing source images
OUTPUT_DIR = "../data/raw_images/augment_images2" # augmented images will be saved here

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ── Augmentation helpers ─────────────────────────────────────────────────────

def adjust_brightness(img: Image.Image, delta: int) -> Image.Image:
	"""Shift every pixel's brightness by *delta* (positive = brighter)."""
	arr = np.array(img, dtype=np.int16)
	arr = np.clip(arr + delta, 0, 255).astype(np.uint8)
	return Image.fromarray(arr)


# ── Augmentation catalogue ───────────────────────────────────────────────────

def get_augmentations():
	"""
	Returns a list of (suffix, transform_fn) pairs.
	Each transform_fn accepts a PIL Image and returns a PIL Image.
	"""
	return [
		("flip_h",      lambda img: ImageOps.mirror(img)),
		("flip_v",      lambda img: ImageOps.flip(img)),
		("bright_up",   lambda img: adjust_brightness(img,  60)),
		("bright_down", lambda img: adjust_brightness(img, -60)),
		("blur",        lambda img: img.filter(ImageFilter.GaussianBlur(radius=2))),
		("grayscale",   lambda img: ImageOps.grayscale(img).convert(img.mode)),
	]


# ── Core processing ──────────────────────────────────────────────────────────

def process_image(src_path: Path, out_dir: Path, augmentations: list):
	"""Load one image, apply every augmentation, and save the results."""
	try:
		img  = Image.open(src_path).convert("RGB")
		stem = src_path.stem
		ext  = src_path.suffix.lower()

		saved = 0
		for suffix, transform in augmentations:
			aug_img  = transform(img)
			out_name = f"{stem}__{suffix}{ext}"
			out_path = out_dir / out_name
			aug_img.save(out_path)
			saved += 1

		print(f"  ✓  {src_path.name}  →  {saved} augmented images")
		return saved

	except Exception as exc:
		print(f"  ✗  {src_path.name}  –  ERROR: {exc}")
		return 0


def main():
	input_dir  = Path(INPUT_DIR)
	output_dir = Path(OUTPUT_DIR)

	# ── Validation ────────────────────────────────────────────────────────────
	if not input_dir.exists():
		print(f"[ERROR] Input directory '{input_dir}' not found.")
		print(f"        Create it and place your images inside, then re-run.")
		return

	images = sorted(
		p for p in input_dir.iterdir()
		if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
	)

	if not images:
		print(f"[ERROR] No supported images found in '{input_dir}'.")
		print(f"        Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
		return

	output_dir.mkdir(parents=True, exist_ok=True)
	augmentations = get_augmentations()

	print("=" * 60)
	print(" Image Data Augmentation (6 transformations)")
	print("=" * 60)
	print(f" Input  : {input_dir.resolve()}")
	print(f" Output : {output_dir.resolve()}")
	print(f" Images : {len(images)}")
	print(f" Augmentations per image: {len(augmentations)}")
	print(f" Expected total output   : {len(images) * len(augmentations)}")
	print("=" * 60)

	total_saved = 0
	for img_path in images:
		total_saved += process_image(img_path, output_dir, augmentations)

	print("=" * 60)
	print(f" Done!  {total_saved} augmented images saved to '{output_dir}'.")
	print("=" * 60)


if __name__ == "__main__":
	main()