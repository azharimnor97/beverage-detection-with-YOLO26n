"""
Image Data Augmentation Script for Computer Vision / Object Detection
======================================================================
Applies multiple augmentation techniques to every image in an input folder
and saves the results to an output folder.

Augmentations applied per image:
  1.  Horizontal flip
  2.  Vertical flip
  3.  Brightness increase  (+60)
  4.  Brightness decrease  (-60)
  5.  High contrast        (factor 1.8)
  6.  Low contrast         (factor 0.5)
  7.  Combined: bright + high contrast
  8.  Combined: dark  + low  contrast
  9.  Rotate 90°
  10. Rotate 180°
  11. Rotate 270°
  12. Gaussian blur        (radius 2)
  13. Sharpen
  14. Grayscale
  15. Salt-and-pepper noise
  16. Zoom-in crop         (80 % of original)
  17. Horizontal flip + brightness increase (compound)

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
OUTPUT_DIR = "../data/raw_images/augment_images"  # augmented images will be saved here

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ── Augmentation helpers ─────────────────────────────────────────────────────

def adjust_brightness(img: Image.Image, delta: int) -> Image.Image:
    """Shift every pixel's brightness by *delta* (positive = brighter)."""
    arr = np.array(img, dtype=np.int16)
    arr = np.clip(arr + delta, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    """
    Scale contrast around the mean pixel value.
    factor > 1 → more contrast; 0 < factor < 1 → less contrast.
    """
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def salt_and_pepper(img: Image.Image, amount: float = 0.02) -> Image.Image:
    """Randomly set *amount* fraction of pixels to pure white or pure black."""
    arr   = np.array(img, dtype=np.uint8)
    total = arr.shape[0] * arr.shape[1]
    n     = int(total * amount)

    # Salt (white)
    rows = np.random.randint(0, arr.shape[0], n)
    cols = np.random.randint(0, arr.shape[1], n)
    arr[rows, cols] = 255

    # Pepper (black)
    rows = np.random.randint(0, arr.shape[0], n)
    cols = np.random.randint(0, arr.shape[1], n)
    arr[rows, cols] = 0

    return Image.fromarray(arr)


def zoom_in_crop(img: Image.Image, scale: float = 0.80) -> Image.Image:
    """Centre-crop the image to *scale* of its original size, then resize back."""
    w, h   = img.size
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    left   = (w - new_w) // 2
    top    = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.LANCZOS)


# ── Augmentation catalogue ───────────────────────────────────────────────────

def get_augmentations():
    """
    Returns a list of (suffix, transform_fn) pairs.
    Each transform_fn accepts a PIL Image and returns a PIL Image.
    """
    return [
        ("flip_h",          lambda img: ImageOps.mirror(img)),
        ("flip_v",          lambda img: ImageOps.flip(img)),
        ("bright_up",       lambda img: adjust_brightness(img,  60)),
        ("bright_down",     lambda img: adjust_brightness(img, -60)),
        ("contrast_high",   lambda img: adjust_contrast(img, 1.8)),
        ("contrast_low",    lambda img: adjust_contrast(img, 0.5)),
        ("bright_contrast", lambda img: adjust_contrast(adjust_brightness(img, 50), 1.6)),
        ("dark_lowcon",     lambda img: adjust_contrast(adjust_brightness(img, -50), 0.6)),
        ("rotate_90",       lambda img: img.rotate(90,  expand=True)),
        ("rotate_180",      lambda img: img.rotate(180, expand=True)),
        ("rotate_270",      lambda img: img.rotate(270, expand=True)),
        ("blur",            lambda img: img.filter(ImageFilter.GaussianBlur(radius=2))),
        ("sharpen",         lambda img: img.filter(ImageFilter.SHARPEN)),
        ("grayscale",       lambda img: ImageOps.grayscale(img).convert(img.mode)),
        ("noise_sp",        lambda img: salt_and_pepper(img, amount=0.02)),
        ("zoom_in",         lambda img: zoom_in_crop(img, scale=0.80)),
        ("flip_h_bright",   lambda img: adjust_brightness(ImageOps.mirror(img), 50)),
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
    print(" Image Data Augmentation")
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