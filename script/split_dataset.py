"""
Dataset Splitter — Train / Validate / Test
==========================================
Splits an image dataset into three partitions while preventing data leakage:
all augmented variants of the same source image are always kept in the
same partition (they are never split across train/val/test).

Default split ratio: 70 % train · 15 % validate · 15 % test
The ratios are fully configurable at the top of this file.

Expected input layout (output of augment_images.py):
  augmented_images/
      cat__flip_h.jpg
      cat__bright_up.jpg
      cat__rotate_90.jpg
      dog__flip_h.jpg
      ...

Output layout:
  dataset/
  ├── train/
  ├── validate/
  └── test/

Dependencies: none beyond the Python standard library
"""

import os
import re
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_DIR  = "../data/raw_images/all_image"        # folder with augmented images
OUTPUT_DIR = "../data/dataset_splited"             # root folder for the three partitions

# Split ratios — must sum to 1.0
TRAIN_RATIO    = 0.70
VALIDATE_RATIO = 0.15
TEST_RATIO     = 0.15

# Set to True to MOVE files instead of copying them
MOVE_FILES = False

# Reproducible shuffling — change or set to None for a random seed
RANDOM_SEED = 42

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Separator used by augment_images.py between the original stem and the
# augmentation suffix (e.g. "cat__flip_h.jpg" → separator is "__")
AUGMENTATION_SEPARATOR = "__"

# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_ratios(train: float, val: float, test: float):
    total = round(train + val + test, 10)
    if total != 1.0:
        raise ValueError(
            f"Split ratios must sum to 1.0, but got {train} + {val} + {test} = {total}"
        )


def group_by_source(image_paths: list[Path]) -> dict[str, list[Path]]:
    """
    Group augmented images by their original source stem.

    'cat__flip_h.jpg'   → group key 'cat'
    'cat__rotate_90.jpg'→ group key 'cat'
    'dog__blur.jpg'     → group key 'dog'

    Images whose name does not contain the separator are treated as
    un-augmented originals and form their own single-item group.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        stem = path.stem
        if AUGMENTATION_SEPARATOR in stem:
            source_key = stem.split(AUGMENTATION_SEPARATOR)[0]
        else:
            source_key = stem          # original / non-augmented image
        groups[source_key].append(path)
    return dict(groups)


def split_groups(
    groups: dict[str, list[Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int | None,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Shuffle source groups and assign each group entirely to one partition.
    This guarantees that no augmented variant of a training image leaks
    into validation or test.
    """
    keys = list(groups.keys())
    rng  = random.Random(seed)
    rng.shuffle(keys)

    n           = len(keys)
    n_train     = max(1, round(n * train_ratio))
    n_val       = max(1, round(n * val_ratio))
    # test gets whatever is left so rounding never loses a group
    n_test      = n - n_train - n_val
    if n_test < 1:
        # Adjust to ensure at least 1 group in every split
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        n_test = n - n_train - n_val

    train_keys = keys[:n_train]
    val_keys   = keys[n_train : n_train + n_val]
    test_keys  = keys[n_train + n_val :]

    def flatten(ks):
        files = []
        for k in ks:
            files.extend(groups[k])
        return files

    return flatten(train_keys), flatten(val_keys), flatten(test_keys)


def transfer(src: Path, dst: Path, move: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), dst)
    else:
        shutil.copy2(src, dst)


def populate_partition(files: list[Path], partition_dir: Path, move: bool):
    for src in files:
        dst = partition_dir / src.name
        # Handle filename collisions (should be rare)
        if dst.exists():
            dst = partition_dir / (src.stem + "_dup" + src.suffix)
        transfer(src, dst, move)


def print_summary(
    label: str,
    files: list[Path],
    groups: dict[str, list[Path]],
    total_files: int,
    total_groups: int,
):
    pct_files  = len(files)  / total_files  * 100 if total_files  else 0
    # Count how many source groups ended up in this partition
    file_names  = {f.name for f in files}
    groups_here = sum(
        1 for paths in groups.values()
        if any(p.name in file_names for p in paths)
    )
    pct_groups = groups_here / total_groups * 100 if total_groups else 0
    print(
        f"  {label:<10}  {len(files):>5} images ({pct_files:5.1f} %)   "
        f"{groups_here:>3} source groups ({pct_groups:5.1f} %)"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    validate_ratios(TRAIN_RATIO, VALIDATE_RATIO, TEST_RATIO)

    input_dir  = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    # ── Collect images ────────────────────────────────────────────────────────
    if not input_dir.exists():
        print(f"[ERROR] Input directory '{input_dir}' not found.")
        return

    all_images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not all_images:
        print(f"[ERROR] No supported images found in '{input_dir}'.")
        return

    # ── Group by source, split ────────────────────────────────────────────────
    groups                       = group_by_source(all_images)
    train_files, val_files, test_files = split_groups(
        groups, TRAIN_RATIO, VALIDATE_RATIO, RANDOM_SEED
    )

    # ── Create output folders & populate ─────────────────────────────────────
    partitions = {
        "train":    (train_files, output_dir / "train"),
        "validate": (val_files,   output_dir / "validate"),
        "test":     (test_files,  output_dir / "test"),
    }

    action = "Moving" if MOVE_FILES else "Copying"
    print("=" * 62)
    print(" Dataset Splitter")
    print("=" * 62)
    print(f" Input     : {input_dir.resolve()}")
    print(f" Output    : {output_dir.resolve()}")
    print(f" Action    : {action} files")
    print(f" Seed      : {RANDOM_SEED}")
    print(f" Ratios    : train={TRAIN_RATIO}  val={VALIDATE_RATIO}  test={TEST_RATIO}")
    print(f" Total     : {len(all_images)} images across {len(groups)} source groups")
    print("=" * 62)

    for name, (files, folder) in partitions.items():
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n[{name.upper()}] {action} {len(files)} files → {folder}")
        populate_partition(files, folder, MOVE_FILES)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print(" Split Summary")
    print("─" * 62)
    print(f"  {'Partition':<10}  {'Images':>5}            {'Source groups':>3}")
    print("─" * 62)
    for name, (files, _) in partitions.items():
        print_summary(name, files, groups, len(all_images), len(groups))
    print("─" * 62)
    print(f"  {'TOTAL':<10}  {len(all_images):>5} images         {len(groups):>3} source groups")
    print("=" * 62)
    print(f"\n Done! Dataset written to '{output_dir}/'")
    print(
        f"  {output_dir}/train/      ← {len(train_files)} images\n"
        f"  {output_dir}/validate/   ← {len(val_files)} images\n"
        f"  {output_dir}/test/       ← {len(test_files)} images"
    )


if __name__ == "__main__":
    main()
