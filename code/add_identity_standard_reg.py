#!/usr/bin/env python3
"""Add identity standard-space registration metadata to first-level FEAT dirs.

Use this only when first-level cope images are already in the same standard
space/grid as the chosen reference image.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEAT_ROOT = ROOT / "outputs/feat/firstlevel"
DEFAULT_STANDARD = Path("/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz")

IDENTITY_MAT = """1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
"""


@dataclass(frozen=True)
class ImageGrid:
    dim1: str
    dim2: str
    dim3: str
    pixdim1: float
    pixdim2: float
    pixdim3: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reg/example_func2standard.mat as identity in FEAT dirs."
    )
    parser.add_argument("--feat-root", type=Path, default=DEFAULT_FEAT_ROOT)
    parser.add_argument("--standard", type=Path, default=DEFAULT_STANDARD)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write registration files. Without this, only report actions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing reg/example_func2standard.mat.",
    )
    return parser.parse_args()


def fslval(image: Path, field: str) -> str:
    result = subprocess.run(
        ["fslval", str(image), field],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def grid(image: Path) -> ImageGrid:
    return ImageGrid(
        dim1=fslval(image, "dim1"),
        dim2=fslval(image, "dim2"),
        dim3=fslval(image, "dim3"),
        pixdim1=float(fslval(image, "pixdim1")),
        pixdim2=float(fslval(image, "pixdim2")),
        pixdim3=float(fslval(image, "pixdim3")),
    )


def same_grid(left: ImageGrid, right: ImageGrid) -> bool:
    return (
        left.dim1 == right.dim1
        and left.dim2 == right.dim2
        and left.dim3 == right.dim3
        and abs(left.pixdim1 - right.pixdim1) < 1e-6
        and abs(left.pixdim2 - right.pixdim2) < 1e-6
        and abs(left.pixdim3 - right.pixdim3) < 1e-6
    )


def completed_feat_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("*.feat")
        if (path / "report.html").exists()
        and (path / "stats/cope1.nii.gz").exists()
        and (path / "example_func.nii.gz").exists()
    )


def write_identity_reg(feat_dir: Path, standard: Path, overwrite: bool) -> None:
    reg_dir = feat_dir / "reg"
    reg_dir.mkdir(exist_ok=True)

    for mat_name in ("example_func2standard.mat", "standard2example_func.mat"):
        mat_path = reg_dir / mat_name
        if mat_path.exists() and not overwrite:
            continue
        mat_path.write_text(IDENTITY_MAT)

    shutil.copyfile(standard, reg_dir / "standard.nii.gz")
    shutil.copyfile(feat_dir / "example_func.nii.gz", reg_dir / "example_func2standard.nii.gz")


def main() -> int:
    args = parse_args()
    feat_root = args.feat_root.resolve()
    standard = args.standard.resolve()

    if not standard.exists():
        print(f"Standard image not found: {standard}", file=sys.stderr)
        return 1

    feat_dirs = completed_feat_dirs(feat_root)
    if not feat_dirs:
        print(f"No completed first-level FEAT directories found in {feat_root}", file=sys.stderr)
        return 1

    standard_grid = grid(standard)
    ready: list[Path] = []
    skipped: list[str] = []

    for feat_dir in feat_dirs:
        mat_path = feat_dir / "reg/example_func2standard.mat"
        if mat_path.exists() and not args.overwrite:
            skipped.append(f"{feat_dir.name}: registration already exists")
            continue

        cope = feat_dir / "stats/cope1.nii.gz"
        cope_grid = grid(cope)
        if not same_grid(cope_grid, standard_grid):
            skipped.append(f"{feat_dir.name}: cope1 grid does not match standard")
            continue

        ready.append(feat_dir)

    print(f"Completed FEAT dirs checked: {len(feat_dirs)}")
    print(f"Ready for identity registration: {len(ready)}")
    if skipped:
        print(f"Skipped: {len(skipped)}")
        for item in skipped[:20]:
            print(f"  {item}")
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more")

    if not args.apply:
        print("Dry run only: no files were changed. Add --apply to write reg files.")
        return 0

    for feat_dir in ready:
        write_identity_reg(feat_dir, standard, args.overwrite)
        print(f"Wrote identity reg: {feat_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
