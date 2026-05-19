#!/usr/bin/env python3
"""Prepare 8-second Go timing files and a corrected run table for rerun."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


BASE = Path("/home/zkavian/fsl_glm")
DATA = BASE / "data"
SOURCE_EV_DIR = BASE / "go_times_3col"
OUT_EV_DIR = BASE / "go_times_3col_8s"
OUT_RUN_TABLE = BASE / "run_table_8s.tsv"

EV_DURATION = "8"

BOLD_RE = re.compile(
    r"sub-pd(?P<sub>\d+)_ses-(?P<ses>\d+)_run-(?P<run>\d+)_"
    r"task-mv_bold_corrected_smoothed_mnireg-2mm(?:-upsampled)?\.nii\.gz$"
)


def rewrite_ev_file(src: Path, dst: Path) -> None:
    rows: list[tuple[str, str, str]] = []
    with src.open() as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Expected 3 columns in {src}: {line!r}")
            rows.append((parts[0], EV_DURATION, parts[2]))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for onset, duration, amp in rows:
            f.write(f"{onset}\t{duration}\t{amp}\n")


def make_8s_evs() -> None:
    if not SOURCE_EV_DIR.exists():
        raise FileNotFoundError(f"Missing EV directory: {SOURCE_EV_DIR}")

    evs = sorted(SOURCE_EV_DIR.glob("*_go.txt"))
    if not evs:
        raise FileNotFoundError(f"No EV files found in {SOURCE_EV_DIR}")

    OUT_EV_DIR.mkdir(parents=True, exist_ok=True)
    for src in evs:
        rewrite_ev_file(src, OUT_EV_DIR / src.name)


def strip_nii_gz(path: Path) -> str:
    text = str(path)
    if text.endswith(".nii.gz"):
        return text[:-7]
    if text.endswith(".nii"):
        return text[:-4]
    return text


def expected_ev(sub: str, ses: str, run: str) -> Path:
    return OUT_EV_DIR / f"PSPD{sub}-ses-{ses}-go-times_run-{run}_go.txt"


def make_run_table() -> list[str]:
    rows: list[dict[str, str]] = []
    errors: list[str] = []

    for bold in sorted(DATA.rglob("*_task-mv_bold_corrected_smoothed_mnireg-2mm*.nii.gz")):
        match = BOLD_RE.match(bold.name)
        if not match:
            continue

        sub = match.group("sub")
        ses = match.group("ses")
        run = match.group("run")
        ev = expected_ev(sub, ses, run)
        label = f"sub-pd{sub}_ses-{ses}_run-{run}"

        if not ev.exists():
            errors.append(f"Missing EV for {label}: {ev}")

        rows.append(
            {
                "sub": f"sub-pd{sub}",
                "ses": f"ses-{ses}",
                "run": f"run-{run}",
                "bold": str(bold),
                "ev": str(ev),
            }
        )

    if not rows:
        errors.append(f"No BOLD files found under {DATA}")

    if errors:
        return errors

    with OUT_RUN_TABLE.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["sub", "ses", "run", "bold", "ev"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    return []


def main() -> int:
    make_8s_evs()
    errors = make_run_table()
    if errors:
        print("Cannot prepare rerun inputs:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    print(f"Wrote 8-second EV files to {OUT_EV_DIR}")
    print(f"Wrote corrected run table to {OUT_RUN_TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
