#!/usr/bin/env python3
"""Create PSC-scaled subject FEAT inputs and a final group FSF."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = BASE / "feat/mixed_model.gfeat/design.fsf"
DEFAULT_SUBJECT_DIR = BASE / "feat_8s/subject_fixed.gfeat"
DEFAULT_PSC_DIR = BASE / "feat_8s/subject_fixed_psc"
DEFAULT_FSF_DIR = BASE / "fsf_8s/final_group"
DEFAULT_OUTPUT_BASE = BASE / "feat_8s/group_excl_leftdom_psc"
DEFAULT_EXCLUDE = {10, 20, 28}

SUBJECT_RE = re.compile(r"sub(?P<sub>\d+)\.gfeat$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PSC-scaled final group inputs.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--subject-dir", type=Path, default=DEFAULT_SUBJECT_DIR)
    parser.add_argument("--psc-dir", type=Path, default=DEFAULT_PSC_DIR)
    parser.add_argument("--fsf-dir", type=Path, default=DEFAULT_FSF_DIR)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--exclude", action="append", help="Subject to exclude, e.g. 10 or sub10.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_subject(value: str) -> int:
    match = re.search(r"\d+", value)
    if not match:
        raise ValueError(f"Could not parse subject number: {value}")
    return int(match.group(0))


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def subject_inputs(subject_dir: Path) -> list[tuple[int, Path]]:
    subjects: list[tuple[int, Path]] = []
    for gfeat in sorted(subject_dir.glob("sub*.gfeat")):
        match = SUBJECT_RE.match(gfeat.name)
        if not match:
            continue
        input_feat = gfeat / "cope1.feat"
        required = (
            input_feat / "stats/cope1.nii.gz",
            input_feat / "stats/varcope1.nii.gz",
            input_feat / "mean_func.nii.gz",
            input_feat / "mask.nii.gz",
        )
        if all(path.exists() for path in required):
            subjects.append((int(match.group("sub")), input_feat))
    return subjects


def make_psc_copy(subject: int, src: Path, dst_root: Path, overwrite: bool) -> Path:
    dst = dst_root / f"sub{subject:02d}_psc.feat"
    if dst.exists():
        if not overwrite:
            return dst
        shutil.rmtree(dst)

    shutil.copytree(src, dst, symlinks=True)

    scale = dst / "psc_scale.nii.gz"
    run(["fslmaths", str(dst / "mean_func.nii.gz"), "-recip", "-mul", "100", "-mas", str(dst / "mask.nii.gz"), str(scale)])
    run([
        "fslmaths",
        str(dst / "stats/cope1.nii.gz"),
        "-mul",
        str(scale),
        "-mas",
        str(dst / "mask.nii.gz"),
        str(dst / "stats/cope1.nii.gz"),
    ])
    run([
        "fslmaths",
        str(dst / "stats/varcope1.nii.gz"),
        "-mul",
        str(scale),
        "-mul",
        str(scale),
        "-mas",
        str(dst / "mask.nii.gz"),
        str(dst / "stats/varcope1.nii.gz"),
    ])
    return dst


def replace_setting(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(set {re.escape(key)}\s+).*$", flags=re.MULTILINE)
    new_text, count = pattern.subn(lambda match: f"{match.group(1)}{value}", text)
    if count == 0:
        raise ValueError(f"Missing setting: {key}")
    return new_text


def make_group_fsf(template: str, inputs: list[Path], output_base: Path) -> str:
    n_inputs = len(inputs)
    fsf = template
    replacements = {
        "fmri(outputdir)": f'"{output_base.resolve()}"',
        "fmri(level)": "2",
        "fmri(analysis)": "2",
        "fmri(featwatcher_yn)": "0",
        "fmri(npts)": str(n_inputs),
        "fmri(multiple)": str(n_inputs),
        "fmri(inputtype)": "1",
        "fmri(stats_yn)": "1",
        "fmri(mixed_yn)": "2",
        "fmri(evs_orig)": "1",
        "fmri(evs_real)": "1",
        "fmri(ncon_orig)": "1",
        "fmri(ncon_real)": "1",
        "fmri(nftests_orig)": "0",
        "fmri(nftests_real)": "0",
        "fmri(poststats_yn)": "0",
        "fmri(ncopeinputs)": "1",
        "fmri(copeinput.1)": "1",
        "fmri(con_mode_old)": "real",
        "fmri(con_mode)": "real",
        "fmri(conname_real.1)": '"group mean PSC"',
        "fmri(con_real1.1)": "1",
        "fmri(overwrite_yn)": "1",
    }
    for key, value in replacements.items():
        fsf = replace_setting(fsf, key, value)

    for index, input_feat in enumerate(inputs, start=1):
        fsf = replace_setting(fsf, f"feat_files({index})", f'"{input_feat.resolve()}"')
        fsf = replace_setting(fsf, f"fmri(evg{index}.1)", "1")
        fsf = replace_setting(fsf, f"fmri(groupmem.{index})", "1")

    return fsf


def main() -> int:
    args = parse_args()
    if not args.template.exists():
        print(f"Template FSF not found: {args.template}", file=sys.stderr)
        return 1

    excluded = set(DEFAULT_EXCLUDE)
    if args.exclude:
        excluded = {parse_subject(value) for value in args.exclude}

    subjects = [(sub, feat) for sub, feat in subject_inputs(args.subject_dir) if sub not in excluded]
    if not subjects:
        print("No subject inputs found after exclusions.", file=sys.stderr)
        return 1

    args.psc_dir.mkdir(parents=True, exist_ok=True)
    args.fsf_dir.mkdir(parents=True, exist_ok=True)

    psc_inputs = [make_psc_copy(sub, feat, args.psc_dir, args.overwrite) for sub, feat in subjects]
    fsf = make_group_fsf(args.template.read_text(), psc_inputs, args.output_base)
    fsf_path = args.fsf_dir / "group_excl_leftdom_psc.fsf"
    fsf_path.write_text(fsf)

    included = ", ".join(f"sub{sub:02d}" for sub, _ in subjects)
    print(f"Included subjects ({len(subjects)}): {included}")
    print(f"Excluded subjects: {', '.join(f'sub{sub:02d}' for sub in sorted(excluded))}")
    print(f"PSC inputs: {args.psc_dir}")
    print(f"Final group FSF: {fsf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
