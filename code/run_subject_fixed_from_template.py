#!/usr/bin/env python3
"""Create and optionally run subject/session fixed-effects FEAT jobs.

The script discovers first-level FEAT outputs named like:

    sub-pd011_ses-1_run-1.feat
    sub-pd011_ses-1_run-2.feat

For each subject/session with both run 1 and run 2, it writes one higher-level
FSF using an existing fixed-effects design as the template. Existing completed
outputs in feat/group.gfeat are skipped unless --overwrite is used. Partial
outputs are regenerated with overwrite enabled for that job.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


BASE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = BASE / "feat/group.gfeat/sub10-ses2.gfeat/design.fsf"
DEFAULT_FIRSTLEVEL_DIR = BASE / "feat/firstlevel"
DEFAULT_GROUP_DIR = BASE / "feat/group.gfeat"
DEFAULT_FSF_DIR = BASE / "fsf/subject_fixed"
DEFAULT_LOG_DIR = BASE / "logs/subject_fixed"

FEAT_DIR_RE = re.compile(r"sub-pd(?P<sub>\d+)_ses-(?P<ses>\d+)_run-(?P<run>\d+)\.feat$")


@dataclass(frozen=True)
class FixedPair:
    sub: int
    ses: int
    run1: Path
    run2: Path

    @property
    def label(self) -> str:
        return f"sub{self.sub:02d}-ses{self.ses}"

    @property
    def bids_label(self) -> str:
        return f"sub-pd{self.sub:03d}_ses-{self.ses}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally run run-pair fixed-effects FEAT FSFs."
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--firstlevel-dir", type=Path, default=DEFAULT_FIRSTLEVEL_DIR)
    parser.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    parser.add_argument("--fsf-dir", type=Path, default=DEFAULT_FSF_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--feat-cmd", default="feat")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run feat for each generated FSF. Without this, only FSFs are written.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of FEAT jobs to run at the same time when --run is used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate and rerun existing outputs, setting fmri(overwrite_yn)=1.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        help="Only process this subject, e.g. --subject sub-pd011 or --subject 11. Can be repeated.",
    )
    parser.add_argument(
        "--session",
        action="append",
        help="Only process this session, e.g. --session ses-1 or --session 1. Can be repeated.",
    )
    return parser.parse_args()


def numeric_filter(values: list[str] | None, label: str) -> set[int]:
    if not values:
        return set()

    parsed: set[int] = set()
    for value in values:
        match = re.search(r"\d+", value)
        if not match:
            raise ValueError(f"Could not parse {label} filter as a number: {value}")
        parsed.add(int(match.group(0)))
    return parsed


def discover_pairs(firstlevel_dir: Path) -> tuple[list[FixedPair], list[str]]:
    runs: dict[tuple[int, int], dict[int, Path]] = {}
    duplicates: list[str] = []

    for feat_dir in sorted(firstlevel_dir.glob("*.feat")):
        match = FEAT_DIR_RE.match(feat_dir.name)
        if not match:
            continue

        sub = int(match.group("sub"))
        ses = int(match.group("ses"))
        run = int(match.group("run"))
        key = (sub, ses)
        by_run = runs.setdefault(key, {})
        if run in by_run:
            duplicates.append(
                f"Duplicate run {run} for sub-pd{sub:03d} ses-{ses}: {by_run[run]} and {feat_dir}"
            )
        by_run[run] = feat_dir

    pairs: list[FixedPair] = []
    incomplete: list[str] = []
    for (sub, ses), by_run in sorted(runs.items()):
        missing = [str(run) for run in (1, 2) if run not in by_run]
        if missing:
            incomplete.append(
                f"sub-pd{sub:03d}_ses-{ses}: missing run(s) {', '.join(missing)}; found {sorted(by_run)}"
            )
            continue
        pairs.append(FixedPair(sub=sub, ses=ses, run1=by_run[1], run2=by_run[2]))

    return pairs, duplicates + incomplete


def filter_pairs(pairs: list[FixedPair], args: argparse.Namespace) -> list[FixedPair]:
    subjects = numeric_filter(args.subject, "subject")
    sessions = numeric_filter(args.session, "session")

    return [
        pair
        for pair in pairs
        if (not subjects or pair.sub in subjects)
        and (not sessions or pair.ses in sessions)
    ]


def replace_setting(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(set {re.escape(key)}\s+).*$", flags=re.MULTILINE)
    new_text, count = pattern.subn(lambda match: f"{match.group(1)}{value}", text)
    if count != 1:
        raise ValueError(f"Expected exactly one setting for {key}, found {count}")
    return new_text


def validate_pair(pair: FixedPair) -> list[str]:
    errors: list[str] = []
    required = (
        "report.html",
        "stats/cope1.nii.gz",
        "stats/varcope1.nii.gz",
        "reg/example_func2standard.mat",
        "reg/standard.nii.gz",
    )
    for feat_dir in (pair.run1, pair.run2):
        for rel_path in required:
            path = feat_dir / rel_path
            if not path.exists():
                errors.append(f"{pair.bids_label}: missing {path}")
    return errors


def output_dir_for(pair: FixedPair, group_dir: Path) -> Path:
    return group_dir / f"{pair.label}.gfeat"


def make_fsf(template: str, pair: FixedPair, output_dir: Path, overwrite: bool) -> str:
    fsf = template
    replacements = {
        "fmri(outputdir)": f'"{output_dir}"',
        "fmri(level)": "2",
        "fmri(analysis)": "2",
        "fmri(featwatcher_yn)": "0",
        "fmri(npts)": "2",
        "fmri(multiple)": "2",
        "fmri(inputtype)": "1",
        "fmri(stats_yn)": "1",
        "fmri(mixed_yn)": "3",
        "fmri(evs_orig)": "1",
        "fmri(evs_real)": "1",
        "fmri(ncon_orig)": "1",
        "fmri(ncon_real)": "1",
        "fmri(nftests_orig)": "0",
        "fmri(nftests_real)": "0",
        "fmri(poststats_yn)": "0",
        "fmri(ncopeinputs)": "1",
        "fmri(copeinput.1)": "1",
        "feat_files(1)": f'"{pair.run1}"',
        "feat_files(2)": f'"{pair.run2}"',
        "fmri(evg1.1)": "1",
        "fmri(evg2.1)": "1",
        "fmri(groupmem.1)": "1",
        "fmri(groupmem.2)": "1",
        "fmri(con_mode_old)": "real",
        "fmri(con_mode)": "real",
        "fmri(conname_real.1)": '"group mean"',
        "fmri(con_real1.1)": "1",
        "fmri(overwrite_yn)": "1" if overwrite else "0",
    }
    for key, value in replacements.items():
        fsf = replace_setting(fsf, key, value)
    return fsf


def run_feat(fsf: Path, log_path: Path, feat_cmd: str) -> tuple[Path, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        result = subprocess.run(
            [feat_cmd, str(fsf)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return fsf, result.returncode


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    template_path = args.template.resolve()
    firstlevel_dir = args.firstlevel_dir.resolve()
    group_dir = args.group_dir.resolve()
    fsf_dir = args.fsf_dir.resolve()
    log_dir = args.log_dir.resolve()

    if args.run and shutil.which(args.feat_cmd) is None:
        print(f"FEAT command not found: {args.feat_cmd}", file=sys.stderr)
        return 1
    if not template_path.exists():
        print(f"Template FSF not found: {template_path}", file=sys.stderr)
        return 1
    if not firstlevel_dir.exists():
        print(f"First-level directory not found: {firstlevel_dir}", file=sys.stderr)
        return 1

    template = template_path.read_text()
    pairs, discovery_warnings = discover_pairs(firstlevel_dir)
    pairs = filter_pairs(pairs, args)
    if not pairs:
        print("No complete run-1/run-2 subject/session pairs matched the requested filters.", file=sys.stderr)
        return 1

    validation_errors: list[str] = []
    for pair in pairs:
        validation_errors.extend(validate_pair(pair))
    if validation_errors:
        print("Cannot continue because some first-level FEAT inputs are incomplete:", file=sys.stderr)
        for error in validation_errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    fsf_dir.mkdir(parents=True, exist_ok=True)
    group_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fsfs: list[Path] = []
    skipped_existing: list[Path] = []
    partial_existing: list[Path] = []
    for pair in pairs:
        out_dir = output_dir_for(pair, group_dir)
        if out_dir.exists() and not args.overwrite:
            if (out_dir / "report.html").exists():
                skipped_existing.append(out_dir)
                continue
            else:
                partial_existing.append(out_dir)

        pair_overwrite = args.overwrite or out_dir.exists()

        fsf_path = fsf_dir / f"{pair.label}_fixed.fsf"
        fsf_path.write_text(make_fsf(template, pair, out_dir, pair_overwrite))
        fsfs.append(fsf_path)

    print(f"Template: {template_path}")
    print(f"First-level FEAT dir: {firstlevel_dir}")
    print(f"Group output dir: {group_dir}")
    print(f"Complete run pairs matched: {len(pairs)}")
    print(f"Generated FSFs: {len(fsfs)} in {fsf_dir}")
    if skipped_existing:
        print(f"Skipped completed existing outputs: {len(skipped_existing)}")
    if partial_existing:
        print(f"Regenerating partial existing outputs with overwrite enabled: {len(partial_existing)}")
        for path in partial_existing[:20]:
            print(f"  {path}")
        if len(partial_existing) > 20:
            print(f"  ... {len(partial_existing) - 20} more")
    if discovery_warnings:
        print(f"Discovery warnings: {len(discovery_warnings)}")
        for warning in discovery_warnings[:20]:
            print(f"  {warning}")
        if len(discovery_warnings) > 20:
            print(f"  ... {len(discovery_warnings) - 20} more")

    if not args.run:
        print("Dry run only: FEAT was not started. Add --run to launch the jobs.")
        return 0
    if not fsfs:
        print("No FEAT jobs to run.")
        return 0

    print(f"Starting FEAT jobs: {len(fsfs)} with --jobs {args.jobs}")
    failures: list[tuple[Path, int]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_feat, fsf, log_dir / f"{fsf.stem}.log", args.feat_cmd): fsf
            for fsf in fsfs
        }
        for future in as_completed(futures):
            fsf, returncode = future.result()
            if returncode == 0:
                print(f"OK: {fsf.name}")
            else:
                print(f"FAILED ({returncode}): {fsf.name}")
                failures.append((fsf, returncode))

    if failures:
        print("\nFailed FEAT jobs:")
        for fsf, returncode in failures:
            print(f"  {fsf} -> return code {returncode}; log: {log_dir / (fsf.stem + '.log')}")
        return 1

    print("All submitted FEAT jobs finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
