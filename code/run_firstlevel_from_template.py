#!/usr/bin/env python3
"""Create and optionally run first-level FEAT jobs from one template design.fsf.

Default behavior writes one run-specific FSF per row in outputs/run_table.tsv
and does not start FEAT. Add --run when the generated FSFs look correct.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "temporary/feat/firstlevel.feat/design.fsf"
DEFAULT_RUN_TABLE = ROOT / "outputs/run_table.tsv"
DEFAULT_FSF_DIR = ROOT / "outputs/fsf/firstlevel"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/feat/firstlevel"
DEFAULT_LOG_DIR = ROOT / "outputs/logs/firstlevel"


@dataclass(frozen=True)
class Run:
    sub: str
    ses: str
    run: str
    bold: Path
    ev: Path

    @property
    def label(self) -> str:
        return f"{self.sub}_{self.ses}_{self.run}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally run first-level FEAT FSFs."
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--run-table", type=Path, default=DEFAULT_RUN_TABLE)
    parser.add_argument("--fsf-dir", type=Path, default=DEFAULT_FSF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        help="Allow FEAT to overwrite existing output directories.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        help="Only process this subject, e.g. --subject sub-pd001. Can be repeated.",
    )
    parser.add_argument(
        "--session",
        action="append",
        help="Only process this session, e.g. --session ses-1. Can be repeated.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        help="Only process this run, e.g. --run-id run-1. Can be repeated.",
    )
    return parser.parse_args()


def read_run_table(path: Path) -> list[Run]:
    if not path.exists():
        raise FileNotFoundError(f"Run table not found: {path}")

    runs: list[Run] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"sub", "ses", "run", "bold", "ev"}
        if set(reader.fieldnames or []) < required:
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")

        for row in reader:
            runs.append(
                Run(
                    sub=row["sub"],
                    ses=row["ses"],
                    run=row["run"],
                    bold=Path(row["bold"]),
                    ev=Path(row["ev"]),
                )
            )

    return runs


def filter_runs(runs: list[Run], args: argparse.Namespace) -> list[Run]:
    subjects = set(args.subject or [])
    sessions = set(args.session or [])
    run_ids = set(args.run_id or [])

    return [
        run
        for run in runs
        if (not subjects or run.sub in subjects)
        and (not sessions or run.ses in sessions)
        and (not run_ids or run.run in run_ids)
    ]


def strip_nii_gz(path: Path) -> str:
    text = str(path)
    if text.endswith(".nii.gz"):
        return text[:-7]
    if text.endswith(".nii"):
        return text[:-4]
    return text


def fslnvols(path: Path) -> int:
    result = subprocess.run(
        ["fslnvols", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return int(result.stdout.strip())


def replace_setting(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf'^(set {re.escape(key)}\s+).*$',
                         flags=re.MULTILINE)
    new_text, count = pattern.subn(lambda match: f"{match.group(1)}{value}", text)
    if count != 1:
        raise ValueError(f"Expected exactly one setting for {key}, found {count}")
    return new_text


def resolve_template(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_TEMPLATE:
        generated = sorted(DEFAULT_FSF_DIR.glob("*_level1.fsf"))
        if generated:
            return generated[0]
    return path


def make_fsf(template: str, run: Run, output_dir: Path, overwrite: bool) -> str:
    npts = fslnvols(run.bold)
    out_base = output_dir / run.label

    fsf = template
    fsf = replace_setting(fsf, "fmri(outputdir)", f'"{out_base}"')
    fsf = replace_setting(fsf, "fmri(npts)", str(npts))
    fsf = replace_setting(fsf, "feat_files(1)", f'"{strip_nii_gz(run.bold)}"')
    fsf = replace_setting(fsf, "fmri(shape1)", "3")
    fsf = replace_setting(fsf, "fmri(convolve1)", "3")
    fsf = replace_setting(fsf, "fmri(custom1)", f'"{run.ev}"')
    fsf = replace_setting(fsf, "fmri(featwatcher_yn)", "0")
    fsf = replace_setting(fsf, "fmri(overwrite_yn)", "1" if overwrite else "0")
    return fsf


def validate_runs(runs: list[Run]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for run in runs:
        if run.label in seen:
            errors.append(f"Duplicate run label: {run.label}")
        seen.add(run.label)
        if not run.bold.exists():
            errors.append(f"Missing BOLD for {run.label}: {run.bold}")
        if not run.ev.exists():
            errors.append(f"Missing EV for {run.label}: {run.ev}")
    return errors


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

    template_path = resolve_template(args.template.resolve())
    run_table_path = args.run_table.resolve()
    fsf_dir = args.fsf_dir.resolve()
    output_dir = args.output_dir.resolve()
    log_dir = args.log_dir.resolve()

    if not template_path.exists():
        print(f"Template FSF not found: {template_path}", file=sys.stderr)
        return 1

    template = template_path.read_text()
    runs = filter_runs(read_run_table(run_table_path), args)
    if not runs:
        print("No runs matched the requested filters.", file=sys.stderr)
        return 1

    errors = validate_runs(runs)
    if errors:
        print("Cannot continue because some inputs are missing:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    fsf_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fsfs: list[Path] = []
    skipped_existing: list[Run] = []
    for run in runs:
        completed_report = output_dir / f"{run.label}.feat" / "report.html"
        if completed_report.exists() and not args.overwrite:
            skipped_existing.append(run)
            continue

        fsf_path = fsf_dir / f"{run.label}_level1.fsf"
        fsf_path.write_text(make_fsf(template, run, output_dir, args.overwrite))
        fsfs.append(fsf_path)

    print(f"Template: {template_path}")
    print(f"Run table: {run_table_path}")
    print(f"Generated FSFs: {len(fsfs)} in {fsf_dir}")
    if skipped_existing:
        print(f"Skipped existing completed outputs: {len(skipped_existing)}")

    if not args.run:
        print("Dry run only: FEAT was not started. Add --run to launch the jobs.")
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
