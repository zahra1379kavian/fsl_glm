#!/usr/bin/env python3
"""Combine session fixed-effect FEAT outputs into subject fixed effects."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "temporary/feat_8s/subject_fixed.gfeat/sub11.gfeat/design.fsf"
DEFAULT_SESSION_DIR = ROOT / "outputs/feat/session_fixed.gfeat"
DEFAULT_SUBJECT_DIR = ROOT / "outputs/feat/subject_fixed.gfeat"
DEFAULT_FSF_DIR = ROOT / "outputs/fsf/subject_fixed"
DEFAULT_LOG_DIR = ROOT / "outputs/logs/subject_fixed"

SESSION_RE = re.compile(r"sub(?P<sub>\d+)-ses(?P<ses>\d+)\.gfeat$")


@dataclass(frozen=True)
class SubjectSessions:
    sub: int
    sessions: tuple[Path, ...]

    @property
    def label(self) -> str:
        return f"sub{self.sub:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and run subject-level fixed-effects FEATs.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION_DIR)
    parser.add_argument("--subject-dir", type=Path, default=DEFAULT_SUBJECT_DIR)
    parser.add_argument("--fsf-dir", type=Path, default=DEFAULT_FSF_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--feat-cmd", default="feat")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--subject", action="append", help="Only process this subject, e.g. 11 or sub11.")
    return parser.parse_args()


def numeric_filter(values: list[str] | None) -> set[int]:
    if not values:
        return set()
    parsed: set[int] = set()
    for value in values:
        match = re.search(r"\d+", value)
        if not match:
            raise ValueError(f"Could not parse subject filter: {value}")
        parsed.add(int(match.group(0)))
    return parsed


def discover_subjects(session_dir: Path) -> tuple[list[SubjectSessions], list[str]]:
    by_subject: dict[int, list[tuple[int, Path]]] = {}
    warnings: list[str] = []

    for gfeat in sorted(session_dir.glob("sub*-ses*.gfeat")):
        match = SESSION_RE.match(gfeat.name)
        if not match:
            continue
        sub = int(match.group("sub"))
        ses = int(match.group("ses"))
        input_feat = gfeat / "cope1.feat"
        required = (
            input_feat / "report.html",
            input_feat / "stats/cope1.nii.gz",
            input_feat / "stats/varcope1.nii.gz",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            warnings.append(f"Skipping incomplete {gfeat}: missing {', '.join(missing)}")
            continue
        by_subject.setdefault(sub, []).append((ses, input_feat))

    subjects = [
        SubjectSessions(sub=sub, sessions=tuple(path for _, path in sorted(sessions)))
        for sub, sessions in sorted(by_subject.items())
    ]
    return subjects, warnings


def replace_setting(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(set {re.escape(key)}\s+).*$", flags=re.MULTILINE)
    new_text, count = pattern.subn(lambda match: f"{match.group(1)}{value}", text)
    if count == 0:
        raise ValueError(f"Missing setting: {key}")
    return new_text


def make_fsf(template: str, subject: SubjectSessions, output_dir: Path, overwrite: bool) -> str:
    n_sessions = len(subject.sessions)
    fsf = template
    replacements = {
        "fmri(outputdir)": f'"{output_dir.resolve()}"',
        "fmri(level)": "2",
        "fmri(analysis)": "2",
        "fmri(featwatcher_yn)": "0",
        "fmri(npts)": str(n_sessions),
        "fmri(multiple)": str(n_sessions),
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
        "fmri(con_mode_old)": "real",
        "fmri(con_mode)": "real",
        "fmri(conname_real.1)": '"subject session mean"',
        "fmri(con_real1.1)": "1",
        "fmri(overwrite_yn)": "1" if overwrite else "0",
    }
    for key, value in replacements.items():
        fsf = replace_setting(fsf, key, value)

    for index, session in enumerate(subject.sessions, start=1):
        fsf = replace_setting(fsf, f"feat_files({index})", f'"{session.resolve()}"')
        fsf = replace_setting(fsf, f"fmri(evg{index}.1)", "1")
        fsf = replace_setting(fsf, f"fmri(groupmem.{index})", "1")

    return fsf


def run_feat(fsf: Path, log_path: Path, feat_cmd: str) -> tuple[Path, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        result = subprocess.run([feat_cmd, str(fsf)], stdout=log, stderr=subprocess.STDOUT, text=True)
    return fsf, result.returncode


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if args.run and shutil.which(args.feat_cmd) is None:
        print(f"FEAT command not found: {args.feat_cmd}", file=sys.stderr)
        return 1
    if not args.template.exists():
        print(f"Template FSF not found: {args.template}", file=sys.stderr)
        return 1

    subjects, warnings = discover_subjects(args.session_dir)
    selected = numeric_filter(args.subject)
    if selected:
        subjects = [subject for subject in subjects if subject.sub in selected]
    if not subjects:
        print("No subject session inputs found.", file=sys.stderr)
        return 1

    args.subject_dir.mkdir(parents=True, exist_ok=True)
    args.fsf_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    template = args.template.read_text()
    fsfs: list[Path] = []
    skipped: list[Path] = []
    for subject in subjects:
        out_dir = args.subject_dir / f"{subject.label}.gfeat"
        if out_dir.exists() and (out_dir / "report.html").exists() and not args.overwrite:
            skipped.append(out_dir)
            continue
        subject_overwrite = args.overwrite or out_dir.exists()
        fsf_path = args.fsf_dir / f"{subject.label}_sessions_fixed.fsf"
        fsf_path.write_text(make_fsf(template, subject, out_dir, subject_overwrite))
        fsfs.append(fsf_path)

    print(f"Session input dir: {args.session_dir}")
    print(f"Subjects discovered: {len(subjects)}")
    print(f"Generated FSFs: {len(fsfs)}")
    if skipped:
        print(f"Skipped completed existing outputs: {len(skipped)}")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings[:20]:
            print(f"  {warning}")

    if not args.run:
        print("Dry run only: FEAT was not started. Add --run to launch jobs.")
        return 0
    if not fsfs:
        print("No FEAT jobs to run.")
        return 0

    failures: list[tuple[Path, int]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_feat, fsf, args.log_dir / f"{fsf.stem}.log", args.feat_cmd): fsf
            for fsf in fsfs
        }
        for future in as_completed(futures):
            fsf, returncode = future.result()
            if returncode == 0:
                print(f"OK: {fsf.name}")
            else:
                print(f"FAILED ({returncode}): {fsf.name}")
                failures.append((fsf, returncode))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
