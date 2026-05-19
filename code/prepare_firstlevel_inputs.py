#!/usr/bin/env python3
"""Prepare kept-trial EV files and the first-level run table.

The cleaned project stores per-run Go timing files in seconds under
data/go_times_second. The skipped-trial MATLAB files use 0 for kept trials and
1 for rejected trials, despite the variable name `rej_trials`.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOLD_DIR = ROOT / "data/bold_data"
DEFAULT_ONSET_DIR = ROOT / "data/go_times_second"
DEFAULT_TRIAL_DIR = ROOT / "data/skipped_trials"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_EV_DIR = DEFAULT_OUTPUT_DIR / "evs"
DEFAULT_RUN_TABLE = DEFAULT_OUTPUT_DIR / "run_table.tsv"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "trial_selection_summary.tsv"

KEEP_VALUE = 0
EVENT_DURATION = "2"
EVENT_AMPLITUDE = "1"

BOLD_RE = re.compile(
    r"sub-pd(?P<sub>\d+)_ses-(?P<ses>\d+)_run-(?P<run>\d+)_"
    r"task-mv_bold_corrected_smoothed_mnireg-2mm(?:-upsampled)?\.nii\.gz$"
)


@dataclass(frozen=True)
class BoldRun:
    sub: str
    ses: int
    run: int
    path: Path

    @property
    def bids_sub(self) -> str:
        return f"sub-pd{self.sub}"

    @property
    def bids_ses(self) -> str:
        return f"ses-{self.ses}"

    @property
    def bids_run(self) -> str:
        return f"run-{self.run}"

    @property
    def pspd(self) -> str:
        return f"PSPD{self.sub}"

    @property
    def label(self) -> str:
        return f"{self.bids_sub}_{self.bids_ses}_{self.bids_run}"


@dataclass(frozen=True)
class SessionSelection:
    state: str
    trial_file: Path
    kept_by_run: tuple[int, int]
    ambiguous: bool
    matched_counts: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate first-level EV files and outputs/run_table.tsv."
    )
    parser.add_argument("--bold-dir", type=Path, default=DEFAULT_BOLD_DIR)
    parser.add_argument("--onset-dir", type=Path, default=DEFAULT_ONSET_DIR)
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL_DIR)
    parser.add_argument("--ev-dir", type=Path, default=DEFAULT_EV_DIR)
    parser.add_argument("--run-table", type=Path, default=DEFAULT_RUN_TABLE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def discover_bold_runs(bold_dir: Path) -> list[BoldRun]:
    runs: list[BoldRun] = []
    for bold in sorted(bold_dir.rglob("*_task-mv_bold_corrected_smoothed_mnireg-2mm*.nii.gz")):
        match = BOLD_RE.match(bold.name)
        if not match:
            continue
        runs.append(
            BoldRun(
                sub=match.group("sub"),
                ses=int(match.group("ses")),
                run=int(match.group("run")),
                path=bold,
            )
        )
    return runs


def onset_path(onset_dir: Path, run: BoldRun) -> Path:
    return onset_dir / f"{run.pspd}-ses-{run.ses}-run-{run.run}-go-times.txt"


def trial_path(trial_dir: Path, sub: str, state: str) -> Path:
    return trial_dir / f"PSPD{sub}_{state}_rejtrials.mat"


def load_trials(path: Path) -> np.ndarray:
    data = loadmat(path)
    if "rej_trials" not in data:
        raise ValueError(f"{path} does not contain variable 'rej_trials'")
    trials = np.asarray(data["rej_trials"])
    if trials.ndim != 2 or trials.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns; found {trials.shape}")
    return trials[:, :2]


def read_ev_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                float(parts[0])
            except ValueError as exc:
                raise ValueError(f"Invalid onset in {path}:{line_number}: {parts[0]!r}") from exc
            rows.append(parts)
    return rows


def clean_number(text: str) -> str:
    value = float(text)
    if value.is_integer():
        return str(int(value))
    return text


def session_onset_counts(onset_dir: Path, sub: str, ses: int) -> tuple[int | None, int | None]:
    counts: list[int | None] = []
    for run in (1, 2):
        path = onset_dir / f"PSPD{sub}-ses-{ses}-run-{run}-go-times.txt"
        counts.append(len(read_ev_rows(path)) if path.exists() else None)
    return counts[0], counts[1]


def choose_session_selection(
    trial_dir: Path,
    onset_dir: Path,
    sub: str,
    ses: int,
) -> SessionSelection:
    states: dict[str, tuple[Path, np.ndarray]] = {}
    for state in ("OFF", "ON"):
        path = trial_path(trial_dir, sub, state)
        if not path.exists():
            raise FileNotFoundError(f"Missing skipped-trials file: {path}")
        states[state] = (path, load_trials(path))

    onset_counts = session_onset_counts(onset_dir, sub, ses)
    matches: list[str] = []
    kept_counts_by_state: dict[str, tuple[int, int]] = {}
    for state, (_, trials) in states.items():
        kept_counts = tuple(int((trials[:, run_index] == KEEP_VALUE).sum()) for run_index in (0, 1))
        kept_counts_by_state[state] = kept_counts
        if kept_counts == onset_counts:
            matches.append(state)

    preferred = "OFF" if ses == 1 else "ON"
    if preferred in matches:
        selected = preferred
    elif matches:
        selected = matches[0]
    else:
        selected = preferred

    path, _ = states[selected]
    return SessionSelection(
        state=selected,
        trial_file=path,
        kept_by_run=kept_counts_by_state[selected],
        ambiguous=len(matches) > 1,
        matched_counts=selected in matches,
    )


def kept_ev_rows(source_rows: list[list[str]], trials: np.ndarray, run: int, source: Path) -> tuple[list[list[str]], str]:
    keep_mask = trials[:, run - 1] == KEEP_VALUE
    n_kept = int(keep_mask.sum())

    if len(source_rows) == len(keep_mask):
        return [row for row, keep in zip(source_rows, keep_mask) if keep], "filtered_full_source"
    if len(source_rows) == n_kept:
        return source_rows, "source_already_kept_only"

    raise ValueError(
        f"{source} has {len(source_rows)} rows, but skipped-trials column run-{run} "
        f"has {len(keep_mask)} trials and {n_kept} kept trials"
    )


def write_ev(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(f"{clean_number(row[0])}\t{EVENT_DURATION}\t{EVENT_AMPLITUDE}\n")


def main() -> int:
    args = parse_args()
    bold_dir = args.bold_dir.resolve()
    onset_dir = args.onset_dir.resolve()
    trial_dir = args.trial_dir.resolve()
    ev_dir = args.ev_dir.resolve()
    run_table = args.run_table.resolve()
    summary = args.summary.resolve()

    runs = discover_bold_runs(bold_dir)
    errors: list[str] = []
    if not runs:
        errors.append(f"No matching BOLD files found under {bold_dir}")

    selection_cache: dict[tuple[str, int], SessionSelection] = {}
    run_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for run in runs:
        source_ev = onset_path(onset_dir, run)
        if not source_ev.exists():
            errors.append(f"Missing onset file for {run.label}: {source_ev}")
            continue

        try:
            selection = selection_cache.setdefault(
                (run.sub, run.ses),
                choose_session_selection(trial_dir, onset_dir, run.sub, run.ses),
            )
            trials = load_trials(selection.trial_file)
            source_rows = read_ev_rows(source_ev)
            rows, selection_mode = kept_ev_rows(source_rows, trials, run.run, source_ev)
        except Exception as exc:
            errors.append(f"{run.label}: {exc}")
            continue

        out_ev = ev_dir / source_ev.name
        write_ev(out_ev, rows)

        run_rows.append(
            {
                "sub": run.bids_sub,
                "ses": run.bids_ses,
                "run": run.bids_run,
                "bold": str(run.path.resolve()),
                "ev": str(out_ev),
            }
        )
        summary_rows.append(
            {
                "sub": run.bids_sub,
                "ses": run.bids_ses,
                "run": run.bids_run,
                "trial_file": str(selection.trial_file),
                "condition": selection.state,
                "source_ev": str(source_ev.resolve()),
                "output_ev": str(out_ev),
                "source_rows": str(len(source_rows)),
                "kept_rows": str(len(rows)),
                "selection_mode": selection_mode,
                "matched_session_counts": str(int(selection.matched_counts)),
                "ambiguous_condition_match": str(int(selection.ambiguous)),
            }
        )

    if errors:
        print("Cannot prepare first-level inputs:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    run_table.parent.mkdir(parents=True, exist_ok=True)
    with run_table.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["sub", "ses", "run", "bold", "ev"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(run_rows)

    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "sub",
                "ses",
                "run",
                "trial_file",
                "condition",
                "source_ev",
                "output_ev",
                "source_rows",
                "kept_rows",
                "selection_mode",
                "matched_session_counts",
                "ambiguous_condition_match",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    ambiguous = sum(row["ambiguous_condition_match"] == "1" for row in summary_rows)
    already_kept = sum(row["selection_mode"] == "source_already_kept_only" for row in summary_rows)

    print(f"Wrote EV files: {len(summary_rows)} to {ev_dir}")
    print(f"Wrote run table: {run_table}")
    print(f"Wrote trial-selection summary: {summary}")
    print(f"EV files already matched kept-trial counts: {already_kept}")
    if ambiguous:
        print(f"Ambiguous ON/OFF count matches resolved by session default: {ambiguous} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
