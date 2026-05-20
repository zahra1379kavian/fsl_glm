#!/usr/bin/env python3
"""Create and optionally run the mixed-effects model across sessions."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "temporary/feat/mixed_model.gfeat/design.fsf"
DEFAULT_SESSION_DIR = ROOT / "outputs/feat/session_fixed.gfeat"
DEFAULT_OUTPUT_BASE = ROOT / "outputs/feat/mixed_model"
DEFAULT_FSF_DIR = ROOT / "outputs/fsf/mixed_model"
DEFAULT_LOG_DIR = ROOT / "outputs/logs/mixed_model"
FALLBACK_TEMPLATE = Path(
    "/usr/local/fsl/lib/python3.12/site-packages/fsl/tests/testdata/"
    "test_feat/2ndlevel_1.gfeat/design.fsf"
)

SESSION_RE = re.compile(r"sub(?P<sub>\d+)-ses(?P<ses>\d+)\.gfeat$")


@dataclass(frozen=True)
class SessionInput:
    sub: int
    ses: int
    path: Path

    @property
    def label(self) -> str:
        return f"sub{self.sub:02d}-ses{self.ses}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally run a mixed-effects FEAT across subject/session inputs."
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION_DIR)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--fsf-dir", type=Path, default=DEFAULT_FSF_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--feat-cmd", default="feat")
    parser.add_argument(
        "--exclude-sub",
        type=int,
        nargs="+",
        default=[],
        help="Subject numbers to omit from the mixed-effects model.",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def replace_setting(text: str, key: str, value: str, append_missing: bool = False) -> str:
    pattern = re.compile(rf"^(set {re.escape(key)}\s+).*$", flags=re.MULTILINE)
    new_text, count = pattern.subn(lambda match: f"{match.group(1)}{value}", text)
    if count == 0:
        if append_missing:
            return f"{text.rstrip()}\nset {key} {value}\n"
        raise ValueError(f"Missing setting: {key}")
    return new_text


def strip_input_settings(text: str) -> str:
    patterns = (
        r'^set feat_files\(\d+\)\s+.*\n?',
        r'^set fmri\(evg\d+\.1\)\s+.*\n?',
        r'^set fmri\(groupmem\.\d+\)\s+.*\n?',
    )
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    return f"{text.rstrip()}\n"


def resolve_template(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_TEMPLATE and FALLBACK_TEMPLATE.exists():
        return FALLBACK_TEMPLATE
    return path


def discover_inputs(session_dir: Path) -> tuple[list[SessionInput], list[str]]:
    inputs: list[SessionInput] = []
    warnings: list[str] = []
    for gfeat in sorted(session_dir.glob("sub*-ses*.gfeat")):
        match = SESSION_RE.match(gfeat.name)
        if not match:
            continue
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
        inputs.append(
            SessionInput(
                sub=int(match.group("sub")),
                ses=int(match.group("ses")),
                path=input_feat,
            )
        )
    return inputs, warnings


def make_fsf(template: str, inputs: list[SessionInput], output_base: Path, overwrite: bool) -> str:
    n_inputs = len(inputs)
    fsf = strip_input_settings(template)
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
        "fmri(ncopeinputs)": "1",
        "fmri(copeinput.1)": "1",
        "fmri(con_mode_old)": "real",
        "fmri(con_mode)": "real",
        "fmri(conname_real.1)": '"group mean"',
        "fmri(con_real1.1)": "1",
        "fmri(overwrite_yn)": "1" if overwrite else "0",
    }
    for key, value in replacements.items():
        fsf = replace_setting(fsf, key, value)

    for index, item in enumerate(inputs, start=1):
        fsf = replace_setting(
            fsf,
            f"feat_files({index})",
            f'"{item.path.resolve()}"',
            append_missing=True,
        )
        fsf = replace_setting(fsf, f"fmri(evg{index}.1)", "1", append_missing=True)
        fsf = replace_setting(fsf, f"fmri(groupmem.{index})", "1", append_missing=True)

    return fsf


def write_input_manifest(fsf_dir: Path, inputs: list[SessionInput], excluded_inputs: list[SessionInput]) -> None:
    manifest = fsf_dir / "mixed_model_inputs.tsv"
    with manifest.open("w") as f:
        f.write("status\tsubject\tsession\tpath\n")
        for item in inputs:
            f.write(f"included\tsub{item.sub:02d}\tses{item.ses}\t{item.path.resolve()}\n")
        for item in excluded_inputs:
            f.write(f"excluded\tsub{item.sub:02d}\tses{item.ses}\t{item.path.resolve()}\n")


def run_feat(fsf: Path, log_path: Path, feat_cmd: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        result = subprocess.run(
            [feat_cmd, str(fsf)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return result.returncode


def gfeat_dir(output_base: Path) -> Path:
    text = str(output_base)
    if text.endswith(".gfeat"):
        return output_base
    return output_base.with_name(output_base.name + ".gfeat")


def main() -> int:
    args = parse_args()
    template_path = resolve_template(args.template.resolve())
    session_dir = args.session_dir.resolve()
    output_base = args.output_base.resolve()
    fsf_dir = args.fsf_dir.resolve()
    log_dir = args.log_dir.resolve()

    if args.run and shutil.which(args.feat_cmd) is None:
        print(f"FEAT command not found: {args.feat_cmd}", file=sys.stderr)
        return 1
    if not template_path.exists():
        print(f"Template FSF not found: {template_path}", file=sys.stderr)
        return 1
    if not session_dir.exists():
        print(f"Session fixed-effects directory not found: {session_dir}", file=sys.stderr)
        return 1

    inputs, warnings = discover_inputs(session_dir)
    if not inputs:
        print("No completed subject/session fixed-effects inputs found.", file=sys.stderr)
        return 1

    requested_exclusions = set(args.exclude_sub)
    excluded_inputs = [item for item in inputs if item.sub in requested_exclusions]
    if requested_exclusions:
        available_subjects = {item.sub for item in inputs}
        missing_subjects = sorted(requested_exclusions - available_subjects)
        inputs = [item for item in inputs if item.sub not in requested_exclusions]
        if missing_subjects:
            print(
                "Warning: requested excluded subjects were not found in completed inputs: "
                + ", ".join(f"sub{sub:02d}" for sub in missing_subjects)
            )
        if not inputs:
            print("All completed inputs were excluded; no model generated.", file=sys.stderr)
            return 1

    out_gfeat = gfeat_dir(output_base)
    if out_gfeat.exists() and (out_gfeat / "report.html").exists() and not args.overwrite:
        print(f"Skipped completed existing output: {out_gfeat}")
        return 0

    fsf_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fsf_path = fsf_dir / "mixed_model.fsf"
    fsf_path.write_text(make_fsf(template_path.read_text(), inputs, output_base, args.overwrite or out_gfeat.exists()))
    write_input_manifest(fsf_dir, inputs, excluded_inputs)

    print(f"Template: {template_path}")
    print(f"Session fixed-effects dir: {session_dir}")
    print(f"Mixed-model inputs: {len(inputs)}")
    if requested_exclusions:
        print("Excluded subjects:", ", ".join(f"sub{sub:02d}" for sub in sorted(requested_exclusions)))
        print(f"Excluded session inputs: {len(excluded_inputs)}")
    print(f"Generated FSF: {fsf_path}")
    print(f"Input manifest: {fsf_dir / 'mixed_model_inputs.tsv'}")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings[:20]:
            print(f"  {warning}")
        if len(warnings) > 20:
            print(f"  ... {len(warnings) - 20} more")

    if not args.run:
        print("Dry run only: FEAT was not started. Add --run to launch the job.")
        return 0

    print("Starting mixed-effects FEAT job")
    returncode = run_feat(fsf_path, log_dir / "mixed_model.log", args.feat_cmd)
    if returncode != 0:
        print(f"FAILED ({returncode}): {fsf_path}; log: {log_dir / 'mixed_model.log'}")
        return 1

    print("Mixed-effects FEAT job finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
