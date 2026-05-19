#!/usr/bin/env python3
"""Prepare robust-FLAME and responder-only FSFs from the original mixed model."""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FSF = ROOT / "outputs/feat/mixed_model.gfeat/design.fsf"
REF_IMG = ROOT / "outputs/feat/mixed_model.gfeat/cope1.feat/example_func.nii.gz"
FSF_DIR = ROOT / "outputs/fsf/mixed_model_sensitivity"
QC_DIR = ROOT / "outputs/analysis_qc"

ROBUST_OUTPUT = ROOT / "outputs/feat/mixed_model_robust"
RESPONDER_OUTPUT = ROOT / "outputs/feat/mixed_model_responder_only"

LEFT_M1_MM = (-38, -24, 56)
RIGHT_M1_MM = (38, -24, 56)
ROI_RADIUS_MM = 8


@dataclass(frozen=True)
class InputFeat:
    index: int
    subject: int
    session: int
    path: Path


@dataclass(frozen=True)
class SubjectResponse:
    subject: int
    n_sessions: int
    mean_z_left: float
    mean_z_right: float

    @property
    def responder(self) -> bool:
        return self.mean_z_left > 0 or self.mean_z_right > 0


def run(cmd: list[str], stdin: str | None = None) -> str:
    return subprocess.check_output(cmd, input=stdin, text=True)


def replace_setting(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(set {re.escape(key)}\s+).*$", flags=re.MULTILINE)
    text, count = pattern.subn(lambda m: f"{m.group(1)}{value}", text)
    if count == 0:
        raise ValueError(f"Missing setting: {key}")
    return text


def mm_to_vox(coord: tuple[int, int, int]) -> tuple[int, int, int]:
    out = run(
        ["std2imgcoord", "-img", str(REF_IMG), "-std", str(REF_IMG), "-vox"],
        stdin=f"{coord[0]} {coord[1]} {coord[2]}\n",
    )
    return tuple(int(round(float(v))) for v in out.split()[:3])


def make_sphere(point_name: str, sphere_name: str, coord: tuple[int, int, int]) -> Path:
    vx, vy, vz = mm_to_vox(coord)
    point = QC_DIR / point_name
    sphere = QC_DIR / sphere_name
    subprocess.run(
        [
            "fslmaths",
            str(REF_IMG),
            "-mul",
            "0",
            "-add",
            "1",
            "-roi",
            str(vx),
            "1",
            str(vy),
            "1",
            str(vz),
            "1",
            "0",
            "1",
            str(point),
            "-odt",
            "char",
        ],
        check=True,
    )
    subprocess.run(
        [
            "fslmaths",
            str(point),
            "-kernel",
            "sphere",
            str(ROI_RADIUS_MM),
            "-fmean",
            "-bin",
            str(sphere),
        ],
        check=True,
    )
    return sphere.with_suffix(".nii.gz")


def make_rois() -> tuple[Path, Path, Path]:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    left = make_sphere("m1_left_point", "m1_left_8mm", LEFT_M1_MM)
    right = make_sphere("m1_right_point", "m1_right_8mm", RIGHT_M1_MM)
    bilateral = QC_DIR / "m1_bilateral_8mm"
    subprocess.run(["fslmaths", str(left), "-add", str(right), "-bin", str(bilateral)], check=True)
    return left, right, bilateral.with_suffix(".nii.gz")


def parse_inputs(fsf: str) -> list[InputFeat]:
    multiple = int(re.search(r"^set fmri\(multiple\)\s+(\d+)", fsf, re.MULTILINE).group(1))
    inputs: list[InputFeat] = []
    for match in re.finditer(r'^set feat_files\((\d+)\)\s+"([^"]+)"', fsf, re.MULTILINE):
        index = int(match.group(1))
        if index > multiple:
            continue
        path = Path(match.group(2))
        label = path.parent.name
        label_match = re.search(r"sub(\d+)-ses(\d+)", label)
        if not label_match:
            raise ValueError(f"Could not parse subject/session from {path}")
        inputs.append(
            InputFeat(
                index=index,
                subject=int(label_match.group(1)),
                session=int(label_match.group(2)),
                path=path,
            )
        )
    return inputs


def fsl_mean(img: Path, mask: Path) -> float:
    out = run(["fslstats", str(img), "-k", str(mask), "-M"])
    return float(out.split()[0])


def classify_responders(inputs: list[InputFeat], left_roi: Path, right_roi: Path) -> dict[int, SubjectResponse]:
    by_subject: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for item in inputs:
        zstat = item.path / "stats/zstat1.nii.gz"
        by_subject[item.subject].append((fsl_mean(zstat, left_roi), fsl_mean(zstat, right_roi)))

    responses: dict[int, SubjectResponse] = {}
    for subject, rows in sorted(by_subject.items()):
        n_sessions = len(rows)
        responses[subject] = SubjectResponse(
            subject=subject,
            n_sessions=n_sessions,
            mean_z_left=sum(row[0] for row in rows) / n_sessions,
            mean_z_right=sum(row[1] for row in rows) / n_sessions,
        )
    return responses


def make_robust_fsf(template: str) -> str:
    fsf = replace_setting(template, "fmri(outputdir)", f'"{ROBUST_OUTPUT}"')
    fsf = replace_setting(fsf, "fmri(robust_yn)", "1")
    fsf = replace_setting(fsf, "fmri(overwrite_yn)", "1")
    return fsf


def make_responder_fsf(template: str, included: list[InputFeat]) -> str:
    fsf = replace_setting(template, "fmri(outputdir)", f'"{RESPONDER_OUTPUT}"')
    fsf = replace_setting(fsf, "fmri(npts)", str(len(included)))
    fsf = replace_setting(fsf, "fmri(multiple)", str(len(included)))
    fsf = replace_setting(fsf, "fmri(robust_yn)", "0")
    fsf = replace_setting(fsf, "fmri(overwrite_yn)", "1")

    for new_index, item in enumerate(included, start=1):
        fsf = replace_setting(fsf, f"feat_files({new_index})", f'"{item.path}"')
        fsf = replace_setting(fsf, f"fmri(evg{new_index}.1)", "1")
        fsf = replace_setting(fsf, f"fmri(groupmem.{new_index})", "1")
    return fsf


def write_summary(inputs: list[InputFeat], responses: dict[int, SubjectResponse]) -> None:
    included_subjects = {sub for sub, response in responses.items() if response.responder}
    summary = QC_DIR / "mixed_model_responder_summary.tsv"
    with summary.open("w") as f:
        f.write("subject\tn_sessions\tmean_z_left_m1\tmean_z_right_m1\tresponder\n")
        for subject, response in sorted(responses.items()):
            f.write(
                f"sub{subject:02d}\t{response.n_sessions}\t"
                f"{response.mean_z_left:.6f}\t{response.mean_z_right:.6f}\t"
                f"{int(response.responder)}\n"
            )

    included = QC_DIR / "mixed_model_responder_included_inputs.txt"
    with included.open("w") as f:
        for item in inputs:
            if item.subject in included_subjects:
                f.write(f"sub{item.subject:02d}\tses{item.session}\t{item.path}\n")

    notes = QC_DIR / "mixed_model_sensitivity_notes.txt"
    with notes.open("w") as f:
        f.write("Derivative analyses from outputs/feat/mixed_model.gfeat/design.fsf\n")
        f.write(f"Robust FLAME output base: {ROBUST_OUTPUT}\n")
        f.write(f"Responder-only output base: {RESPONDER_OUTPUT}\n")
        f.write(
            "Responder rule: subject-average mean zstat1 > 0 in either 8 mm sphere "
            "centered at left M1 (-38,-24,56) or right M1 (38,-24,56), computed from "
            "the original session-level inputs. This is circular and should be treated "
            "as descriptive/sensitivity only.\n"
        )


def main() -> int:
    if not SOURCE_FSF.exists():
        raise FileNotFoundError(SOURCE_FSF)
    if not REF_IMG.exists():
        raise FileNotFoundError(REF_IMG)

    template = SOURCE_FSF.read_text()
    inputs = parse_inputs(template)
    left_roi, right_roi, _ = make_rois()
    responses = classify_responders(inputs, left_roi, right_roi)

    responder_subjects = {sub for sub, response in responses.items() if response.responder}
    included = [item for item in inputs if item.subject in responder_subjects]

    FSF_DIR.mkdir(parents=True, exist_ok=True)
    (FSF_DIR / "mixed_model_robust.fsf").write_text(make_robust_fsf(template))
    (FSF_DIR / "mixed_model_responder_only.fsf").write_text(make_responder_fsf(template, included))
    write_summary(inputs, responses)

    print(f"Inputs in original mixed model: {len(inputs)}")
    print(f"Responder subjects: {len(responder_subjects)} / {len(responses)}")
    print("Included responder subjects:", ", ".join(f"sub{sub:02d}" for sub in sorted(responder_subjects)))
    print(f"Included responder session inputs: {len(included)}")
    print(f"FSFs written to: {FSF_DIR}")
    print(f"Responder table: {QC_DIR / 'mixed_model_responder_summary.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
