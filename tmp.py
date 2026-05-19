from pathlib import Path
import re

RAW = Path("/home/zkavian/fsl_glm/go_times")
OUT = Path("/home/zkavian/fsl_glm/go_times_3col")

DURATION = 9
AMPLITUDE = 1

OUT.mkdir(parents=True, exist_ok=True)

def read_numbers_by_line(path):
    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nums = [float(x) for x in line.split()]
            lines.append(nums)
    return lines

def clean_number(x):
    if float(x).is_integer():
        return str(int(x))
    return str(x)

def write_3col(out_file, onsets):
    with open(out_file, "w") as f:
        for onset in onsets:
            f.write(f"{clean_number(onset)}\t{DURATION}\t{AMPLITUDE}\n")

files = sorted([p for p in RAW.rglob("*") if p.is_file()])

if not files:
    raise RuntimeError(f"No files found in {RAW}")

for path in files:
    lines = read_numbers_by_line(path)

    if len(lines) == 2:
        run1 = lines[0]
        run2 = lines[1]
    else:
        all_nums = [x for row in lines for x in row]
        if len(all_nums) == 180:
            run1 = all_nums[:90]
            run2 = all_nums[90:]
        else:
            print(f"SKIPPING {path}")
            print(f"  Found {len(lines)} non-empty lines and {len(all_nums)} total numbers.")
            print("  Expected either 2 lines or 180 total numbers.")
            continue

    # Try to extract subject/session from filename.
    # Expected filename contains something like sub-pd004_ses-1
    m = re.search(r"(sub-[A-Za-z0-9]+)_(ses-[A-Za-z0-9]+)", path.stem)

    if m:
        prefix = f"{m.group(1)}_{m.group(2)}"
    else:
        # Fallback: use the original filename stem
        prefix = path.stem
        print(f"WARNING: Could not find sub/ses pattern in {path.name}; using prefix {prefix}")

    out1 = OUT / f"{prefix}_run-1_go.txt"
    out2 = OUT / f"{prefix}_run-2_go.txt"

    write_3col(out1, run1)
    write_3col(out2, run2)

    print(f"Converted {path.name}")
    print(f"  run-1: {len(run1)} events -> {out1.name}")
    print(f"  run-2: {len(run2)} events -> {out2.name}")