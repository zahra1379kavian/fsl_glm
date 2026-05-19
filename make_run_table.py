from pathlib import Path
import re

BASE = Path("/home/zkavian/fsl_glm")
DATA = BASE / "data"

# Use go_times_3col if you converted files. Otherwise use go_times.
GO = BASE / "go_times_3col"

OUT = BASE / "run_table.tsv"

pat = re.compile(r"(sub-[^_]+)_(ses-[^_]+)_(run-[^_]+)_task-mv_bold_corrected_smoothed_mnireg-2mm(?:-upsampled)?\.nii\.gz$")

rows = []
missing = []

for bold in sorted(DATA.rglob("*_task-mv_bold_corrected_smoothed_mnireg-2mm*.nii.gz")):
    m = pat.match(bold.name)
    if not m:
        continue

    sub, ses, run = m.groups()
    key = f"{sub}_{ses}_{run}"

    # first try a straightforward substring match
    evs = sorted(GO.rglob(f"*{key}*"))

    # if no matches, try a relaxed match: look for subject digits, ses and run
    if len(evs) == 0:
        sub_digits_m = re.search(r"(\d+)", sub)
        if sub_digits_m:
            sd = sub_digits_m.group(1)
        else:
            sd = None

        ses_token = ses
        run_token = run
        alt_ses = ses.replace("-", "")
        alt_run = run.replace("-", "")

        candidates = []
        if sd:
            for f in GO.rglob("*"):
                name = f.name.lower()
                if sd in name and (ses_token in name or alt_ses in name) and (run_token in name or alt_run in name):
                    candidates.append(f)

        evs = sorted(candidates)

    if len(evs) == 0:
        missing.append(key)
        ev = ""
    else:
        ev = str(evs[0])

    rows.append((sub, ses, run, str(bold), ev))

with OUT.open("w") as out:
    out.write("sub\tses\trun\tbold\tev\n")
    for r in rows:
        out.write("\t".join(r) + "\n")

print(f"Wrote {len(rows)} rows to {OUT}")

if missing:
    print("\nMissing Go timing file for these runs:")
    for x in missing:
        print("  ", x)
else:
    print("All BOLD files have matching Go timing files.")
