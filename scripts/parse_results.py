#!/usr/bin/env python3
"""
Parse Nest output log and append results to a per-model CSV file.

Normal usage (called per run):
    python3 parse_results.py --log_file llama2_mbs1_output_64.log \
                             --num_accelerators 64 \
                             --runtime 142 \
                             --csv_file out/llama2_mbs1/llama2_mbs1.csv

Summary usage (called once at the end of the bash script):
    python3 parse_results.py --summary \
                             --models bert llama2 llama3 gpt3 mixtral \
                             --mbs 1 \
                             --out_dir out/
"""

import argparse
import csv
import os
import re

# Canonical column order for throughput baselines
BASELINE_COLS = ["Nest", "Manual", "Phaze", "MCMC"]


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_log(log_file):
    """
    Parse a Nest output log and return (nest_throughput, baselines, error).
    All missing values are returned as None (not empty string).
    """
    if not os.path.exists(log_file):
        print(f"WARNING: log file not found: {log_file}")
        return None, {}, "log file not found"

    with open(log_file, "r") as f:
        content = f.read()

    # --- Nest throughput ---
    throughput_match = re.search(r"throughput=([\d.e+\-]+)", content)
    nest_throughput = float(throughput_match.group(1)) if throughput_match else None

    # --- Baseline throughputs from fixed_strategy_throughput dict ---
    baselines = {}
    fixed_match = re.search(r"fixed_strategy_throughput=\{([^}]+)\}", content)
    if fixed_match:
        pairs = re.findall(r"'([^']+)':\s*([\d.e+\-]+)", fixed_match.group(1))
        for name, value in pairs:
            baselines[name] = float(value)

    if nest_throughput is None and not baselines:
        return None, {}, "no throughput data found"

    return nest_throughput, baselines, None


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_csv(csv_file):
    """Load an existing CSV into {num_accelerators: row_dict}."""
    rows = {}
    fieldnames = []
    if os.path.exists(csv_file):
        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                rows[int(row["num_accelerators"])] = dict(row)
    return rows, fieldnames


def build_fieldnames(extra_baselines=None):
    """Return the canonical fieldnames for a model CSV."""
    cols = ["num_accelerators"] + BASELINE_COLS
    if extra_baselines:
        for b in extra_baselines:
            if b not in cols:
                cols.append(b)
    cols.append("runtime_s")
    return cols


def write_csv(csv_file, rows, fieldnames):
    """Write rows sorted by num_accelerators."""
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for acc in sorted(rows.keys()):
            row = rows[acc]
            # Fill any missing column with None (written as 'None')
            for fn in fieldnames:
                if fn not in row or row[fn] == "":
                    row[fn] = "None"
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Per-run update
# ---------------------------------------------------------------------------

def update_csv(csv_file, num_accelerators, nest_throughput, baselines, runtime):
    rows, existing_fieldnames = load_csv(csv_file)

    # Determine all baseline columns seen so far
    all_baselines = list(BASELINE_COLS)
    for b in baselines:
        if b not in all_baselines:
            all_baselines.append(b)
    for fn in existing_fieldnames:
        if fn not in ("num_accelerators", "runtime_s") and fn not in all_baselines:
            all_baselines.append(fn)

    fieldnames = build_fieldnames(
        [b for b in all_baselines if b not in BASELINE_COLS]
    )

    new_row = {
        "num_accelerators": num_accelerators,
        "Nest": nest_throughput if nest_throughput is not None else "None",
        "runtime_s": runtime if runtime is not None else "None",
    }
    for col in BASELINE_COLS[1:]:  # Manual, Phaze, MCMC
        new_row[col] = baselines.get(col, "None")
    for b, v in baselines.items():
        if b not in new_row:
            new_row[b] = v

    rows[num_accelerators] = new_row
    write_csv(csv_file, rows, fieldnames)

    print(f"Saved results for num_accelerators={num_accelerators} to {csv_file}")
    print(f"  Nest:    {new_row['Nest']}")
    for col in BASELINE_COLS[1:]:
        print(f"  {col:8s}: {new_row[col]}")
    print(f"  Runtime: {runtime}s")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def safe_div(a, b):
    """Return a/b as a formatted string, or 'None' if either value is missing."""
    try:
        fa, fb = float(a), float(b)
        if fb == 0:
            return "None"
        return f"{fa / fb:.3f}x"
    except (TypeError, ValueError):
        return "None"


def format_runtime(seconds):
    try:
        s = int(seconds)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h:
            return f"{h}h {m}m {sec}s"
        if m:
            return f"{m}m {sec}s"
        return f"{sec}s"
    except (TypeError, ValueError):
        return "None"


def print_summary(models, mbs, out_dir):
    """
    Print a per-model summary table:
      Model | num_devices | Manual | MCMC | Phaze | NEST| Total Runtime
    """

    # Column widths
    COL_MODEL   = 10
    COL_DEVICES =  12
    COL_SPEEDUP =  20
    COL_RUNTIME =  15

    header = (
        f"{'Model':<{COL_MODEL}} "
        f"{'num_devices':<{COL_DEVICES}} "
        f"{'Manual':<{COL_SPEEDUP}} "
        f"{'MCMC':<{COL_SPEEDUP}} "
        f"{'Phaze':<{COL_SPEEDUP}} "
        f"{'NEST':<{COL_SPEEDUP}} "
        f"{'Runtime':<{COL_RUNTIME}}"
    )
    separator = "*" * len(header)

    print(separator)
    print(header)
    print(separator)

    for model in models:
        csv_file = os.path.join(out_dir, f"{model}_mbs{mbs}", f"{model}_mbs{mbs}.csv")
        if not os.path.exists(csv_file):
            print(f"  [WARNING] CSV not found for {model}: {csv_file}")
            continue

        rows, _ = load_csv(csv_file)
        first_row = True

        for acc in sorted(rows.keys()):
            row = rows[acc]
            nest    = row.get("Nest",      "None")
            manual  = row.get("Manual",    "None")
            mcmc    = row.get("MCMC",      "None")
            phaze   = row.get("Phaze",     "None")
            runtime = row.get("runtime_s", "None")

            speedup_manual = safe_div(manual, manual)
            speedup_mcmc   = safe_div(mcmc, manual)
            speedup_phaze  = safe_div(phaze, manual)
            speedup_nest   = safe_div(nest, manual)
            runtime_str    = format_runtime(runtime) if runtime != "None" else "None"

            model_col = model if first_row else ""
            first_row = False

            print(
                f"{model_col:<{COL_MODEL}} "
                f"{acc:<{COL_DEVICES}} "
                f"{speedup_manual:<{COL_SPEEDUP}} "
                f"{speedup_mcmc:<{COL_SPEEDUP}} "
                f"{speedup_phaze:<{COL_SPEEDUP}} "
                f"{speedup_nest:<{COL_SPEEDUP}} "
                f"{runtime_str:<{COL_RUNTIME}}"
            )

        print(separator)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # Per-run mode
    parser.add_argument("--log_file",         help="Path to Nest output log file")
    parser.add_argument("--num_accelerators", type=int, help="Number of accelerators for this run")
    parser.add_argument("--runtime",          type=int, default=None, help="Wall-clock runtime in seconds")
    parser.add_argument("--csv_file",         help="Output CSV file path")

    # Summary mode
    parser.add_argument("--summary",  action="store_true", help="Print summary table instead of parsing a log")
    parser.add_argument("--models",   nargs="+", help="List of models for summary")
    parser.add_argument("--mbs",      type=int,  default=1, help="Micro-batch size (used to locate CSVs)")
    parser.add_argument("--out_dir",  help="Base output directory (parent of model subdirs)")

    args = parser.parse_args()

    if args.summary:
        if not args.models or not args.out_dir:
            parser.error("--summary requires --models and --out_dir")
        print_summary(args.models, args.mbs, args.out_dir)
        return

    # Per-run mode
    if not args.log_file or args.num_accelerators is None or not args.csv_file:
        parser.error("Per-run mode requires --log_file, --num_accelerators, and --csv_file")

    nest_throughput, baselines, err = parse_log(args.log_file)
    if err and nest_throughput is None:
        print(f"WARNING: {err}. Writing None values to CSV.")

    update_csv(args.csv_file, args.num_accelerators, nest_throughput, baselines, args.runtime)


if __name__ == "__main__":
    main()