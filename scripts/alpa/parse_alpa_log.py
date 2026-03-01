"""
parse_alpa_log.py

Parses an Alpa pipeshard log file, extracts the best solution cost
(time per batch), computes throughput, and appends to a CSV summary.

Usage:
    python parse_alpa_log.py --log <log_file> --model <model> \
                             --num_devices <N> --mbs <M> \
                             [--out_dir results/parsed_results] \
                             [--out_file result_summary.csv]

The script can also be called directly from the shell script by passing
the same --model / --num_devices args, with --mbs inferred from
NUM_MICROBATCH if not supplied explicitly.

Output CSV columns:
    model, num_devices, mbs, time_per_batch, tput
"""

import os
import re
import csv
import argparse


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_best_cost(log_path: str) -> float | None:
    """
    Scan the log file for a line like:
        Best solution cost: 5611.959...
    Return the float value, or None if not found.
    """
    pattern = re.compile(r"Best solution cost:\s*([\d.eE+\-]+)")
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# CSV append
# ---------------------------------------------------------------------------

FIELDNAMES = ["model", "num_devices", "mbs", "time_per_batch", "tput"]


def append_to_csv(out_path: str, row: dict) -> None:
    """
    Append one row to the CSV. Creates the file with a header if it
    does not exist yet; otherwise just appends without re-writing the header.
    """
    file_exists = os.path.isfile(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse an Alpa pipeshard log and append results to a CSV.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--log", required=True,
                        help="Path to the log file to parse.")
    parser.add_argument("--model", required=True,
                        help="Model name (bert, gpt3, llama2, llama3, mixtral).")
    parser.add_argument("--num_devices", required=True, type=int,
                        help="Total number of devices used in the run.")
    parser.add_argument("--mbs", required=True, type=int,
                        help="Number of microbatches (NUM_MICROBATCH in the shell script).")
    parser.add_argument("--out_dir", default="results/parsed_results",
                        help="Directory for the output CSV (default: results/parsed_results).")
    parser.add_argument("--out_file", default="result_summary.csv",
                        help="Output CSV filename (default: result_summary.csv).")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Parse log ──────────────────────────────────────────────────────────
    if not os.path.isfile(args.log):
        print(f"[ERROR] Log file not found: {args.log}")
        return 1

    time_per_batch = parse_best_cost(args.log)
    if time_per_batch is None:
        print(f"[ERROR] Could not find 'Best solution cost:' in {args.log}")
        return 1

    tput = args.mbs / time_per_batch

    print(f"[INFO] Parsed log      : {args.log}")
    print(f"[INFO] Model           : {args.model}")
    print(f"[INFO] num_devices     : {args.num_devices}")
    print(f"[INFO] mbs             : {args.mbs}")
    print(f"[INFO] time_per_batch  : {time_per_batch:.6f} s")
    print(f"[INFO] throughput      : {tput:.6f} samples/s")

    # ── Write CSV ──────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_file)

    row = {
        "model":          args.model.lower(),
        "num_devices":    args.num_devices,
        "mbs":            args.mbs,
        "time_per_batch": round(time_per_batch, 6),
        "tput":           round(tput, 6),
    }
    append_to_csv(out_path, row)
    print(f"[INFO] Appended to     : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())