"""
plot_results.py

Usage examples:
    python plot_results.py --model bert --num_devices 8 64 128 --mbs 32 --setup_name my_setup
    python plot_results.py --model gpt3 --mbs 8 --setup_name my_setup
    python plot_results.py --model bert --num_devices 8 --mbs 2 4 8 --setup_name my_setup --plot_mbs

Default mode:  y = tput / MANUAL_TPUT[model]
--plot_mbs:    y = tput / REFERENCE_TPUT[model]
"""

import os
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANUAL_TPUT = {
    "bert":    1.239425674,
    "gpt3":    0.00061296,
    "llama2":  0.003286446,
    "llama3":  0.000228341,
    "mixtral": 0.001815775,
}

REFERENCE_TPUT = {
    "bert":   4.76487369435058,
    "llama2": 0.012634556,
    "llama3": 0.000659740461264875,
}

BAR_COLORS = {
    "Manual": "darkblue",
    "MCMC":   "orange",
    "Alpa-E": "darkgreen",
    "Phaze":  "gold",
    "Nest":   "lightblue",
}

BAR_ORDER = ["Manual", "MCMC", "Alpa-E", "Phaze", "Nest"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return f if f > 0 else None
    except (ValueError, TypeError):
        return None


def safe_div(numerator, denominator):
    n = safe_float(numerator)
    d = safe_float(denominator)
    if n is None or d is None:
        return None
    return n / d


def format_runtime(seconds):
    if seconds is None:
        return "None"
    try:
        s = float(seconds)
    except (ValueError, TypeError):
        return "None"
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h}h {m}m {sec:.1f}s"


def _placeholder_bar(ax, x, bar_width, ref_height, color):
    """Dashed outline + bold X. No text label."""
    h = max(ref_height * 0.15, 1e-9)
    ax.bar(x, h, width=bar_width, color="none", edgecolor=color,
           linewidth=1.4, linestyle="--", zorder=3)
    ax.text(x, h / 2, "X", ha="center", va="center",
            fontsize=13, color=color, alpha=0.75, zorder=4, fontweight="bold")


def _ref_height(values):
    pos = [v for v in values if v is not None and v > 0]
    return max(pos) if pos else 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_setup_csv(setup_name, model, mbs):
    path = os.path.join(setup_name, "out", f"{model}_mbs{mbs}", f"{model}_mbs{mbs}.csv")
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return {}
    results = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                num_acc = int(float(row.get("num_accelerators",
                                           row.get("num_devices", 0))))
            except (ValueError, TypeError):
                continue
            results[num_acc] = row
    return results


def load_alpa_csv(alpa_csv="alpa/results/parsed_results/result_summary_reference.csv"):
    if not os.path.exists(alpa_csv):
        print(f"[WARNING] Alpa CSV not found: {alpa_csv}")
        return {}
    results = {}
    with open(alpa_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        model_col = next((c for c in fieldnames if "model" in c.lower()), None)
        dev_col   = next((c for c in fieldnames
                          if "num_acc" in c.lower() or "num_dev" in c.lower()), None)
        mbs_col   = next((c for c in fieldnames if "mbs" in c.lower()
                          or "batch" in c.lower()), None)
        tput_col  = next((c for c in fieldnames
                          if "tput" in c.lower() or "throughput" in c.lower()), None)
        if not (model_col and dev_col and tput_col):
            print(f"[WARNING] Could not identify required columns in alpa CSV. Found: {fieldnames}")
            return {}
        if not mbs_col:
            print(f"[WARNING] No mbs column found in alpa CSV ({fieldnames}); "
                  "alpa data will be keyed without mbs.")
        for row in reader:
            mdl = row[model_col].strip().lower()
            try:
                nd = int(float(row[dev_col]))
            except (ValueError, TypeError):
                continue
            mbs_val = None
            if mbs_col:
                try:
                    mbs_val = int(float(row[mbs_col]))
                except (ValueError, TypeError):
                    pass
            tput = safe_float(row.get(tput_col))
            results[(mdl, nd, mbs_val)] = tput
    return results


def get_alpa_tput(alpa_data, model, num_devices, mbs):
    key_with_mbs = (model.lower(), num_devices, mbs)
    key_no_mbs   = (model.lower(), num_devices, None)
    return alpa_data.get(key_with_mbs, alpa_data.get(key_no_mbs))


def get_tputs_for_device(model, num_devices, mbs, setup_data, alpa_data):
    row = setup_data.get(num_devices, {})
    return {
        "Manual": safe_float(row.get("Manual")),
        "MCMC":   safe_float(row.get("MCMC")),
        "Alpa-E": get_alpa_tput(alpa_data, model, num_devices, mbs),
        "Phaze":  safe_float(row.get("Phaze")),
        "Nest":   safe_float(row.get("Nest")),
    }


def get_improvements(tputs, ref):
    return {b: safe_div(tputs[b], ref) for b in BAR_ORDER}


# ---------------------------------------------------------------------------
# Plot: x = num_devices, y = tput / MANUAL_TPUT  (default mode)
# ---------------------------------------------------------------------------

def plot_multi_device(model, mbs, setup_name, alpa_data,
                      num_devices_filter=None, out_dir="."):
    setup_data = load_setup_csv(setup_name, model, mbs)
    if not setup_data:
        print(f"[WARNING] No data for {model} mbs={mbs}")
        return None

    all_devices = sorted(setup_data.keys())
    if num_devices_filter:
        devices = [d for d in num_devices_filter if d in setup_data]
        missing = [d for d in num_devices_filter if d not in setup_data]
        if missing:
            print(f"[WARNING] num_devices {missing} not found in CSV for {model} mbs={mbs}")
        if not devices:
            print("[ERROR] None of the requested num_devices found in CSV.")
            return None
    else:
        devices = all_devices

    ref = MANUAL_TPUT.get(model.lower(), 1.0)

    n_dev  = len(devices)
    n_base = len(BAR_ORDER)
    group_width = 0.8
    bar_width = group_width / n_base

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * n_dev), 5))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f8f8f8")
    x_positions = np.arange(n_dev)

    all_vals = []
    for nd in devices:
        impr = get_improvements(get_tputs_for_device(model, nd, mbs, setup_data, alpa_data), ref)
        all_vals.extend(impr.values())
    ref_h = _ref_height(all_vals)

    for bi, baseline in enumerate(BAR_ORDER):
        offset = (bi - n_base / 2 + 0.5) * bar_width
        color  = BAR_COLORS[baseline]
        for di, nd in enumerate(devices):
            val = get_improvements(
                get_tputs_for_device(model, nd, mbs, setup_data, alpa_data), ref)[baseline]
            x = x_positions[di] + offset
            if val is not None:
                ax.bar(x, val, width=bar_width, color=color,
                       edgecolor="white", linewidth=0.5, zorder=3)
            else:
                _placeholder_bar(ax, x, bar_width, ref_h, color)

    ax.axhline(1.0, color="#555", linewidth=1.0, linestyle=":", zorder=2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(d) for d in devices], fontsize=13)
    ax.set_xlabel("Number of Devices", fontsize=14)
    ax.set_ylabel("Throughput Improvement", fontsize=14)
    ax.set_title(f"{model.upper()}  |  mbs={mbs}", fontsize=15,
                 fontweight="bold", pad=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    patches = [mpatches.Patch(color=BAR_COLORS[b], label=b) for b in BAR_ORDER]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.01, 1),
              borderaxespad=0, fontsize=12, framealpha=0.6)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    dev_tag  = "_".join(str(d) for d in devices)
    fname    = f"{model}_mbs{mbs}_devices{dev_tag}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Plot: x = baselines, sub-bars = mbs, y = tput / REFERENCE_TPUT  (--plot_mbs)
# ---------------------------------------------------------------------------

def plot_mbs_mode(model, mbs_list, setup_name, alpa_data,
                  num_devices, out_dir="."):
    MBS_PALETTE = ["darkblue", "orange", "darkgreen", "gold",
                   "mediumpurple", "crimson", "teal", "sienna"]
    mbs_colors = {m: MBS_PALETTE[i % len(MBS_PALETTE)]
                  for i, m in enumerate(mbs_list)}

    ref = REFERENCE_TPUT.get(model.lower(), 1.0)

    improvements = {}
    for mbs in mbs_list:
        setup_data = load_setup_csv(setup_name, model, mbs)
        tputs = get_tputs_for_device(model, num_devices, mbs, setup_data, alpa_data)
        improvements[mbs] = get_improvements(tputs, ref)

    n_base      = len(BAR_ORDER)
    n_mbs       = len(mbs_list)
    group_width = 0.8
    bar_width   = group_width / n_mbs

    fig, ax = plt.subplots(figsize=(max(9, 1.8 * n_base), 5))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f8f8f8")
    x_positions = np.arange(n_base)

    all_vals = [v for m in mbs_list for v in improvements[m].values()]
    ref_h    = _ref_height(all_vals)

    for mi, mbs in enumerate(mbs_list):
        offset = (mi - n_mbs / 2 + 0.5) * bar_width
        color  = mbs_colors[mbs]
        for bi, baseline in enumerate(BAR_ORDER):
            val = improvements[mbs][baseline]
            x   = x_positions[bi] + offset
            if val is not None:
                ax.bar(x, val, width=bar_width, color=color,
                       edgecolor="white", linewidth=0.5, zorder=3)
            else:
                _placeholder_bar(ax, x, bar_width, ref_h, color)

    ax.axhline(1.0, color="#555", linewidth=1.0, linestyle=":", zorder=2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(BAR_ORDER, fontsize=13)
    ax.set_xlabel("Baseline", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Throughput Improvement", fontsize=14)
    ax.set_title(
        f"{model.upper()}  |  num_devices={num_devices}  |  Throughput Improvement",
        fontsize=14, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    patches = [mpatches.Patch(color=mbs_colors[m], label=f"mbs={m}")
               for m in mbs_list]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.01, 1),
              borderaxespad=0, fontsize=12, framealpha=0.6,
              title="Micro-batch size", title_fontsize=11)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    mbs_tag  = "_".join(str(m) for m in mbs_list)
    fname    = f"{model}_dev{num_devices}_mbs{mbs_tag}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(models, mbs_list, setup_name,
                  alpa_data=None, num_devices_filter=None, use_ref_tput=False):
    """
    Layout (one row per model/device/mbs combination):

    =====================================================================...
    Model      | num_devices | mbs |       Throughput Improvement        |     Runtime
               |             |     |  Manual |    MCMC |  Alpa-E |  Phaze |    Nest |
    =====================================================================...
    bert       |           8 |   1 |  3.84x  |  1.20x  |    None |  2.10x |  5.01x | 0h 12m 3.0s
    -----------------------------------------------------------------...
    ...
    =====================================================================...

    - Values are tput / MANUAL_TPUT (default) or tput / REFERENCE_TPUT (--plot_mbs)
    - 2 decimal places with 'x' suffix (e.g. 1.50x); missing values show as 'None'
    - | column separators throughout; thin --- between rows; thick === between models
    """
    if alpa_data is None:
        alpa_data = {}

    # ---- column widths ----
    COL_MODEL   = 10
    COL_DEVICES = 11
    COL_MBS     = 3
    COL_VAL     = 7     # e.g. " 1.50x" or "  None"
    COL_RUNTIME = 15
    G = "    "          # extra gap before Runtime column

    def fmt_val(v):
        return f"{v:.2f}x" if v is not None else "None"

    def make_val_cells(impr):
        return " | ".join(f"{fmt_val(impr[b]):>{COL_VAL}}" for b in BAR_ORDER)

    def row_str(model_col, acc, mbs, val_cells, runtime_str):
        return (
            f"{str(model_col):<{COL_MODEL}} | "
            f"{str(acc):>{COL_DEVICES}} | "
            f"{str(mbs):>{COL_MBS}} | "
            f"{val_cells}"
            f"{G}{runtime_str}"
        )

    # ---- build header ----
    val_header  = " | ".join(f"{b:>{COL_VAL}}" for b in BAR_ORDER)
    tput_width  = len(val_header)
    tput_label  = "Throughput Improvement"

    top_header  = row_str("Model", "num_devices", "mbs",
                          tput_label.center(tput_width) + " " * len(G), "Runtime")
    sub_header  = row_str("", "", "", val_header, "")

    thick = "=" * len(top_header)
    thin  = "-" * len(top_header)

    print(thick)
    print(top_header)
    print(sub_header)
    print(thick)

    for model in models:
        model_key = model.lower()
        ref       = (REFERENCE_TPUT if use_ref_tput else MANUAL_TPUT).get(model_key, 1.0)
        first_model_row = True

        for mbs in mbs_list:
            setup_data = load_setup_csv(setup_name, model, mbs)
            if not setup_data:
                continue

            accs = sorted(setup_data.keys())
            if num_devices_filter:
                accs = [a for a in accs if a in num_devices_filter] or accs

            for acc in accs:
                row_data = setup_data[acc]
                tputs    = get_tputs_for_device(model, acc, mbs, setup_data, alpa_data)
                impr     = get_improvements(tputs, ref)
                runtime  = row_data.get("runtime_s")

                model_col = model if first_model_row else ""
                first_model_row = False

                print(row_str(model_col, acc, mbs,
                              make_val_cells(impr),
                              format_runtime(safe_float(runtime))))
                print(thin)

        print(thick)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot throughput improvement bar charts.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--num_devices", type=int, nargs="+", default=None)
    parser.add_argument("--mbs", type=int, nargs="+", required=True)
    parser.add_argument("--setup_name", required=True)
    parser.add_argument("--out_dir", default="plots")
    parser.add_argument("--alpa_csv",
                        default="alpa/results/parsed_results/result_summary_reference.csv")
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--plot_mbs", action="store_true",
                        help="x=baselines, sub-bars=mbs. Requires exactly one --num_devices.")
    return parser.parse_args()


def main():
    args = parse_args()
    model     = args.model.lower()
    alpa_data = load_alpa_csv(args.alpa_csv)
    models    = list(MANUAL_TPUT.keys()) if args.all_models else [model]

    if args.plot_mbs:
        if not args.num_devices or len(args.num_devices) != 1:
            print("[ERROR] --plot_mbs requires exactly one --num_devices value.")
            return
        nd = args.num_devices[0]
        mbs_list = sorted(args.mbs)
        plot_mbs_mode(model, mbs_list, args.setup_name, alpa_data,
                      num_devices=nd, out_dir=args.out_dir)
        print_summary(models, mbs_list, args.setup_name,
                      alpa_data=alpa_data, num_devices_filter=[nd],
                      use_ref_tput=True)
        return

    if len(args.mbs) > 1:
        print("[WARNING] Multiple --mbs values ignored in default mode; "
              "using first value only. Use --plot_mbs for multi-mbs plots.")
    mbs = args.mbs[0]

    plot_multi_device(model, mbs, args.setup_name, alpa_data,
                      num_devices_filter=args.num_devices,
                      out_dir=args.out_dir)
    print_summary(models, [mbs], args.setup_name,
                  alpa_data=alpa_data,
                  num_devices_filter=args.num_devices,
                  use_ref_tput=False)


if __name__ == "__main__":
    main()