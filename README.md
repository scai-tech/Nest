# NEST

**NEST** is a network-, compute-, and memory-aware automatic distributed training configuration and device placement framework for large-scale training of large language models (LLMs). It co-optimizes parallelism strategies (including data, tensor, pipeline, expert, and ZeRO parallelism) together with device placement using structured dynamic programming.

---

## Table of Contents

- [Installation](#installation-30-minutes)
- [Obtaining a Gurobi License](#obtaining-a-gurobi-license)
- [Experiments](#experiments)
  - [Experiment 1 — Figure 5](#experiment-1-reproducing-results-in-figure-5)
  - [Experiment 2 — Figure 6](#experiment-2-reproducing-results-in-figure-6)
  - [Experiment 3 — Table 3](#experiment-3-reproducing-results-in-table-3)
- [Quick Start](#quick-start)
- [Running NEST Directly](#running-nest-directly)
- [Troubleshooting](#troubleshooting)
- [Code Structure](#code-structure)

---

## Installation (~30 minutes)

### Prerequisite
The artifact is tested using Anaconda.

NEST is compatible with any NVIDIA GPU supporting PyTorch 2.5 and CUDA 12.4 with at least 100 GB of RAM. When using the provided operator graphs and estimates included as part of this artifact, most experiments can also be executed in CPU-only environments with at least 64 GB of RAM.

### Setup
```bash
git clone https://github.com/scai-tech/Nest.git

cd Nest
conda env create -f environment.yml
conda activate nestenv
```

Verify that `$CONDA_PREFIX` is set correctly:
```bash
echo $CONDA_PREFIX
```


If it is not set, update it to point to your conda environment directory. Then install the remaining packages:
```bash
./setup.sh
source $CONDA_PREFIX/etc/conda/activate.d/nest_paths.sh
```


**[Not required for AE]** If running on a GPU (that supports CUDA 12.4) and want to extract graphs from scratch, also install the APEX library:

```bash
./setup.sh --apex_only
```

> **Note:** The `apex` installation step commonly fails. See [Troubleshooting Apex Installation](#1-troubleshooting-apex-installation) for fixes.

After installation, verify that `PYTHONPATH` is configured correctly and all listed paths are valid:
```bash
echo $PYTHONPATH
```

---

## Obtaining a Gurobi License

NEST uses Gurobi to solve ILP formulations. To run the ILP solver, obtain a license from the [Gurobi website](https://www.gurobi.com/).

- Most academic users can obtain a free Gurobi WLS license at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/). Once downloaded, place the `gurobi.lic` file in your home directory (e.g., `/home/yourusername`).
- For AE reviewers who are unable to obtain a license, please contact the authors via HotCRP.

---

## Experiments

### A note on Alpa-E (Optional)

NEST evaluates against an offline variant of Alpa called **Alpa-E** (Alpa Estimator), which retains Alpa's core optimization while replacing its hardware-dependent profiler with NEST's unified estimator. The full functional estimator is provided in [`scripts/alpa/`](./scripts/alpa/), along with a setup tutorial.

Due to Alpa-E's long runtime (single experiments can take over 24 hours) and separate resource requirements, they are excluded from the main artifact evaluation. Pre-computed Alpa-E results are provided in [`scripts/alpa/results/parsed_results/result_summary_reference.csv`](./scripts/alpa/results/parsed_results/result_summary_reference.csv) and are used directly for comparison. Interested users are welcome to follow the tutorial in [`scripts/alpa`](./scripts/alpa/) to set up a separate conda environment and run the experiments independently.

---

### A Note on Extracted Graphs and Estimates (Optional)

For the artifact evaluation, we have provided pre-extracted operator graphs and operator latency estimates. Extracted graphs are placed in [`nest/GraphExtractor/out/<model>/`](./nest/GraphExtractor/out/) and estimates in [`nest/Estimator/estimates_<setup_name>/<model>/`](./nest/Estimator/).

To re-extract graphs and estimates from scratch, delete the existing files (`.pickle` files for graphs and `.json` files for estimates) from these directories. The graphs and estimates will then be regenerated automatically when running the experiments.

> **Note:** Graph extraction and estimation require GPU access (CUDA 12.4) and the APEX library to be installed (see [Installation](#installation)). This process may add approximately 30 minutes to the workflow. For the artifact evaluation, we recommend using the provided files.

---

## Experiment 1: Reproducing Results in Figure 5

> **Experiments 1 and 2 run from the same directory. Navigate there once before starting:**
> ```bash
> cd scripts/tpuv4_fatTree
> ```

### Experiment 1.1 — Llama2 Results (~10 minutes)

Runs NEST and baselines (Manual, MCMC, Phaze) for Llama2-7B across device scales of 64, 128, 256, 512, and 1024.
```bash
./run_eval.sh llama2
```

**Outputs:**
```
scripts/tpuv4_fatTree/
├── plots/
│   └── llama2_mbs1_devices64_128_256_512_1024.png   # Output plot - corresponding to the Llama2-7B plot in Figure 5
└── out/
    └── llama2_mbs1/
        ├── llama2_mbs1_output_<num_device>.log      # Simulation log files
        └── llama2_mbs1.csv                          # Aggregated Summary
```

The output plot is saved to `scripts/tpuv4_fatTree/plots/llama2_mbs1_devices64_128_256_512_1024.png`, which corresponds to the Llama2-7B plot in Figure 5. 
Console output reports throughput improvements for all baselines relative to the Manual baseline at 64 devices.

Expected results can also be found in table format [`scripts/tpuv4_fatTree/reference_results.md`](./scripts/tpuv4_fatTree/reference_results.md)

---

### Experiment 1.2 — All Models at Scale 512 (~2 hours)

Runs NEST and baselines for all evaluated models (BertLarge, Llama2-7B, Llama3-70B, Mixtral-8x7B, GPT3-175B) at 512 devices.
```bash
./run_eval.sh "" 512
```

**Outputs:**
```
scripts/tpuv4_fatTree/
├── plots/                                            # Plot outputs
└── out/
    └── <model>_mbs1/
        ├── <model>_mbs1_output_<num_device>.log      # Simulation log files
        └── <model>_mbs1.csv                          # Aggregated Summary
```

Figures are saved to `scripts/tpuv4_fatTree/plots/` in the format `<model>_mbs1_devices512.png`. These correspond to the 512-device sub-graph of Figure 5 for each model.
Expected results can also be found in table format [`scripts/tpuv4_fatTree/reference_results.md`](./scripts/tpuv4_fatTree/reference_results.md)

---

### [Optional] Experiment 1.3 — Full Figures for All Models ( ~7 hours)

Reproduces full per-model figures across all device scales:
```bash
./run_eval.sh <model>    # Options: bert, llama2, llama3, mixtral, gpt3
```

Figures are saved to `scripts/tpuv4_fatTree/plots/`, in the format `<model>_mbs1_devices64_128_256_512_1024.png`, and can be compared to the full plot for each model in Figure 5.
Expected results can also be found in table format [`scripts/tpuv4_fatTree/reference_results.md`](./scripts/tpuv4_fatTree/reference_results.md)

> **Note:** GPT3 at 1024 devices will require higher RAM to run.

---

## Experiment 2: Reproducing Results in Figure 6

> **Still in `scripts/tpuv4_fatTree`** — no need to change directories.

### Experiment 2.1 — Llama2-70B (~20 minutes)

Runs NEST and baselines for Llama2-70B with microbatch sizes 1, 2, 4, and 8 at 256 devices.
```bash
./run_eval_mbs.sh llama2 256 1,2,4,8
```

**Outputs:**
```
scripts/tpuv4_fatTree/
├── plots/
│   └── llama2_dev256_mbs1_2_4_8.png               # Output plot: Llama2-7B (Figure 6)
└── out/
    └── llama2_mbs<mbs>/
        ├── llama2_mbs<mbs>_output_256.log         # Simulation log files
        └── llama2_mbs<mbs>.csv                    # Aggregated Summary
```

The output plot is saved to `scripts/tpuv4_fatTree/plots/llama2_dev256_mbs1_2_4_8.png` corresponding to the Llama2-7B plot in Figure 6.

Expected results can also be found in table format [`scripts/tpuv4_fatTree/reference_results.md`](./scripts/tpuv4_fatTree/reference_results.md)

---

### [Optional] Experiment 2.2 — BertLarge and Llama3-70B (~40 minutes)
```bash
./run_eval_mbs.sh bert 256 1,2,4,8     # BertLarge
./run_eval_mbs.sh llama3 256 1,2,4,8   # Llama3-70B
```

Figures are saved to `scripts/tpuv4_fatTree/plots/`, in the format `scripts/tpuv4_fatTree/plots/<model>_dev256_mbs1_2_4_8.png`

---

## Experiment 3: Reproducing Results in Table 3
> **Note: Experiment 3 is run from a different directory.**
> ```bash
> cd scripts/h100_spineLeaf
> # If currently in scripts/tpuv4_fatTree:
> # cd ../../scripts/h100_spineLeaf
> ```

### Experiment 3.1 — H100 Mixtral (~5 minutes)

Runs NEST and baselines for Mixtral-8x7B at 1024 devices using profiled H100 GPU data.
```bash
./run_eval.sh mixtral 1024
```

> **Note:** If this experiment fails due to memory constraints, use the smaller-scale alternative below.

**Outputs:**
```
scripts/h100_spineLeaf/out/mixtral_mbs1/
  ├── mixtral_mbs1_output_1024.log
  ├── mixtral_mbs1.csv/
```

**Expected Console Output:**

| Model   | Devices | Manual | MCMC   | Phaze  | NEST   | Runtime |
|---------|---------|--------|--------|--------|--------|---------|
| mixtral | 1024    | 1.000x | 1.324x | 1.215x | 1.519x | 4m 0s   |

---

### [Optional] Experiment 3.2 — Mixtral at 128 Devices (~1 minute)
```bash
./run_eval.sh mixtral 128
```

**Expected Console Output:**

| Model   | Devices | Manual | MCMC   | Phaze  | NEST   | Runtime |
|---------|---------|--------|--------|--------|--------|---------|
| mixtral | 128     | 1.000x | 0.749x | 1.202x | 1.503x | 45s     |

---

### [Optional] Experiment 3.3 — GPT3 at 128 Devices (~12 minutes)

Running GPT3 at 1024 devices requires significant memory. Use 128 devices as an alternative:
```bash
./run_eval.sh gpt3 128
```

**Expected Console Output:**

| Model | Devices | Manual | MCMC   | Phaze  | NEST   | Runtime |
|-------|---------|--------|--------|--------|--------|---------|
| gpt3  | 128     | 1.000x | 1.000x | 1.143x | 1.334x | 11m 1s  |

Results are consistent with the 1024-scale results shown in Table 3 of the paper.

---

## Quick Start

Scripts for the TPUv4 (fat-tree) and H100 (spine-leaf) setups described in the paper are provided in [`scripts/`](./scripts/).
```bash
cd scripts/tpuv4_fatTree   # or scripts/h100_spineLeaf
./run_eval.sh <model> <num_devices>
```

| Argument | Options |
|----------|---------|
| `model` | `bert`, `llama2`, `llama3`, `mixtral`, `gpt3` (leave blank to run all) |
| `num_devices` | `64`, `128`, `256`, `512`, `1024` (leave blank to run all) |

To sweep over multiple microbatch sizes:
```bash
./run_eval_mbs.sh <model> <num_devices> <mbs>
```

| Argument | Options |
|----------|---------|
| `model` | `bert`, `llama2`, `llama3`, `mixtral`, `gpt3` |
| `num_devices` | `64`, `128`, `256`, `512`, `1024` |
| `mbs` | `1`, `2`, `4`, `8` (use commas to specify multiple, e.g. `1,2,4,8`) |

---

## Running NEST Directly

You can invoke [`nest.py`](./nest.py) directly for custom configurations:
```bash
python3 nest.py \
    --model_names $MODEL_NAME \
    --exec_type $EXEC_TYPE \
    --micro_batch_size $MBS \
    --sequence_length $SEQ_LEN \
    --hbm_size $HBM_SIZE \
    --num_accelerators $NUM_ACC \
    --setup_name "$SETUP_NAME" \
    --devices_per_level $DEVICES_PER_LEVEL \
    --bandwidth_per_level $BW_PER_LEVEL \
    --use_ilp
```

| Argument | Description |
|----------|-------------|
| `--model_names` | Model to run (`Bert`, `Llama2`, `llama3`, `GPT`, `Mixtral` variants) |
| `--sequence_length` | Input sequence length |
| `--micro_batch_size` | List of microbatch sizes to explore |
| `--num_accelerators` | Number of accelerators in the cluster |
| `--setup_name` | Name for the network/accelerator config; used when saving runtime estimates, outputs, and baseline configurations |

The full list of arguments is in [`nest/arguments.py`](./nest/arguments.py).

---

## Code Structure
```
/                               # NEST root
├── nest/                       # Core source code
│   ├── GraphExtractor/         # Extracts model operator graphs
│   ├── Estimator/              # Generates architectures and estimates latencies
│   └── Solver/                 # ILP and DP solvers
├── third_party_for_nest/
│   ├── Wham/                   # Operator mapping and area estimation
│   ├── Sunstone/               # Operator latency estimation
│   ├── Megatron/               # Megatron model support
│   ├── Astra-sim/              # Network modeling (Chakra + OpenMPI)
│   ├── Chakra/                 # Network modeling
│   └── Openmpi/                # Network modeling
├── nest.py                     # Main entry point
└── scripts/
    ├── alpa/                   # Alpa-E source code and scripts
    ├── tpuv4_fatTree/          # Scripts for TPUv4 + fat-tree setup
    └── h100_spineLeaf/         # Scripts for H100 + spine-leaf setup
```

---

## Troubleshooting

### 1. Troubleshooting Apex Installation

**Error:** `Cuda extensions are being compiled with a version of Cuda that does not match the version used to compile Pytorch binaries...`

For minor version mismatches, it is generally safe to skip the strict version check:

1. Open `apex/setup.py`.
2. Comment out lines 84–92 (the `if bare_metal_version != torch_binary_version:` block).
3. Save the file and re-run the apex installation step:
```bash
   ./setup.sh --apex-only
```

> **Note:** Compiling these extensions from source typically takes 15–20 minutes depending on your hardware.

---

### 2. Troubleshooting C++ Build Errors

If you see an error like:
```
x86_64-conda-linux-gnu-cc: fatal error: cannot execute 'cc1plus'
```

Run the following before retrying:
```bash
export CXX=$(which g++)
export CC=$(which gcc)
```
---

## More Information

For AE reviewers, please contact the authors through HotCRP for any questions. For other users, please open an issue publicly or contact Irene Wang (irene.wang@gatech.edu) for any technical questions.
