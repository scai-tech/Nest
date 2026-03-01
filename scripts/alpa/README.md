# Alpa-Estimator (Alpa-E)
NEST evaluates against an offline variant of Alpa called **Alpa-E** (Alpa Estimator), which retains Alpa's core optimization while replacing its hardware-dependent profiler with NEST's unified estimator. Here, we provide the full guide to setup and run Alpa Experiments 

Note: Alpa Requires one GPU with CUDA 11.3, Therefore users must check what CUDA version their GPU supports (For example, A100, V100, RTX 6000) (H100s will not work)

## Installation 

```bash
cd Nest/scripts/alpa
conda env create -f environment.yml
conda activate alpa_env
```
Verify that `$CONDA_PREFIX` is set correctly:

```bash
echo $CONDA_PREFIX
```

If it is not set, update it to point to your conda environment directory. Then install the remaining packages:

```bash
./setup.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

### Test Setup

```bash
./test.sh
```
If the output shows that the tests have passed, you are good to go!

## Running Alpa

Use the following command to run experiments with Alpa-E

```bash
./run_test.sh --model <model> --num_devices <num_devices> --mbs <microbatch size>
```
Alpa-E supports the same set of models as NEST models BertLarge (`bert`), Llama2-7b (`llama2`), Llama3-70b (`llama3`), GPT3-175B(`gpt3`), Mixtral-7bx8 (`mixtral`).

- Run logs are saved to `results/logs/`
- Results are parsed into [`results/parsed_results/result_summary.csv`](./results/parsed_results/result_summary.csv)
- Reference results from NEST evaluations are in [`results/parsed_results/result_summary_reference.csv`](./results/parsed_results/result_summary_reference.csv)

> Deviations in the final cost calculated by Alpa-E are expected to be less than 1%.

### Note on Alpa Runtime
Alpa can have long execution times to up to 20 hours per experiment, and requires a lot of memory to run large models like GPT3-175B. We have simplified parts of our scripts (eg run less layers), valideated to not affect the results,  to enable the experiments to all finish within 24 hours. 

## Code Structure

```
/alpa                           
├── alpa/                              # Alpa-E root
│   ├── estimator/                     # Estimates latencies 
│   ├── pipeline_parallel/             # Stage Construction and profiling logic
│   ├──<Other>/                        # Rest remain similar to original Alpa
│   └── Pipeshard_parallelism_<model>  # Python Scripts to run Alpa-E with each model
├── scripts/
│    ├── logs/                         # Alpa-E source code and scripts
│    └── parsed_results/               # where parsed_results and reference results are saved
└── <Third Party>                      # Other third party source code required for Alpa-E

```