# Expected Throughput Results

While the paper primarily presents figures for our results, we summarize the expected throughput improvement results that the console will output when running the TPUv4 Fat Tree configuration. Plots for each result are generated in `/plots` and can be compared directly with each figure in the paper.

> **Note:** Expected overall runtimes are based on H100 GPU execution on an HPC cluster with network-mounted storage. Runtimes may vary by environment: local machines with SSD storage may run faster. First-time runs may also be slower due to Python and CUDA compilation caching. Actual Nest solving times are also reported in the execution logs, as "total solving time".

---

## Experiment 1: Reproducing Results in Figure 5

Expected results for all models across different scales.

> To compare with Experiment 1.2, use the **512 device** row for each model.

### BERT
| Model | num_devices | mbs | Manual | MCMC | Alpa-E | Phaze | Nest | Runtime |
|-------|-------------|-----|--------|------|--------|-------|------|---------|
| bert | 64 | 1 | 1.00x | 0.55x | 0.42x | 1.35x | 1.35x | 0h 0m 54s |
| bert | 128 | 1 | 1.97x | 1.05x | 0.65x | 2.70x | 2.70x | 0h 1m 2s |
| bert | 256 | 1 | 3.84x | 0.99x | 0.85x | 5.40x | 5.40x | 0h 1m 57s |
| bert | 512 | 1 | 7.31x | 9.39x | 0.69x | 10.81x | 10.81x | 0h 3m 37s |
| bert | 1024 | 1 | 13.31x | 16.61x | — | 21.61x | 21.61x | 0h 7m 4s |

### Llama 2

| Model | num_devices | mbs | Manual | MCMC | Alpa-E | Phaze | Nest | Runtime |
|-------|-------------|-----|--------|------|--------|-------|------|---------|
| llama2 | 64 | 1 | 1.00x | 0.62x | 1.25x | 1.04x | 1.25x | 0h 3m 1s |
| llama2 | 128 | 1 | 1.97x | 1.23x | 2.23x | 2.15x | 2.47x | 0h 0m 52s |
| llama2 | 256 | 1 | 3.84x | 3.19x | 2.40x | 4.20x | 4.81x | 0h 1m 0s |
| llama2 | 512 | 1 | 7.31x | 8.65x | 3.72x | 8.01x | 9.14x | 0h 1m 9s |
| llama2 | 1024 | 1 | 13.31x | 12.97x | — | 14.42x | 16.63x | 0h 1m 36s |

### Llama 3

| Model | num_devices | mbs | Manual | MCMC | Alpa-E | Phaze | Nest | Runtime |
|-------|-------------|-----|--------|------|--------|-------|------|---------|
| llama3 | 64 | 1 | — | 1.01x | — | 1.01x | 1.01x | 0h 2m 4s |
| llama3 | 128 | 1 | 1.00x | 1.00x | 3.00x | 2.97x | 2.97x | 0h 0m 59s |
| llama3 | 256 | 1 | 2.89x | 2.89x | 5.23x | 5.77x | 5.77x | 0h 1m 36s |
| llama3 | 512 | 1 | 5.48x | 5.48x | 8.38x | 10.93x | 10.94x | 0h 2m 33s |
| llama3 | 1024 | 1 | 9.92x | 9.92x | — | 19.06x | 20.51x | 0h 4m 39s |

### Mixtral

| Model | num_devices | mbs | Manual | MCMC | Alpa-E | Phaze | Nest | Runtime |
|-------|-------------|-----|--------|------|--------|-------|------|---------|
| mixtral | 64 | 1 | — | 0.68x | 0.49x | 0.68x | 1.02x | 0h 3m 28s |
| mixtral | 128 | 1 | 1.00x | 1.00x | 0.88x | 1.33x | 2.00x | 0h 2m 50s |
| mixtral | 256 | 1 | 1.94x | 2.60x | 1.41x | 2.60x | 3.89x | 0h 2m 48s |
| mixtral | 512 | 1 | 3.68x | 3.68x | 1.83x | 4.92x | 7.38x | 0h 3m 30s |
| mixtral | 1024 | 1 | 6.63x | 6.63x | — | 8.90x | 13.35x | 0h 4m 41s |


### GPT-3

| Model | num_devices | mbs | Manual | MCMC | Alpa-E | Phaze | Nest | Runtime |
|-------|-------------|-----|--------|------|--------|-------|------|---------|
| gpt3 | 64 | 1 | — | 0.57x | — | 0.57x | 0.67x | 0h 5m 7s |
| gpt3 | 128 | 1 | 1.00x | 1.12x | 0.54x | 1.14x | 1.33x | 0h 5m 21s |
| gpt3 | 256 | 1 | 1.99x | 1.59x | 1.23x | 2.27x | 2.65x | 0h 18m 19s |
| gpt3 | 512 | 1 | 3.91x | 4.58x | 1.86x | 4.47x | 5.22x | 1h 14m 45s |
| gpt3 | 1024 | 1 | 7.60x | 8.70x | — | 9.03x | 10.15x | 3h 42m 41s |
             


---

## Experiment 2: Reproducing Results in Figure 6

Expected results for BertLarge, Llama2-7B, and Llama3-70B across micro-batch sizes.

### BERT

| Model | num_devices | mbs | Manual | MCMC | Alpa-E | Phaze | Nest | Runtime |
|-------|-------------|-----|--------|------|--------|-------|------|---------|
| bert | 256 | 1 | 1.00x | 0.26x | 0.22x | 1.41x | 1.41x | 0h 3m 24s |
| bert | 256 | 2 | 0.95x | 0.11x | 0.21x | 1.41x | 1.41x | 0h 1m 43s |
| bert | 256 | 4 | 0.87x | 0.87x | 0.26x | 1.41x | 1.41x | 0h 1m 35s |
| bert | 256 | 8 | 0.73x | 0.24x | 0.39x | 1.41x | 1.41x | 0h 1m 20s |

### Llama 2

| Model | num_devices | mbs | Manual | MCMC | Alpa-E | Phaze | Nest | Runtime |
|-------|-------------|-----|--------|------|--------|-------|------|---------|
| llama2 | 256 | 1 | 1.00x | 0.83x | 0.63x | 1.09x | 1.25x | 0h 1m 35s |
| llama2 | 256 | 2 | — | 0.62x | 0.90x | 1.62x | 1.83x | 0h 0m 40s |
| llama2 | 256 | 4 | — | — | 1.17x | — | 2.25x | 0h 0m 35s |
| llama2 | 256 | 8 | — | — | — | — | — | — |

### Llama 3

| Model | num_devices | mbs | Manual | MCMC | Alpa-E | Phaze | Nest | Runtime |
|-------|-------------|-----|--------|------|--------|-------|------|---------|
| llama3 | 256 | 1 | 1.00x | 1.00x | 1.81x | 2.00x | 2.00x | 0h 3m 3s |
| llama3 | 256 | 2 | — | — | 1.96x | 3.37x | 3.37x | 0h 3m 49s |
| llama3 | 256 | 4 | — | — | — | — | — | — |
| llama3 | 256 | 8 | — | — | — | — | — | — |