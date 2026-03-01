"""
Distributed BERT Training with Alpa (Pipeshard Parallelism + Estimator Mode)
=============================================================================
Uses Alpa with Nest estimators instead of a real GPU cluster.
Developed to compare Alpa-E against Nest for the TPUv4 configuration.

Usage:
    python script.py --num_nodes <int> --num_layers <int> --num_microbatch <int>

Example:
    python script.py --num_nodes 1 --num_layers 24 --num_microbatch 4096
"""

import argparse
from time import time

import alpa
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax import random
import optax
import ray
from transformers import FlaxBertModel, BertConfig
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Alpa BERT training with estimator mode.")
    parser.add_argument("--num_nodes",     type=int, default=1,    help="Number of nodes in the fake cluster.")
    parser.add_argument("--num_layers",    type=int, default=24,   help="Number of hidden layers in BERT.")
    parser.add_argument("--num_microbatch",type=int, default=4096, help="Number of micro-batches for pipeline parallelism.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Cluster Initialization ---
    ray.init(address="auto")
    alpa.init(cluster="fake", num_nodes=args.num_nodes, num_devices_per_node=8)
    alpa.util.disable_tqdm_globally()

    # --- Configuration ---
    config = BertConfig.from_pretrained("bert-large-uncased", num_hidden_layers=args.num_layers)
    batch_size = 4096
    seq_len    = 512
    hidden_dim = config.hidden_size  # 1024 for bert-large

    # --- Data Generation ---
    rngkey = jax.random.PRNGKey(0)
    k_ids, k_noise = random.split(rngkey)

    batch = {
        "input_ids":      random.randint(k_ids, (batch_size, seq_len), 0, config.vocab_size),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "y":              random.normal(k_noise, (batch_size, hidden_dim)),
    }

    # --- Model & Optimizer Initialization ---
    model  = FlaxBertModel.from_pretrained("bert-large-uncased")
    tx     = optax.adam(learning_rate=1e-4)
    state  = TrainState.create(apply_fn=model.__call__, params=model.params, tx=tx)

    # --- Parallelization Method ---
    method = alpa.PipeshardParallel(
        num_micro_batches=args.num_microbatch,
        layer_option=alpa.AutoLayerOption(layer_num=args.num_layers),
        stage_option="auto",
    )

    # --- Training Step ---
    @alpa.parallelize(method=method)
    def train_step(state, batch):
        def loss_func(params):
            outputs = state.apply_fn(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                params=params,
            )
            loss = jnp.mean((outputs.pooler_output - batch["y"]) ** 2)
            return loss

        grads     = alpa.grad(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state

    # --- Execution ---
    print(
        f"Running with: num_nodes={args.num_nodes}, total devices={args.num_nodes * 8}, "
        f"num_layers={args.num_layers}, "
        f"num_microbatch={args.num_microbatch}"
    )
    print("Compiling and running parallelized training step...")

    tic = time()
    try:
        state = train_step(state, batch)
        print("Training step completed successfully.")
    except Exception as e:
        print(f"Training step encountered an error: {e}")
    finally:
        toc = time()
        print(f"Elapsed time: {toc - tic:.2f}s")
        print("-" * 50)
        # alpa.shutdown()


if __name__ == "__main__":
    main()