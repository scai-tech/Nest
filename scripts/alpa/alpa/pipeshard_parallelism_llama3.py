"""
Distributed Llama-3 Training with Alpa (Pipeshard Parallelism + Estimator Mode)
================================================================================
Uses Alpa with Nest estimators instead of a real GPU cluster.
Developed to compare Alpa-E against Nest for the TPUv4 configuration.

Usage:
    python llama3_alpa.py --num_nodes <int> --num_layers <int> --num_microbatch <int>

Example:
    python llama3_alpa.py --num_nodes 16 --num_layers 20 --num_microbatch 1024
"""

import argparse
import os
from time import time

import alpa
from alpa.model.model_util import DynamicScale, TrainState
import jax
import jax.numpy as jnp
from jax import random
import optax
import ray
from transformers import AutoConfig, FlaxLlamaForCausalLM


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
jax.config.update('jax_enable_x64', False)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Alpa Llama-3 training with estimator mode.")
    parser.add_argument("--num_nodes",      type=int, default=16,   help="Number of nodes in the fake cluster.")
    parser.add_argument("--num_layers",     type=int, default=20,   help="Number of hidden layers in Llama-3.")
    parser.add_argument("--num_microbatch", type=int, default=4096, help="Number of micro-batches for pipeline parallelism.")
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
    alpa.global_config.flax_always_use_fp16_embedding = True

    # --- Configuration ---
    MODEL_NAME = "meta-llama/Llama-3.1-70B"
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        token='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',
        attn_implementation="eager",
        num_hidden_layers=args.num_layers,
    )

    batch_size = 4096
    seq_len    = 4096

    # --- Data Generation ---
    rngkey = jax.random.PRNGKey(0)
    k_ids, k_noise, k_model_init = random.split(rngkey, 3)

    batch = {
        "input_ids":      random.randint(k_ids, (batch_size, seq_len), 0, config.vocab_size),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
    }

    # --- Model & Optimizer Initialization ---
    tx = optax.adam(learning_rate=1e-4)

    def create_state_fn():
        model  = FlaxLlamaForCausalLM(config=config, dtype=jax.numpy.bfloat16)
        params = model.init_weights(k_model_init, batch["input_ids"].shape)
        return TrainState.create(
            apply_fn=model.__call__,
            params=params,
            tx=tx,
            dynamic_scale=DynamicScale(),
            use_master_copy=True,
        )

    state = jax.eval_shape(create_state_fn)

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
            return jnp.sum(outputs.logits)

        grads     = alpa.grad(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state

    # --- Execution ---
    print(
        f"Running with: num_nodes={args.num_nodes}, total_devices={args.num_nodes * 8}, "
        f"num_layers={args.num_layers}, num_microbatch={args.num_microbatch}"
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