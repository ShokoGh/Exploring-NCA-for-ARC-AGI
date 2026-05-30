# This file is part of a JAX/Flax Neural Cellular Automata implementation for ARC-AGI.
# Portions of this file are adapted from and inspired by:
# Google Research, self-organising-systems, notebooks/growing_ca.ipynb
#
# Related article:
# "Growing Neural Cellular Automata: Differentiable Model of Morphogenesis"
# by Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, and Michael Levin,
# Distill, 2020.
# https://distill.pub/2020/growing-ca/
#
# Reference implementation:
# https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb
#
# Original Google Research code:
# Copyright 2020 Google LLC
# Licensed under the Apache License, Version 2.0.
#
# This file has been modified for an ARC-AGI project.
#
# Adapted/inspired components:
# - Model initialization for Neural Cellular Automata training.
# - Sample-pool/state-pool based training.
# - Replacing the worst-performing pool element with the initial state.
# - Per-parameter gradient normalization for training stability.
#
# ARC-specific modifications:
#- Adapted the state pool to ARC input-output training pairs.
#- Uses one pool initialized from each ARC input grid.
#- Reimplemented training utilities in JAX/Flax/Optax.
#- Adds project-specific output directories, learning-rate schedules, and ARC task loading.
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import tree_util

from .model import NCAModel
from .config import CHANNELS
from .output_utils import save_training_outputs

def load_arc_task(task_path: str) -> dict:
    """Load a single ARC task from a JSON file."""
    with open(task_path) as file:
        return json.load(file)

def get_task_name(task_path: str) -> str:
    """Extract the ARC task name from the file path."""
    return os.path.splitext(os.path.basename(task_path))[0]

def prepare_output_dirs(task_name: str, run_id: int) -> dict:
    """Create and return all output directories used for a task run."""

    log_root = f"Training_Logs_FF_pad_train1/run_{run_id}"
    trace_root = f"json_traces_FF_pad_train1/run_{run_id}"
    io_root = f"json_io_FF_pad_train1/run_{run_id}"

    for directory in [log_root, trace_root, io_root]:
        os.makedirs(directory, exist_ok=True)

    base_trace_dir = os.path.join(trace_root, task_name)
    base_io_dir = os.path.join(io_root, task_name)

    os.makedirs(base_trace_dir, exist_ok=True)
    os.makedirs(base_io_dir, exist_ok=True)

    return {
        "log_root": log_root,
        "trace_root": trace_root,
        "io_root": io_root,
        "base_trace_dir": base_trace_dir,
        "base_io_dir": base_io_dir,
    }

def initialize_model(rng_key, input_state_nhwc, fire_rate: float):
    """
    Initialize the NCA model and parameter tree.
    """
    model = NCAModel(
        channels=CHANNELS,
        fire_rate=fire_rate,
    )

    params = model.init(
        {"params": rng_key, "fire": jax.random.PRNGKey(1)},
        input_state_nhwc,
    )["params"]

    return model, params

def make_scheduler(name: str, num_iters: int):
    """
    Create a learning-rate scheduler by name.
    """
    scheduler_name = name.lower()

    if scheduler_name == "step_decay_2000":
        initial_lr = 1e-3
        step_size = 2000
        decay_factor = 0.3
        num_drops = max(1, num_iters // step_size)

        boundaries = {
            step_size * i: decay_factor
            for i in range(1, num_drops + 1)
        }

        return optax.piecewise_constant_schedule(
            init_value=initial_lr,
            boundaries_and_scales=boundaries,
        )

    raise ValueError(f"Unknown scheduler: {name}")

def normalize_gradient_tree(grads):
    """
    Normalize each parameter tensor gradient independently using its L2 norm.

    This follows the per-variable gradient normalization strategy used in
    Growing Neural Cellular Automata to stabilize NCA training:
    https://distill.pub/2020/growing-ca/
    """
    def normalize_gradient(grad):
        grad_norm = jnp.sqrt(jnp.sum(grad * grad))
        return jnp.where(grad_norm > 0, grad / (grad_norm + 1e-8), grad)

    return tree_util.tree_map(normalize_gradient, grads)

def update_state_pools(
    final_state_list,
    pool_loss_list,
    input_states,
    target_states,
    state_trace_list,
    collect_trace: bool,
    iteration_index: int,
    num_unroll_steps: int,
    trace_train_dir: str,
    io_train_dir: str,
) -> list:
    """
    Reset the worst-performing pool element for each training pair and return the
    updated state pools.

    If collect_trace is True, save training traces and I/O bundles before resetting.
    """
    updated_state_pools = []

    for pair_index, (final_state, pool_element_loss) in enumerate(
        zip(final_state_list, pool_loss_list)
    ):
        worst_pool_index = int(np.asarray(pool_element_loss).argmax())

        if collect_trace:
            save_training_outputs(
                iteration_index=iteration_index,
                pair_index=pair_index,
                num_unroll_steps=num_unroll_steps,
                input_state=input_states[pair_index],
                target_state=target_states[pair_index],
                final_state=final_state,
                state_trace=state_trace_list[pair_index],
                trace_output_dir=trace_train_dir,
                io_output_dir=io_train_dir,
            )

        # Reset the worst-performing pool element to the original input state.
        reset_state = input_states[pair_index][0]
        final_state = final_state.at[worst_pool_index].set(reset_state)
        updated_state_pools.append(final_state)

    return updated_state_pools

