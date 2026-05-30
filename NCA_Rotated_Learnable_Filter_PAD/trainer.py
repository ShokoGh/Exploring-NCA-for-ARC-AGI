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
# - Training by repeatedly unrolling the same learned NCA update rule.
# - Randomized number of NCA update steps during training.
# - Sample-pool/state-pool based training.
# - Updating the state pool with final NCA states after each training iteration.
# - Resetting the worst-performing pool element.
# - Per-parameter gradient normalization before optimizer updates.
#
# ARC-specific modifications:
# - Trains on ARC input-output task pairs instead of fixed image-growth targets.
# - Supports multiple training pairs per ARC task.
# - Uses padded ARC grids, spatial masks, and target masks.
# - Computes ARC-specific masked loss and palette-based accuracy.
# - Uses rotation-conditioned training for rotation-aware ARC experiments.
# - Saves ARC-specific traces, predictions, metrics, and evaluation outputs.
# - Reimplemented the training loop using JAX/Flax/Optax.
import os
import time


import jax
import jax.numpy as jnp
import optax
from jax import tree_util
from functools import partial

from .config import NUM_TRAIN_ITERS, MIN_UNROLL_STEPS ,MAX_UNROLL_STEPS, POOL_SIZE, ROTATION_CHANNEL_INDEX
from .data_utils import build_pair_tensors, get_task_padding_size
from .model import unroll_nca, apply_spatial_mask
from .metrics import compute_train_metrics, compute_test_metrics, masked_mse_rgba_per_state
from .output_utils import save_test_outputs, save_test_metrics_csv, save_train_metrics_csv, plot_lr_and_loss, \
    save_padding_debug_log
from .training_utils import load_arc_task, prepare_output_dirs, get_task_name, initialize_model, make_scheduler, \
    normalize_gradient_tree, update_state_pools


def evaluate_on_test_pair(
    params,
    model,
    test_pairs,
    rng_key,
    fire_rate: float,
    step_size: float,
    lr_log_dir: str,
    trace_test_dir: str,
    io_test_dir: str,
    task_name: str,
    run_id: int,
    task_runtime_seconds: float,
    max_h: int,
    max_w: int,
    padding_log_lines: list[str],
) -> None:
    """
    Evaluate the trained model on the first test pair of an ARC task and save
    metrics, trace frames, and the final prediction.

    The test pair is padded to the task size, and the same
    masking rules are used as in training.

    Args:
        params: Trained model parameters.
        model: NCA model.
        test_pairs: Test pairs from the ARC task.
        rng_key: JAX random key.
        fire_rate: Stochastic cell update rate. Set to 1.0 for deterministic testing.
        step_size: Scale factor for each residual update.
        lr_log_dir: Directory for metric logs.
        trace_test_dir: Directory for test trace files.
        io_test_dir: Directory for test I/O files.
        task_name: ARC task name.
        run_id: Run index.
        task_runtime_seconds: Total training time for this task.
        max_h: Task padding height.
        max_w: Task padding width.
        padding_log_lines: List used to collect padding/debug messages generated during training and evaluation.
    """
    if not test_pairs:
        return

    input_state_nhwc, target_state_nhwc, input_valid_mask, target_valid_mask, debug_message = build_pair_tensors(
        test_pairs[0],
        max_h,
        max_w,
        debug=True,
        pair_name="test pair 0",
    )

    if debug_message is not None:
        padding_log_lines.append(debug_message)

    spatial_mask = jnp.maximum(input_valid_mask, target_valid_mask)

    num_test_steps = 63
    _, eval_key = jax.random.split(rng_key)

    # Use deterministic evaluation by updating all cells during the test run.
    fire_rate = 1.0

    # Evaluate the unrotated test task.
    input_state_nhwc = input_state_nhwc.at[..., ROTATION_CHANNEL_INDEX].set(0.0)
    target_state_nhwc = target_state_nhwc.at[..., ROTATION_CHANNEL_INDEX].set(0.0)

    final_state, state_trace = unroll_nca(
        params,
        model,
        eval_key,
        input_state_nhwc,
        steps=num_test_steps,
        fire_rate=fire_rate,
        step_size=step_size,
        collect_trace=True,
        spatial_mask=spatial_mask,
        visible_target_mask=target_valid_mask,
        rotation_index=0,
    )

    metrics = compute_test_metrics(
        state_trace=state_trace,
        final_state=final_state,
        target_state_nhwc=target_state_nhwc,
        target_valid_mask=target_valid_mask,
    )

    results_dir = "run_results_RLF4_pad_train1"
    os.makedirs(results_dir, exist_ok=True)

    results_csv = os.path.join(results_dir, f"run_{run_id:02d}.csv")
    with open(results_csv, "a") as file:
        file.write(
            f"{task_name},{metrics['final_accuracy']:.6f},"
            f"{metrics['final_exact_match']:.1f},"
            f"{metrics['final_foreground_accuracy']:.6f},"
            f"{metrics['first_solve_step']},"
            f"{task_runtime_seconds:.6f}\n"
        )

    save_test_metrics_csv(
        lr_log_dir=lr_log_dir,
        metrics=metrics,
    )

    save_test_outputs(
        input_state_nhwc=input_state_nhwc,
        target_state_nhwc=target_state_nhwc,
        final_state=final_state,
        state_trace=state_trace,
        trace_test_dir=trace_test_dir,
        io_test_dir=io_test_dir,
    )


# ---------------- Training ----------------

def train_one_task(
    task_path,
    rng_key,
    num_iters=NUM_TRAIN_ITERS,
    fire_rate=0.5,
    step_size=1.0,
    run_id=0,
):
    """
    Train the NCA model on a single ARC task.
    """
    task_start_time = time.time()
    task = load_arc_task(task_path)
    task_name = get_task_name(task_path)
    output_dirs = prepare_output_dirs(task_name, run_id)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])
    num_train_pairs = len(train_pairs)

    assert num_train_pairs > 0, "Task contains no training pairs."

    padding_log_lines = []
    max_h, max_w = get_task_padding_size(task)
    padding_log_lines.append(f"Task {task_name}: padding all grids to {max_h}x{max_w}")

    # Build padded tensors for all training pairs using the shared task size.

    input_states = []
    target_states = []
    target_masks = []
    spatial_masks = []

    for pair_index, pair in enumerate(train_pairs):

        input_state_nhwc, target_state_nhwc, input_valid_mask, target_valid_mask, debug_message = build_pair_tensors(
            pair,
            max_h,
            max_w,
            debug=True,
            pair_name=f"train pair {pair_index}",
        )

        if debug_message is not None:
            padding_log_lines.append(debug_message)

        spatial_mask = jnp.maximum(input_valid_mask, target_valid_mask)

        input_states.append(input_state_nhwc)
        target_states.append(target_state_nhwc)
        target_masks.append(target_valid_mask)
        spatial_masks.append(spatial_mask)

    model, initial_params = initialize_model(
        rng_key,
        input_states[0],
        fire_rate=fire_rate,
    )

    scheduler_names = ["step_decay_2000"]

    for scheduler_name in scheduler_names:
        params = tree_util.tree_map(lambda x: x.copy(), initial_params)

        # Expand each single input state from shape (1, H, W, C) to pool of parallel state (POOL_SIZE, H, W, C).
        state_pools = []
        for i in range(num_train_pairs):
            pool = jnp.tile(input_states[i], (POOL_SIZE, 1, 1, 1))
            pool_mask = jnp.tile(spatial_masks[i], (POOL_SIZE, 1, 1, 1))
            pool = apply_spatial_mask(pool, pool_mask)
            state_pools.append(pool)

        scheduler_tag = scheduler_name
        lr_log_dir = os.path.join(output_dirs["log_root"], task_name, "lr_runs", scheduler_tag)

        trace_train_dir = os.path.join(output_dirs["base_trace_dir"], scheduler_tag, "train")
        trace_test_dir = os.path.join(output_dirs["base_trace_dir"], scheduler_tag, "test")
        io_train_dir = os.path.join(output_dirs["base_io_dir"], scheduler_tag, "train")
        io_test_dir = os.path.join(output_dirs["base_io_dir"], scheduler_tag, "test")

        for directory in [lr_log_dir, trace_train_dir, trace_test_dir, io_train_dir, io_test_dir]:
            os.makedirs(directory, exist_ok=True)

        lr_log_csv = os.path.join(lr_log_dir, "lr_and_loss.csv")
        with open(lr_log_csv, "w") as file:
            file.write("iter,lr,loss,log_loss,accuracy\n")

        padding_log_path = os.path.join(lr_log_dir, "padding_debug.txt")

        lr_schedule = make_scheduler(scheduler_name, num_iters)
        optimizer = optax.adamw(learning_rate=lr_schedule)
        opt_state = optimizer.init(params)

        @partial(jax.jit, static_argnums=(5,))
        def train_one_step(
                params,
                opt_state,
                rng_key,
                state_pools,
                target_states,
                num_unroll_steps,
                fire_rate,
                step_size,
                collect_trace: bool,
                target_masks,
                spatial_masks,
        ):
            """
            Run one training iteration and update the model parameters across all training pairs.

            Returns:
                new_params, new_opt_state, loss, (
                    final_state_list,
                    pool_loss_list,
                    state_trace_list
                )
            """

            def loss_fn(current_params, current_key):
                total_loss = 0.0
                loop_key = current_key

                final_state_list = []
                pool_loss_list = []
                state_trace_list = []

                # Loop over all training pairs in this task.
                for pair_index in range(num_train_pairs):

                    loop_key, pair_key = jax.random.split(loop_key)

                    pool_state_nhwc = state_pools[pair_index]

                    # Build the target state pool and the matching masks.
                    target_pool_nhwc = jnp.tile(
                        target_states[pair_index],
                        (POOL_SIZE, 1, 1, 1),
                    )

                    target_pool_mask = jnp.tile(
                        target_masks[pair_index],
                        (POOL_SIZE, 1, 1, 1),
                    )

                    spatial_pool_mask = jnp.tile(
                        spatial_masks[pair_index],
                        (POOL_SIZE, 1, 1, 1),
                    )

                    loop_key, rotation_key = jax.random.split(loop_key)
                    rotation_index = jax.random.randint(rotation_key, (), 0, 4, dtype=jnp.int32)
                    rotation_value = rotation_index.astype(jnp.float32) / 3.0

                    input_state_with_rotation = pool_state_nhwc.at[..., ROTATION_CHANNEL_INDEX].set(rotation_value)
                    target_state_with_rotation = target_pool_nhwc.at[..., ROTATION_CHANNEL_INDEX].set(rotation_value)

                    final_state, state_trace = unroll_nca(
                        current_params,
                        model,
                        pair_key,
                        input_state_with_rotation,
                        steps=num_unroll_steps,
                        fire_rate=fire_rate,
                        step_size=step_size,
                        collect_trace=collect_trace,
                        spatial_mask=spatial_pool_mask,
                        visible_target_mask=target_pool_mask,
                        rotation_index=rotation_index,
                    )

                    pool_element_loss = masked_mse_rgba_per_state(
                        final_state,
                        target_state_with_rotation,
                        target_pool_mask,
                    )


                    total_loss += jnp.mean(pool_element_loss)

                    final_state_list.append(final_state)
                    pool_loss_list.append(pool_element_loss)
                    state_trace_list.append(state_trace)

                mean_loss = total_loss / jnp.maximum(1, num_train_pairs)

                return mean_loss, (
                    tuple(final_state_list),
                    tuple(pool_loss_list),
                    tuple(state_trace_list),
                )

            (loss, training_outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, rng_key)

            grads = normalize_gradient_tree(grads)

            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return new_params, new_opt_state, loss, training_outputs

        training_key = rng_key
        # Track the current scheduler step for learning-rate logging.
        optimizer_step = 0

        for iteration_index in range(1, num_iters + 1):

            training_key, step_key = jax.random.split(training_key)
            num_unroll_steps = int(jax.random.randint(step_key, (), MIN_UNROLL_STEPS, MAX_UNROLL_STEPS))

            # Store traces only on the final iteration for visualization.
            collect_trace = iteration_index == num_iters

            params, opt_state, loss, training_outputs = train_one_step(
                params,
                opt_state,
                training_key,
                state_pools,
                target_states,
                num_unroll_steps,
                fire_rate,
                step_size,
                collect_trace,
                target_masks,
                spatial_masks,
            )

            final_state_list, pool_loss_list, state_trace_list = training_outputs

            train_metrics = compute_train_metrics(
                loss=loss,
                final_state_list=final_state_list,
                target_states=target_states,
                target_masks=target_masks,
            )

            current_lr = float(lr_schedule(optimizer_step))
            save_train_metrics_csv(
                lr_log_csv=lr_log_csv,
                iteration_index=iteration_index,
                current_lr=current_lr,
                loss=loss,
                metrics=train_metrics,
            )

            optimizer_step += 1

            state_pools = update_state_pools(
                final_state_list=final_state_list,
                pool_loss_list=pool_loss_list,
                input_states=input_states,
                target_states=target_states,
                state_trace_list=state_trace_list,
                collect_trace=collect_trace,
                iteration_index=iteration_index,
                num_unroll_steps=num_unroll_steps,
                trace_train_dir=trace_train_dir,
                io_train_dir=io_train_dir,
            )


        plot_lr_and_loss(lr_log_csv, lr_log_dir)

        task_runtime_seconds = time.time() - task_start_time

        evaluate_on_test_pair(
            params=params,
            model=model,
            test_pairs=test_pairs,
            rng_key=rng_key,
            fire_rate=fire_rate,
            step_size=step_size,
            lr_log_dir=lr_log_dir,
            trace_test_dir=trace_test_dir,
            io_test_dir=io_test_dir,
            task_name=task_name,
            run_id=run_id,
            task_runtime_seconds=task_runtime_seconds,
            max_h=max_h,
            max_w=max_w,
            padding_log_lines=padding_log_lines,
        )
        save_padding_debug_log(padding_log_lines, padding_log_path)

    return params