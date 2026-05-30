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
# - Neural Cellular Automata with visible and hidden state channels.
# - Fixed local perception using identity/Sobel-style filters.
# - Repeated NCA unrolling, stochastic fire-rate updates, state-pool training,
# and gradient normalization.
#
# ARC-specific modifications:
# - ARC grid and palette representation.
# - Extended fixed filters and learnable depthwise perception.
# - Cell-wise selector that chooses between fixed-filter and learnable-perception update rules.
# - ARC-specific loss, accuracy, evaluation, trace saving, and logging.
# - JAX/Flax/Optax implementation.


import glob
import json
import os
import time
import csv
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from jax import tree_util
from tqdm import tqdm


# ---------------- Configuration ----------------

CHANNELS = 50
POOL_SIZE = 8
NUM_TRAIN_ITERS = 3000
VISIBLE_CHANNELS = 4
HIDDEN_CHANNELS = CHANNELS - VISIBLE_CHANNELS

MIN_UNROLL_STEPS = 32
MAX_UNROLL_STEPS = 64

# ---------------- Data utilities ----------------
PALETTE = np.array(
    [
        [0, 0, 0],        # 0: black (background)
        [0, 0, 255],      # 1: blue
        [255, 0, 0],      # 2: red
        [0, 255, 0],      # 3: green
        [255, 255, 0],    # 4: yellow
        [128, 128, 128],  # 5: gray
        [255, 0, 255],    # 6: magenta
        [255, 165, 0],    # 7: orange
        [0, 255, 255],    # 8: cyan
        [128, 0, 0],      # 9: maroon
    ],
    dtype=np.float32,) / 255.0

def rgb_to_palette_index(rgb_image: jnp.ndarray) -> jnp.ndarray:
    """
    Map RGB values to the nearest ARC palette index.

    This function assigns each RGB pixel to the closest color in the fixed ARC
    palette using squared Euclidean distance.

    Args:
        rgb_image: RGB tensor with shape (..., H, W, 3).

    Returns:
        Tensor of palette indices with shape (..., H, W).
    """

    # ARC palette with shape (10, 3), where 10 is the number of ARC colors.
    palette = jnp.asarray(PALETTE)

    # Add a palette-comparison dimension (..., H, W, 1, 3).
    # rgb_expanded holds the RGB value of each pixel.
    rgb_expanded = rgb_image[..., None, :]

    # Expand palette for broadcasting across all pixels. Resulting shape: (1, 1, 1, 10, 3).
    # palette_expanded holds all palette colors
    palette_expanded = palette[None, None, None, :, :]

    # Squared distance from each pixel to each palette color.
    squared_distance = jnp.sum((rgb_expanded - palette_expanded) ** 2, axis=-1)

    # Choose the nearest palette color index. Output shape: (..., H, W).
    return jnp.argmin(squared_distance, axis=-1)


def palette_index_to_rgb(palette_index: jnp.ndarray) -> jnp.ndarray:
    """
    Convert ARC palette indices to RGB values in [0, 1].
    """
    palette_index = palette_index.astype(jnp.int32)
    return jnp.asarray(PALETTE)[palette_index]


def snap_to_palette_rgb(state_nhwc: jnp.ndarray) -> jnp.ndarray:
    """
    Snap the RGB channels of an NCA state tensor to the nearest ARC palette color.
    """
    palette_index = rgb_to_palette_index(state_nhwc[..., :3])
    snapped_rgb = palette_index_to_rgb(palette_index)
    return state_nhwc.at[..., :3].set(snapped_rgb)


def arc_to_nca(grid_2d: np.ndarray, num_hidden: int = HIDDEN_CHANNELS, pad_value: int = -1) -> jnp.ndarray:
    """
    Convert a 2D ARC grid to the NCA state representation.

    The returned tensor contains four visible channels (RGB + alpha)
    followed by a configurable number of hidden channels.

    Args:
        grid_2d: Integer ARC grid of shape (H, W).
        num_hidden: Number of hidden channels to append.
        pad_value: Value used for padded cells, if padding is present.

    Returns:
        A JAX array of shape (C, H, W), where C is the total number of channels (4 visible + num_hidden =50).
    """
    grid = np.asarray(grid_2d, dtype=np.int32)
    height, width = grid.shape

    rgb = PALETTE[np.clip(grid, 0, 9)]
    alpha = ((grid > 0) & (grid != pad_value)).astype(np.float32)[..., None]

    visible = np.concatenate([rgb, alpha], axis=-1)
    visible = np.moveaxis(visible, -1, 0).astype(np.float32)

    hidden = np.zeros((num_hidden, height, width), dtype=np.float32)
    state = np.concatenate([visible, hidden], axis=0)

    return jnp.array(state)


def extract_rgb_image(state_nhwc: jnp.ndarray) -> np.ndarray:
    """
    Extract the RGB image from the first leading element of an NHWC state tensor
    and return it as a NumPy array.
    """
    rgb = jnp.clip(state_nhwc[0, ..., :3], 0.0, 1.0)
    return np.asarray(rgb, dtype=np.float32)


def extract_rgb_images(state_nhwc: jnp.ndarray) -> list[np.ndarray]:
    """
    Extract RGB images from all leading elements of an NHWC state tensor .
    """
    rgb_tensor = jnp.clip(state_nhwc[..., :3], 0.0, 1.0)
    rgb_array = np.asarray(rgb_tensor, dtype=np.float32)
    return [rgb_array[i] for i in range(rgb_array.shape[0])]


def extract_trace_rgb(state_trace: jnp.ndarray) -> np.ndarray:
    """
    Extract RGB channels from a full state history.

    Args:
        state_trace: Tensor with shape (T, N, H, W, C).

    Returns:
        NumPy array with shape (T, N, H, W, 3).
    """
    rgb_tensor = jnp.clip(state_trace[..., :3], 0.0, 1.0)
    return np.asarray(rgb_tensor , dtype=np.float32)



# ---------------- I/O utilities ----------------

def trace_to_frame_list(trace_rgb: np.ndarray) -> list:
    """
    Convert a trace array of shape (T, H, W, 3) into a list of RGB frames.
    """
    trace_rgb = np.asarray(trace_rgb)
    trace_rgb = np.clip(trace_rgb, 0.0, 1.0)
    return [trace_rgb[t] for t in range(trace_rgb.shape[0])]

def save_trace_json(rgb_frames: list, output_path: str) -> None:
    """
    Save a sequence of RGB frames to a JSON file.

    Each frame is stored as an (H, W, 3) array, and the full sequence is written
    as a JSON list.

    Args:
        rgb_frames: List of RGB arrays with shape (H, W, 3).
        output_path: Destination JSON file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as file:
        file.write("[\n")
        for index, frame in enumerate(rgb_frames):
            json.dump(frame.tolist(), file)
            file.write(",\n" if index < len(rgb_frames) - 1 else "\n")
        file.write("]\n")


def save_io_bundle_json(
    input_images: list,
    target_images: list,
    predicted_images: list,
    output_path: str,
) -> None:
    """
    Save input, target, and predicted RGB images to a JSON file.

    Args:
        input_images: List of input RGB images.
        target_images: List of target RGB images.
        predicted_images: List of predicted RGB images.
        output_path: Destination JSON file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bundle = {
        "input": [image.tolist() for image in input_images],
        "target": [image.tolist() for image in target_images],
        "pred": [image.tolist() for image in predicted_images],
    }

    with open(output_path, "w") as file:
        json.dump(bundle, file)


def save_training_trace_outputs(
    iteration_index: int,
    pair_index: int,
    num_unroll_steps: int,
    input_state,
    target_state,
    final_state,
    state_trace,
    trace_output_dir: str,
    io_output_dir: str,
) -> None:
    """
    Save training traces and input/target/prediction bundles for one training pair.
    """

    snapped_trace = snap_to_palette_rgb(state_trace)
    trace_rgb = extract_trace_rgb(snapped_trace)
    _, pool_size, _, _, _ = trace_rgb.shape

    for pool_index in range(pool_size):
        rgb_frames = trace_to_frame_list(trace_rgb[:, pool_index])
        filename = f"iter_{iteration_index:04d}_pair{pair_index}_P{pool_index}_{num_unroll_steps}x.json"
        save_trace_json(rgb_frames, os.path.join(trace_output_dir, filename))

    input_rgb = extract_rgb_image(input_state)
    target_rgb = extract_rgb_image(target_state)
    predicted_rgb = extract_rgb_images(snap_to_palette_rgb(final_state))

    input_images = [input_rgb] * pool_size
    target_images = [target_rgb] * pool_size

    io_filename = f"iter_{iteration_index:04d}_pair{pair_index}_{num_unroll_steps}x_io.json"
    save_io_bundle_json(
        input_images,
        target_images,
        predicted_rgb,
        os.path.join(io_output_dir, io_filename),
    )

def save_test_trace_outputs(
    input_state_nhwc: jnp.ndarray,
    target_state_nhwc: jnp.ndarray,
    final_state: jnp.ndarray,
    state_trace: jnp.ndarray,
    trace_test_dir: str,
    io_test_dir: str,
) -> None:
    """
    Save test trace frames and final input/target/prediction bundle.
    """

    snapped_trace = snap_to_palette_rgb(state_trace)
    trace_rgb = extract_trace_rgb(snapped_trace)[:, 0]
    rgb_frames = trace_to_frame_list(trace_rgb)
    save_trace_json(rgb_frames, os.path.join(trace_test_dir, "iter_final_test.json"))

    input_rgb = extract_rgb_image(input_state_nhwc)
    target_rgb = extract_rgb_image(target_state_nhwc)
    predicted_rgb = extract_rgb_image(snap_to_palette_rgb(final_state))

    save_io_bundle_json(
        [input_rgb],
        [target_rgb],
        [predicted_rgb],
        os.path.join(io_test_dir, "iter_final_test_io.json"),
    )

def save_train_metrics_csv(
    lr_log_csv: str,
    iteration_index: int,
    current_lr: float,
    loss: jnp.ndarray,
    log_loss: float,
    mean_accuracy: float,
) -> None:
    """
    Append one training metrics row to the training CSV log.
    """
    with open(lr_log_csv, "a") as file:
        file.write(
            f"{iteration_index},{current_lr:.8e},{float(loss):.8f},"
            f"{log_loss:.8f},{mean_accuracy:.4f}\n"
        )

def save_test_metrics_csv(
    lr_log_dir: str,
    step_loss: jnp.ndarray,
    step_log_loss: jnp.ndarray,
    step_accuracy: jnp.ndarray,
    final_loss: float,
    final_log_loss: float,
    final_accuracy: float,
) -> None:
    """
    Save per-step and final test metrics to CSV.
    """
    test_metrics_csv = os.path.join(lr_log_dir, "test_metrics.csv")

    with open(test_metrics_csv, "w") as file:
        file.write("step,loss,log_loss,acc\n")
        for step_index in range(step_loss.shape[0]):
            file.write(
                f"{step_index},{float(step_loss[step_index]):.10f},"
                f"{float(step_log_loss[step_index]):.10f},"
                f"{float(step_accuracy[step_index]):.4f}\n"
            )

        file.write(
            f"final,{final_loss:.10f},{final_log_loss:.10f},{final_accuracy:.4f}\n"
        )


def plot_lr_and_loss(log_csv_path: str, output_dir: str) -> None:
    """
    Plot learning rate, loss, log-loss, and accuracy curves from a CSV log file.

    The CSV file is expected to contain the columns:
    iter, lr, loss, log_loss, accuracy

    Missing optional values are handled gracefully when possible.

    Args:
        log_csv_path: Path to the training log CSV file.
        output_dir: Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    steps, learning_rates, losses = [], [], []
    log_losses, accuracies = [], []

    with open(log_csv_path, newline="") as file:
        reader = csv.reader(file)
        next(reader, None)

        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue

            try:
                step = int(row[0])
                learning_rate = float(row[1])
                loss = float(row[2])
            except (ValueError, IndexError):
                continue

            steps.append(step)
            learning_rates.append(learning_rate)
            losses.append(loss)

            if len(row) >= 4 and row[3].strip():
                try:
                    log_losses.append(float(row[3]))
                except ValueError:
                    pass
            elif log_losses:
                log_losses.append(log_losses[-1])

            if len(row) >= 5 and row[4].strip():
                try:
                    accuracies.append(float(row[4]))
                except ValueError:
                    pass
            elif accuracies:
                accuracies.append(accuracies[-1])

    if steps:
        plt.figure()
        plt.plot(steps, learning_rates)
        plt.xlabel("Iteration")
        plt.ylabel("Learning rate")
        plt.title("Learning rate schedule")
        plt.savefig(os.path.join(output_dir, "lr.png"))
        plt.close()

        plt.figure()
        plt.plot(steps, losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training loss")
        plt.savefig(os.path.join(output_dir, "loss.png"))
        plt.close()

    if log_losses and len(log_losses) == len(steps):
        plt.figure()
        plt.plot(steps, log_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Log loss")
        plt.title("Training log-loss")
        plt.savefig(os.path.join(output_dir, "log_loss.png"))
        plt.close()

    if accuracies and len(accuracies) == len(steps):
        plt.figure()
        plt.plot(steps, accuracies)
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Training accuracy")
        plt.savefig(os.path.join(output_dir, "accuracy.png"))
        plt.close()


# ---------------- Model ----------------

def fixed_filter_perception(
    state_nhwc: jnp.ndarray,
    angle: float = 0.0,
) -> jnp.ndarray:
    """
    Compute fixed-filter perception features for the NCA state.

    This perception module applies a set of predefined depthwise filters to each
    channel independently. The filter bank includes:
    - identity
    - rotated Sobel x/y filters
    - Laplacian-style filters
    - blur and cross-shaped filters
    - two fixed Gabor filters

    Args:
        state_nhwc: State tensor with shape (N, H, W, C).
        angle: Rotation angle used for the Sobel filters.

    Returns:
        Concatenated perception features with shape (N, H, W, K * C), where K is
        the number of fixed filters applied per channel.
    """
    _, _, _, channels = state_nhwc.shape

    base = jnp.array([0, 1, 0], dtype=jnp.float32)
    identity_kernel = jnp.outer(base, base)

    sobel_x = (
        jnp.outer(
            jnp.array([1, 2, 1], dtype=jnp.float32),
            jnp.array([-1, 0, 1], dtype=jnp.float32),
        )
        / 8.0
    )
    sobel_y = sobel_x.T

    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)

    rotated_sobel_x = cos_angle * sobel_x - sin_angle * sobel_y
    rotated_sobel_y = sin_angle * sobel_x + cos_angle * sobel_y


    laplacian_8  = jnp.array(
        [
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1],
        ],
        dtype=jnp.float32,
    ) / 8.0

    blur_kernel = jnp.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=jnp.float32,
    ) / 9.0

    cross_kernel = jnp.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=jnp.float32,
    ) / 5.0

    def gabor_kernel(
        kernel_size: int = 5,
        sigma: float = 1.5,
        theta: float = 0.0,
        wavelength: float = 3.0,
        gamma: float = 0.5,
        phase: float = 0.0,
    ) -> jnp.ndarray:
        """
        Create a normalized zero-mean Gabor filter.
        """
        axis = jnp.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        xx, yy = jnp.meshgrid(axis, axis, indexing="xy")

        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        x_theta = xx * cos_theta + yy * sin_theta
        y_theta = -xx * sin_theta + yy * cos_theta

        gaussian = jnp.exp(
            -(x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2)
        )
        wave = jnp.cos(2 * jnp.pi * x_theta / wavelength + phase)

        kernel = gaussian * wave
        kernel = kernel - jnp.mean(kernel)
        kernel = kernel / (jnp.sqrt(jnp.sum(kernel * kernel)) + 1e-8)
        return kernel.astype(jnp.float32)

    gabor_0 = gabor_kernel(kernel_size=5, theta=0.0)
    gabor_45 = gabor_kernel(kernel_size=5, theta=jnp.pi / 4)

    def tile_depthwise_kernel(kernel: jnp.ndarray) -> jnp.ndarray:
        """
        Tile a spatial kernel across all channels for depthwise convolution.
        """
        kernel = kernel[:, :, None, None]
        kernel = jnp.repeat(kernel, channels, axis=3)
        return kernel

    identity_dw = tile_depthwise_kernel(identity_kernel)
    sobel_x_dw = tile_depthwise_kernel(rotated_sobel_x)
    sobel_y_dw = tile_depthwise_kernel(rotated_sobel_y)
    laplacian_dw = tile_depthwise_kernel(laplacian_8)
    cross_dw = tile_depthwise_kernel(cross_kernel)
    blur_dw = tile_depthwise_kernel(blur_kernel)
    gabor_0_dw = tile_depthwise_kernel(gabor_0)
    gabor_45_dw = tile_depthwise_kernel(gabor_45)

    def depthwise_convolution(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.conv_general_dilated(
            lhs=x,
            rhs=kernel,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=channels,
        )

    identity_features = depthwise_convolution(state_nhwc, identity_dw)
    sobel_x_features = depthwise_convolution(state_nhwc, sobel_x_dw)
    sobel_y_features = depthwise_convolution(state_nhwc, sobel_y_dw)
    laplacian_features = depthwise_convolution(state_nhwc, laplacian_dw)
    cross_features = depthwise_convolution(state_nhwc, cross_dw)
    blur_features = depthwise_convolution(state_nhwc, blur_dw)
    gabor_0_features = depthwise_convolution(state_nhwc, gabor_0_dw)
    gabor_45_features = depthwise_convolution(state_nhwc, gabor_45_dw)

    return jnp.concatenate(
        [
            identity_features,
            sobel_x_features,
            sobel_y_features,
            laplacian_features,
            cross_features,
            blur_features,
            gabor_0_features,
            gabor_45_features,
        ],
        axis=-1,
    )


class LearnablePerception(nn.Module):
    """
    Learnable depthwise perception expert used by the selector NCA.
    """

    channels: int
    filters_per_channel: int = 4
    kernel_size: int = 3

    @nn.compact
    def __call__(self, state_nhwc: jnp.ndarray) -> jnp.ndarray:
        """
        Apply learnable depthwise perception filters to the NCA state.

        Args:
            state_nhwc: Input state tensor with shape (N, H, W, C).

        Returns:
            Learnable perception features with shape
            (N, H, W, C * filters_per_channel).
        """
        return nn.Conv(
            features=self.channels * self.filters_per_channel,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="SAME",
            feature_group_count=self.channels,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.05),
        )(state_nhwc)


class NCAModel(nn.Module):
    """
    Mixed-rule NCA model with a cell-wise selector.

    Each cell chooses between two update rules:
    1. a fixed-filter NCA rule
    2. a learnable-perception NCA rule

    During training, the selector uses straight-through Gumbel-softmax.
    During evaluation, it uses deterministic argmax selection.
    """

    channels: int
    fire_rate: float = 0.5
    num_rules: int = 2

    @nn.compact
    def __call__(
        self,
        state_nhwc: jnp.ndarray,
        *,
        angle: float = 0.0,
        step_size: float = 1.0,
        fire_rate: Optional[float] = None,
        tau: float = 1.0,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Apply one selector-based NCA update step.

        Args:
            state_nhwc: Input state tensor with shape (N, H, W, C).
            angle: Rotation angle used by the fixed-filter expert.
            step_size: Scale factor for the residual state update.
            fire_rate: Optional override for the stochastic cell update rate.
            tau: Temperature for the Gumbel-softmax selector during training.
            training: If True, use stochastic Gumbel-softmax selection. If False,
                use deterministic argmax selection.

        Returns:
            Updated state tensor with the same shape as the input.
        """
        if fire_rate is None:
            fire_rate = self.fire_rate

        selector_logits = nn.Conv(
            features=self.num_rules,
            kernel_size=(1, 1),
            padding="SAME",
        )(state_nhwc)

        if training:
            gumbel_rng = self.make_rng("gumbel")
            uniform_noise = jax.random.uniform(
                gumbel_rng,
                selector_logits.shape,
                minval=1e-6,
                maxval=1.0 - 1e-6,
            )
            gumbel_noise = -jnp.log(-jnp.log(uniform_noise))
            selector_soft = jax.nn.softmax((selector_logits + gumbel_noise) / tau, axis=-1)

            selected_rule = jnp.argmax(selector_soft, axis=-1)
            selector_hard = jax.nn.one_hot(selected_rule, self.num_rules)

            # Straight-through estimator: hard rule choice in the forward pass,
            # but gradients follow the soft selector probabilities.
            selector_weights = jax.lax.stop_gradient(selector_hard - selector_soft) + selector_soft
        else:
            selected_rule = jnp.argmax(selector_logits, axis=-1)
            selector_weights = jax.nn.one_hot(selected_rule, self.num_rules)

        fixed_features = fixed_filter_perception(state_nhwc, angle=angle)
        fixed_delta = self._update_block(
            jnp.concatenate([fixed_features, state_nhwc], axis=-1)
        )

        learned_features = LearnablePerception(
            channels=self.channels,
            filters_per_channel=4,
        )(state_nhwc)
        learned_delta = self._update_block(
            jnp.concatenate([learned_features, state_nhwc], axis=-1)
        )

        state_delta = (
            selector_weights[..., 0:1] * fixed_delta
            + selector_weights[..., 1:2] * learned_delta
        )

        fire_rng = self.make_rng("fire")
        update_mask = (
            jax.random.uniform(fire_rng, shape=state_nhwc[..., :1].shape) <= fire_rate
        ).astype(jnp.float32)

        return state_nhwc + step_size * state_delta * update_mask

    def _update_block(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Shared update-network structure used by each rule expert.
        """
        hidden = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            padding="SAME",
        )(features)
        hidden = nn.relu(hidden)
        return nn.Conv(
            features=self.channels,
            kernel_size=(1, 1),
            padding="SAME",
            kernel_init=nn.initializers.zeros,
        )(hidden)


def unroll_nca(
    params,
    model,
    rng_key,
    initial_state_nhwc,
    steps: int = 64,
    fire_rate: float = 0.5,
    angle: float = 0.0,
    step_size: float = 1.0,
    collect_trace: bool = False,
    tau: float = 1.0,
    training: bool = True,
):
    """
    Unroll the selector-based NCA for a fixed number of steps.

    Args:
        params: Model parameters.
        model: NCA model.
        rng_key: JAX random key.
        initial_state_nhwc: Initial state tensor with shape (N, H, W, C).
        steps: Number of update steps.
        fire_rate: Stochastic cell update rate.
        angle: Rotation angle for the fixed-filter expert.
        step_size: Scale factor for each residual update.
        collect_trace: Whether to store all intermediate states for visualization.
        tau: Temperature used by the Gumbel-softmax selector during training.
        training: If True, use stochastic Gumbel-softmax selection. If False,
            use deterministic argmax selection.

    Returns:
        A tuple (final_state, state_trace), where:
        - final_state has shape (N, H, W, C)
        - state_trace has shape (steps, N, H, W, C)
    """
    num_states, height, width, channels = initial_state_nhwc.shape

    state_trace = jnp.zeros(
        (steps, num_states, height, width, channels),
        dtype=initial_state_nhwc.dtype,
    )

    def body_fn(step_index, carry):
        current_state, current_key, trace_buffer = carry
        current_key, step_key = jax.random.split(current_key)
        fire_key, gumbel_key = jax.random.split(step_key)

        next_state = model.apply(
            {"params": params},
            current_state,
            rngs={"fire": fire_key, "gumbel": gumbel_key},
            angle=angle,
            step_size=step_size,
            fire_rate=fire_rate,
            tau=tau,
            training=training,
        )

        trace_buffer = jax.lax.cond(
            collect_trace,
            lambda buffer: buffer.at[step_index].set(next_state),
            lambda buffer: buffer,
            trace_buffer,
        )

        return next_state, current_key, trace_buffer

    final_state, _, state_trace = jax.lax.fori_loop(
        0,
        steps,
        body_fn,
        (initial_state_nhwc, rng_key, state_trace),
    )

    return final_state, state_trace

# ---------------- Loss and metrics ----------------

def mse_rgba(predicted_state: jnp.ndarray, target_state: jnp.ndarray) -> jnp.ndarray:
    """
    Compute mean squared error (MSE) over the first four visible channels (RGBA).

    Args:
        predicted_state: Predicted NCA state tensor.
        target_state: Target NCA state tensor.

    Returns:
        Mean squared error over channels 0-3.
    """
    diff = predicted_state[..., :4] - target_state[..., :4]
    return jnp.mean(diff ** 2)

def mse_rgba_per_state(
    predicted_state: jnp.ndarray,
    target_state: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute RGBA MSE for each state in the first dimension.

    Args:
        predicted_state: Predicted state tensor with shape (N, H, W, C).
        target_state: Target state tensor with shape (N, H, W, C).

    Returns:
        Per-state MSE with shape (N,).
    """
    diff = predicted_state[..., :4] - target_state[..., :4]
    return jnp.mean(diff ** 2, axis=(1, 2, 3))

def mse_rgba_per_step(
    state_trace: jnp.ndarray,
    target_state_nhwc: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute RGBA MSE for each step in a state history.

    Args:
        state_trace: State trace with shape (T, N, H, W, C).
        target_state_nhwc: Target state with shape (N, H, W, C).

    Returns:
        Per-step MSE with shape (T,).
    """
    num_steps = state_trace.shape[0]
    trace_rgba = state_trace[..., :4]
    target_rgba = jnp.tile(target_state_nhwc[..., :4], (num_steps, 1, 1, 1, 1))
    return jnp.mean((trace_rgba - target_rgba) ** 2, axis=(1, 2, 3, 4))

def safe_log_loss(loss_value: jnp.ndarray) -> jnp.ndarray:

    """Compute log(loss) with a small floor."""
    return jnp.log(jnp.maximum(loss_value, 1e-12))


def compute_train_metrics(
    loss: jnp.ndarray,
    final_state_list,
    target_states,
) -> tuple[float, float]:
    """
    Compute training log-loss and mean palette accuracy across all training pairs.

    Args:
        loss: Mean training loss for the current iteration.
        final_state_list: Final predicted states for all training pairs.
        target_states: Target states for all training pairs.

    Returns:
        log_loss: Log-transformed training loss.
        mean_accuracy: Mean palette-based accuracy across all training pairs.
    """
    log_loss = float(safe_log_loss(loss))

    accuracy_values = []
    for final_state, target_state in zip(final_state_list, target_states):
        predicted_rgb = final_state[..., :3]
        target_rgb = target_state[..., :3]
        accuracy_values.append(palette_accuracy(predicted_rgb, target_rgb))

    mean_accuracy = float(jnp.mean(jnp.array(accuracy_values)))
    return log_loss, mean_accuracy

def compute_test_metrics(
    state_trace: jnp.ndarray,
    final_state: jnp.ndarray,
    target_state_nhwc: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, float, float, jnp.ndarray, float]:
    """
    Compute per-step and final loss/accuracy metrics for test evaluation.
    """

    step_loss = mse_rgba_per_step(state_trace, target_state_nhwc)
    step_log_loss = safe_log_loss(step_loss)

    final_loss = mse_rgba(final_state, target_state_nhwc)
    final_log_loss = safe_log_loss(final_loss)

    final_loss = float(final_loss)
    final_log_loss = float(final_log_loss)

    trace_rgb = state_trace[..., :3]
    target_rgb = target_state_nhwc[..., :3]
    final_rgb = final_state[..., :3]

    step_accuracy = palette_accuracy_per_step(trace_rgb, target_rgb)
    final_accuracy = float(palette_accuracy(final_rgb, target_rgb))

    return step_loss, step_log_loss, final_loss, final_log_loss, step_accuracy, final_accuracy



def palette_accuracy(predicted_rgb: jnp.ndarray, target_rgb: jnp.ndarray) -> jnp.ndarray:
    """
    Compute palette-based accuracy by comparing nearest ARC palette colors.

    Args:
        predicted_rgb: Predicted RGB tensor with shape (..., H, W, 3).
        target_rgb: Target RGB tensor with shape (..., H, W, 3).

    Returns:
        Mean accuracy over all compared pixels.
    """
    predicted_indices = rgb_to_palette_index(predicted_rgb)
    target_indices = rgb_to_palette_index(target_rgb)
    return jnp.mean(predicted_indices == target_indices)

def palette_accuracy_per_step(
    predicted_rgb_trace: jnp.ndarray,
    target_rgb: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute palette-based accuracy for each step in an RGB trace.

    Args:
        predicted_rgb_trace: Predicted RGB trace with shape (T, N, H, W, 3).
        target_rgb: Target RGB tensor with shape (N, H, W, 3).

    Returns:
        Accuracy per step with shape (T,).
    """
    predicted_indices = rgb_to_palette_index(predicted_rgb_trace)
    target_indices = rgb_to_palette_index(target_rgb)
    return jnp.mean(predicted_indices == target_indices, axis=(1, 2, 3))

# ---------------- Training helpers ----------------

def load_arc_task(task_path: str) -> dict:
    """Load a single ARC task from a JSON file."""
    with open(task_path) as file:
        return json.load(file)

def get_task_name(task_path: str) -> str:
    """Extract the ARC task name from the file path."""
    return os.path.splitext(os.path.basename(task_path))[0]

def prepare_output_dirs(task_name: str, run_id: int) -> dict:
    """Create and return all output directories used for a task run."""

    log_root = f"Training_g_Logs_MNCA_selector/run_{run_id}"
    trace_root = f"json_g_traces_MNCA_selector/run_{run_id}"
    io_root = f"json_g_io_MNCA_selector/run_{run_id}"

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

def build_pair_tensors(pair: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert one ARC input-output pair to NCA state tensors in NHWC format.

    Each input and target state has shape (1, H, W, C). This matches the
    general NCA state format (N, H, W, C), where N is the number of parallel
    states. Here N is 1 because each pair starts as a single state. During
    training, this dimension is later expanded to POOL_SIZE when creating the
    state pools. During evaluation, it remains 1.

    Returns:
        input_state_nhwc: Input state tensor with shape (1, H, W, C)
        target_state_nhwc: Target state tensor with shape (1, H, W, C)
    """
    input_grid = np.array(pair["input"], dtype=np.float32)
    target_grid = np.array(pair["output"], dtype=np.float32)

    input_chw = arc_to_nca(input_grid)
    target_chw = arc_to_nca(target_grid)

    input_state_nhwc = jnp.moveaxis(input_chw[None], 1, -1)
    target_state_nhwc = jnp.moveaxis(target_chw[None], 1, -1)

    return input_state_nhwc, target_state_nhwc


def initialize_model(rng_key, input_state_nhwc, fire_rate: float):
    """
    Initialize the selector-based NCA model and parameter tree.
    """
    model = NCAModel(
        channels=CHANNELS,
        fire_rate=fire_rate,
    )

    params = model.init(
        {
            "params": rng_key,
            "fire": jax.random.PRNGKey(1),
            "gumbel": jax.random.PRNGKey(2),
        },
        input_state_nhwc,
        training=True,
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
            save_training_trace_outputs(
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

def normalize_gradient_tree(grads):
    """
    Normalize each gradient leaf independently.
    """
    def normalize_gradient(grad):
        grad_norm = jnp.sqrt(jnp.sum(grad * grad))
        return jnp.where(grad_norm > 0, grad / (grad_norm + 1e-8), grad)

    return tree_util.tree_map(normalize_gradient, grads)

def evaluate_on_test_pair(
    params,
    model,
    test_pairs,
    rng_key,
    fire_rate: float,
    angle: float,
    step_size: float,
    lr_log_dir: str,
    trace_test_dir: str,
    io_test_dir: str,
    task_name: str,
    run_id: int,
) -> None:
    """
    Evaluate the trained model on the first test pair of an ARC task and save
    metrics, trace frames, and the final prediction.
    """
    if not test_pairs:
        return

    input_state_nhwc, target_state_nhwc = build_pair_tensors(test_pairs[0])

    num_test_steps = 63
    _, eval_key = jax.random.split(rng_key)

    final_state, state_trace = unroll_nca(
        params,
        model,
        eval_key,
        input_state_nhwc,
        steps=num_test_steps,
        fire_rate=1.0,
        angle=angle,
        step_size=step_size,
        collect_trace=True,
        tau=1.0,
        training=False,
    )

    (
        step_loss,
        step_log_loss,
        final_loss,
        final_log_loss,
        step_accuracy,
        final_accuracy,
    ) = compute_test_metrics(
        state_trace=state_trace,
        final_state=final_state,
        target_state_nhwc=target_state_nhwc,
    )

    solved = 1 if final_accuracy == 1.0 else 0

    results_dir = "run_results_MNCA_selector"
    os.makedirs(results_dir, exist_ok=True)

    results_csv = os.path.join(results_dir, f"run_{run_id:02d}.csv")
    with open(results_csv, "a") as file:
        file.write(f"{task_name},{solved},{final_accuracy:.6f}\n")

    save_test_metrics_csv(
        lr_log_dir=lr_log_dir,
        step_loss=step_loss,
        step_log_loss=step_log_loss,
        step_accuracy=step_accuracy,
        final_loss=final_loss,
        final_log_loss=final_log_loss,
        final_accuracy=final_accuracy,
    )

    save_test_trace_outputs(
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
    angle=0.0,
    step_size=1.0,
    run_id=0,
):
    """
    Train the NCA model on a single ARC task.
    """
    task = load_arc_task(task_path)
    task_name = get_task_name(task_path)
    output_dirs = prepare_output_dirs(task_name, run_id)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])
    num_train_pairs = len(train_pairs)

    assert num_train_pairs > 0, "Task contains no training pairs."

    input_states = []
    target_states = []

    for pair in train_pairs:
        input_state_nhwc, target_state_nhwc = build_pair_tensors(pair)
        input_states.append(input_state_nhwc)
        target_states.append(target_state_nhwc)

    model, initial_params = initialize_model(
        rng_key,
        input_states[0],
        fire_rate=fire_rate,
    )

    scheduler_names = ["step_decay_2000"]

    for scheduler_name in scheduler_names:
        params = tree_util.tree_map(lambda x: x.copy(), initial_params)

        # Expand each single input state from shape (1, H, W, C) to pool of parallel state (POOL_SIZE, H, W, C).
        state_pools = [
            jnp.tile(input_states[i], (POOL_SIZE, 1, 1, 1))
            for i in range(num_train_pairs)
        ]

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
                angle,
                step_size,
                collect_trace: bool,
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

                for pair_index in range(num_train_pairs):

                    loop_key, pair_key = jax.random.split(loop_key)

                    pool_state_nhwc = state_pools[pair_index]
                    target_pool_nhwc = jnp.tile(
                        target_states[pair_index],
                        (POOL_SIZE, 1, 1, 1),
                    )

                    final_state, state_trace = unroll_nca(
                        current_params,
                        model,
                        pair_key,
                        pool_state_nhwc,
                        steps=num_unroll_steps,
                        fire_rate=fire_rate,
                        angle=angle,
                        step_size=step_size,
                        collect_trace=collect_trace,
                        tau=1.0,
                        training=True,
                    )

                    pool_element_loss = mse_rgba_per_state(final_state, target_pool_nhwc)

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
            iteration_start_time = time.time()

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
                angle,
                step_size,
                collect_trace,
            )

            final_state_list, pool_loss_list, state_trace_list = training_outputs

            log_loss, mean_accuracy = compute_train_metrics(
                loss=loss,
                final_state_list=final_state_list,
                target_states=target_states,
            )

            current_lr = float(lr_schedule(optimizer_step))
            save_train_metrics_csv(
                lr_log_csv=lr_log_csv,
                iteration_index=iteration_index,
                current_lr=current_lr,
                loss=loss,
                log_loss=log_loss,
                mean_accuracy=mean_accuracy,
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


        evaluate_on_test_pair(
            params=params,
            model=model,
            test_pairs=test_pairs,
            rng_key=rng_key,
            fire_rate=fire_rate,
            angle=angle,
            step_size=step_size,
            lr_log_dir=lr_log_dir,
            trace_test_dir=trace_test_dir,
            io_test_dir=io_test_dir,
            task_name=task_name,
            run_id=run_id,
        )

    return params



# ---------------- Main ----------------
def main():
    """Run training over all ARC tasks for the configured number of runs."""
    task_root = "data/test"
    task_paths = sorted(glob.glob(os.path.join(task_root, "*.json")))

    rng_key = jax.random.PRNGKey(0)
    num_runs = 1
    print("Available JAX devices:", jax.devices())

    for run_index in range(1, num_runs + 1):
        print(f"\n===== Run {run_index}/{num_runs} =====")

        results_dir = "run_results_MNCA_selector"
        os.makedirs(results_dir, exist_ok=True)

        results_csv = os.path.join(results_dir, f"run_{run_index:02d}.csv")
        with open(results_csv, "w") as file:
            file.write("task,solved,acc\n")

        for task_path in tqdm(
            task_paths,
            desc=f"Run {run_index} | Training tasks",
            ncols=80,
        ):
            rng_key, task_key = jax.random.split(rng_key)
            train_one_task(
                task_path=task_path,
                rng_key=task_key,
                num_iters=NUM_TRAIN_ITERS,
                run_id=run_index,
            )


if __name__ == "__main__":
    main()
