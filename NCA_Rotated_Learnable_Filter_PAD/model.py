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
# - Neural Cellular Automata architecture with visible and hidden state channels.
# - Repeated NCA unrolling over multiple update steps.
# - Stochastic cell updates using a fire-rate mask.
# - Residual state updates through a learned 1x1 convolutional update network.
# - The general idea of local NCA perception and local cell updates.
#
# ARC-specific modifications:
# - Reimplemented the model in JAX/Flax.
# - Replaced the original fixed Sobel-style perception with learnable depthwise perception filters.
# - Added rotated learnable perception filters using 0, 90, 180, and 270 degree kernel rotations.
# - Added a rotation-conditioning channel for ARC rotation-aware experiments.
# - Adapted the state representation for ARC grids and ARC color palettes.
# - Uses ARC input grids as the initial NCA state instead of growing from a seed cell.
# - Adds spatial masks and visible-target masks for padded ARC grids.
# - Predicts ARC output grids instead of growing a fixed target image.

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

class LearnablePerception(nn.Module):
    channels: int
    filters_per_channel: int = 4
    kernel_size: int = 3

    @nn.compact
    def __call__(
        self,
        state_nhwc: jnp.ndarray,
        rotation_index: int = 0,
    ) -> jnp.ndarray:
        """
        Apply learnable depthwise perception filters, optionally rotated by
        multiples of 90 degrees.

        Args:
            state_nhwc: Input state tensor with shape (N, H, W, C).
            rotation_index: Integer in {0, 1, 2, 3} selecting 0°, 90°, 180°,
                or 270° kernel rotation.

        Returns:
            Learnable perception features with shape
            (N, H, W, C * filters_per_channel).
        """
        channels = self.channels
        filters_per_channel = self.filters_per_channel
        kernel_size = self.kernel_size

        assert kernel_size == 3, "This implementation assumes 3x3 kernels."

        kernel = self.param(
            "kernel",
            nn.initializers.normal(stddev=0.05),
            (kernel_size, kernel_size, 1, channels * filters_per_channel),
        )

        rotation_index = jnp.asarray(rotation_index, dtype=jnp.int32)

        def rotate_0(kernel_tensor):
            return kernel_tensor

        def rotate_90(kernel_tensor):
            return jnp.rot90(kernel_tensor, k=1, axes=(0, 1))

        def rotate_180(kernel_tensor):
            return jnp.rot90(kernel_tensor, k=2, axes=(0, 1))

        def rotate_270(kernel_tensor):
            return jnp.rot90(kernel_tensor, k=3, axes=(0, 1))

        rotated_kernel = jax.lax.switch(
            rotation_index,
            (rotate_0, rotate_90, rotate_180, rotate_270),
            kernel,
        )

        return jax.lax.conv_general_dilated(
            lhs=state_nhwc,
            rhs=rotated_kernel,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=channels,
        )


class NCAModel(nn.Module):
    channels: int
    fire_rate: float = 0.5
    filters_per_channel: int = 4
    perception_kernel_size: int = 3

    @nn.compact
    def __call__(
        self,
        state_nhwc: jnp.ndarray,
        *,
        step_size: float = 1.0,
        fire_rate: Optional[float] = None,
        rotation_index: int = 0,
    ) -> jnp.ndarray:
        """
        Apply one NCA update step using learnable depthwise perception filters
        and a rotation-conditioning channel.

        The model uses the rotation index in two ways:
        1. to rotate the learnable perception kernels
        2. to store the normalized rotation value in a dedicated state channel

        Args:
            state_nhwc: Input state tensor of shape (N, H, W, C).
            step_size: Scale factor for the residual state update.
            fire_rate: Optional override for the stochastic cell update rate.
            rotation_index: Integer in {0, 1, 2, 3} selecting the rotation
                applied to the learnable perception kernels.

        Returns:
            Updated state tensor with the same shape as the input.
        """

        if fire_rate is None:
            fire_rate = self.fire_rate

        learned_features = LearnablePerception(
            channels=self.channels,
            filters_per_channel=self.filters_per_channel,
            kernel_size=self.perception_kernel_size,
        )(state_nhwc, rotation_index=rotation_index)

        update_features = jnp.concatenate([learned_features, state_nhwc], axis=-1)
        update_features = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            padding="SAME",
        )(update_features)
        update_features = nn.relu(update_features)

        state_delta = nn.Conv(
            features=self.channels,
            kernel_size=(1, 1),
            padding="SAME",
            kernel_init=nn.initializers.zeros,
        )(update_features)

        fire_rng = self.make_rng("fire")
        update_mask = (
            jax.random.uniform(fire_rng, shape=state_nhwc[..., :1].shape) <= fire_rate
        ).astype(jnp.float32)

        updated_state = state_nhwc + step_size * state_delta * update_mask

        return updated_state


def apply_spatial_mask(state_nhwc: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Keep state only inside the valid spatial region.
    Args:
        state_nhwc: State tensor with shape (N, H, W, C).
        valid_mask: Spatial mask with shape (N, H, W, 1) or (1, H, W, 1).

    Returns:
        State tensor with zeros outside the valid area.
    """
    return state_nhwc * valid_mask

def apply_visible_target_mask(
    state_nhwc: jnp.ndarray,
    target_valid_mask: jnp.ndarray,
    visible_channels: int = 4,
) -> jnp.ndarray:
    """
    Keep visible channels only in the target area.

    The remaining channels, including the rotation-conditioning channel and hidden channels,
    are kept unchanged.

    Args:
        state_nhwc: State tensor with shape (N, H, W, C).
        target_valid_mask: Target mask with shape (N, H, W, 1) or (1, H, W, 1).
        visible_channels: Number of visible channels at the start of the state.

    Returns:
        State tensor where visible channels are zero outside the target area.
    """
    visible = state_nhwc[..., :visible_channels] * target_valid_mask
    hidden = state_nhwc[..., visible_channels:]
    return jnp.concatenate([visible, hidden], axis=-1)

def unroll_nca(
    params,
    model,
    rng_key,
    initial_state_nhwc,
    steps: int = 64,
    fire_rate: float = 0.5,
    step_size: float = 1.0,
    collect_trace: bool = False,
    spatial_mask: Optional[jnp.ndarray] = None,
    visible_target_mask: Optional[jnp.ndarray] = None,
    rotation_index: int = 0,
):
    """
    Unroll the NCA for a fixed number of steps.

    Args:
        params: Model parameters.
        model: NCA model.
        rng_key: JAX random key.

        initial_state_nhwc: Initial state tensor of shape (N, H, W, C), where N is the number of
        parallel states processed together. This is 1 during evaluation and
        POOL_SIZE during training.

        steps: Number of update steps.
        fire_rate: Stochastic cell update rate.
        step_size: Scale factor for each residual update.
        collect_trace: Whether to store all intermediate states for visualization.
        spatial_mask: Optional mask used to keep the state inside the valid area.
        visible_target_mask: Optional mask used to keep visible channels inside the target area.
        rotation_index: Integer in {0, 1, 2, 3} selecting the rotation applied
            to the learnable perception kernels during the rollout.

    Returns:
        A tuple (final_state, state_trace), where:
        - final_state has shape (N, H, W, C)
        - state_trace has shape (steps, N, H, W, C)

        If collect_trace is False, the trace buffer is still returned with a fixed
        shape, but it remains filled with zeros.
    """

    # num_states is the first state dimension (N, H, W, C): This is 1 during evaluation and POOL_SIZE during training.

    num_states, height, width, channels = initial_state_nhwc.shape

    # Create a buffer to store the state history after each step.
    # This has one extra dimension for time: (steps, N, H, W, C)
    state_trace = jnp.zeros(
        (steps, num_states, height, width, channels),
        dtype=initial_state_nhwc.dtype,
    )

    def body_fn(step_index, carry):
        # carry stores the values passed from one step to the next
        current_state, current_key, trace_buffer = carry
        current_key, step_key = jax.random.split(current_key)

        # Use a new random key for the stochastic cell update mask at this NCA step.
        rngs = {"fire": step_key}

        # Apply one NCA step to get the next state.
        next_state = model.apply(
            {"params": params},
            current_state,
            rngs=rngs,
            step_size=step_size,
            fire_rate=fire_rate,
            rotation_index=rotation_index,
        )

        # Keep the full state only inside the valid spatial area.
        if spatial_mask is not None:
            next_state = apply_spatial_mask(next_state, spatial_mask)

        # Keep visible channels only inside the target area.
        if visible_target_mask is not None:
            next_state = apply_visible_target_mask(next_state, visible_target_mask)

        current_state = next_state

        # trace_buffer: the buffer used to store state history at this time step if collect_trace is True.
        trace_buffer = jax.lax.cond(
            collect_trace,
            lambda buffer: buffer.at[step_index].set(current_state),
            lambda buffer: buffer,
            trace_buffer,
        )

        return current_state, current_key, trace_buffer

    # Run the loop from step 0 up to step `steps - 1`.
    final_state, _, state_trace = jax.lax.fori_loop(
        0,
        steps,
        body_fn,
        (initial_state_nhwc, rng_key, state_trace),
    )

    return final_state, state_trace