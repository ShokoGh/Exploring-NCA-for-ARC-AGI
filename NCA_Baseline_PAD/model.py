# This file implements a JAX/Flax Neural Cellular Automata model for ARC-AGI.
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
# - Reimplemented the NCA model/training logic in JAX/Flax.
# - Neural Cellular Automata architecture with visible and hidden state channels.
# - Local perception using identity/Sobel-style filters.
# - Repeated NCA unrolling over multiple update steps.
# - Stochastic cell updates using a fire-rate mask.
# - Sample-pool/state-pool based training.
# - Per-parameter gradient normalization for training stability.
#
# ARC-specific modifications:
# - ARC grid and palette representation.
# - Input-output task setup instead of growing a fixed target image from a seed.
# - Padding support for variable-sized ARC grids.
# - Spatial masks and target masks.
# - ARC-specific loss, accuracy, evaluation, trace saving, and logging.
# - JAX/Flax/Optax implementation.

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

def perceive(state_nhwc: jnp.ndarray, angle: float = 0.0) -> jnp.ndarray:
    """
    Compute perception features for the NCA state.

    The function applies three depthwise filters per channel:
    an identity filter and two Sobel-based gradient filters rotated
    by the given angle.

    Args:
        state_nhwc: State tensor of shape (N, H, W, C), where N is the number of
        parallel states processed together. This is 1 during evaluation and
        POOL_SIZE during training.

    Returns:
        A tensor of concatenated perception features with shape (N, H, W, 3 * C).
    """
    num_states, height, width, channels = state_nhwc.shape

    base = jnp.array([0, 1, 0], dtype=jnp.float32)
    identity_kernel = jnp.outer(base, base)

    sobel_x = (
        jnp.outer(jnp.array([1, 2, 1], dtype=jnp.float32),
                  jnp.array([-1, 0, 1], dtype=jnp.float32))
        / 8.0
    )
    sobel_y = sobel_x.T

    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)

    rotated_sobel_x = cos_angle * sobel_x - sin_angle * sobel_y
    rotated_sobel_y = sin_angle * sobel_x + cos_angle * sobel_y

    def tile_depthwise_kernel(kernel: jnp.ndarray) -> jnp.ndarray:
        kernel = kernel[:, :, None, None]
        kernel = jnp.repeat(kernel, channels, axis=3)
        return kernel

    identity_kernel_dw = tile_depthwise_kernel(identity_kernel)
    sobel_x_dw = tile_depthwise_kernel(rotated_sobel_x)
    sobel_y_dw = tile_depthwise_kernel(rotated_sobel_y)

    def depthwise_convolution(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.conv_general_dilated(
            lhs=x,
            rhs=kernel,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=channels,
        )

    identity_features = depthwise_convolution(state_nhwc, identity_kernel_dw)
    sobel_x_features = depthwise_convolution(state_nhwc, sobel_x_dw)
    sobel_y_features = depthwise_convolution(state_nhwc, sobel_y_dw)

    return jnp.concatenate(
        [identity_features, sobel_x_features, sobel_y_features],
        axis=-1,
    )

class NCAModel(nn.Module):
    channels: int
    fire_rate: float = 0.5

    @nn.compact
    def __call__(
        self,
        state_nhwc: jnp.ndarray,
        *,
        angle: float = 0.0,
        step_size: float = 1.0,
        fire_rate: Optional[float] = None,
    ) -> jnp.ndarray:

        """
        Apply one NCA update step to the input state.

        Args:
            state_nhwc: Input state tensor of shape (N, H, W, C), where N is the number of
            parallel states processed together. N is 1 during evaluation and
            POOL_SIZE during training.

            angle: Rotation angle used in the perception filters.
            step_size: Scale factor for the residual state update.
            fire_rate: Optional override for the stochastic cell update rate.

        Returns:
            Updated state tensor with the same shape as the input.
        """
        if fire_rate is None:
            fire_rate = self.fire_rate

        perception_features = perceive(state_nhwc, angle=angle)

        update_features = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            padding="SAME",
        )(perception_features)
        update_features = nn.relu(update_features)

        state_delta = nn.Conv(
            features=self.channels,
            kernel_size=(1, 1),
            padding="SAME",
            kernel_init=nn.initializers.zeros,
        )(update_features)

        # Only a subset of cells update at each step

        fire_rng = self.make_rng("fire")
        update_mask = (
            jax.random.uniform(fire_rng, shape=state_nhwc[..., :1].shape) <= fire_rate
        ).astype(jnp.float32)

        updated_state = state_nhwc + step_size * state_delta * update_mask
        return updated_state


def apply_spatial_mask(state_nhwc: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Keep the full NCA state only inside the valid task region, excluding padded cells.
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

    The hidden channels are kept unchanged.

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
    angle: float = 0.0,
    step_size: float = 1.0,
    collect_trace: bool = False,
    spatial_mask: Optional[jnp.ndarray] = None,
    visible_target_mask: Optional[jnp.ndarray] = None,
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
        angle: Rotation angle for perception filters.
        step_size: Scale factor for each residual update.
        collect_trace: Whether to store all intermediate states for visualization.
        spatial_mask: Optional mask used to keep the state inside the valid area.
        visible_target_mask: Optional mask used to keep visible channels inside the target area.

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
            angle=angle,
            step_size=step_size,
            fire_rate=fire_rate,
        )

        # Keep the full state only inside the valid spatial area.
        if spatial_mask is not None:
            next_state = apply_spatial_mask(next_state, spatial_mask)

        ## Keep visible output channels only inside the valid target region.
        # Hidden channels are allowed to persist outside the target region but still inside the spatial mask.
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