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
# - Local perception using identity and Sobel-style filters.
# - Rotated Sobel perception idea.
# - Repeated NCA unrolling over multiple update steps.
# - Stochastic cell updates using a fire-rate mask.
# - Residual state updates through a learned 1x1 convolutional update network.
#

# ARC-specific modifications:
# - Reimplemented the model in JAX/Flax.
# - Adapted the state representation for ARC grids and ARC color palettes.
# - Uses ARC input grids as the initial NCA state instead of growing from a seed cell.
# - Adds extended fixed perception filters: Laplacian, blur, cross, and Gabor filters.
# - Adds spatial masks and visible-target masks for padded ARC grids.
# - Predicts ARC output grids instead of growing a fixed target image.



from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


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

    laplacian_8 = jnp.array(
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
            -(x_theta ** 2 + (gamma ** 2) * y_theta ** 2) / (2 * sigma ** 2)
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
        Apply one NCA update step using fixed depthwise perception filters.

        This variant uses a larger fixed perception filter bank than the baseline model.
        The model first extracts fixed-filter perception features from the current
        state, then predicts a residual update with two 1x1 convolutions and a
        ReLU nonlinearity. A stochastic update mask controls which cells are
        updated at this step.

        Args:
            state_nhwc: Input state tensor of shape (N, H, W, C), where N is the
                number of parallel states. N is 1 during evaluation and
                POOL_SIZE during training.
            angle: Rotation angle used by the fixed Sobel perception filters.
            step_size: Scale factor for the residual state update.
            fire_rate: Optional override for the stochastic cell update rate.

        Returns:
            Updated state tensor with the same shape as the input.
        """
        if fire_rate is None:
            fire_rate = self.fire_rate

        perception_features = fixed_filter_perception(state_nhwc, angle=angle)

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

        # Sample a stochastic mask that decides which cells update at this step.
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