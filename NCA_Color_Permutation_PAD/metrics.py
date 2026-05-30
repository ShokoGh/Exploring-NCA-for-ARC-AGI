import jax.numpy as jnp

from .palette_utils import rgb_to_palette_index, palette_index_to_rgb

def safe_log_loss(loss_value: jnp.ndarray) -> jnp.ndarray:
    """Compute log(loss) with a small floor."""
    return jnp.log(jnp.maximum(loss_value, 1e-12))


def masked_mse_rgba(
    predicted_state: jnp.ndarray,
    target_state: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute masked MSE over RGBA channels.
    Only cells inside the valid mask are used.

    Args:
        predicted_state: Predicted state tensor.
        target_state: Target state tensor.
        valid_mask: Mask with shape (N, H, W, 1).

    Returns:
        Masked mean squared error.
    """
    diff2 = (predicted_state[..., :4] - target_state[..., :4]) ** 2
    masked_diff2 = diff2 * valid_mask

    valid_value_count = jnp.sum(valid_mask) * 4.0
    return jnp.where(valid_value_count > 0, jnp.sum(masked_diff2) / valid_value_count, 0.0)


def masked_mse_rgba_per_state(
    predicted_state: jnp.ndarray,
    target_state: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute masked RGBA MSE for each state.

    Args:
        predicted_state: Predicted state tensor with shape (N, H, W, C).
        target_state: Target state tensor with shape (N, H, W, C).
        valid_mask: Mask with shape (N, H, W, 1).

    Returns:
        Per-state masked MSE with shape (N,).
    """
    diff2 = (predicted_state[..., :4] - target_state[..., :4]) ** 2
    masked_diff2 = diff2 * valid_mask

    masked_error_sum = jnp.sum(masked_diff2, axis=(1, 2, 3))
    valid_value_count = jnp.sum(valid_mask, axis=(1, 2, 3)) * 4.0

    return jnp.where(valid_value_count > 0, masked_error_sum / valid_value_count, 0.0)


def masked_mse_rgba_per_step(
    state_trace: jnp.ndarray,
    target_state_nhwc: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute masked RGBA MSE for each step in the trace.

    Args:
        state_trace: State history with shape (T, N, H, W, C).
        target_state_nhwc: Target state with shape (N, H, W, C).
        valid_mask: Mask with shape (N, H, W, 1).

    Returns:
        Per-step masked MSE with shape (T,).

    """

    trace_rgba = state_trace[..., :4]
    target_rgba = jnp.broadcast_to(target_state_nhwc[..., :4], trace_rgba.shape)

    mask = jnp.broadcast_to(valid_mask[None, ...], trace_rgba.shape[:-1] + (1,))
    diff2 = (trace_rgba - target_rgba) ** 2
    masked_diff2 = diff2 * mask

    masked_error_sum = jnp.sum(masked_diff2, axis=(1, 2, 3, 4))
    valid_value_count = jnp.sum(mask, axis=(1, 2, 3, 4)) * 4.0

    return jnp.where(valid_value_count > 0, masked_error_sum / valid_value_count, 0.0)

def masked_palette_accuracy(
    predicted_rgb: jnp.ndarray,
    target_rgb: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute palette accuracy inside the valid area.

    Args:
        predicted_rgb: Predicted RGB tensor with shape (..., H, W, 3).
        target_rgb: Target RGB tensor with shape (..., H, W, 3).
        valid_mask: Mask with shape (..., H, W, 1).

    Returns:
        Mean palette accuracy over valid cells.
    """
    predicted_indices = rgb_to_palette_index(predicted_rgb)
    target_indices = rgb_to_palette_index(target_rgb)

    valid = valid_mask[..., 0] > 0
    correct = predicted_indices == target_indices

    valid_count = jnp.sum(valid)
    return jnp.where(valid_count > 0, jnp.sum(correct & valid) / valid_count, 1.0)


def masked_palette_accuracy_per_step(
    predicted_rgb_trace: jnp.ndarray,
    target_rgb: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute palette accuracy for each step inside the valid area.

    Args:
        predicted_rgb_trace: Predicted RGB trace with shape (T, N, H, W, 3).
        target_rgb: Target RGB tensor with shape (N, H, W, 3).
        valid_mask: Mask with shape (N, H, W, 1).

    Returns:
        Per-step palette accuracy with shape (T,).
    """

    predicted_indices = rgb_to_palette_index(predicted_rgb_trace)
    target_indices = rgb_to_palette_index(target_rgb)

    valid = valid_mask[..., 0] > 0                      # (N,H,W)
    valid = jnp.broadcast_to(valid[None, ...], predicted_indices.shape)

    correct = predicted_indices == target_indices
    correct_count = jnp.sum(correct & valid, axis=(1, 2, 3))
    valid_count = jnp.sum(valid, axis=(1, 2, 3))

    return jnp.where(valid_count > 0, correct_count / valid_count, 1.0)


def masked_exact_match(
    predicted_rgb: jnp.ndarray,
    target_rgb: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute exact-match accuracy inside the valid area.

    Each state is counted as correct only if all valid cells match. The returned value is averaged over states.

    Args:
        predicted_rgb: Predicted RGB tensor with shape (..., H, W, 3).
        target_rgb: Target RGB tensor with shape (..., H, W, 3).
        valid_mask: Mask with shape (..., H, W, 1).

    Returns:
        Mean exact-match accuracy.
    """

    predicted_indices = rgb_to_palette_index(predicted_rgb)
    target_indices = rgb_to_palette_index(target_rgb)

    valid = valid_mask[..., 0] > 0
    matches = (predicted_indices == target_indices) | (~valid)

    all_match = jnp.all(matches, axis=(-2, -1))
    return jnp.mean(all_match.astype(jnp.float32))


def masked_exact_match_per_step(
    predicted_rgb_trace: jnp.ndarray,
    target_rgb: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute exact-match accuracy for each step inside the valid area.

    For each step, each state is correct only if all valid cells match; the returned value is averaged over states.

    Args:
        predicted_rgb_trace: Predicted RGB trace with shape (T, N, H, W, 3).
        target_rgb: Target RGB tensor with shape (N, H, W, 3).
        valid_mask: Mask with shape (N, H, W, 1).

    Returns:
        Per-step exact-match accuracy with shape (T,).
    """
    predicted_indices = rgb_to_palette_index(predicted_rgb_trace)
    target_indices = rgb_to_palette_index(target_rgb)

    valid = valid_mask[..., 0] > 0
    valid = jnp.broadcast_to(valid[None, ...], predicted_indices.shape)

    matches = (predicted_indices == target_indices) | (~valid)
    return jnp.mean(jnp.all(matches, axis=(2, 3)).astype(jnp.float32), axis=1)


def masked_foreground_accuracy(
    predicted_rgb: jnp.ndarray,
    target_rgb: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute foreground accuracy inside the valid area.

    Only valid target cells with non-zero color are used.

    Args:
        predicted_rgb: Predicted RGB tensor with shape (..., H, W, 3).
        target_rgb: Target RGB tensor with shape (..., H, W, 3).
        valid_mask: Mask with shape (..., H, W, 1).

    Returns:
        Mean foreground accuracy over valid foreground cells.
    """
    predicted_indices = rgb_to_palette_index(predicted_rgb)
    target_indices = rgb_to_palette_index(target_rgb)

    valid = valid_mask[..., 0] > 0
    foreground_mask = (target_indices != 0) & valid
    correct = predicted_indices == target_indices

    foreground_count = jnp.sum(foreground_mask)
    return jnp.where(foreground_count > 0, jnp.sum(correct & foreground_mask) / foreground_count, 1.0)


def masked_foreground_accuracy_per_step(
    predicted_rgb_trace: jnp.ndarray,
    target_rgb: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute foreground accuracy for each step inside the valid area.

    Only valid target cells with non-zero color are used.

    Args:
        predicted_rgb_trace: Predicted RGB trace with shape (T, N, H, W, 3).
        target_rgb: Target RGB tensor with shape (N, H, W, 3).
        valid_mask: Mask with shape (N, H, W, 1).

    Returns:
        Per-step foreground accuracy with shape (T,).
    """
    predicted_indices = rgb_to_palette_index(predicted_rgb_trace)
    target_indices = rgb_to_palette_index(target_rgb)

    valid = valid_mask[..., 0] > 0
    valid = jnp.broadcast_to(valid[None, ...], predicted_indices.shape)

    foreground_mask = (target_indices != 0) & valid
    correct = predicted_indices == target_indices

    foreground_correct = jnp.sum(correct & foreground_mask, axis=(1, 2, 3))
    foreground_count = jnp.sum(foreground_mask, axis=(1, 2, 3))

    return jnp.where(foreground_count > 0, foreground_correct / foreground_count, 1.0)

def masked_palette_loss_rgb_target(
    predicted_state: jnp.ndarray,
    target_state: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute masked RGB loss toward the target palette color.

    Instead of only measuring raw RGBA reconstruction error, this loss
    measures the squared RGB distance to the ARC palette color that the
    target corresponds to. Only valid cells are used.

    Args:
        predicted_state: Predicted state tensor with shape (N, H, W, C).
        target_state: Target state tensor with shape (N, H, W, C).
        valid_mask: Mask with shape (N, H, W, 1).

    Returns:
        Mean masked target-palette RGB loss.
    """
    predicted_rgb = jnp.clip(predicted_state[..., :3], 0.0, 1.0)
    target_rgb = target_state[..., :3]

    target_palette_index = rgb_to_palette_index(target_rgb)
    target_palette_rgb = palette_index_to_rgb(target_palette_index)

    squared_distance = jnp.sum((predicted_rgb - target_palette_rgb) ** 2, axis=-1, keepdims=True)
    masked_squared_distance = squared_distance * valid_mask

    valid_count = jnp.sum(valid_mask)
    return jnp.where(
        valid_count > 0,
        jnp.sum(masked_squared_distance) / valid_count,
        0.0,
    )

def masked_palette_loss_rgb_target_per_step(
    state_trace: jnp.ndarray,
    target_state_nhwc: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute masked target-palette RGB loss for each step in the trace.

    Args:
        state_trace: State history with shape (T, N, H, W, C).
        target_state_nhwc: Target state with shape (N, H, W, C).
        valid_mask: Mask with shape (N, H, W, 1).

    Returns:
        Per-step masked palette loss with shape (T,).
    """
    predicted_rgb = jnp.clip(state_trace[..., :3], 0.0, 1.0)
    target_rgb = target_state_nhwc[..., :3]

    target_palette_index = rgb_to_palette_index(target_rgb)
    target_palette_rgb = palette_index_to_rgb(target_palette_index)
    target_palette_rgb = jnp.broadcast_to(target_palette_rgb, predicted_rgb.shape)

    mask = jnp.broadcast_to(valid_mask[None, ...], predicted_rgb.shape[:-1] + (1,))

    squared_distance = jnp.sum((predicted_rgb - target_palette_rgb) ** 2, axis=-1, keepdims=True)
    masked_squared_distance = squared_distance * mask

    masked_error_sum = jnp.sum(masked_squared_distance, axis=(1, 2, 3, 4))
    valid_count = jnp.sum(mask, axis=(1, 2, 3, 4))

    return jnp.where(valid_count > 0, masked_error_sum / valid_count, 0.0)
#-------------


def compute_train_metrics(
    loss: jnp.ndarray,
    final_state_list,
    target_states,
    target_masks,
) -> dict:
    """
    Compute training metrics from the final states.

    The accuracy is computed only inside the target masks.

    Args:
        loss: Mean training loss for the current iteration.
        final_state_list: Final predicted states for all training pairs.
        target_states: Target states for all training pairs.
        target_masks: Target masks for all training pairs.

    Returns:
        Dictionary with log-loss and mean accuracy.

    """

    metrics = {
        "log_loss": float(safe_log_loss(loss)),
    }

    accuracy_values = []

    for final_state, target_state, target_mask in zip(final_state_list, target_states, target_masks):
        predicted_rgb = final_state[..., :3]
        target_rgb = target_state[..., :3]

        pool_mask = jnp.tile(target_mask, (final_state.shape[0], 1, 1, 1))
        accuracy_values.append(masked_palette_accuracy(predicted_rgb, target_rgb, pool_mask))

    metrics["mean_accuracy"] = float(jnp.mean(jnp.array(accuracy_values)))
    return metrics



def compute_test_metrics(
    state_trace: jnp.ndarray,
    final_state: jnp.ndarray,
    target_state_nhwc: jnp.ndarray,
    target_valid_mask: jnp.ndarray,
) -> dict:
    """
    Compute test metrics using the target mask.

    The reported loss combines masked RGBA MSE with an additional
    target-palette RGB loss. All losses and accuracies are computed only
    inside the valid target area.

    Args:
        state_trace: State history from the test run.
        final_state: Final predicted state.
        target_state_nhwc: Target state.
        target_valid_mask: Mask for the valid target area.

    Returns:
        Dictionary with per-step and final test metrics.
    """

    step_mse_loss = masked_mse_rgba_per_step(state_trace, target_state_nhwc, target_valid_mask)
    final_mse_loss = masked_mse_rgba(final_state, target_state_nhwc, target_valid_mask)

    step_palette_loss = masked_palette_loss_rgb_target_per_step(
        state_trace,
        target_state_nhwc,
        target_valid_mask,
    )
    final_palette_loss = masked_palette_loss_rgb_target(
        final_state,
        target_state_nhwc,
        target_valid_mask,
    )

    step_loss = step_mse_loss + 0.2 * step_palette_loss
    final_loss = final_mse_loss + 0.2 * final_palette_loss

    step_log_loss = safe_log_loss(step_loss)
    final_log_loss = safe_log_loss(final_loss)

    trace_rgb = state_trace[..., :3]
    target_rgb = target_state_nhwc[..., :3]
    final_rgb = final_state[..., :3]

    step_accuracy = masked_palette_accuracy_per_step(trace_rgb, target_rgb, target_valid_mask)
    final_accuracy = masked_palette_accuracy(final_rgb, target_rgb, target_valid_mask)

    step_exact_match = masked_exact_match_per_step(trace_rgb, target_rgb, target_valid_mask)
    final_exact_match = masked_exact_match(final_rgb, target_rgb, target_valid_mask)

    step_foreground_accuracy = masked_foreground_accuracy_per_step(trace_rgb, target_rgb, target_valid_mask)
    final_foreground_accuracy = masked_foreground_accuracy(final_rgb, target_rgb, target_valid_mask)

    has_solution = jnp.any(step_exact_match == 1.0)
    first_solve_step = jnp.where(
        has_solution,
        jnp.argmax(step_exact_match == 1.0),
        -1,
    )

    return {
        "step_loss": step_loss,
        "step_log_loss": step_log_loss,
        "step_accuracy": step_accuracy,
        "step_exact_match": step_exact_match,
        "step_foreground_accuracy": step_foreground_accuracy,

        "final_loss": float(final_loss),
        "final_log_loss": float(final_log_loss),
        "final_accuracy": float(final_accuracy),
        "final_exact_match": float(final_exact_match),
        "final_foreground_accuracy": float(final_foreground_accuracy),
        "first_solve_step": int(first_solve_step),
    }
