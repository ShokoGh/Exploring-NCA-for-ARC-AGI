import jax.numpy as jnp
import numpy as np
from typing import Optional

from .palette_utils import PALETTE
from .config import HIDDEN_CHANNELS

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
        A JAX array of shape (C, H, W), where C is the total number of channels (4 + num_hidden channels, 50 with the default configuration).
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

def get_task_padding_size(task: dict) -> tuple[int, int]:
    """
    Find the maximum height and width across all input/output grids in a task.

    Args:
        task: ARC task dictionary.

    Returns:
        A tuple (max_h, max_w).
    """
    max_h, max_w = 0, 0

    for split_name in ["train", "test"]:
        for pair in task.get(split_name, []):
            for key in ["input", "output"]:
                if key in pair:
                    grid = np.asarray(pair[key], dtype=np.int32)
                    h, w = grid.shape
                    max_h = max(max_h, h)
                    max_w = max(max_w, w)

    return max_h, max_w


def pad_grid(
    grid_2d: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: int = -1,
) -> np.ndarray:
    """
    Pad a 2D ARC grid to (target_h, target_w) with pad_value.

    The original grid is placed in the top-left corner.

    Args:
        grid_2d: Input grid with shape (H, W).
        target_h: Target height.
        target_w: Target width.
        pad_value: Value used in padded cells.

    Returns:
        Padded grid with shape (target_h, target_w).

    """
    grid = np.asarray(grid_2d, dtype=np.int32)
    h, w = grid.shape

    if h > target_h or w > target_w:
        raise ValueError(
            f"Grid shape {(h, w)} is larger than target shape {(target_h, target_w)}"
        )

    padded = np.full((target_h, target_w), pad_value, dtype=np.int32)
    padded[:h, :w] = grid
    return padded


def make_valid_mask(
    original_grid_2d: np.ndarray,
    target_h: int,
    target_w: int,
) -> jnp.ndarray:
    """
    Create a mask for the original grid area.

    Valid cells are set to 1.0 and padded cells are set to 0.0.

    Args:
        original_grid_2d: Original unpadded grid.
        target_h: Target height.
        target_w: Target width.

    Returns:
        Mask with shape (1, H, W, 1).
    """
    grid = np.asarray(original_grid_2d)
    h, w = grid.shape

    mask = np.zeros((target_h, target_w), dtype=np.float32)
    mask[:h, :w] = 1.0

    return jnp.array(mask[None, ..., None])

def format_padding_debug_message(
    input_grid: np.ndarray,
    target_grid: np.ndarray,
    max_h: int,
    max_w: int,
    pair_name: str = "",
) -> str:
    """
    Build one debug line for input and output padding.
    """
    in_h, in_w = input_grid.shape
    out_h, out_w = target_grid.shape

    input_will_pad = (in_h != max_h) or (in_w != max_w)
    target_will_pad = (out_h != max_h) or (out_w != max_w)

    label = f"[{pair_name}] " if pair_name else ""

    return (
        f"{label}input {in_h}x{in_w} -> {max_h}x{max_w} "
        f"{'(PADDED)' if input_will_pad else '(no padding)'} | "
        f"output {out_h}x{out_w} -> {max_h}x{max_w} "
        f"{'(PADDED)' if target_will_pad else '(no padding)'}"
    )


def build_pair_tensors(
        pair: dict,
        max_h: int,
        max_w: int,
        pad_value: int = -1,
        debug: bool = False,
        pair_name: str = "",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[str]]:
    """
    Convert one ARC input-output pair to padded NCA state tensors in NHWC format.

    Each input and target state has shape (1, H, W, C). This matches the
    general NCA state format (N, H, W, C), where N is the number of parallel
    states. Here N is 1 because each pair starts as a single state. During
    training, this dimension is later expanded to POOL_SIZE when creating the
    state pools. During evaluation, it remains 1.

    Both input and output are padded to the same task size.
    The function also builds masks for the original input area
    and the original target area.

    Args:
        pair: ARC input-output pair.
        max_h: Task height after padding.
        max_w: Task width after padding.
        pad_value: Value used in padded cells.
        debug: If True, return size/padding debug information.
        pair_name: Optional name used in the debug message.

    Returns:
        input_state_nhwc: Input state with shape (1, H, W, C).
        target_state_nhwc: Target state with shape (1, H, W, C).
        input_valid_mask: Input mask with shape (1, H, W, 1).
        target_valid_mask: Target mask with shape (1, H, W, 1)
        debug_message: Optional padding/debug message.
    """
    input_grid = np.array(pair["input"], dtype=np.int32)
    target_grid = np.array(pair["output"], dtype=np.int32)

    debug_message = None
    if debug:
        debug_message = format_padding_debug_message(
            input_grid=input_grid,
            target_grid=target_grid,
            max_h=max_h,
            max_w=max_w,
            pair_name=pair_name,
        )

    input_grid_padded = pad_grid(input_grid, max_h, max_w, pad_value=pad_value)
    target_grid_padded = pad_grid(target_grid, max_h, max_w, pad_value=pad_value)

    input_valid_mask = make_valid_mask(input_grid, max_h, max_w)
    target_valid_mask = make_valid_mask(target_grid, max_h, max_w)

    input_chw = arc_to_nca(input_grid_padded, pad_value=pad_value)
    target_chw = arc_to_nca(target_grid_padded, pad_value=pad_value)

    input_state_nhwc = jnp.moveaxis(input_chw[None], 1, -1)
    target_state_nhwc = jnp.moveaxis(target_chw[None], 1, -1)

    return input_state_nhwc, target_state_nhwc, input_valid_mask, target_valid_mask, debug_message


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
