import jax.numpy as jnp
import numpy as np

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