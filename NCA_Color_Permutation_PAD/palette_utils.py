import jax.numpy as jnp
import numpy as np
import jax

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

def make_color_permutation(rng_key: jnp.ndarray) -> jnp.ndarray:
    """
    Create a random ARC color permutation.

    Background color 0 is kept fixed, while colors 1..9 are randomly permuted.

    Args:
        rng_key: JAX random key.

    Returns:
        A permutation array with shape (10,), where:
        - perm[0] = 0
        - perm[1:10] is a random permutation of 1..9
    """
    permuted_colors = jax.random.permutation(rng_key, jnp.arange(1, 10, dtype=jnp.int32))
    return jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), permuted_colors],
        axis=0,
    )


def permute_nhwc_by_palette(state_nhwc: jnp.ndarray, permutation: jnp.ndarray) -> jnp.ndarray:
    """
    Apply a palette permutation to the RGB channels of an NCA state tensor.

    The RGB channels are first mapped to their nearest ARC palette indices,
    then permuted, and finally mapped back to RGB values.
    The alpha and hidden channels are kept unchanged.

    Args:
        state_nhwc: State tensor with shape (..., H, W, C).
        permutation: Palette permutation array with shape (10,).

    Returns:
        State tensor with permuted RGB channels and unchanged non-RGB channels.
    """
    rgb = state_nhwc[..., :3]
    alpha = state_nhwc[..., 3:4]
    hidden = state_nhwc[..., 4:]

    palette_index = rgb_to_palette_index(rgb)
    permuted_palette_index = permutation[palette_index]
    permuted_rgb = palette_index_to_rgb(permuted_palette_index)

    return jnp.concatenate([permuted_rgb, alpha, hidden], axis=-1)