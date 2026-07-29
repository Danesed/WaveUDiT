# wavedit/evaluation/visualization.py
#
# Triplane (axial/coronal/sagittal) inpainting comparison grids for wandb / disk.

import numpy as np
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _mask_centroid_axis(mask_3d: np.ndarray, axis: int) -> int:
    """Centroid index of the mask along `axis`. Falls back to the array centre
    if the mask is empty."""
    sum_axes = tuple(a for a in range(3) if a != axis)
    profile = mask_3d.sum(axis=sum_axes)
    total = profile.sum()
    if total <= 0:
        return mask_3d.shape[axis] // 2
    coords = np.arange(mask_3d.shape[axis])
    return int(round((profile * coords).sum() / total))


def _to01(slice_norm: np.ndarray) -> np.ndarray:
    """Map a [-1,1] slice to [0,1] for plotting."""
    return np.clip((slice_norm + 1.0) / 2.0, 0.0, 1.0)


def _take_plane(vol_3d: np.ndarray, plane: str, idx: int) -> np.ndarray:
    if plane == "axial":
        return vol_3d[idx, :, :]
    if plane == "coronal":
        return vol_3d[:, idx, :]
    if plane == "sagittal":
        return vol_3d[:, :, idx]
    raise ValueError(plane)


_PLANE_ROT_K = {"axial": -1, "coronal": 2, "sagittal": 2}


def _take_plane_rot(vol_3d: np.ndarray, plane: str, idx: int) -> np.ndarray:
    slice_2d = _take_plane(vol_3d, plane, idx)
    k = _PLANE_ROT_K.get(plane, 0)
    if k:
        slice_2d = np.rot90(slice_2d, k=k)
    return slice_2d


def _thick_mask_border(mask_2d: np.ndarray, thickness: int = 3) -> np.ndarray:
    """Dilated outer ring of `mask_2d` as a binary 2D map."""
    mask_bool = mask_2d > 0.5
    if not mask_bool.any():
        return np.zeros_like(mask_2d, dtype=bool)
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(mask_bool, iterations=int(max(1, thickness)))
    return dilated & ~mask_bool


def visualize_inpainting_samples(
    i_gt: np.ndarray,
    i_voided: np.ndarray,
    mask: np.ndarray,
    i_pred: np.ndarray,
    out_path: str | None = None,
    title_prefix: str = "",
    max_samples: int = 4,
    wandb_key: str | None = None,
    mask_border_thickness: int = 3,
):
    """Triplane comparison grid: GT | GT+mask border | pred | |pred-GT|.

    Inputs are numpy arrays (B,1,D,H,W) or (B,D,H,W) in the model's [-1,1] range.
    For each sample, slices are cut at the mask centroid along each axis.
    """
    if i_gt.ndim == 5:
        i_gt = i_gt[:, 0]
        mask = mask[:, 0]
        i_pred = i_pred[:, 0]

    n = min(max_samples, i_gt.shape[0])
    if n == 0:
        return None

    planes = ("axial", "coronal", "sagittal")
    plane_axis = {"axial": 0, "coronal": 1, "sagittal": 2}
    cols = 4
    rows = n * len(planes)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.6), squeeze=False)

    for s in range(n):
        gt_v, mk_v, pr_v = i_gt[s], mask[s], i_pred[s]
        for p, plane in enumerate(planes):
            idx = _mask_centroid_axis(mk_v, axis=plane_axis[plane])
            r = s * len(planes) + p

            gt_s = _to01(_take_plane_rot(gt_v, plane, idx))
            pr_s = _to01(_take_plane_rot(pr_v, plane, idx))
            mk_s = (_take_plane_rot(mk_v, plane, idx) > 0.5).astype(np.float32)
            diff_s = np.clip(np.abs(pr_s - gt_s), 0.0, 1.0)
            border = _thick_mask_border(mk_s, thickness=mask_border_thickness)

            row_tag = f"{title_prefix}s{s} {plane} @{idx}"

            axs[r, 0].imshow(gt_s, cmap="gray", vmin=0, vmax=1)
            axs[r, 0].set_ylabel(row_tag, fontsize=8)
            axs[r, 0].set_title("GT" if r == 0 else "")

            axs[r, 1].imshow(gt_s, cmap="gray", vmin=0, vmax=1)
            border_img = np.zeros((*border.shape, 4), dtype=np.float32)
            border_img[..., 0] = 1.0
            border_img[..., 3] = border.astype(np.float32)
            axs[r, 1].imshow(border_img, interpolation="nearest")
            axs[r, 1].set_title("mask" if r == 0 else "")

            axs[r, 2].imshow(pr_s, cmap="gray", vmin=0, vmax=1)
            axs[r, 2].set_title("pred" if r == 0 else "")

            axs[r, 3].imshow(diff_s, cmap="magma", vmin=0, vmax=0.5)
            axs[r, 3].set_title("|pred-GT|" if r == 0 else "")

            for c in range(cols):
                axs[r, c].set_xticks([]); axs[r, c].set_yticks([])

    plt.tight_layout(pad=0.4)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    if out_path is not None:
        try:
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            logger.info(f"Saved inpainting viz to {out_path}")
        except Exception as e:
            logger.warning(f"Could not save inpainting viz: {e}")

    if wandb_key is not None and WANDB_AVAILABLE:
        try:
            wandb.log({wandb_key: wandb.Image(rgb, caption=title_prefix.strip(" |"))})
        except Exception:
            pass

    plt.close(fig)
    return rgb
