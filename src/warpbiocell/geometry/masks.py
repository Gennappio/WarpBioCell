"""Tissue regions from segmentation masks (optional dependencies: scipy, nibabel).

    pip install -e ".[masks]"

``sdf_from_mask`` turns a binary voxel mask into a signed distance field (negative inside)
with two Euclidean distance transforms; ``region_from_mask`` resamples it onto a
WarpBioCell grid; ``load_nifti_mask`` reads a NIfTI label image. No clinical claim is
attached to any of this: a mask is a geometry, nothing more.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from warpbiocell.fields.scalar_field import GridGeometry
from warpbiocell.geometry.region import TissueRegion


def sdf_from_mask(mask: np.ndarray, voxel_size: tuple[float, float, float] | float) -> np.ndarray:
    """Signed distance [um] on the mask's own voxel grid, negative inside the mask."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("sdf_from_mask needs scipy: pip install -e '.[masks]'") from exc
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 3:
        raise ValueError("mask must be a 3-D array")
    spacing = (voxel_size,) * 3 if np.isscalar(voxel_size) else tuple(float(s) for s in voxel_size)
    if not mask.any():
        raise ValueError("mask is empty")
    outside = distance_transform_edt(~mask, sampling=spacing)  # distance to the tissue, outside
    inside = distance_transform_edt(mask, sampling=spacing)  # distance to the exterior, inside
    # Half a voxel shifts the zero level onto the face between an inside and an outside voxel.
    half = 0.5 * min(spacing)
    return np.where(mask, -(inside - half), outside - half).astype(np.float32)


def resample_nearest(values: np.ndarray, voxel_size, mask_origin, geometry: GridGeometry, fill: float) -> np.ndarray:
    """Nearest-voxel resampling of ``values`` (defined on the mask grid) onto ``geometry``."""
    spacing = np.array((voxel_size,) * 3 if np.isscalar(voxel_size) else voxel_size, dtype=float)
    x, y, z = np.meshgrid(*geometry.node_coordinates(), indexing="ij")
    idx = np.stack([x, y, z], axis=-1) - np.asarray(mask_origin, dtype=float)
    idx = np.rint(idx / spacing).astype(int)
    inside = np.all((idx >= 0) & (idx < np.array(values.shape)), axis=-1)
    out = np.full(geometry.shape, fill, dtype=np.float32)
    ii = idx[inside]
    out[inside] = values[ii[:, 0], ii[:, 1], ii[:, 2]]
    return out


def region_from_mask(
    mask: np.ndarray,
    voxel_size,
    geometry: GridGeometry,
    mask_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    device=None,
    name: str = "mask",
) -> TissueRegion:
    """Build a TissueRegion on ``geometry`` from a binary mask whose voxel (0,0,0) sits at ``mask_origin`` [um]."""
    sdf = sdf_from_mask(mask, voxel_size)
    far = float(np.abs(sdf).max()) + 10.0 * float(np.max(np.atleast_1d(voxel_size)))
    return TissueRegion(geometry, resample_nearest(sdf, voxel_size, mask_origin, geometry, fill=far), device=device, name=name)


def load_nifti_mask(path: str | Path, label: int | None = None) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Binary mask and voxel size [um] from a NIfTI label image (voxel sizes are stored in mm)."""
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_nifti_mask needs nibabel: pip install -e '.[masks]'") from exc
    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    mask = (data == label) if label is not None else (data > 0)
    voxel_mm = image.header.get_zooms()[:3]
    return mask, tuple(1000.0 * float(v) for v in voxel_mm)
