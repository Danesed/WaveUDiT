#!/usr/bin/env python3
r"""Inference-time prediction averaging (TTA) for the BraTS inpainting models.

USAGE
-----
    from scripts.tta_ensemble import predict_tta, TTASpec
    pred = predict_tta(model, v, m, "mirror")            # == predict_mirror_consistent
    pred = predict_tta(model, v, m, "crop")              # crop-augmented
    pred, info = predict_tta(model, v, m, TTASpec(n_crops=2, shifts=((0, 4, 4),)),
                             return_info=True)

Drop-in for the existing submission path (scripts/gen_val_predictions.py calls
`model.predict_mirror_consistent`) with NO edit to that script. bind the attribute
after `load(...)`:

    from scripts.tta_ensemble import predict_tta
    model.predict_mirror_consistent = lambda v, m: predict_tta(model, v, m, "crop")

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

__all__ = [
    "BG_VALUE", "LR_DIM", "Flip", "Shift", "Crop", "View", "TTASpec", "PRESETS",
    "build_views", "predict_tta",
]

# Background value of a T1 volume after `_normalise_t1` (raw 0 -> 2*0/max - 1 = -1).
BG_VALUE = -1.0
# Spatial dims of the (B, C, D, H, W) layout, and the anatomical left-right axis.
_SPATIAL = (-3, -2, -1)
LR_DIM = -2


# ---------------------------------------------------------------------------
# Atomic, exactly-invertible ops.  Each op is a re-indexing of the voxel grid:
# fwd(x, fill) maps the volume into the view's frame, inv(y, fill) maps back.
# `fill` is the value written where the op has no source voxel; the support map
# is derived by running ones through fwd+inv with fill=0, so a voxel that ever
# touched `fill` automatically gets weight 0.
# --------------------------------------------------------------------------- #
class _Op:
    name = "op"

    def fwd(self, x: torch.Tensor, fill: float) -> torch.Tensor:
        raise NotImplementedError

    def inv(self, x: torch.Tensor, fill: float) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


class Flip(_Op):
    """Mirror along one axis. An involution, so inv is fwd. `torch.flip` is a pure
    gather (no arithmetic) -> the round trip is bit-exact on every voxel."""

    def __init__(self, dim: int = LR_DIM):
        self.dim = int(dim)
        self.name = "mirror" if self.dim == LR_DIM else f"flip{self.dim}"

    def fwd(self, x: torch.Tensor, fill: float) -> torch.Tensor:
        return torch.flip(x, dims=(self.dim,))

    inv = fwd


class Shift(_Op):
    """Integer translation along (D, H, W).

    mode="fill": voxels shifted in from outside the FOV take `fill`.  Content that
        leaves the FOV is gone, so the round trip is bit-exact everywhere EXCEPT one
        |shift|-wide band per axis: the TRAILING band for a positive shift, the
        LEADING band for a negative one (the band whose source fell outside).  The
        support map marks exactly that band as 0.  NB the support tracks index
        bookkeeping, not receptive fields: a voxel with support 1 may still have had
        `fill` inside its receptive field.  The fill value IS
        background, and `_shift_is_lossless` guarantees the discarded band was
        background too, so nothing anatomical is invented or destroyed.
    mode="roll": circular (torch.roll); bit-exact on EVERY voxel, but anatomy wraps
        around the FOV.  Harmless for small shifts on a background-padded volume and
        useful as a zero-loss-of-support reference.
    """

    def __init__(self, shift: Sequence[int], mode: str = "fill"):
        if mode not in ("fill", "roll"):
            raise ValueError(f"shift mode must be 'fill' or 'roll', got {mode!r}")
        self.shift = tuple(int(s) for s in shift)
        if len(self.shift) != 3:
            raise ValueError("shift must have 3 components (dD, dH, dW)")
        self.mode = mode
        self.name = f"shift{self.shift}" + ("" if mode == "fill" else ":roll")

    def _apply(self, x: torch.Tensor, s: Tuple[int, int, int], fill: float) -> torch.Tensor:
        if self.mode == "roll":
            return torch.roll(x, shifts=s, dims=_SPATIAL)
        out = torch.full_like(x, fill)
        src: List[Union[slice, int]] = [slice(None)] * x.dim()
        dst: List[Union[slice, int]] = [slice(None)] * x.dim()
        for ax, d in zip(_SPATIAL, s):
            n = x.shape[ax]
            if abs(d) >= n:
                return out                       # everything shifted out of the FOV
            if d >= 0:
                src[ax], dst[ax] = slice(0, n - d), slice(d, n)
            else:
                src[ax], dst[ax] = slice(-d, n), slice(0, n + d)
        out[tuple(dst)] = x[tuple(src)]
        return out

    def fwd(self, x: torch.Tensor, fill: float) -> torch.Tensor:
        return self._apply(x, self.shift, fill)

    def inv(self, x: torch.Tensor, fill: float) -> torch.Tensor:
        return self._apply(x, tuple(-s for s in self.shift), fill)


class Crop(_Op):
    """Take a fixed axis-aligned sub-volume; the inverse pastes it back into a
    full-size tensor of `fill`.  Both directions are pure slicing -> bit-exact
    inside the box, support 0 outside it."""

    def __init__(self, box: Sequence[Tuple[int, int]], full_spatial: Sequence[int]):
        self.box = tuple((int(a), int(b)) for a, b in box)
        self.full = tuple(int(s) for s in full_spatial)
        if len(self.box) != 3 or len(self.full) != 3:
            raise ValueError("box and full_spatial must be 3-element")
        for (a, b), f in zip(self.box, self.full):
            if not (0 <= a < b <= f):
                raise ValueError(f"invalid crop box {self.box} for volume {self.full}")
        self.name = "crop(" + ",".join(f"{a}:{b}" for a, b in self.box) + ")"

    @property
    def size(self) -> Tuple[int, int, int]:
        return tuple(b - a for a, b in self.box)

    def _sl(self):
        return (Ellipsis,) + tuple(slice(a, b) for a, b in self.box)

    def fwd(self, x: torch.Tensor, fill: float) -> torch.Tensor:
        return x[self._sl()]

    def inv(self, x: torch.Tensor, fill: float) -> torch.Tensor:
        out = x.new_full(tuple(x.shape[:-3]) + self.full, fill)
        out[self._sl()] = x
        return out


@dataclass(frozen=True)
class View:
    name: str
    ops: Tuple[_Op, ...]
    n_voxels: int                     # voxels the network actually processes

    def fwd_inputs(self, voided: torch.Tensor, mask: torch.Tensor, bg: float):
        """Map (voided, mask) into the view frame.  The mask fill is 0 (no void
        outside the FOV); the voided fill is the background intensity."""
        v, m = voided, mask
        for op in self.ops:
            v = op.fwd(v, bg)
            m = op.fwd(m, 0.0)
        return v, m

    def inv_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """Map a prediction back to the original frame (fill 0; those voxels have
        support 0 and are discarded by the weighted mean)."""
        p = pred
        for op in reversed(self.ops):
            p = op.inv(p, 0.0)
        return p

    def support(self, shape, device, dtype=torch.float32) -> torch.Tensor:
        """0/1 map of voxels whose prediction is real AND landed back in place.
        Derived from the ops themselves, never hand-written."""
        w = torch.ones(shape, device=device, dtype=dtype)
        for op in self.ops:
            w = op.fwd(w, 0.0)
        for op in reversed(self.ops):
            w = op.inv(w, 0.0)
        return w



@dataclass(frozen=True)
class TTASpec:
    """Declarative description of a transform set.

    full         include the uncropped, unshifted view.
    mirror       also run the L-R mirror of every base view (doubles the count).
    shifts       extra uncropped views, each translated by (dD, dH, dW).
    n_crops      number of training-sized crop views placed around the void.
    crop_size    the size the checkpoint was TRAINED on (144, 208, 208 for all
                 current launchers).
    crop_jitter  offset (voxels) between successive crop windows, mirroring the
                 -+16 centroid jitter used during training.
    shift_mode   "fill" (default, semantically honest) or "roll" (circular).
    size_multiple  crop dims must be divisible by this (16 = DWT stride 2 x 2^3
                 downsamples for the levels=4 wavelet backbones).  0 disables.
    """
    full: bool = True
    mirror: bool = True
    shifts: Tuple[Tuple[int, int, int], ...] = ()
    n_crops: int = 0
    crop_size: Tuple[int, int, int] = (144, 208, 208)
    crop_jitter: int = 16
    shift_mode: str = "fill"
    size_multiple: int = 16


PRESETS: Dict[str, TTASpec] = {
    # 1 fwd — no averaging at all (reference / ablation baseline).
    "none":      TTASpec(mirror=False),
    # 2 fwd — exactly what predict_mirror_consistent does today.
    "mirror":    TTASpec(),
    # 4 fwd — mirror + one sub-stride translation (phase decorrelation).
    "shift1":    TTASpec(shifts=((0, 4, 4),)),
    # 6 fwd — mirror + two opposite sub-stride translations.
    "shift":     TTASpec(shifts=((0, 4, 4), (0, -4, -4))),
    # 4 fwd — mirror + one crop at the TRAINED input size.
    "crop":      TTASpec(n_crops=1),
    # 8 fwd — mirror + three jittered training-size crops.
    "crop3":     TTASpec(n_crops=3),
    # 6 fwd — crops only: run the net exclusively on its training-size grid.
    "crop_only": TTASpec(full=False, n_crops=3),
    # 12 fwd — everything.
    "max":       TTASpec(shifts=((0, 4, 4), (0, -4, -4)), n_crops=3),
}


def _void_bbox(mask: torch.Tensor) -> Optional[Tuple[List[int], List[int]]]:
    """Inclusive (lo, hi) voxel bounding box of the void, or None if the mask is
    empty.  Uses sample 0 (crop views are single-case by construction)."""
    idx = (mask[0, 0] > 0.5).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return None
    return idx.min(dim=0).values.tolist(), idx.max(dim=0).values.tolist()


def _crop_boxes(mask: torch.Tensor, full_spatial: Tuple[int, int, int],
                crop_size: Tuple[int, int, int], jitter: int, n: int
                ) -> List[Tuple[Tuple[int, int], ...]]:
    """Up to `n` distinct crop windows around the void.

    Every returned window fully contains the void bounding box: the crop is the
    only view that can fail to cover the void, and a partially covered void would
    put a seam through the synthesised region (some voxels averaged over k views,
    their neighbours over k-1).  If the void does not fit inside `crop_size` at
    all, returns [] — the caller then falls back to the full-volume views.
    """
    cs = tuple(min(c, f) for c, f in zip(crop_size, full_spatial))
    bb = _void_bbox(mask)
    if bb is None:                                     # degenerate: nothing to inpaint
        centre = [f // 2 for f in full_spatial]
        lo = hi = centre
    else:
        lo, hi = bb
        centre = [(a + b) // 2 for a, b in zip(lo, hi)]
    j = int(jitter)
    deltas = [(0, 0, 0), (0, j, j), (0, -j, -j), (0, j, -j), (0, -j, j),
              (j, 0, 0), (-j, 0, 0), (0, 0, j), (0, 0, -j), (0, j, 0), (0, -j, 0)]
    boxes: List[Tuple[Tuple[int, int], ...]] = []
    for d in deltas:
        if len(boxes) >= n:
            break
        start = []
        for ax in range(3):
            c, f = cs[ax], full_spatial[ax]
            s_min = max(0, hi[ax] - c + 1)             # window must contain the void...
            s_max = min(f - c, lo[ax])                 # ...and stay inside the volume
            if s_min > s_max:
                return []                              # void larger than the crop
            s = centre[ax] - c // 2 + d[ax]
            start.append(int(min(max(s, s_min), s_max)))
        box = tuple((s, s + c) for s, c in zip(start, cs))
        if box not in boxes:
            boxes.append(box)
    return boxes


def build_views(spec: TTASpec, full_spatial: Tuple[int, int, int],
                mask: torch.Tensor) -> List[View]:
    """Materialise the transform set for ONE case (crop placement depends on the
    void location, so views are built per case)."""
    n_full = int(full_spatial[0]) * int(full_spatial[1]) * int(full_spatial[2])
    base: List[Tuple[str, Tuple[_Op, ...], int]] = []
    if spec.full:
        base.append(("id", (), n_full))
    for s in spec.shifts:
        op = Shift(s, mode=spec.shift_mode)
        base.append((op.name, (op,), n_full))
    if spec.n_crops > 0:
        if spec.size_multiple:
            bad = [c for c in spec.crop_size if c % spec.size_multiple]
            if bad:
                raise ValueError(
                    f"crop_size {spec.crop_size} not divisible by {spec.size_multiple}; "
                    "the levels=4 wavelet backbones need dims divisible by 16 "
                    "(DWT stride 2 x 2^3 strided downsamples).")
        for box in _crop_boxes(mask, full_spatial, spec.crop_size,
                               spec.crop_jitter, spec.n_crops):
            op = Crop(box, full_spatial)
            base.append((op.name, (op,), op.size[0] * op.size[1] * op.size[2]))
    if not base:
        # Only reachable with full=False and a void too large for `crop_size`.
        # Degrade to the full-volume view rather than crash a 219-case submission
        # loop; the fallback is visible in info["views"].
        base.append(("id[crop-fallback]", (), n_full))

    views: List[View] = []
    for name, ops, nv in base:
        views.append(View(name, ops, nv))
        if spec.mirror:
            # Mirror LAST: the crop window stays anchored on the void in ORIGINAL
            # coordinates (so void coverage is unaffected by the mirror) and the
            # network sees the mirrored crop, exactly as in training augmentation.
            views.append(View(f"{name}+mirror", ops + (Flip(LR_DIM),), nv))
    return views


def _shift_is_lossless(voided: torch.Tensor, op: Shift, bg: float, tol: float) -> bool:
    """True if translating by `op` only pushes BACKGROUND out of the FOV.

    The padded volume has 8+8 voxels of headroom along H/W (240 -> 256) but only
    2/3 along D (155 -> 160), so an unchecked shift can amputate brain tissue —
    that is information destruction, not a benign re-framing, and no amount of
    averaging repairs it.  Cheap to check, so we check it per case.
    """
    if op.mode == "roll":
        return True                                    # nothing leaves the FOV
    for ax, d in zip(_SPATIAL, op.shift):
        if d == 0:
            continue
        n = voided.shape[ax]
        if abs(d) >= n:
            return False
        sl: List[Union[slice, int]] = [slice(None)] * voided.dim()
        sl[ax] = slice(n - d, n) if d > 0 else slice(0, -d)
        band = voided[tuple(sl)]
        if band.numel() and float(band.max()) > bg + tol:
            return False
    return True


@torch.no_grad()
def predict_tta(model, voided: torch.Tensor, mask: torch.Tensor,
                spec: Union[str, TTASpec] = "mirror", *,
                bg_value: float = BG_VALUE,
                require_void_coverage: bool = True,
                check_shift_clipping: bool = True,
                shift_bg_tol: float = 0.02,
                check: bool = True,
                return_info: bool = False):
    """Average `model(voided, mask)` over a set of exactly-invertible test-time views.

    Args:
        model:   anything with `forward(voided, mask) -> (B,1,D,H,W)`; the output is
                 assumed to be a full volume whose VOID content is the prediction
                 (DirectInpaintModel composites internally — irrelevant here, we
                 re-composite ourselves and only the void is read).
        voided:  (B,1,D,H,W) float, normalised to [-1,1] (background = `bg_value`).
        mask:    (B,1,D,H,W) BINARY void mask (1 = synthesise).
        spec:    preset name (see PRESETS) or a TTASpec.
    Returns:
        (B,1,D,H,W) tensor, same dtype/device as `voided`: the mean prediction inside
        the void, the ORIGINAL voided values bit-for-bit outside it.
        With return_info=True, also a dict {views, skipped, n_forward, voxel_cost}.
    """
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise KeyError(f"unknown TTA preset {spec!r}; have {sorted(PRESETS)}")
        spec = PRESETS[spec]
    if voided.dim() != 5 or mask.shape != voided.shape:
        raise ValueError(f"expected (B,1,D,H,W) with matching shapes, got "
                         f"{tuple(voided.shape)} and {tuple(mask.shape)}")
    if check:
        uniq = torch.unique(mask)
        if not bool(((uniq == 0) | (uniq == 1)).all()):
            raise ValueError("mask must be binary {0,1}; got values "
                             f"{uniq[:8].tolist()}...")
    if spec.n_crops > 0 and voided.shape[0] != 1:
        raise ValueError("crop views place the window on the void of sample 0; "
                         "call with batch size 1 (the submission path is per-case).")

    full_spatial = tuple(int(s) for s in voided.shape[-3:])
    views = build_views(spec, full_spatial, mask)
    void = mask > 0.5
    has_void = bool(void.any())

    num = torch.zeros(voided.shape, device=voided.device, dtype=torch.float32)
    den = torch.zeros_like(num)
    used: List[str] = []
    skipped: List[str] = []
    cost = 0

    for view in views:
        shift_ops = [o for o in view.ops if isinstance(o, Shift)]
        if check_shift_clipping and any(
                not _shift_is_lossless(voided, o, bg_value, shift_bg_tol) for o in shift_ops):
            skipped.append(f"{view.name}[clips brain]")
            continue
        w = view.support(voided.shape, voided.device, torch.float32)
        if require_void_coverage and has_void and not bool((w[void] == 1).all()):
            skipped.append(f"{view.name}[partial void support]")
            continue
        v_t, m_t = view.fwd_inputs(voided, mask, bg_value)
        p_t = model(v_t, m_t)
        p = view.inv_pred(p_t.float())
        num = num + p * w
        den = den + w
        used.append(view.name)
        cost += view.n_voxels

    if not used:
        raise RuntimeError("every TTA view was skipped; nothing to average")
    if has_void and not bool((den[void] > 0).all()):
        raise RuntimeError("some void voxels are not covered by any view "
                           "(use require_void_coverage=True or add the full view)")
    if check and require_void_coverage and has_void:
        dv = den[void]
        assert float(dv.min()) == float(dv.max()) == float(len(used)), (
            "void denominator must be exactly the number of used views")

    avg = (num / den.clamp_min(1.0)).to(voided.dtype)
    out = torch.where(void, avg, voided)               # pure select: known tissue exact

    if check:
        known = ~void
        assert torch.equal(out[known], voided[known]), \
            "known tissue was modified — composite is broken"

    if return_info:
        n_full = full_spatial[0] * full_spatial[1] * full_spatial[2]
        return out, {"views": used, "skipped": skipped, "n_forward": len(used),
                     "voxel_cost": cost / float(n_full)}
    return out


# Test

def _main() -> int:  # pragma: no cover - exercised via __main__
    import itertools
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, ok: bool, note: str = ""):
        results.append((name, bool(ok), note))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({note})" if note else ""),
              flush=True)

    # ---------------- stub models with closed-form outputs ----------------- #
    class CountingModel(nn.Module):
        """Wrapper that counts forward passes and records the shapes it saw."""

        def __init__(self, inner):
            super().__init__()
            self.inner, self.n, self.shapes = inner, 0, []

        def forward(self, v, m):
            self.n += 1
            self.shapes.append(tuple(v.shape))
            return self.inner(v, m)

    class ConstStub(nn.Module):
        """raw == c everywhere.  Output is independent of the view -> the TTA mean
        must equal the single-view prediction EXACTLY (idempotence)."""

        def __init__(self, c=0.3):
            super().__init__()
            self.c = c

        def forward(self, v, m):
            return v * (1 - m) + torch.full_like(v, self.c) * m

    class RampStub(nn.Module):
        """raw(h) == ramp of the ABSOLUTE H index of the tensor it is given.
        Content-independent, so the prediction of any view is known in closed form
        and a wrong inverse (wrong axis / wrong sign / wrong offset) is detectable
        analytically rather than statistically."""

        @staticmethod
        def ramp(H, device, dtype):
            return torch.arange(H, device=device, dtype=dtype).view(1, 1, 1, H, 1)

        def forward(self, v, m):
            r = self.ramp(v.shape[-2], v.device, v.dtype).expand_as(v)
            return v * (1 - m) + r * m

    class BlurStub(nn.Module):
        """raw = 3x3x3 box filter of `voided`.  Exactly equivariant to L-R flip and
        (in the interior) to translation and cropping.  For an EQUIVARIANT model the
        TTA mean must reproduce the identity prediction — any mis-inverted transform
        breaks that equality, which makes this the strongest end-to-end check."""

        def forward(self, v, m):
            return v * (1 - m) + F.avg_pool3d(v, 3, stride=1, padding=1) * m

    class GarbageStub(nn.Module):
        """Adversarial: returns noise everywhere and does NOT composite.  Used to
        prove known tissue is preserved by predict_tta itself, not by the model."""

        def forward(self, v, m):
            return torch.randn_like(v) * 10.0

    def make_case(shape=(32, 48, 48), box=((10, 18), (16, 28), (16, 28)), device="cpu"):
        v = torch.randn(1, 1, *shape, device=device).clamp(-1, 1)
        v[..., :4, :, :] = BG_VALUE          # background rims so shifts are lossless
        v[..., -4:, :, :] = BG_VALUE
        v[..., :, :6, :] = BG_VALUE
        v[..., :, -6:, :] = BG_VALUE
        v[..., :, :, :6] = BG_VALUE
        v[..., :, :, -6:] = BG_VALUE
        m = torch.zeros_like(v)
        m[..., box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]] = 1.0
        return v, m

    print("\n=== 1. atomic ops: exact round trip ===")
    x = torch.randn(2, 1, 16, 20, 24)

    f = Flip(LR_DIM)
    check("Flip(dim=-2) inverse(forward(x)) == x  bit-exact",
          torch.equal(f.inv(f.fwd(x, 0.0), 0.0), x))
    check("Flip is not the identity (the test has power)",
          not torch.equal(f.fwd(x, 0.0), x))

    ok_roll = True
    for s in [(0, 4, 4), (0, -4, -4), (2, -3, 5), (1, 1, 1)]:
        op = Shift(s, mode="roll")
        ok_roll &= torch.equal(op.inv(op.fwd(x, 0.0), 0.0), x)
    check("Shift(roll) inverse(forward(x)) == x  bit-exact, ALL voxels", ok_roll)

    ok_fill, ok_supp = True, True
    for s in [(0, 4, 4), (0, -4, -4), (2, -3, 5)]:
        op = Shift(s, mode="fill")
        rt = op.inv(op.fwd(x, BG_VALUE), 0.0)
        sup = op.inv(op.fwd(torch.ones_like(x), 0.0), 0.0)
        b = sup > 0.5
        ok_fill &= torch.equal(rt[b], x[b])
        exp = torch.ones_like(x)
        for ax, d in zip(_SPATIAL, s):
            if d == 0:
                continue
            n = exp.shape[ax]
            idx: List[Union[slice, int]] = [slice(None)] * exp.dim()
            idx[ax] = slice(n - d, n) if d > 0 else slice(0, -d)
            exp[tuple(idx)] = 0.0
        ok_supp &= torch.equal(sup, exp)
        ok_supp &= bool(((sup == 0) | (sup == 1)).all())
    check("Shift(fill) round trip bit-exact on its support", ok_fill)
    check("Shift(fill) support == analytic band structure, values in {0,1}", ok_supp)

    cop = Crop(((2, 10), (4, 16), (0, 16)), (16, 20, 24))
    rt = cop.inv(cop.fwd(x, 0.0), 0.0)
    sup = cop.inv(cop.fwd(torch.ones_like(x), 0.0), 0.0)
    b = sup > 0.5
    check("Crop round trip bit-exact inside the box", torch.equal(rt[b], x[b]))
    exp = torch.zeros_like(x)
    exp[..., 2:10, 4:16, 0:16] = 1.0
    check("Crop support == box indicator, values in {0,1}",
          torch.equal(sup, exp) and bool(((sup == 0) | (sup == 1)).all()))

    # composed chains
    ok_chain = True
    chains = [(Crop(((2, 10), (4, 16), (0, 16)), (16, 20, 24)), Flip(LR_DIM)),
              (Shift((0, 3, -2), "fill"), Flip(LR_DIM)),
              (Shift((0, 3, -2), "roll"), Crop(((0, 8), (2, 18), (2, 22)), (16, 20, 24)))]
    for ops in chains:
        vw = View("chain", tuple(ops), 1)
        y, _ = vw.fwd_inputs(x, torch.zeros_like(x), 0.0)
        rt = vw.inv_pred(y)
        sup = vw.support(x.shape, x.device)
        b = sup > 0.5
        ok_chain &= torch.equal(rt[b], x[b]) and bool(((sup == 0) | (sup == 1)).all())
    check("composed chains round trip bit-exact on their support", ok_chain)

    print("\n=== 2. requirement 1: known tissue preserved EXACTLY ===")
    v_big = torch.randn(1, 1, 160, 256, 256).clamp(-1, 1)
    m_big = torch.zeros_like(v_big)
    m_big[..., 70:90, 100:140, 110:150] = 1.0
    g = GarbageStub()
    out_big = predict_tta(g, v_big, m_big, "mirror")
    known = m_big <= 0.5
    check("production shape (1,1,160,256,256): out == voided outside the void, bit-exact",
          torch.equal(out_big[known], v_big[known]),
          f"{int(known.sum())} voxels compared, model output was pure noise")
    check("void was actually written (out != voided somewhere inside)",
          not torch.equal(out_big[~known], v_big[~known]))
    del v_big, m_big, out_big

    v, m = make_case()
    for preset in ["none", "mirror", "shift", "crop", "crop3", "crop_only", "max"]:
        sp = PRESETS[preset]
        sp = TTASpec(**{**sp.__dict__, "crop_size": (16, 32, 32), "crop_jitter": 4})
        o = predict_tta(GarbageStub(), v, m, sp)
        if not torch.equal(o[m <= 0.5], v[m <= 0.5]):
            check(f"known tissue exact for preset '{preset}'", False)
            break
    else:
        check("known tissue exact for every preset (garbage model)", True)

    print("\n=== 3. requirement 2: idempotence of the average ===")
    blur = BlurStub()
    single = predict_tta(blur, v, m, "none")
    ok_exact, ok_ulp, worst_rel = True, True, 0.0
    for n in (2, 4, 8):
        views = [View(f"id{i}", (), 1) for i in range(n)]
        num = torch.zeros_like(v, dtype=torch.float32)
        den = torch.zeros_like(num)
        for vw in views:
            vt, mt = vw.fwd_inputs(v, m, BG_VALUE)
            w = vw.support(v.shape, v.device)
            num = num + vw.inv_pred(blur(vt, mt).float()) * w
            den = den + w
        avg = (num / den).to(v.dtype)
        got = torch.where(m > 0.5, avg, v)
        rel = float(((got - single).abs() / single.abs().clamp_min(1e-6)).max())
        worst_rel = max(worst_rel, rel)
        if n == 2:
            ok_exact &= torch.equal(got, single)
        # n>=3 accumulates 3p, 5p, ... which are not representable, so the mean is
        # exact only up to rounding; assert it never exceeds 2 ulp of float32.
        ok_ulp &= rel <= 2 * 2 ** -23
    check("averaging N identical predictions == the single prediction (N=2, bit-exact)",
          ok_exact)
    check("...and for N=4,8 to within 2 ulp (sequential fp summation)",
          ok_ulp, f"max rel dev = {worst_rel:.2e}")

    check("preset 'none' == plain composited model forward, bit-exact",
          torch.equal(single, torch.where(m > 0.5, blur(v, m), v)))

    print("\n=== 4. equivariant model: every transform is inverted correctly ===")
    ok_eq, worst = True, 0.0
    for preset in ["mirror", "shift1", "shift", "crop", "crop3", "crop_only", "max"]:
        sp = PRESETS[preset]
        sp = TTASpec(**{**sp.__dict__, "crop_size": (16, 32, 32), "crop_jitter": 4})
        o = predict_tta(blur, v, m, sp)
        d = float((o - single).abs().max())
        worst = max(worst, d)
        ok_eq &= d < 1e-6
    check("equivariant stub: TTA mean == identity prediction for all presets",
          ok_eq, f"max |diff| = {worst:.2e}")

    # NEGATIVE CONTROL: the check above only has power if a WRONG inverse fails it.
    class WrongFlip(Flip):
        def inv(self, xx, fill):
            return torch.flip(xx, dims=(-1,))          # inverts the wrong axis

    class WrongShift(Shift):
        def inv(self, xx, fill):
            return self._apply(xx, self.shift, fill)   # forgets to negate

    bad = []
    for ops, tag in [((WrongFlip(LR_DIM),), "flip inverted on the wrong axis"),
                     ((WrongShift((0, 4, 4), "fill"),), "shift inverse not negated")]:
        vw = View("bad", ops, 1)
        w = vw.support(v.shape, v.device)
        vt, mt = vw.fwd_inputs(v, m, BG_VALUE)
        p = vw.inv_pred(blur(vt, mt).float())
        num = single.float() * torch.ones_like(w) + p * w
        den = torch.ones_like(w) + w
        o = torch.where(m > 0.5, (num / den).to(v.dtype), v)
        bad.append(float((o - single).abs().max()) > 1e-4)
    check("negative control: deliberately wrong inverses DO break the equality",
          all(bad), "flip-wrong-axis and un-negated shift both detected")

    print("\n=== 5. analytic mean over non-equivariant predictions ===")
    ramp = RampStub()
    H = v.shape[-2]
    r = RampStub.ramp(H, v.device, v.dtype).expand_as(v)
    # mirror: prediction of the flipped view, un-flipped, is ramp(H-1-h)
    o = predict_tta(ramp, v, m, TTASpec(mirror=True))
    exp = torch.where(m > 0.5, 0.5 * (r + torch.flip(r, dims=(-2,))), v)
    check("mirror: mean == 0.5*(ramp(h) + ramp(H-1-h))  (closed form)",
          torch.allclose(o, exp, atol=1e-6), f"max|d|={float((o - exp).abs().max()):.2e}")
    # shift by +dh along H: un-shifted prediction is ramp(h + dh) on the support
    dh = 4
    o = predict_tta(ramp, v, m, TTASpec(mirror=False, shifts=((0, dh, 0),)))
    r_sh = Shift((0, dh, 0), "fill").inv(r, 0.0)       # ramp(h+dh) on the support
    exp = torch.where(m > 0.5, 0.5 * (r + r_sh), v)
    check(f"shift(+{dh} along H): mean == 0.5*(ramp(h) + ramp(h+{dh}))  (closed form)",
          torch.allclose(o, exp, atol=1e-6), f"max|d|={float((o - exp).abs().max()):.2e}")
    # a sign error would give ramp(h - dh); confirm that is a different tensor
    r_wrong = Shift((0, -dh, 0), "fill").inv(r, 0.0)
    check("shift sign is observable (ramp(h+dh) != ramp(h-dh) inside the void)",
          not torch.allclose((r_sh * m)[m > 0.5], (r_wrong * m)[m > 0.5], atol=1e-6))
    # crop at offset y0: un-cropped prediction is ramp(h - y0) inside the box
    sp = TTASpec(full=False, mirror=False, n_crops=1, crop_size=(16, 32, 32), crop_jitter=4)
    views = build_views(sp, tuple(v.shape[-3:]), m)
    cbox = views[0].ops[0].box
    o = predict_tta(ramp, v, m, sp)
    r_cr = torch.zeros_like(v)
    r_cr[..., cbox[0][0]:cbox[0][1], cbox[1][0]:cbox[1][1], cbox[2][0]:cbox[2][1]] = \
        RampStub.ramp(cbox[1][1] - cbox[1][0], v.device, v.dtype)
    exp = torch.where(m > 0.5, r_cr, v)
    check("crop: prediction lands back at ramp(h - y0) inside the window (closed form)",
          torch.allclose(o, exp, atol=1e-6), f"box={cbox}")

    print("\n=== 6. view construction, coverage, cost accounting ===")
    sp = TTASpec(n_crops=3, crop_size=(16, 32, 32), crop_jitter=4)
    views = build_views(sp, tuple(v.shape[-3:]), m)
    lo, hi = _void_bbox(m)
    ok_cov = True
    for vw in views:
        for op in vw.ops:
            if isinstance(op, Crop):
                ok_cov &= all(op.box[a][0] <= lo[a] and hi[a] < op.box[a][1] for a in range(3))
    check("every crop window fully contains the void bounding box", ok_cov,
          f"{sum(1 for w in views if any(isinstance(o, Crop) for o in w.ops))} crop views")

    o, info = predict_tta(blur, v, m, sp, return_info=True)
    check("void denominator is constant (== n views) -> no seam inside the void",
          info["n_forward"] == len(views) and not info["skipped"],
          f"n_forward={info['n_forward']} voxel_cost={info['voxel_cost']:.2f}")

    cm = CountingModel(blur)
    predict_tta(cm, v, m, "mirror")
    check("preset 'mirror' issues exactly 2 forward passes", cm.n == 2)
    cm = CountingModel(blur)
    _, info = predict_tta(cm, v, m,
                          TTASpec(shifts=((0, 4, 4), (0, -4, -4)), n_crops=3,
                                  crop_size=(16, 32, 32), crop_jitter=4),
                          return_info=True)
    check("preset 'max' issues exactly 12 forward passes", cm.n == 12,
          f"shapes seen: {sorted(set(cm.shapes))}")

    # void too large for the crop -> crop views are dropped, result still valid
    v2, m2 = make_case(box=((4, 28), (4, 44), (4, 44)))
    o2, info2 = predict_tta(blur, v2, m2,
                            TTASpec(n_crops=3, crop_size=(16, 32, 32), crop_jitter=4),
                            return_info=True)
    check("void bigger than the crop -> crop views dropped, full views still used",
          info2["n_forward"] == 2 and torch.equal(o2[m2 <= 0.5], v2[m2 <= 0.5]),
          f"views={info2['views']}")

    v3, m3 = make_case()
    v3[..., -1, :, :] = 0.9                             # tissue on the LAST D slice
    _, i_pos = predict_tta(blur, v3, m3,
                           TTASpec(mirror=False, shifts=((4, 0, 0),)), return_info=True)
    _, i_neg_ok = predict_tta(blur, v3, m3,
                              TTASpec(mirror=False, shifts=((-4, 0, 0),)), return_info=True)
    v4, m4 = make_case()
    v4[..., 0, :, :] = 0.9                              # tissue on the FIRST D slice
    _, j_neg = predict_tta(blur, v4, m4,
                           TTASpec(mirror=False, shifts=((-4, 0, 0),)), return_info=True)
    check("shift that would push tissue out of the FOV is skipped (both directions)",
          any("clips brain" in s for s in i_pos["skipped"])
          and any("clips brain" in s for s in j_neg["skipped"]),
          f"+4: {i_pos['skipped']} | -4: {j_neg['skipped']}")
    check("the opposite-direction shift over the same tissue is NOT skipped "
          "(band selection has power)",
          not i_neg_ok["skipped"] and i_neg_ok["n_forward"] == 2,
          f"views={i_neg_ok['views']}")

    print("\n=== 7. shape/size: view construction + documented cost table ===")
    prod = (160, 256, 256)
    m_p = torch.zeros(1, 1, *prod)
    m_p[..., 70:96, 96:144, 104:152] = 1.0              # a realistic unilateral void
    n_full_p = prod[0] * prod[1] * prod[2]
    expected = {"none": (1, 1.00), "mirror": (2, 2.00), "shift1": (4, 4.00),
                "shift": (6, 6.00), "crop": (4, 3.19), "crop3": (8, 5.56),
                "crop_only": (6, 3.56), "max": (12, 9.56)}
    ok_tbl, rows = True, []
    for preset, (n_exp, c_exp) in expected.items():
        vws = build_views(PRESETS[preset], prod, m_p)
        cost = sum(w.n_voxels for w in vws) / n_full_p
        ok_tbl &= (len(vws) == n_exp and abs(cost - c_exp) < 0.005)
        rows.append(f"{preset}={len(vws)}fwd/{cost:.2f}x")
    check("docstring cost table matches build_views at 160x256x256 / crop 144x208x208",
          ok_tbl, "  ".join(rows))

    vws = build_views(PRESETS["crop3"], prod, m_p)
    boxes = [o.box for w in vws for o in w.ops if isinstance(o, Crop)]
    lo_p, hi_p = _void_bbox(m_p)
    check("production crops: 3 distinct 144x208x208 windows, all containing the void",
          len(set(boxes)) == 3
          and all(b[a][1] - b[a][0] == (144, 208, 208)[a] for b in boxes for a in range(3))
          and all(b[a][0] <= lo_p[a] and hi_p[a] < b[a][1] for b in boxes for a in range(3)),
          f"{sorted(set(boxes))}")

    # full=False + a void too large to crop must degrade, not crash
    m_big2 = torch.ones(1, 1, *prod)
    vws = build_views(PRESETS["crop_only"], prod, m_big2)
    check("full=False with an uncroppable void falls back to the full view (no crash)",
          [w.name for w in vws] == ["id[crop-fallback]", "id[crop-fallback]+mirror"],
          f"{[w.name for w in vws]}")
    del m_p, m_big2

    print("\n=== 8. determinism, dtype/device, guards ===")
    a = predict_tta(blur, v, m, "mirror")
    b = predict_tta(blur, v, m, "mirror")
    check("deterministic: identical inputs -> bit-identical output", torch.equal(a, b))
    check("dtype/device/shape preserved",
          a.dtype == v.dtype and a.device == v.device and a.shape == v.shape)
    check("output is finite", bool(torch.isfinite(a).all()))

    class BF16Stub(BlurStub):
        def forward(self, vv, mm):
            return super().forward(vv, mm).to(torch.bfloat16)

    ab = predict_tta(BF16Stub(), v, m, "mirror")
    check("bf16 model output: result stays float32, known tissue still exact",
          ab.dtype == v.dtype and torch.equal(ab[m <= 0.5], v[m <= 0.5])
          and float((ab - a).abs().max()) < 1e-2,
          f"max|bf16 - fp32| = {float((ab - a).abs().max()):.2e}")

    try:
        predict_tta(blur, v, (m * 0.5), "mirror")
        ok = False
    except ValueError:
        ok = True
    check("non-binary mask is rejected", ok)
    try:
        predict_tta(blur, v, m, "does_not_exist")
        ok = False
    except KeyError:
        ok = True
    check("unknown preset is rejected", ok)
    try:
        predict_tta(blur, v, m, TTASpec(n_crops=1, crop_size=(16, 30, 32)))
        ok = False
    except ValueError:
        ok = True
    check("crop size not divisible by 16 is rejected", ok)
    try:
        predict_tta(blur, torch.cat([v, v]), torch.cat([m, m]),
                    TTASpec(n_crops=1, crop_size=(16, 32, 32)))
        ok = False
    except ValueError:
        ok = True
    check("crop views with batch>1 are rejected", ok)
    ok_b = torch.equal(
        predict_tta(blur, torch.cat([v, v]), torch.cat([m, m]), "mirror")[1:],
        predict_tta(blur, v, m, "mirror"))
    check("batch>1 works for flip/shift views and is per-sample consistent", ok_b)

    print("\n=== 9. parity with the production path (predict_mirror_consistent) ===")
    class MirrorRef(BlurStub):
        @torch.no_grad()
        def predict_mirror_consistent(self, vv, mm):
            p1 = self(vv, mm)
            p2 = torch.flip(self(torch.flip(vv, dims=[-2]), torch.flip(mm, dims=[-2])),
                            dims=[-2])
            return 0.5 * (p1 + p2)

    ref = MirrorRef()
    ours = predict_tta(ref, v, m, "mirror")
    theirs = ref.predict_mirror_consistent(v, m)
    check("preset 'mirror' reproduces predict_mirror_consistent inside the void",
          torch.equal(ours[m > 0.5], theirs[m > 0.5]), "bit-exact")

    print("\n=== 10. real DirectInpaintModel (wudit) on CPU ===")
    try:
        from wavedit.models.direct_unet import DirectInpaintModel
        torch.manual_seed(1)
        real = DirectInpaintModel(base=8, levels=2, dropout=0.0, arch="wudit",
                                  udit_blocks=1, udit_downsample=1, udit_d_head=8,
                                  in_contra=True, nonlocal_healthy=True,
                                  udit_tokenrep=True, contra_attn=True).eval()
        vr, mr = make_case(shape=(32, 48, 48), box=((12, 20), (18, 30), (18, 30)))
        ref_id = predict_tta(real, vr, mr, "none")
        ok_real, notes = True, []
        for preset, sp in [("mirror", PRESETS["mirror"]),
                           ("shift1", PRESETS["shift1"]),
                           ("crop", TTASpec(n_crops=1, crop_size=(16, 32, 32), crop_jitter=4)),
                           ("crop_only", TTASpec(full=False, n_crops=2,
                                                 crop_size=(16, 32, 32), crop_jitter=4))]:
            o, inf = predict_tta(real, vr, mr, sp, return_info=True)
            ok_real &= (o.shape == vr.shape and o.dtype == vr.dtype
                        and bool(torch.isfinite(o).all())
                        and torch.equal(o[mr <= 0.5], vr[mr <= 0.5])
                        and float(o.abs().max()) <= 1.0 + 1e-6)
            notes.append(f"{preset}:{inf['n_forward']}fwd")
        check("real wudit model: shapes/dtype/range OK, known tissue exact, all presets run",
              ok_real, ", ".join(notes))
        d = float((predict_tta(real, vr, mr, "mirror") - ref_id).abs().max())
        check("real model is NOT flip-equivariant -> mirror TTA genuinely averages",
              d > 1e-4, f"max|TTA - identity| = {d:.3e} (a real ensemble, not a no-op)")
    except Exception as e:                                # pragma: no cover
        check("real wudit model on CPU", False, f"{type(e).__name__}: {e}")

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n===== {n_ok}/{len(results)} checks PASS =====")
    for name, ok, note in results:
        if not ok:
            print(f"  FAILED: {name} {note}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
