"""Mixup augmentation utilities for single-cell Perturb-seq using cell-load batches."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._perturbation import PerturbationDataset

from dataclasses import dataclass
from typing import Iterable, List, MutableMapping, Sequence, Tuple

import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pl = None  # type: ignore


def _as_broadcastable_lambda(lam: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape/squeeze `lam` for broadcasting onto `x`.

    - If x is (B, G) -> lam -> (B, 1)
    - If x is (B, N, G) -> lam -> (B, 1, 1)
    - If x is (B, ...) arbitrary rank >= 2 -> lam -> (B, *[1]*(x.ndim-1))
    """
    assert x.ndim >= 2, "x must have batch dimension first"
    B = x.shape[0]
    lam = lam.reshape(B)
    shape = [B] + [1] * (x.ndim - 1)
    return lam.view(*shape)


def _mix_linear(
    a: torch.Tensor, b: torch.Tensor, lam_bcast: torch.Tensor
) -> torch.Tensor:
    """Linear-space mix: a,b ~ counts or embeddings."""
    return lam_bcast * a + (1.0 - lam_bcast) * b


def _mix_log(
    a: torch.Tensor, b: torch.Tensor, lam_bcast: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Log-space mix: a,b ~ ln(counts + eps)"""
    return torch.log(lam_bcast * torch.exp(a) + (1.0 - lam_bcast) * torch.exp(b) + eps)


def _mix_log1p(
    a: torch.Tensor, b: torch.Tensor, lam_bcast: torch.Tensor, eps: float = 0.0
) -> torch.Tensor:
    """Log1p-space mix: a,b ~ ln(1 + counts)."""
    return torch.log1p(
        lam_bcast * torch.expm1(a) + (1.0 - lam_bcast) * torch.expm1(b) + eps
    )


def _choose_mix_fn(space: str):
    space = space.lower()
    if space == "linear":
        return _mix_linear
    if space == "log":
        return _mix_log
    if space == "log1p":
        return _mix_log1p
    raise ValueError(
        f"Unknown space '{space}'. Choose from ['linear', 'log', 'log1p']."
    )


@torch.no_grad()
def _resolve_pert_ids(
    batch: MutableMapping[str, torch.Tensor],
    pert_key_candidates: Sequence[str] = ("pert_id", "pert", "pert_idx", "pert_onehot"),
) -> torch.Tensor | None:
    """Try to extract a per-sample perturbation id vector of shape (B,).

    Supports:
      - Integer ids: LongTensor (B,) or (B,1) or (B,N) with identical across N
      - One-hot: FloatTensor (B, P) -> argmax
    Returns None if no candidate key is present.
    """
    key = next((k for k in pert_key_candidates if k in batch), None)
    if key is None:
        return None
    t = batch[key]
    if not isinstance(t, torch.Tensor):  # pragma: no cover - defensive
        return None

    if t.ndim == 1:
        return t.long()
    if t.ndim >= 2 and t.dtype in (torch.long, torch.int32, torch.int64):
        # If (B, N) or (B, 1): take first along non-batch dims
        return t.select(dim=1, index=0) if t.ndim >= 2 else t
    if t.ndim == 2 and t.dtype.is_floating_point:
        # one-hot / logits: (B, P)
        return torch.argmax(t, dim=1).long()
    # As a last resort: squeeze to (B,)
    return t.reshape(t.shape[0], -1)[:, 0].long()


@torch.no_grad()
def _groupwise_permutation(
    pert_ids: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Create a permutation index `perm` such that each index i is paired with a j
    from the same perturbation id when possible; else fallback to global shuffle.

    Args:
      pert_ids: (B,) LongTensor of group ids
      device:   torch.device for the output

    Returns:
      perm: (B,) LongTensor of partner indices
    """
    B = pert_ids.shape[0]
    perm = torch.empty(B, dtype=torch.long, device=device)

    # Build mapping from id -> indices
    unique_ids = pert_ids.unique()
    global_perm = torch.randperm(B, device=device)

    for pid in unique_ids.tolist():
        idx = torch.nonzero(pert_ids == pid, as_tuple=False).flatten()
        if idx.numel() == 1:
            # fallback: use global perm for this singleton
            perm[idx] = global_perm[idx]
            continue
        # random cycle within group ensuring no self-pairs
        shuffled = idx[torch.randperm(idx.numel(), device=device)]
        # If by chance any position equals itself, rotate by one (rare)
        clashes = shuffled == idx
        if clashes.any():
            shuffled = torch.roll(shuffled, shifts=1, dims=0)
        perm[idx] = shuffled

    return perm


@dataclass
class MixupSpec:
    alpha: float = 0.4
    p: float = 1.0  # probability to apply on a given batch
    space: str = "log1p"  # 'linear' | 'log' | 'log1p'
    same_pert_only: bool = True
    pert_key_candidates: Tuple[str, ...] = (
        "pert_id",
        "pert",
        "pert_idx",
        "pert_onehot",
    )
    feature_keys: Tuple[str, ...] = ("x_pre",)  # tensors to mix as inputs
    target_keys: Tuple[str, ...] = ("x_post",)  # tensors to mix as targets/labels
    extra_mix_keys: Tuple[str, ...] = tuple()  # any additional tensors to mix


@torch.no_grad()
def apply_mixup_to_batch(
    batch: MutableMapping[str, torch.Tensor], spec: MixupSpec
) -> MutableMapping[str, torch.Tensor]:
    """Apply Mixup to `batch` in-place (and also return it).

    - Samples lambda ~ Beta(alpha, alpha) independently per-sample.
    - Builds a pairing permutation (within-pert groups if enabled).
    - Mixes tensors for all keys in feature_keys, target_keys, extra_mix_keys.
    - Non-tensor entries are ignored.
    """
    if torch.rand(()) > spec.p:
        return batch

    device = None
    mix_keys: List[str] = []
    for k in (*spec.feature_keys, *spec.target_keys, *spec.extra_mix_keys):
        if k in batch and isinstance(batch[k], torch.Tensor):
            mix_keys.append(k)
            if device is None:
                device = batch[k].device
    if not mix_keys:
        return batch  # nothing to do

    if device is None:
        # default to CPU if somehow tensors are CPU and we didn't detect; defensive
        device = torch.device("cpu")

    # Determine batch size from first tensor
    ref = batch[mix_keys[0]]
    assert ref.ndim >= 2, "Expected tensors with batch dimension first"
    B = ref.shape[0]

    # Pairing permutation
    if spec.same_pert_only:
        pert_ids = _resolve_pert_ids(batch, spec.pert_key_candidates)
        if pert_ids is None:
            # Fall back to global shuffle if no perturbation id present
            perm = torch.randperm(B, device=device)
        else:
            perm = _groupwise_permutation(pert_ids.to(device=device), device)
    else:
        perm = torch.randperm(B, device=device)

    # Beta-distributed lambdas per sample
    lam = torch.distributions.Beta(spec.alpha, spec.alpha).sample((B,)).to(device)

    # Choose mixing function
    mix_fn = _choose_mix_fn(spec.space)

    # Apply mix for each key
    for k in mix_keys:
        x = batch[k]
        assert x.shape[0] == B, f"Key '{k}' has mismatched batch size"
        lam_b = _as_broadcastable_lambda(lam, x)
        x_perm = x[perm]
        batch[k] = mix_fn(x, x_perm, lam_b)

    batch.setdefault("mixup_lam", lam)
    batch.setdefault("mixup_perm", perm)

    return batch


class MixupCallback(nn.Module):
    """PyTorch Lightning-compatible callback-like module that applies Mixup.

    Args:
        alpha (float): Beta distribution parameter (default: 0.4).
        p (float): Probability of applying Mixup to a given batch (default: 1.0).
        space (str): Space in which to mix: 'linear', 'log', or 'log1p' (default: 'log1p').
        same_pert_only (bool): If True, only mix samples with the same perturbation id when possible (default: True).
        pert_key_candidates (Sequence[str]): Candidate keys to identify perturbation ids
            (default: ('pert_id', 'pert', 'pert_idx', 'pert_onehot')).
        feature_keys (Sequence[str]): Keys of tensors to mix as features (default: ('x_pre',)).
        target_keys (Sequence[str]): Keys of tensors to mix as targets/labels (default: ('x_post',)).
        extra_mix_keys (Sequence[str]): Any additional tensor keys to mix (default: ()).
    """

    def __init__(
        self,
        alpha: float = 0.4,
        p: float = 1.0,
        space: str = "log1p",
        same_pert_only: bool = True,
        pert_key_candidates: Sequence[str] = (
            "pert_id",
            "pert",
            "pert_idx",
            "pert_onehot",
        ),
        feature_keys: Sequence[str] = ("x_pre",),
        target_keys: Sequence[str] = ("x_post",),
        extra_mix_keys: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.spec = MixupSpec(
            alpha=alpha,
            p=p,
            space=space,
            same_pert_only=same_pert_only,
            pert_key_candidates=tuple(pert_key_candidates),
            feature_keys=tuple(feature_keys),
            target_keys=tuple(target_keys),
            extra_mix_keys=tuple(extra_mix_keys),
        )

    def forward(
        self, batch: MutableMapping[str, torch.Tensor]
    ) -> MutableMapping[str, torch.Tensor]:
        return apply_mixup_to_batch(batch, self.spec)

    if pl is not None:  # type: ignore

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
            """Apply Mixup in-place on the training batch before the step."""
            if isinstance(batch, MutableMapping):
                apply_mixup_to_batch(batch, self.spec)


def wrap_collate_with_mixup(
    base_collate_fn,
    *,
    alpha: float = 0.4,
    p: float = 1.0,
    space: str = "log1p",
    same_pert_only: bool = True,
    pert_key_candidates: Sequence[str] = ("pert_id", "pert", "pert_idx", "pert_onehot"),
    feature_keys: Sequence[str] = ("x_pre",),
    target_keys: Sequence[str] = ("x_post",),
    extra_mix_keys: Sequence[str] = (),
):
    """Return a collate_fn that first uses `base_collate_fn` then applies Mixup."""

    spec = MixupSpec(
        alpha=alpha,
        p=p,
        space=space,
        same_pert_only=same_pert_only,
        pert_key_candidates=tuple(pert_key_candidates),
        feature_keys=tuple(feature_keys),
        target_keys=tuple(target_keys),
        extra_mix_keys=tuple(extra_mix_keys),
    )

    def _collate_with_mixup(samples: Iterable):
        batch = base_collate_fn(samples)
        if isinstance(batch, MutableMapping):
            apply_mixup_to_batch(batch, spec)
        return batch

    return _collate_with_mixup


class MixupPerturbationDataset(torch.utils.data.Dataset):
    """Applies Mixup over a `PerturbationDataset` (or a `torch.utils.data.Subset` of it)
    *before* batching. It groups indices by perturbation **code** using
    `metadata_cache.pert_codes`—so partner selection is efficient and does not
    touch HDF5 during grouping.

    Defaults (override as needed):
      - Mix features:  'pert_cell_emb', 'ctrl_cell_emb', 'pert_emb' (soft-labels)
      - Mix targets:   'pert_cell_counts', 'ctrl_cell_counts'
      - Spaces:        'linear' for embeddings/one-hots, 'log1p' for *_counts

    If your counts are already log1p or log, set `spaces_by_key={'pert_cell_counts': 'linear', ...}`.
    """

    def __init__(
        self,
        base: torch.utils.data.Dataset,  # PerturbationDataset or Subset
        *,
        alpha: float = 0.4,
        p: float = 1.0,
        same_pert_only: bool = True,
        feature_keys: Sequence[str] = ("pert_cell_emb", "ctrl_cell_emb", "pert_emb"),
        target_keys: Sequence[str] = ("pert_cell_counts", "ctrl_cell_counts"),
        spaces_by_key: dict[str, str]
        | None = None,  # per-key: 'linear' | 'log' | 'log1p'
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.base = base
        self.alpha = alpha
        self.p = p
        self.same_pert_only = same_pert_only
        self.feature_keys = tuple(feature_keys)
        self.target_keys = tuple(target_keys)
        self.spaces_by_key = spaces_by_key or {}
        self._rng = torch.Generator().manual_seed(seed)

        # unwrap Subset → (dataset, indices)
        if isinstance(base, torch.utils.data.Subset):
            self._pd = base.dataset  # type: ignore
            self._indices = list(base.indices)  # dataset-relative indices
            self._subset_mode = True
        else:
            self._pd = base  # type: ignore
            self._indices = None
            self._subset_mode = False

        # precompute groups by perturbation code using metadata_cache
        self._groups: dict[int, List[int]] = {}
        self._build_groups()

    def __len__(self) -> int:
        return len(self.base)  # type: ignore

    def _file_idx_from_local(self, local_idx: int) -> int:
        # local_idx -> base dataset index -> file_idx used inside H5
        if self._subset_mode and self._indices is not None:
            base_idx = self._indices[local_idx]
        else:
            base_idx = local_idx
        return int(self._pd.all_indices[base_idx])  # type: ignore

    def _build_groups(self) -> None:
        mc = self._pd.metadata_cache  # type: ignore
        groups: dict[int, List[int]] = {}
        for i in range(len(self)):
            file_idx = self._file_idx_from_local(i)
            pid = int(mc.pert_codes[file_idx])
            groups.setdefault(pid, []).append(i)
        self._groups = groups

    def _draw_partner(self, i: int) -> int:
        N = len(self)
        if N <= 1:
            return i

        if self.same_pert_only:
            pid = int(self._pd.metadata_cache.pert_codes[self._file_idx_from_local(i)])  # type: ignore
            lst = self._groups.get(pid, [])
            if len(lst) > 1:
                # Remove current index from candidates
                candidates = [idx for idx in lst if idx != i]
                if candidates:
                    r = int(
                        torch.randint(
                            0, len(candidates), (1,), generator=self._rng
                        ).item()
                    )
                    return candidates[r]

        # fallback: global partner
        j = int(torch.randint(0, N - 1, (1,), generator=self._rng).item())
        if j >= i:
            j += 1  # Skip the current index
        return j

    def _space_for(self, key: str) -> str:
        if key in self.spaces_by_key:
            return self.spaces_by_key[key]
        # sensible defaults
        if key.endswith("_counts"):
            return "log1p"
        return "linear"

    def __getitem__(self, idx: int):
        item = self.base[idx]
        # Respect probability p
        if torch.rand((), generator=self._rng) > self.p:
            return item

        mate_idx = self._draw_partner(idx)
        mate = self.base[mate_idx]

        lam = (
            torch.distributions.Beta(self.alpha, self.alpha)
            .sample(
                (1,)
            )  # Remove generator parameter as it's not supported in older PyTorch
            .item()
        )

        out = dict(item)
        for k in (*self.feature_keys, *self.target_keys):
            x = item.get(k, None)
            y = mate.get(k, None)
            if x is None or y is None:
                continue
            if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
                continue

            # Ensure tensors are compatible
            if x.shape != y.shape:
                continue  # Skip incompatible shapes

            # Ensure both tensors are on the same device
            if x.device != y.device:
                y = y.to(x.device)

            mix_fn = _choose_mix_fn(self._space_for(k))
            lam_t = torch.tensor(lam, dtype=x.dtype, device=x.device)
            out[k] = mix_fn(x, y, lam_t)  # scalar broadcasts to vector/matrix

        out["mixup_lam"] = torch.tensor(lam)
        out["mixup_partner_local_idx"] = torch.tensor(mate_idx)
        try:
            name_a = item.get("pert_name", None)
            name_b = mate.get("pert_name", None)
            if name_a is not None and name_b is not None:
                out.setdefault("mixup_pert_pair", (name_a, name_b))
        except (KeyError, TypeError, AttributeError):
            # More specific exception handling for expected errors
            pass
        return out
