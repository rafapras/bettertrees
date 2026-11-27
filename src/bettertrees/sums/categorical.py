"""Optimal categorical splits via Fisher / Newton ratio ordering (Spec R5).

For binary classification under second-order Newton loss, each category ``c``
inside a leaf carries sufficient statistics ``(G_c, H_c, W_c)``. Ordering active
categories (``W_c > 0``) by the smoothed Newton ratio

    r_c = G_c / (H_c + cat_smooth)

and scanning the ``C - 1`` prefixes finds a binary subset partition
``x_j in S_L`` vs ``x_j not in S_L`` in ``O(C log C)`` time. Because
``G^2 / (H + lam)`` is jointly convex in ``(G, H)``, when ``cat_smooth = 0``
the scan recovers the exact globally optimal subset partition among all
``2^{C-1} - 1`` bipartitions (matching exhaustive search).

Numeric features keep standard ``uint8`` bins; without categorical features,
``CategoricalFIGSClassifier`` delegates to ``FIGSClassifier`` bit for bit.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
from numba import njit

from .._typing import ArrayLike, FeatureNames, FloatArray, SelfT
from ..bins import fit_bin_edges, transform_bins_row_major
from ._common import as_target, base_margin, bin_threshold
from ._kernels import node_hist_margin
from .smalltrees import InterleavedTreeClassifier, SmallTree

MAX_CAT_CODES = 253


@njit(cache=True)
def _cat_score(G, H, lam):
    return G * G / (H + lam)


@njit(cache=True)
def _best_num_hist(hist, nb, lam, min_weight):
    G, H, W = 0.0, 0.0, 0.0
    for b in range(nb):
        G += hist[b, 0]
        H += hist[b, 1]
        W += hist[b, 2]
    parent = _cat_score(G, H, lam)
    best_gain = 0.0
    best_t = -1
    GL, HL, WL = 0.0, 0.0, 0.0
    for t in range(nb - 1):
        GL += hist[t, 0]
        HL += hist[t, 1]
        WL += hist[t, 2]
        if min_weight > WL or min_weight > W - WL:
            continue
        gain = _cat_score(GL, HL, lam) + _cat_score(G - GL, H - HL, lam) - parent
        if gain > best_gain:
            best_gain = gain
            best_t = t
    return best_gain, best_t


@njit(cache=True)
def _best_cat_hist(
    hist,
    nc,
    lam,
    cat_smooth,
    min_weight,
    min_cat_weight=0.0,
    cat_dof_penalty=0.0,
    max_subset_size=0,
):
    """Fisher-ordered scan over active categories (0..nc-1); returns (gain, left_mask)."""
    G, H, W = 0.0, 0.0, 0.0
    n_kept = 0
    n_rare = 0
    G_rare, H_rare, W_rare = 0.0, 0.0, 0.0
    cat_to_grp = np.full(nc, -1, dtype=np.int64)
    for c in range(nc):
        wc = hist[c, 2]
        G += hist[c, 0]
        H += hist[c, 1]
        W += wc
        if wc > 0.0:
            if wc < min_cat_weight:
                G_rare += hist[c, 0]
                H_rare += hist[c, 1]
                W_rare += wc
                n_rare += 1
            else:
                cat_to_grp[c] = n_kept
                n_kept += 1
    has_rare = 1 if n_rare > 0 else 0
    n_groups = n_kept + has_rare
    mask = np.ones(nc, dtype=np.bool_)
    if n_groups < 2:
        return 0.0, mask
    if has_rare:
        for c in range(nc):
            if hist[c, 2] > 0.0 and hist[c, 2] < min_cat_weight:
                cat_to_grp[c] = n_kept

    grp_G = np.empty(n_groups, dtype=np.float64)
    grp_H = np.empty(n_groups, dtype=np.float64)
    grp_W = np.empty(n_groups, dtype=np.float64)
    grp_id = np.empty(n_groups, dtype=np.int64)
    pos = 0
    for c in range(nc):
        wc = hist[c, 2]
        if wc > 0.0 and wc >= min_cat_weight:
            grp_G[pos] = hist[c, 0]
            grp_H[pos] = hist[c, 1]
            grp_W[pos] = wc
            grp_id[pos] = c
            pos += 1
    if has_rare:
        grp_G[pos] = G_rare
        grp_H[pos] = H_rare
        grp_W[pos] = W_rare
        grp_id[pos] = nc

    order = np.empty(n_groups, dtype=np.int64)
    ratios = np.empty(n_groups, dtype=np.float64)
    for u in range(n_groups):
        order[u] = u
        denom = grp_H[u] + cat_smooth
        if denom > 0.0:
            ratios[u] = grp_G[u] / denom
        elif grp_G[u] == 0.0:
            ratios[u] = 0.0
        elif grp_G[u] > 0.0:
            ratios[u] = 1e30
        else:
            ratios[u] = -1e30
    # Stable insertion sort (n_groups <= 255) by ratio ascending, tie-break by group id.
    for i in range(1, n_groups):
        key_r = ratios[i]
        key_u = order[i]
        key_id = grp_id[key_u]
        j = i - 1
        while j >= 0 and (
            ratios[j] > key_r or (ratios[j] == key_r and grp_id[order[j]] > key_id)
        ):
            ratios[j + 1] = ratios[j]
            order[j + 1] = order[j]
            j -= 1
        ratios[j + 1] = key_r
        order[j + 1] = key_u

    lam_eff = lam + cat_smooth
    parent = _cat_score(G, H, lam_eff)
    penalty = cat_dof_penalty * (n_groups - 2) if n_groups > 2 else 0.0
    best_gain = 0.0
    best_k = -1
    GL, HL, WL = 0.0, 0.0, 0.0
    best_WL = 0.0
    for k in range(n_groups - 1):
        u = order[k]
        GL += grp_G[u]
        HL += grp_H[u]
        WL += grp_W[u]
        if min_weight > WL or min_weight > W - WL:
            continue
        if max_subset_size > 0:
            left_cnt = k + 1
            right_cnt = n_groups - 1 - k
            m_sub = left_cnt if left_cnt <= right_cnt else right_cnt
            if m_sub > max_subset_size:
                continue
        raw_gain = (
            _cat_score(GL, HL, lam_eff)
            + _cat_score(G - GL, H - HL, lam_eff)
            - parent
        )
        gain = raw_gain - penalty
        if gain > best_gain:
            best_gain = gain
            best_k = k
            best_WL = WL
    if best_k < 0:
        return 0.0, mask
    grp_right = np.zeros(n_groups, dtype=np.bool_)
    for k in range(best_k + 1, n_groups):
        grp_right[order[k]] = True
    default_left = best_WL >= (W - best_WL)
    for c in range(nc):
        g_idx = cat_to_grp[c]
        if g_idx >= 0:
            if grp_right[g_idx]:
                mask[c] = False
        else:
            mask[c] = default_left
    return best_gain, mask


@njit(cache=True)
def best_mixed_cut_1d(
    hist,
    nb,
    is_cat,
    lam,
    cat_smooth,
    min_weight,
    min_cat_weight=0.0,
    cat_dof_penalty=0.0,
    max_subset_size=0,
):
    """Best cut across numeric and categorical features in one leaf histogram."""
    p = hist.shape[0]
    B = hist.shape[1]
    gains = np.zeros(p)
    num_cuts = np.full(p, -1, dtype=np.int64)
    cat_masks = np.ones((p, B), dtype=np.bool_)
    for j in range(p):
        if is_cat[j]:
            gain, m = _best_cat_hist(
                hist[j],
                nb[j],
                lam,
                cat_smooth,
                min_weight,
                min_cat_weight,
                cat_dof_penalty,
                max_subset_size,
            )
            gains[j] = gain
            for c in range(nb[j]):
                cat_masks[j, c] = m[c]
        else:
            gains[j], num_cuts[j] = _best_num_hist(hist[j], nb[j], lam, min_weight)
    return gains, num_cuts, cat_masks


@njit(cache=True)
def cat_tree_leaf_ids(Xb, feature, threshold, left, right, is_cat, cat_masks):
    """Leaf node of each row for a tree mixing numeric and categorical cuts."""
    n = Xb.shape[0]
    ids = np.empty(n, dtype=np.int64)
    for i in range(n):
        node = 0
        while left[node] != -1:
            f = feature[node]
            t = threshold[node]
            if is_cat[f]:
                go_left = cat_masks[t, Xb[i, f]]
            else:
                go_left = Xb[i, f] <= t
            node = left[node] if go_left else right[node]
        ids[i] = node
    return ids


def _is_missing_scalar(v):
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        return bool(np.isnan(v))
    try:
        import pandas as pd

        return bool(pd.isna(v))
    except Exception:
        return False


def _is_cat_dtype(dtype):
    try:
        import pandas as pd

        if isinstance(dtype, pd.CategoricalDtype):
            return True
        return bool(pd.api.types.is_string_dtype(dtype) or getattr(dtype, "kind", "") == "O")
    except Exception:
        return getattr(dtype, "kind", "") == "O"


def _to_float32_col(col, n):
    """Fast C-level numeric column conversion preserving NaN."""
    dtype = getattr(col, "dtype", None)
    if dtype is not None and getattr(dtype, "kind", "") in ("f", "i", "u", "b"):
        arr = np.asarray(col, dtype=np.float32)
        if arr.shape == (n,):
            return arr
    try:
        import pandas as pd

        if isinstance(col, pd.Series):
            return pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float32)
    except Exception:
        pass
    return np.fromiter(
        (np.nan if _is_missing_scalar(v) else float(v) for v in col),
        count=n,
        dtype=np.float32,
    )


def _fit_cat_col(col, n, max_categories):
    """Vectorized fit-time category mapping: 0=missing, 1=other/unseen, 2..C+1=kept."""
    try:
        import pandas as pd

        if isinstance(col, pd.Series) and isinstance(col.dtype, pd.CategoricalDtype):
            codes = col.cat.codes.to_numpy(dtype=np.int64, copy=False)
            uniques = col.cat.categories.to_list()
        else:
            codes, uniques_idx = pd.factorize(col, sort=False)
            codes = np.asarray(codes, dtype=np.int64)
            uniques = uniques_idx.tolist()
        if len(uniques) == 0:
            return [], {}, np.zeros(n, dtype=np.uint8)
        counts = np.bincount(codes[codes >= 0], minlength=len(uniques))
        active = [(uniques[i], int(counts[i])) for i in range(len(uniques)) if counts[i] > 0]
        ranked = sorted(active, key=lambda kv: (-kv[1], str(kv[0]), repr(kv[0])))
        kept = sorted((kv[0] for kv in ranked[:max_categories]), key=lambda v: (str(v), repr(v)))
        cmap = {val: code for code, val in enumerate(kept, 2)}
        lut = np.ones(len(uniques) + 1, dtype=np.uint8)
        lut[0] = 0  # codes == -1 -> missing (0); non-missing unkept -> other (1)
        for idx_u, val in enumerate(uniques):
            c = cmap.get(val, 1)
            lut[idx_u + 1] = c
        return kept, cmap, lut[codes + 1]
    except Exception:
        counts = {}
        for v in col:
            if _is_missing_scalar(v):
                continue
            counts[v] = counts.get(v, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]), repr(kv[0])))
        kept = sorted((kv[0] for kv in ranked[:max_categories]), key=lambda v: (str(v), repr(v)))
        cmap = {val: code for code, val in enumerate(kept, 2)}
        col_codes = np.fromiter(
            (0 if _is_missing_scalar(v) else cmap.get(v, 1) for v in col),
            count=n,
            dtype=np.uint8,
        )
        return kept, cmap, col_codes


def _transform_cat_col(col, n, kept, cmap):
    """Vectorized predict-time lookup: 0=missing, 1=other/unseen, 2..C+1=kept."""
    if not kept:
        try:
            import pandas as pd

            miss = pd.isna(col)
            return np.where(miss, 0, 1).astype(np.uint8)
        except Exception:
            return np.fromiter(
                (0 if _is_missing_scalar(v) else 1 for v in col),
                count=n,
                dtype=np.uint8,
            )
    try:
        import pandas as pd

        if isinstance(col, pd.Series) and isinstance(col.dtype, pd.CategoricalDtype):
            codes = col.cat.codes.to_numpy(dtype=np.int64, copy=False)
            cats = col.cat.categories.to_list()
            lut = np.ones(len(cats) + 1, dtype=np.uint8)
            lut[0] = 0
            for idx_u, val in enumerate(cats):
                lut[idx_u + 1] = cmap.get(val, 1)
            return lut[codes + 1]
        idx = pd.Index(kept).get_indexer(col)
        miss = np.asarray(pd.isna(col), dtype=bool)
        out = np.ones(n, dtype=np.uint8)
        ok = idx >= 0
        out[ok] = (idx[ok] + 2).astype(np.uint8)
        out[miss] = 0
        return out
    except Exception:
        return np.fromiter(
            (0 if _is_missing_scalar(v) else cmap.get(v, 1) for v in col),
            count=n,
            dtype=np.uint8,
        )


def _fit_lex_ord_col(col, n):
    """Map non-missing values to sorted string ranks (float32, NaN preserved)."""
    vals = []
    miss = np.zeros(n, dtype=bool)
    for i, v in enumerate(col):
        if _is_missing_scalar(v):
            miss[i] = True
            vals.append("")
        else:
            vals.append(str(v))
    arr = np.asarray(vals, dtype=object)
    non_miss = arr[~miss]
    if len(non_miss) == 0:
        return np.empty(0, dtype=object), np.full(n, np.nan, dtype=np.float32)
    vocab = np.unique(non_miss)
    ranks = np.full(n, np.nan, dtype=np.float32)
    ranks[~miss] = np.searchsorted(vocab, non_miss).astype(np.float32)
    return vocab, ranks


def _transform_lex_ord_col(col, n, vocab):
    """Map test strings to lexicographical position via searchsorted (leak-free)."""
    ranks = np.full(n, np.nan, dtype=np.float32)
    if len(vocab) == 0:
        return ranks
    vals = []
    miss = np.zeros(n, dtype=bool)
    for i, v in enumerate(col):
        if _is_missing_scalar(v):
            miss[i] = True
            vals.append("")
        else:
            vals.append(str(v))
    arr = np.asarray(vals, dtype=object)
    non_miss = arr[~miss]
    if len(non_miss):
        ranks[~miss] = np.searchsorted(vocab, non_miss, side="left").astype(np.float32)
    return ranks


def _top_k_coverage(col, max_categories):
    """Return (n_unique_non_null, top_k_coverage_fraction) for near-unique detection."""
    try:
        import pandas as pd

        if isinstance(col, pd.Series) and isinstance(col.dtype, pd.CategoricalDtype):
            codes = col.cat.codes.to_numpy(dtype=np.int64, copy=False)
            valid = codes[codes >= 0]
            if len(valid) == 0:
                return 0, 1.0
            counts = np.bincount(valid)
            counts = counts[counts > 0]
        else:
            codes, _ = pd.factorize(col, sort=False)
            codes = np.asarray(codes, dtype=np.int64)
            valid = codes[codes >= 0]
            if len(valid) == 0:
                return 0, 1.0
            counts = np.bincount(valid)
            counts = counts[counts > 0]
        n_uniq = int(len(counts))
        if n_uniq <= max_categories:
            return n_uniq, 1.0
        k_cut = n_uniq - max_categories
        top_sum = int(np.partition(counts, k_cut)[k_cut:].sum())
        return n_uniq, float(top_sum) / float(len(valid))
    except Exception:
        counts_d = {}
        total = 0
        for v in col:
            if _is_missing_scalar(v):
                continue
            counts_d[v] = counts_d.get(v, 0) + 1
            total += 1
        if total == 0 or len(counts_d) <= max_categories:
            return len(counts_d), 1.0
        top_sum = sum(sorted(counts_d.values(), reverse=True)[:max_categories])
        return len(counts_d), float(top_sum) / float(total)


class _CatTreeProxy:
    """Thin wrapper binding a ``SmallTree`` with mixed categorical/numeric leaf lookup."""

    def __init__(self, tree, is_cat, masks_arr):
        self._tree = tree
        self.feature = tree.feature
        self.threshold = tree.threshold
        self.left = tree.left
        self.right = tree.right
        self.is_cat = is_cat
        self.masks_arr = masks_arr

    @property
    def value(self):
        return self._tree.value

    @value.setter
    def value(self, val):
        self._tree.value = val

    def leaf_ids(self, Xb):
        return cat_tree_leaf_ids(
            Xb, *self._tree.arrays(), self.is_cat, self.masks_arr
        )


class CategoricalInterleavedTreeClassifier(InterleavedTreeClassifier):
    """FIGS with Fisher-optimal Newton categorical splits alongside numeric cuts.

    Parameters
    ----------
    max_splits, max_trees, lam, min_weight, max_bins, backfit_sweeps,
    learning_rate, max_delta_step
        As in ``FIGSClassifier``.
    categorical_features : "auto", sequence of int/str, or None, default="auto"
        Which columns are categorical. ``"auto"`` detects ``category``,
        ``object``, and ``string`` columns in a pandas DataFrame (or object/str
        columns in a NumPy array).
    cat_smooth : float, default=10.0
        Additive smoothing in the category ordering ratio ``G_c / (H_c + cat_smooth)``.
        ``0.0`` recovers exact unconstrained Fisher partitioning.
    max_categories : int, default=253
        Maximum number of explicit category codes ``2..C+1`` kept per column;
        ``NaN`` uses code ``0`` and rarer/unseen non-missing categories use
        separate code ``1`` (so unseen categories never inherit missing risk).
    min_cat_weight : float, default=10.0
        Categories inside a leaf with weight below ``min_cat_weight`` are
        pooled into one rare-in-leaf group before Fisher ratio ordering.
    cat_dof_penalty : float or "auto", default="auto"
        Rissanen MDL subset complexity penalty subtracted per extra category
        group above 2 (``cat_dof_penalty * max(0, n_groups - 2)``). ``"auto"``
        resolves to ``0.0`` when ``cat_smooth == 0.0`` (preserving exact
        unconstrained Fisher partitioning) and ``2 * ln(2)`` otherwise.
    max_subset_size : int >= 1 or None, default=None
        Optional cap on ``min(|S|, K_eff - |S|)`` per categorical split for
        compact human-readable rules (e.g. ``1`` for strict one-vs-rest
        ``x == c``, ``3`` for at most 3 categories per rule). ``None`` allows
        unconstrained Fisher+MDL subset partitions.
    min_cat_coverage : float in [0, 1], default=0.5
        Guardrail used when ``categorical_features="auto"``: if a string column
        has more than ``max_categories`` distinct values and the top
        ``max_categories`` cover less than ``min_cat_coverage`` of non-null
        rows (e.g. ISO-8601 timestamps or near-unique IDs), it is binned via
        leak-free lexicographical rank instead of collapsing into ``other``.
    """

    def __init__(
        self,
        *,
        max_splits: int = 16,
        max_trees: int | None = None,
        lam: float | Literal["auto"] = "auto",
        min_weight: float = 20.0,
        max_bins: int = 32,
        backfit_sweeps: int = 1,
        learning_rate: float = 1.0,
        max_delta_step: float | None = None,
        categorical_features: Literal["auto"] | Sequence[int | str] | None = "auto",
        cat_smooth: float = 10.0,
        max_categories: int = 253,
        min_cat_weight: float = 10.0,
        cat_dof_penalty: float | Literal["auto"] = "auto",
        max_subset_size: int | None = None,
        min_cat_coverage: float = 0.5,
    ) -> None:
        super().__init__(
            max_splits=max_splits,
            max_trees=max_trees,
            lam=lam,
            min_weight=min_weight,
            max_bins=max_bins,
            backfit_sweeps=backfit_sweeps,
            learning_rate=learning_rate,
            max_delta_step=max_delta_step,
        )
        self.categorical_features = categorical_features
        self.cat_smooth = cat_smooth
        self.max_categories = max_categories
        self.min_cat_weight = min_cat_weight
        self.cat_dof_penalty = cat_dof_penalty
        self.max_subset_size = max_subset_size
        self.min_cat_coverage = min_cat_coverage

    def _validate_cat(self):
        if (
            not isinstance(self.cat_smooth, Real)
            or isinstance(self.cat_smooth, bool)
            or not np.isfinite(self.cat_smooth)
            or float(self.cat_smooth) < 0.0
        ):
            raise ValueError("cat_smooth must be finite and >= 0.")
        if (
            isinstance(self.max_categories, bool)
            or not isinstance(self.max_categories, Integral)
            or not 1 <= self.max_categories <= MAX_CAT_CODES
        ):
            raise ValueError(f"max_categories must be in [1, {MAX_CAT_CODES}].")
        if (
            not isinstance(self.min_cat_weight, Real)
            or isinstance(self.min_cat_weight, bool)
            or not np.isfinite(self.min_cat_weight)
            or float(self.min_cat_weight) < 0.0
        ):
            raise ValueError("min_cat_weight must be finite and >= 0.")
        if self.max_subset_size is not None and (
            isinstance(self.max_subset_size, bool)
            or not isinstance(self.max_subset_size, Integral)
            or int(self.max_subset_size) < 1
        ):
            raise ValueError("max_subset_size must be None or an integer >= 1.")
        if (
            not isinstance(self.min_cat_coverage, Real)
            or isinstance(self.min_cat_coverage, bool)
            or not np.isfinite(self.min_cat_coverage)
            or not 0.0 <= float(self.min_cat_coverage) <= 1.0
        ):
            raise ValueError("min_cat_coverage must be finite and in [0, 1].")
        if self.cat_dof_penalty == "auto":
            self.cat_dof_penalty_ = (
                0.0 if float(self.cat_smooth) == 0.0 else float(2.0 * np.log(2.0))
            )
        elif (
            isinstance(self.cat_dof_penalty, Real)
            and not isinstance(self.cat_dof_penalty, bool)
            and np.isfinite(self.cat_dof_penalty)
            and float(self.cat_dof_penalty) >= 0.0
        ):
            self.cat_dof_penalty_ = float(self.cat_dof_penalty)
        else:
            raise ValueError("cat_dof_penalty must be 'auto' or finite and >= 0.")
        if (
            float(self.cat_smooth) == 0.0
            and self.cat_dof_penalty == "auto"
            and float(self.min_cat_weight) == 10.0
        ):
            self.min_cat_weight_ = 0.0
        else:
            self.min_cat_weight_ = float(self.min_cat_weight)

    def _resolve_cat_indices(self, X):
        cols = getattr(X, "columns", None)
        if cols is not None:
            p = len(cols)
            col_list = list(cols)
        else:
            arr = np.asarray(X)
            if arr.ndim != 2:
                raise ValueError("X must be a 2D array or DataFrame.")
            p = arr.shape[1]
            col_list = None

        spec = self.categorical_features
        if spec is None:
            return []
        if spec == "auto":
            if cols is not None:
                return [j for j in range(p) if _is_cat_dtype(X.iloc[:, j].dtype)]
            arr = np.asarray(X)
            if arr.dtype == object or arr.dtype.kind in ("U", "S"):
                # Detect per column in object arrays: non-numeric string/mixed cols
                out = []
                for j in range(p):
                    col = arr[:, j]
                    if arr.dtype.kind in ("U", "S") or any(isinstance(v, str) for v in col):
                        out.append(j)
                return out
            return []
        if isinstance(spec, (str, bytes)) or not hasattr(spec, "__iter__"):
            raise ValueError(
                "categorical_features must be 'auto', None, or a sequence of ints/names."
            )
        indices = []
        seen = set()
        for item in spec:
            if isinstance(item, str):
                if col_list is None or item not in col_list:
                    raise ValueError(f"Unknown categorical column {item!r}.")
                idx = int(col_list.index(item))
            elif isinstance(item, Integral) and not isinstance(item, bool):
                idx = int(item)
                if not 0 <= idx < p:
                    raise ValueError(f"Categorical index {idx} out of range [0, {p}).")
            else:
                raise ValueError(f"Invalid categorical feature specifier {item!r}.")
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        return indices

    def _extract_columns(self, X, reset):
        cols = getattr(X, "columns", None)
        if cols is not None:
            p = len(cols)
            n = len(X)
            names = np.asarray(cols, dtype=object)
            has_names = all(isinstance(c, str) for c in names)
            if reset:
                self.n_features_in_ = p
                if has_names:
                    self.feature_names_in_ = names
                else:
                    self.__dict__.pop("feature_names_in_", None)
            else:
                if p != self.n_features_in_:
                    raise ValueError(
                        f"X has {p} features, but {type(self).__name__} is expecting "
                        f"{self.n_features_in_} features as input."
                    )
                expected = getattr(self, "feature_names_in_", None)
                if expected is not None and has_names and list(names) != list(expected):
                    raise ValueError("Feature names passed at predict do not match fit.")
            col_arrays = [X.iloc[:, j] for j in range(p)]
            return n, p, col_arrays

        dtype = object if (not reset and self.is_categorical_.any()) else None
        arr = np.asarray(X, dtype=dtype)
        if arr.ndim != 2:
            raise ValueError("X must be a 2D array or DataFrame.")
        n, p = arr.shape
        if reset:
            self.n_features_in_ = p
            self.__dict__.pop("feature_names_in_", None)
        elif p != self.n_features_in_:
            raise ValueError(
                f"X has {p} features, but {type(self).__name__} is expecting "
                f"{self.n_features_in_} features as input."
            )
        return n, p, [arr[:, j] for j in range(p)]

    def _fit_prepare_mixed(self, X, y, sample_weight, y_soft):
        from sklearn.utils.multiclass import check_classification_targets, type_of_target
        from sklearn.utils.validation import _check_sample_weight

        cat_idx = set(self._resolve_cat_indices(X))
        n, p, col_arrays = self._extract_columns(X, reset=True)
        y_arr = np.asarray(y)
        if len(y_arr) != n:
            raise ValueError(
                f"Found input variables with inconsistent numbers of samples: {[n, len(y_arr)]}."
            )
        check_classification_targets(y_arr)
        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )
        classes, encoded = np.unique(y_arr, return_inverse=True)
        if len(classes) != 2:
            raise ValueError("Binary classification needs two classes; y has only one class.")
        target = encoded.astype(np.float64) if y_soft is None else as_target(y_soft, n)
        w = _check_sample_weight(sample_weight, np.empty((n, 1)), dtype=np.float64,
                                 ensure_non_negative=True)
        if not w.sum() > 0:
            raise ValueError("sample_weight sums to zero.")

        self._ord_vocabs = {}
        if self.categorical_features == "auto" and float(self.min_cat_coverage) > 0.0:
            guarded = set()
            for j in sorted(cat_idx):
                n_uniq, cov = _top_k_coverage(col_arrays[j], int(self.max_categories))
                if n_uniq > int(self.max_categories) and cov < float(self.min_cat_coverage):
                    guarded.add(j)
            cat_idx = cat_idx - guarded
        else:
            guarded = set()

        is_cat = np.array([j in cat_idx for j in range(p)], dtype=bool)
        self.is_categorical_ = is_cat
        self.classes_ = classes
        self.categories_ = {}
        self._cat_maps = {}
        self.nan_features_ = np.zeros(p, dtype=bool)
        edges = [np.empty(0, dtype=np.float64) for _ in range(p)]
        Xb = np.empty((n, p), dtype=np.uint8)
        nb = np.empty(p, dtype=np.int64)

        num_cols = [j for j in range(p) if not is_cat[j]]
        if num_cols:
            X_num = np.empty((n, len(num_cols)), dtype=np.float32)
            for idx_num, j in enumerate(num_cols):
                if j in guarded:
                    vocab, ranks = _fit_lex_ord_col(col_arrays[j], n)
                    self._ord_vocabs[j] = vocab
                    X_num[:, idx_num] = ranks
                else:
                    X_num[:, idx_num] = _to_float32_col(col_arrays[j], n)
            X_num32 = np.ascontiguousarray(X_num, dtype=np.float32)
            num_edges = fit_bin_edges(X_num32, self.max_bins)
            Xb_num = transform_bins_row_major(X_num32, num_edges)
            for idx_num, j in enumerate(num_cols):
                edges[j] = num_edges[idx_num]
                Xb[:, j] = Xb_num[:, idx_num]
                nb[j] = len(num_edges[idx_num]) + 2
                self.nan_features_[j] = bool(np.isnan(X_num32[:, idx_num]).any())

        for j in range(p):
            if not is_cat[j]:
                continue
            kept, cmap, col_codes = _fit_cat_col(col_arrays[j], n, int(self.max_categories))
            self.categories_[j] = kept
            self._cat_maps[j] = cmap
            Xb[:, j] = col_codes
            nb[j] = len(kept) + 2
            self.nan_features_[j] = bool((col_codes == 0).any())

        self.bin_edges_ = tuple(edges)
        return np.ascontiguousarray(Xb), nb, target, np.ascontiguousarray(w)

    def _transform_mixed(self, X):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "trees_")
        ord_vocabs = getattr(self, "_ord_vocabs", {})
        if not self.is_categorical_.any() and not ord_vocabs:
            from ._common import predict_input, rebin

            return rebin(predict_input(self, X, "trees_"), self.bin_edges_)

        n, p, col_arrays = self._extract_columns(X, reset=False)
        Xb = np.empty((n, p), dtype=np.uint8)
        num_cols = [j for j in range(p) if not self.is_categorical_[j]]
        if num_cols:
            X_num = np.empty((n, len(num_cols)), dtype=np.float32)
            for idx_num, j in enumerate(num_cols):
                if j in ord_vocabs:
                    X_num[:, idx_num] = _transform_lex_ord_col(col_arrays[j], n, ord_vocabs[j])
                else:
                    X_num[:, idx_num] = _to_float32_col(col_arrays[j], n)
            num_edges = tuple(self.bin_edges_[j] for j in num_cols)
            X_num32 = np.ascontiguousarray(X_num, dtype=np.float32)
            Xb_num = transform_bins_row_major(X_num32, num_edges)
            for idx_num, j in enumerate(num_cols):
                Xb[:, j] = Xb_num[:, idx_num]
        for j in range(p):
            if not self.is_categorical_[j]:
                continue
            Xb[:, j] = _transform_cat_col(
                col_arrays[j], n, self.categories_[j], self._cat_maps[j]
            )
        return np.ascontiguousarray(Xb)

    def _cat_masks_matrix(self):
        max_nc = int(max((len(v) + 2 for v in self.categories_.values()), default=2))
        if not self.cat_splits_:
            return np.ones((1, max_nc), dtype=np.bool_)
        arr = np.ones((len(self.cat_splits_), max_nc), dtype=np.bool_)
        for idx, mask in enumerate(self.cat_splits_):
            arr[idx, : len(mask)] = mask
        return arr

    def _tree_leaf_ids(self, tree, Xb, masks_arr=None):
        if not self.is_categorical_.any():
            return tree.leaf_ids(Xb)
        if masks_arr is None:
            masks_arr = self._cat_masks_matrix()
        return cat_tree_leaf_ids(Xb, *tree.arrays(), self.is_categorical_, masks_arr)

    def _grow_mixed(self, Xb, nb, target, w):
        n, B = len(Xb), int(nb.max())
        trees, contribs, ids_of = [], [], []
        margin = np.full(n, self.base_margin_, dtype=np.float64)
        target = np.ascontiguousarray(target, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        root_ids, zero = np.zeros(n, dtype=np.int64), np.zeros(n)
        is_cat = np.ascontiguousarray(self.is_categorical_, dtype=np.bool_)
        cat_smooth = float(self.cat_smooth)
        min_cat_w = float(self.min_cat_weight_)
        dof_pen = float(self.cat_dof_penalty_)
        max_sub = 0 if self.max_subset_size is None else int(self.max_subset_size)
        growth = []

        for _ in range(self.max_splits):
            # (gain, tree_idx, leaf, feature, num_threshold, cat_mask)
            best = (0.0, None, None, None, None, None)
            can_add = self.max_trees is None or len(trees) < self.max_trees
            candidates = list(range(len(trees))) + ([None] if can_add else [])
            for k in candidates:
                if k is None:
                    ids, contrib = root_ids, zero
                    leaves, n_nodes = [0], 1
                else:
                    ids, contrib = ids_of[k], contribs[k]
                    leaves, n_nodes = trees[k].leaves, len(trees[k].feature)
                hist = node_hist_margin(Xb, target, w, margin, contrib, ids, n_nodes, B)
                for leaf in leaves:
                    gains, num_cuts, cat_masks = best_mixed_cut_1d(
                        hist[leaf],
                        nb,
                        is_cat,
                        self.lam_,
                        cat_smooth,
                        self.min_weight,
                        min_cat_w,
                        dof_pen,
                        max_sub,
                    )
                    f = int(np.argmax(gains))
                    if gains[f] > best[0]:
                        if is_cat[f]:
                            best = (
                                float(gains[f]),
                                k,
                                leaf,
                                f,
                                -1,
                                cat_masks[f, : nb[f]].copy(),
                            )
                        else:
                            best = (float(gains[f]), k, leaf, f, int(num_cuts[f]), None)
            _, k, leaf, f, t, mask = best
            if k is None and leaf is None:
                break
            if k is None:
                trees.append(SmallTree())
                contribs.append(np.zeros(n))
                ids_of.append(None)
                k = len(trees) - 1
            if mask is not None:
                sid = len(self.cat_splits_)
                self.cat_splits_.append(mask)
                trees[k].split(leaf, f, sid)
            else:
                trees[k].split(leaf, f, t)
            growth.append((k, int(leaf)))
            masks_arr = self._cat_masks_matrix()
            ids_of[k] = self._tree_leaf_ids(trees[k], Xb, masks_arr)
            contribs[k], margin = self._newton_step(
                trees[k], Xb, target, margin, contribs[k], w, ids_of[k]
            )
            if self.backfit_sweeps:
                margin = self._backfit(trees, contribs, Xb, target, w, self.backfit_sweeps, ids_of)
        self.growth_ = growth
        return trees

    def fit(
        self: SelfT,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        y_soft: ArrayLike | None = None,
    ) -> SelfT:
        """Fit FIGS with optional categorical Fisher partitioning."""
        self._validate_cat()
        cat_idx = self._resolve_cat_indices(X)
        if not cat_idx:
            res = super().fit(X, y, sample_weight=sample_weight, y_soft=y_soft)
            self.is_categorical_ = np.zeros(self.n_features_in_, dtype=bool)
            self.categories_ = {}
            self._cat_maps = {}
            self._ord_vocabs = {}
            self.cat_splits_ = []
            return res

        self.lam_ = 2.0 * self.max_splits if self.lam == "auto" else float(self.lam)
        self.learning_rate_ = float(self.learning_rate)
        self.cat_splits_ = []
        Xb, nb, target, w = self._fit_prepare_mixed(X, y, sample_weight, y_soft)
        self.base_margin_ = base_margin(target, w)
        self.trees_ = self._grow_mixed(Xb, nb, target, w)
        return self

    def decision_function(self, X: ArrayLike) -> FloatArray:
        if not self.is_categorical_.any() and not getattr(self, "_ord_vocabs", None):
            return super().decision_function(X)
        Xb = self._transform_mixed(X)
        masks_arr = self._cat_masks_matrix()
        m = np.full(len(Xb), self.base_margin_, dtype=np.float64)
        for tree in self.trees_:
            m += tree.value[self._tree_leaf_ids(tree, Xb, masks_arr)]
        return m

    def predict_contributions(self, X: ArrayLike) -> FloatArray:
        if not self.is_categorical_.any() and not getattr(self, "_ord_vocabs", None):
            return super().predict_contributions(X)
        Xb = self._transform_mixed(X)
        masks_arr = self._cat_masks_matrix()
        out = np.zeros((len(Xb), len(self.trees_)), dtype=np.float64)
        for k, tree in enumerate(self.trees_):
            out[:, k] = tree.value[self._tree_leaf_ids(tree, Xb, masks_arr)]
        return out

    def _format_cat_condition(self, f, allowed, names):
        nm = str(names[f]) if names is not None else f"x{f}"
        known = self.categories_[f]
        in_cats = [known[c - 2] for c in range(2, len(known) + 2) if allowed[c]]
        out_cats = [known[c - 2] for c in range(2, len(known) + 2) if not allowed[c]]
        inc_null = bool(allowed[0])
        inc_other = bool(allowed[1])
        if inc_other:
            if not out_cats:
                return f"{nm} is any" if inc_null else f"{nm} is present"
            vals = ", ".join(repr(v) for v in out_cats)
            if inc_null:
                return f"({nm} not in {{{vals}}} or missing)"
            return f"({nm} is present and {nm} not in {{{vals}}})"
        if not in_cats:
            return f"{nm} is missing" if inc_null else f"{nm} in {{}}"
        vals = ", ".join(repr(v) for v in in_cats)
        if inc_null:
            return f"({nm} in {{{vals}}} or missing)"
        return f"{nm} in {{{vals}}}"

    def rules(
        self, feature_names: FeatureNames | None = None, precision: int = 6
    ) -> list[tuple[int, list[str], float]]:
        if not self.is_categorical_.any():
            return super().rules(feature_names=feature_names, precision=precision)
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "trees_")
        names = self._names(feature_names)
        nan = getattr(self, "nan_features_", None)
        name = (lambda j: str(names[j])) if names is not None else (lambda j: f"x{j}")
        num = lambda v: f"{v:.{precision}g}"
        out = []
        for k, tree in enumerate(self.trees_):
            def walk(node, path, tree=tree, k=k):
                if tree.left[node] == -1:
                    bounds = {}
                    cat_bounds = {}
                    order = []
                    for f, t, went_left in path:
                        if f not in order:
                            order.append(f)
                        if self.is_categorical_[f]:
                            mask = self.cat_splits_[t]
                            allowed = cat_bounds.setdefault(
                                f, np.ones(len(mask), dtype=bool)
                            )
                            allowed &= mask if went_left else ~mask
                        else:
                            lo, hi = bounds.setdefault(f, [None, None])
                            if went_left:
                                bounds[f][1] = t if hi is None else min(hi, t)
                            else:
                                bounds[f][0] = t if lo is None else max(lo, t)
                    conds = []
                    for f in order:
                        if self.is_categorical_[f]:
                            conds.append(self._format_cat_condition(f, cat_bounds[f], names))
                            continue
                        lo, hi = bounds[f]
                        nm = name(f)
                        nan_ok = nan is not None and bool(nan[f])
                        if lo is None:
                            if hi == 0:
                                conds.append(f"{nm} is missing")
                            else:
                                txt = f"{nm} <= {num(bin_threshold(self.bin_edges_[f], hi))}"
                                conds.append(f"({txt} or missing)" if nan_ok else txt)
                        elif hi is None:
                            conds.append(
                                f"{nm} is present"
                                if lo == 0
                                else f"{nm} > {num(bin_threshold(self.bin_edges_[f], lo))}"
                            )
                        else:
                            low = (
                                ""
                                if lo == 0
                                else f"{num(bin_threshold(self.bin_edges_[f], lo))} < "
                            )
                            conds.append(
                                f"{low}{nm} <= {num(bin_threshold(self.bin_edges_[f], hi))}"
                            )
                    out.append((k, conds, float(tree.value[node])))
                    return
                f, t = int(tree.feature[node]), int(tree.threshold[node])
                walk(tree.left[node], [*path, (f, t, True)])
                walk(tree.right[node], [*path, (f, t, False)])

            walk(0, [])
        return out

    def export_text(self, feature_names=None, precision=4):
        if not self.is_categorical_.any():
            return super().export_text(feature_names=feature_names, precision=precision)
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "trees_")
        names = self._names(feature_names)
        name = (lambda j: str(names[j])) if names is not None else (lambda j: f"x{j}")
        lines = [f"base (logit): {float(self.base_margin_):+.{precision}f}"]
        for k, tree in enumerate(self.trees_, 1):
            lines.append(f"tree {k}")

            def walk(node, depth, tree=tree):
                pad = "|   " * depth
                if tree.left[node] == -1:
                    lines.append(f"{pad}|--- value: {tree.value[node]:+.{precision}f}")
                    return
                f, t = int(tree.feature[node]), int(tree.threshold[node])
                nm = name(f)
                if self.is_categorical_[f]:
                    mask = self.cat_splits_[t]
                    left_lbl = self._format_cat_condition(f, mask, names)
                    right_lbl = self._format_cat_condition(f, ~mask, names)
                else:
                    thr = bin_threshold(self.bin_edges_[f], t)
                    if np.isneginf(thr):
                        left_lbl, right_lbl = f"{nm} is missing", f"{nm} is present"
                    else:
                        v = f"{thr:.{precision}g}"
                        left_lbl, right_lbl = f"{nm} <= {v}", f"{nm} >  {v}"
                lines.append(f"{pad}|--- {left_lbl}")
                walk(tree.left[node], depth + 1)
                lines.append(f"{pad}|--- {right_lbl}")
                walk(tree.right[node], depth + 1)

            walk(0, 0)
        return "\n".join(lines)

    def to_sql(
        self,
        table: str = "input",
        feature_names: FeatureNames | None = None,
        precision: int = 6,
        keep_columns: bool = True,
        *,
        null_safe: bool = True,
        score_col: str = "score",
        proba_col: str = "p",
    ) -> str:
        """SQL export supporting categorical ``IN (...)`` / ``NOT IN (...)`` with SQL escaping."""
        if not self.is_categorical_.any() and not getattr(self, "_ord_vocabs", None):
            return super().to_sql(
                table=table,
                feature_names=feature_names,
                precision=precision,
                keep_columns=keep_columns,
                null_safe=null_safe,
                score_col=score_col,
                proba_col=proba_col,
            )
        from sklearn.utils.validation import check_is_fitted

        from .explain import _sql_ident

        check_is_fitted(self, "trees_")
        names = self._names(feature_names)
        nan = getattr(self, "nan_features_", None)
        name = (lambda j: _sql_ident(names[j])) if names is not None else (lambda j: f"x{j}")
        num = lambda v: f"{float(v):.{precision}g}"
        def cut(v):
            return np.format_float_positional(np.float64(v), unique=True, trim="-")

        def sql_lit(v):
            if isinstance(v, (bool, np.bool_)):
                return "1" if v else "0"
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            if isinstance(v, (float, np.floating)):
                return cut(v)
            return "'" + str(v).replace("'", "''") + "'"

        cols, blocks = [], []
        for k, tree in enumerate(self.trees_, 1):
            feats = []
            for f, lft in zip(tree.feature, tree.left):
                if lft != -1 and int(f) not in feats:
                    feats.append(int(f))
            if feats:
                suffix = "_".join(
                    "".join(c if c.isalnum() else "_" for c in str(names[f]))
                    if names is not None else f"x{f}"
                    for f in feats
                )
                col = _sql_ident(f"t{k}_{suffix}"[:60])
            else:
                col = _sql_ident(f"t{k}")

            leaves = []

            def walk(node, path, tree=tree, leaves=leaves):
                if tree.left[node] == -1:
                    leaves.append((path, float(tree.value[node])))
                    return
                f, t = int(tree.feature[node]), int(tree.threshold[node])
                walk(tree.left[node], [*path, (f, t, True)])
                walk(tree.right[node], [*path, (f, t, False)])

            walk(0, [])
            if len(leaves) == 1:
                blocks.append(f"    {num(leaves[0][1])} AS {col}")
                cols.append(col)
                continue
            lines = [f"    CASE  -- tree {k}"]
            for i, (path, v) in enumerate(leaves):
                num_bounds = {}
                cat_bounds = {}
                for f, t, went_left in path:
                    if self.is_categorical_[f]:
                        mask = self.cat_splits_[t]
                        allowed = cat_bounds.setdefault(f, np.ones(len(mask), dtype=bool))
                        allowed &= mask if went_left else ~mask
                    else:
                        lo, hi = num_bounds.setdefault(f, [None, None])
                        if went_left:
                            num_bounds[f][1] = t if hi is None else min(hi, t)
                        else:
                            num_bounds[f][0] = t if lo is None else max(lo, t)
                conds = []
                ord_vocabs = getattr(self, "_ord_vocabs", {})
                for f, (lo, hi) in num_bounds.items():
                    nm = name(f)
                    nan_ok = bool(null_safe) or (nan is not None and bool(nan[f]))
                    if f in ord_vocabs:
                        vocab = ord_vocabs[f]

                        def _ord_lit(bin_idx, vocab=vocab, edges_f=self.bin_edges_[f]):
                            thr = bin_threshold(edges_f, bin_idx)
                            k_idx = min(max(int(np.floor(thr)), 0), len(vocab) - 1)
                            return sql_lit(vocab[k_idx])

                        if lo is None:
                            if hi == 0:
                                conds.append(f"{nm} IS NULL")
                                continue
                            c = f"{nm} <= {_ord_lit(hi)}"
                            conds.append(f"({c} OR {nm} IS NULL)" if nan_ok else c)
                        elif hi is None:
                            conds.append(
                                f"{nm} IS NOT NULL" if lo == 0 else f"{nm} > {_ord_lit(lo)}"
                            )
                        else:
                            if lo != 0:
                                conds.append(f"{nm} > {_ord_lit(lo)}")
                            else:
                                conds.append(f"{nm} IS NOT NULL")
                            conds.append(f"{nm} <= {_ord_lit(hi)}")
                        continue
                    if lo is None:
                        if hi == 0:
                            conds.append(f"{nm} IS NULL")
                            continue
                        c = f"{nm} <= {cut(bin_threshold(self.bin_edges_[f], hi))}"
                        conds.append(f"({c} OR {nm} IS NULL)" if nan_ok else c)
                    elif hi is None:
                        conds.append(
                            f"{nm} IS NOT NULL"
                            if lo == 0
                            else f"{nm} > {cut(bin_threshold(self.bin_edges_[f], lo))}"
                        )
                    else:
                        if lo != 0:
                            conds.append(f"{nm} > {cut(bin_threshold(self.bin_edges_[f], lo))}")
                        else:
                            conds.append(f"{nm} IS NOT NULL")
                        conds.append(f"{nm} <= {cut(bin_threshold(self.bin_edges_[f], hi))}")
                for f, allowed in cat_bounds.items():
                    nm = name(f)
                    known = self.categories_[f]
                    in_cats = [known[c - 2] for c in range(2, len(known) + 2) if allowed[c]]
                    out_cats = [known[c - 2] for c in range(2, len(known) + 2) if not allowed[c]]
                    inc_null = bool(allowed[0])
                    inc_other = bool(allowed[1])
                    if inc_other:
                        if not out_cats:
                            if not inc_null:
                                conds.append(f"{nm} IS NOT NULL")
                        else:
                            lits = ", ".join(sql_lit(val) for val in out_cats)
                            if inc_null:
                                conds.append(f"({nm} IS NULL OR {nm} NOT IN ({lits}))")
                            else:
                                conds.append(f"({nm} IS NOT NULL AND {nm} NOT IN ({lits}))")
                    else:
                        if not in_cats:
                            conds.append(f"{nm} IS NULL" if inc_null else "1 = 0")
                        else:
                            lits = ", ".join(sql_lit(val) for val in in_cats)
                            if inc_null:
                                conds.append(f"({nm} IS NULL OR {nm} IN ({lits}))")
                            else:
                                conds.append(f"{nm} IN ({lits})")
                if i == len(leaves) - 1:
                    lines.append(f"      ELSE {num(v)}")
                elif not conds:
                    lines.append(f"      WHEN 1 = 1 THEN {num(v)}")
                else:
                    lines.append(f"      WHEN {' AND '.join(conds)} THEN {num(v)}")
            lines.append(f"    END AS {col}")
            blocks.append("\n".join(lines))
            cols.append(col)

        s_col, p_col = _sql_ident(score_col), _sql_ident(proba_col)
        head = (
            f"-- logit P(y = {self.classes_[1]}) = base + sum of {len(self.trees_)} tree columns "
            f"({self.n_splits_} cuts); {p_col} = 1 / (1 + EXP(-{s_col}))"
        )
        keep = "*, " if keep_columns else ""
        if not blocks:
            contrib_body = ["  SELECT *"] if keep_columns else ["  SELECT 0 AS _bt_dummy"]
            scored_head = f"  SELECT {keep}{num(float(self.base_margin_))}  -- base"
            final_line = (
                f"SELECT *, 1.0 / (1.0 + EXP(-{s_col})) AS {p_col}"
                if keep_columns
                else f"SELECT {s_col}, 1.0 / (1.0 + EXP(-{s_col})) AS {p_col}"
            )
            return "\n".join([
                head,
                "WITH contributions AS (",
                *contrib_body,
                f"  FROM {table}",
                "), scored AS (",
                scored_head,
                f"    AS {s_col}",
                "  FROM contributions",
                ")",
                final_line,
                "FROM scored;",
            ])
        final_line = (
            f"SELECT *, 1.0 / (1.0 + EXP(-{s_col})) AS {p_col}"
            if keep_columns
            else f"SELECT {s_col}, 1.0 / (1.0 + EXP(-{s_col})) AS {p_col}"
        )
        return "\n".join([
            head,
            "WITH contributions AS (",
            f"  SELECT {keep}".rstrip(),
            ",\n".join(blocks),
            f"  FROM {table}",
            "), scored AS (",
            f"  SELECT {keep}{num(float(self.base_margin_))}  -- base",
            *[f"    + {c}" for c in cols],
            f"    AS {s_col}",
            "  FROM contributions",
            ")",
            final_line,
            "FROM scored;",
        ])

    def shape_functions(self) -> dict[int, tuple[FloatArray, int]]:
        """Main effects per feature (numeric bins or category codes)."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "trees_")
        if not self.is_categorical_.any():
            return super().shape_functions()
        out = {}
        masks_arr = self._cat_masks_matrix()
        p = self.n_features_in_
        for tree in self.trees_:
            feats = {int(f) for f, lft in zip(tree.feature, tree.left) if lft != -1}
            if len(feats) != 1:
                continue
            f = feats.pop()
            nb = (len(self.categories_[f]) + 2) if self.is_categorical_[f] else (
                len(self.bin_edges_[f]) + 2
            )
            Xb = np.zeros((nb, p), dtype=np.uint8)
            Xb[:, f] = np.arange(nb, dtype=np.uint8)
            vals, count = out.get(f, (np.zeros(nb, dtype=np.float64), 0))
            out[f] = (vals + tree.value[self._tree_leaf_ids(tree, Xb, masks_arr)], count + 1)
        return out

    def plot_contributions(
        self,
        x: ArrayLike,
        feature_names: FeatureNames | None = None,
        max_terms: int = 12,
        ax: Any = None,
    ) -> Any:
        """Waterfall of one prediction, supporting mixed numeric/categorical rows."""
        from .explain import _pyplot, leaf_order

        if not self.is_categorical_.any() and not getattr(self, "_ord_vocabs", None):
            return super().plot_contributions(
                x, feature_names=feature_names, max_terms=max_terms, ax=ax
            )
        plt = _pyplot()
        if hasattr(x, "to_frame"):
            x_row = x.to_frame().T
        elif hasattr(x, "columns"):
            x_row = x.iloc[:1]
        else:
            x_row = np.asarray(x, dtype=object).reshape(1, -1)
        contrib = self.predict_contributions(x_row)[0]
        Xb = self._transform_mixed(x_row)
        masks_arr = self._cat_masks_matrix()
        all_rules = self.rules(feature_names=feature_names, precision=4)
        rules_by_tree = {}
        for tidx, conds, val in all_rules:
            rules_by_tree.setdefault(tidx, []).append((conds, val))
        labels = []
        for tidx, tree in enumerate(self.trees_):
            leaf = int(self._tree_leaf_ids(tree, Xb, masks_arr)[0])
            tr_rules = rules_by_tree.get(tidx, [])
            labels.append(" and ".join(tr_rules[leaf_order(tree).index(leaf)][0]) or "always")
        order = np.argsort(-np.abs(contrib), kind="stable")
        keep, rest = order[:max_terms], order[max_terms:]
        steps = [("base", float(self.base_margin_))]
        steps += [(labels[k], float(contrib[k])) for k in keep]
        if len(rest):
            steps.append((f"other {len(rest)} trees", float(contrib[rest].sum())))
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 0.45 * len(steps) + 1.2))
        pos = 0.0
        for i, (_, v) in enumerate(steps):
            start = 0.0 if i == 0 else pos
            ax.barh(i, v, left=start, color="C0" if i == 0 else ("C2" if v >= 0 else "C3"))
            pos = start + v
        ax.set_yticks(range(len(steps)), [s[0] for s in steps], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(pos, color="0.3", linestyle="--", linewidth=1)
        p = 1.0 / (1.0 + np.exp(-pos))
        ax.set_xlabel(f"cumulative logit (final {pos:+.3f}, P = {p:.3f})")
        return ax


CategoricalFIGSClassifier = CategoricalInterleavedTreeClassifier
