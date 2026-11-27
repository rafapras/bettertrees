"""RashomonMCR: constructive Model Class Reliance via Node-Swap & Drop-and-Regrow."""

import copy

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from ..sums._common import (
    as_weights,
    grad_hess,
    log_loss_margin as _log_loss,
    predict_input,
    rebin,
)
from ..sums._kernels import best_cut_1d, hist_1d


def _trees_of(model):
    if hasattr(model, "_explain_trees"):
        return model._explain_trees()
    return getattr(model, "trees_", [])


def _active_features(model):
    out = set()
    for tree in _trees_of(model):
        for f, lft in zip(tree.feature, tree.left):
            if lft != -1:
                out.add(int(f))
    return out


def feature_importance_shares(model, X):
    """Normalized importance (p,) of a fitted sum on X: each tree's mean |contribution|
    split equally among the distinct features cut in that tree (sums to 1 if any cut)."""
    check_is_fitted(model)
    p = int(model.n_features_in_)
    trees = _trees_of(model)
    if not trees:
        return np.zeros(p, dtype=np.float64)
    contribs = model.predict_contributions(X)
    scores = np.zeros(p, dtype=np.float64)
    for k, tree in enumerate(trees):
        feats = {int(f) for f, lft in zip(tree.feature, tree.left) if lft != -1}
        if not feats:
            continue
        share = float(np.mean(np.abs(contribs[:, k]))) / len(feats)
        for f in feats:
            scores[f] += share
    total = float(scores.sum())
    if total > 1e-12:
        scores /= total
    return scores


def _mask_feature(X, j):
    if hasattr(X, "copy") and hasattr(X, "iloc"):
        Xm = X.copy()
        Xm.iloc[:, j] = 0.0
        return Xm
    Xm = np.array(X, copy=True)
    Xm[:, j] = 0.0
    return Xm


def _node_swap_without_feature(model, j, X_tr, y_tr, w_tr):
    """Copy ``model``, replace every split on feature ``j`` with the best substitute
    cut among ``f != j`` on the rows reaching that node, and refit all leaves."""
    base_est = model.model_ if hasattr(model, "model_") else model
    if not hasattr(base_est, "bin_edges_") or not hasattr(base_est, "_refit_binned"):
        return None
    cand = copy.deepcopy(model)
    inner = cand.model_ if hasattr(cand, "model_") else cand
    Xv = predict_input(inner, X_tr, "trees_")
    Xb = rebin(Xv, inner.bin_edges_)
    target = (np.asarray(y_tr) == inner.classes_[1]).astype(np.float64)
    nb = np.array([len(e) + 2 for e in inner.bin_edges_], dtype=np.int64)
    B = int(nb.max())
    lam = float(getattr(inner, "lam_", None) or 1.0)
    min_w = float(getattr(inner, "min_weight", 1.0))

    kept_trees = []
    for t_idx, tree in enumerate(inner.trees_):
        parent = {
            c: (k, s)
            for k, (lc, rc) in enumerate(zip(tree.left, tree.right))
            if lc != -1
            for c, s in ((lc, 0), (rc, 1))
        }
        others = inner.base_margin_ + sum(
            (u.value[u.leaf_ids(Xb)] for m_idx, u in enumerate(inner.trees_) if m_idx != t_idx),
            np.zeros(len(Xb), dtype=np.float64),
        )
        drop_whole_tree = False
        for node in range(len(tree.feature)):
            if tree.left[node] == -1 or int(tree.feature[node]) != j:
                continue
            rows = np.ones(len(Xb), dtype=bool)
            cur = node
            while cur in parent:
                p_node, side = parent[cur]
                go_left = Xb[:, tree.feature[p_node]] <= tree.threshold[p_node]
                rows &= go_left if side == 0 else ~go_left
                cur = p_node
            if rows.sum() < 2:
                if node == 0:
                    drop_whole_tree = True
                    break
                tree.left[node] = tree.right[node] = -1
                tree.feature[node] = tree.threshold[node] = -1
                continue
            g, h = grad_hess(target[rows], others[rows], w_tr[rows])
            gains, cuts = best_cut_1d(
                hist_1d(np.ascontiguousarray(Xb[rows]), g, h, w_tr[rows], B), nb, lam, min_w
            )
            gains[j] = -1.0
            f_best = int(np.argmax(gains))
            if gains[f_best] > 0:
                tree.feature[node] = f_best
                tree.threshold[node] = int(cuts[f_best])
            elif node == 0:
                drop_whole_tree = True
                break
            else:
                tree.left[node] = tree.right[node] = -1
                tree.feature[node] = tree.threshold[node] = -1
        if not drop_whole_tree:
            from ..sums.edit import _reachable
            clean = _reachable(tree)
            if clean.n_splits > 0:
                kept_trees.append(clean)

    inner.trees_ = kept_trees
    inner._refit_binned(Xb, target, w_tr, lam, sweeps=4)
    if j in _active_features(cand):
        return None
    return cand


def rashomon_mcr(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    *,
    epsilon=0.01,
    sample_weight_train=None,
    sample_weight_val=None,
):
    """Constructive Model Class Reliance for a fitted bettertrees sum."""
    check_is_fitted(model)
    n_tr = len(X_train)
    w_tr = as_weights(sample_weight_train, n_tr)
    if X_val is None:
        X_va, y_va, w_va = X_train, y_train, w_tr
    else:
        X_va = X_val
        y_va = y_val
        w_va = as_weights(sample_weight_val, len(X_va))

    target_va = (np.asarray(y_va) == model.classes_[1]).astype(np.float64)
    base_loss = float(_log_loss(target_va, model.decision_function(X_va), w_va))
    threshold_loss = base_loss * (1.0 + float(epsilon))

    p = int(model.n_features_in_)
    names = (
        list(model.feature_names_in_)
        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None
        else [f"x{j}" for j in range(p)]
    )
    active_set = _active_features(model)
    base_shares = feature_importance_shares(model, X_va)

    alt_losses = {}
    best_alts = {}
    rashomon_models = [model]

    for j in sorted(active_set):
        cands = []
        # 1. Node-swap candidate
        swap_m = _node_swap_without_feature(model, j, X_train, y_train, w_tr)
        if swap_m is not None:
            loss_swap = float(_log_loss(target_va, swap_m.decision_function(X_va), w_va))
            cands.append((loss_swap, swap_m))

        # 2. Drop-and-regrow candidate (column j masked to constant 0)
        X_tr_masked = _mask_feature(X_train, j)
        regrow_m = clone(model)
        regrow_m.fit(X_tr_masked, y_train, sample_weight=w_tr)
        if j not in _active_features(regrow_m):
            loss_regrow = float(_log_loss(target_va, regrow_m.decision_function(X_va), w_va))
            cands.append((loss_regrow, regrow_m))

        best_loss, best_m = min(cands, key=lambda pair: pair[0])
        alt_losses[j] = best_loss
        if best_loss <= threshold_loss:
            best_alts[j] = best_m
            rashomon_models.append(best_m)

    share_matrix = np.vstack([feature_importance_shares(m, X_va) for m in rashomon_models])
    mcr_min_vec = share_matrix.min(axis=0)
    mcr_max_vec = share_matrix.max(axis=0)

    features_info = {}
    alternatives = {}
    for j, fname in enumerate(names):
        is_active = j in active_set
        base_imp = float(base_shares[j])
        if not is_active:
            features_info[fname] = dict(
                index=j,
                active=False,
                base_importance=0.0,
                mcr_min=0.0,
                mcr_max=float(mcr_max_vec[j]),
                alt_loss=base_loss,
                relative_loss_gap=0.0,
                indispensable=False,
            )
            continue
        a_loss = alt_losses[j]
        in_rashomon = j in best_alts
        if in_rashomon:
            alternatives[fname] = best_alts[j]
        features_info[fname] = dict(
            index=j,
            active=True,
            base_importance=base_imp,
            mcr_min=0.0 if in_rashomon else float(mcr_min_vec[j]),
            mcr_max=float(mcr_max_vec[j]),
            alt_loss=a_loss,
            relative_loss_gap=float((a_loss - base_loss) / max(base_loss, 1e-12)),
            indispensable=not in_rashomon,
        )

    return dict(
        base_loss=base_loss,
        epsilon=float(epsilon),
        features=features_info,
        alternatives=alternatives,
    )
