"""G0 gates for the experimental ITM product terms (PLANO_DEGRAU3.md v2)."""

import copy
import json
import sqlite3

import numpy as np
import pytest
from sklearn.base import clone

from bettertrees import FIGSClassifier
from bettertrees.experimental import InteractingTreeClassifier
from bettertrees.experimental._itm_kernels import affine_grad_hess, joint_system
from bettertrees.experimental.itm import CUT, NODE, TREE, _subtree_leaves
from bettertrees.sums._common import grad_hess
from bettertrees.sums.smalltrees import SmallTree

FORMS = ("M1", "M2", "M3", "M4", "M5")


def data(seed=0, n=1200, p=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    eta = 1.5 * np.sign(X[:, 0]) + np.sign(X[:, 1]) + .5 * X[:, 2]
    soft = 1 / (1 + np.exp(-eta))
    y = (rng.random(n) < soft).astype(int)
    return X, y, soft


def product_data(n=6000, seed=8, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    g1 = np.digitize(X[:, 0], [-1, -.4, 0, .4, 1]) - 2.5
    g2 = np.digitize(X[:, 1], [-1, -.4, 0, .4, 1]) - 2.5
    soft = 1 / (1 + np.exp(-(.3 * g1 + .3 * g2 + .25 * g1 * g2 + .8 * (X[:, 2] > 1) * g2 / 2.5)))
    y = (rng.random(n) < soft).astype(int)
    return X, y, soft


def assert_same(a, b, X):
    assert a.base_margin_ == b.base_margin_
    assert len(a.trees_) == len(b.trees_)
    for ta, tb in zip(a.trees_, b.trees_):
        for name in ("feature", "threshold", "left", "right", "value"):
            np.testing.assert_array_equal(getattr(ta, name), getattr(tb, name))
    np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


def rule(budget):
    return dict(max_splits=budget, learning_rate=1. if budget <= 8 else .3,
                max_delta_step=4. if budget <= 8 else None)


# ------------------------------------------------------------ no terms = FIGS, bit for bit

@pytest.mark.parametrize("budget", [0, 4, 8, 16])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("off", [dict(forms=()), dict(max_interactions=0)])
def test_no_terms_is_figs_bitwise(budget, weighted, off):
    X, y, soft = data()
    X[::15, 0] = np.nan
    X[:, -1] = 1
    original = X.copy()
    w = np.arange(len(y)) % 4 if weighted else None
    a = FIGSClassifier(**rule(budget)).fit(X, y, sample_weight=w, y_soft=soft)
    b = InteractingTreeClassifier(**rule(budget), **off).fit(X, y, sample_weight=w, y_soft=soft)
    assert b.n_interactions_ == 0
    assert_same(a, b, X)
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("forms", [("M1",), ("M2",), ("M3",), ("M1", "M2", "M3")])
def test_no_candidate_keeps_figs_bitwise(forms):
    # one tree: no second tree to pair with, so the grower runs the production path
    X, y, _ = data()
    a = FIGSClassifier(max_splits=8, max_trees=1, learning_rate=.3).fit(X, y)
    b = InteractingTreeClassifier(max_splits=8, max_trees=1, learning_rate=.3, forms=forms).fit(X, y)
    assert b.n_interactions_ == 0
    assert_same(a, b, X)


# ------------------------------------------------------------ scoring = dense reference

def raw_columns(est, Xb, values, ids_of, form):
    """Every candidate of ``form`` as (factor1, factor2, raw1, raw2), built densely."""
    K = len(est.trees_)
    trees = [(TREE, k, -1) for k in range(K)]
    leaves = [(NODE, j, a) for j in range(K) for a in est.trees_[j].leaves]
    cuts = sorted({(CUT, int(f), int(t)) for tr in est.trees_
                   for f, t, lft in zip(tr.feature, tr.threshold, tr.left) if lft != -1})
    if form == "M1":
        pairs = [(trees[j], trees[k]) for j in range(K) for k in range(j + 1, K)]
    elif form == "M2":
        pairs = [(lf, tr) for lf in leaves for tr in trees if lf[1] != tr[1]]
    elif form == "M3":
        pairs = [(a, b) for a in leaves for b in leaves if a[1] < b[1]]
    elif form == "M5":
        pairs = [(a, b) for i, a in enumerate(cuts) for b in cuts[i + 1:] if a[1] != b[1]]
    else:  # M4: every (feature, bin) cut x every tree
        pairs = [((CUT, f, t), tr) for f in range(Xb.shape[1]) for t in range(int(Xb[:, f].max()))
                 for tr in trees]
    return [(f1, f2, est._raw(f1, values, ids_of, Xb), est._raw(f2, values, ids_of, Xb))
            for f1, f2 in pairs]


def dense_best(est, Xb, values, ids_of, g, h, w, form):
    """(best gain, {key: gain}) over every valid candidate of ``form``."""
    lam, mw = est.interaction_lam_, est.min_weight
    best, table = (0.0, None), {}
    for f1, f2, r1, r2 in raw_columns(est, Xb, values, ids_of, form):
        cols = []
        for r in (r1, r2):
            c = h @ r / h.sum()
            s = np.sqrt(h @ (r - c) ** 2 / h.sum())
            cols.append(None if s <= 1e-6 else (r - c) / s)
        if cols[0] is None or cols[1] is None:
            continue
        if form in ("M3", "M5"):
            cell = w @ (r1 * r2)
            if cell < mw or w.sum() - cell < mw:
                continue
        if form == "M4":
            right = w @ r1
            if right < mw or w.sum() - right < mw:
                continue
        u = cols[0] * cols[1]
        gain = (g @ u) ** 2 / (h @ (u * u) + lam)
        table[frozenset([f1, f2])] = gain
        if gain > best[0]:
            best = (gain, frozenset([f1, f2]))
    return best[0], table


@pytest.mark.parametrize("form", FORMS)
def test_candidate_scores_match_dense_reference(form):
    X, y, soft = product_data(n=3000)
    X[::17, 1] = np.nan
    base = FIGSClassifier(max_splits=7, learning_rate=.3, max_bins=8).fit(X, y)
    est = InteractingTreeClassifier.from_structure(
        base, base.trees_, [], X, y, forms=(form,), max_splits=7, max_bins=8, min_weight=20,
        max_partners=99)
    from bettertrees.sums._common import rebin
    Xb = rebin(X.astype(np.float64), est.bin_edges_)
    nb = np.asarray([len(e) + 2 for e in est.bin_edges_], dtype=np.int64)
    values, ids_of = est._values(Xb)
    margin = est._margin(values, ids_of, Xb)
    target, w = y.astype(float), np.ones(len(y))
    g, h = grad_hess(target, margin, w)
    gain, spec = est._best_term(values, ids_of, Xb, nb, g, h, w)
    ref_gain, table = dense_best(est, Xb, values, ids_of, g, h, w, form)
    assert ref_gain > 0 and spec is not None
    assert gain == pytest.approx(ref_gain, rel=1e-9)
    # the chosen term is a valid candidate with that gain (ties, e.g. the two sides of
    # a stump, may pick either)
    assert table[frozenset([spec["f1"], spec["f2"]])] == pytest.approx(ref_gain, rel=1e-9)


def test_joint_system_matches_dense():
    rng = np.random.default_rng(21)
    U = np.ascontiguousarray(rng.normal(size=(73, 3)))
    g, h = rng.normal(size=73), rng.uniform(.1, .3, 73)
    G, H = joint_system(U, g, h, 1.7)
    np.testing.assert_allclose(G, U.T @ g)
    np.testing.assert_allclose(H, (U * h[:, None]).T @ U + 1.7 * np.eye(3))


# ------------------------------------------------------------ derivatives

def mixed_model(n=2000):
    """A model with one term of every form, built by hand."""
    X, y, soft = product_data(n=n)
    base = FIGSClassifier(max_splits=2, learning_rate=.3).fit(X, y)  # for the bins
    rng = np.random.default_rng(3)
    t0 = SmallTree.from_nested((0, 10, None, (2, 12, None, None)))
    t1 = SmallTree.from_nested((1, 15, (3, 8, None, None), None))
    for t in (t0, t1):
        t.value = rng.normal(size=len(t.feature))
    base = copy.deepcopy(base)
    base.trees_ = [t0, t1]
    c0, c1 = (0, 10), (1, 15)
    terms = [
        dict(f1=(TREE, 0, -1), f2=(TREE, 1, -1), center=(.1, -.2), scale=(.7, .9), gamma=.4),
        dict(f1=(NODE, 0, t0.leaves[0]), f2=(TREE, 1, -1), center=(.3, .1), scale=(.5, .8), gamma=-.3),
        dict(f1=(NODE, 0, t0.leaves[-1]), f2=(NODE, 1, t1.leaves[0]), center=(.2, .4), scale=(.4, .5),
             gamma=.25),
        dict(f1=(CUT, 3, 4), f2=(TREE, 0, -1), center=(.5, 0.), scale=(.5, 1.1), gamma=.2),
        dict(f1=(CUT, *c0), f2=(CUT, *c1), center=(.4, .6), scale=(.5, .5), gamma=-.15),
    ]
    est = InteractingTreeClassifier.from_structure(base, base.trees_, terms, X, y)
    return est, X, y


def test_multiplier_and_coefficients_match_finite_differences():
    est, X, y = mixed_model()
    from bettertrees.sums._common import rebin
    Xb = rebin(X.astype(np.float64), est.bin_edges_)
    values, ids_of = est._values(Xb)
    P1, P2 = est._phis(values, ids_of, Xb)
    eps = 1e-6
    for k in range(len(est.trees_)):
        a = est._multiplier(k, P1, P2)
        plus, minus = values.copy(), values.copy()
        plus[:, k] += eps
        minus[:, k] -= eps
        fd = (est._margin(plus, ids_of, Xb) - est._margin(minus, ids_of, Xb)) / (2 * eps)
        np.testing.assert_allclose(fd, a, rtol=1e-7, atol=1e-7)
    for e in range(est.n_interactions_):
        base_gamma = est.interaction_coefficients_.copy()
        est.interaction_coefficients_ = base_gamma + eps * np.eye(len(base_gamma))[e]
        up = est._margin(values, ids_of, Xb)
        est.interaction_coefficients_ = base_gamma - eps * np.eye(len(base_gamma))[e]
        down = est._margin(values, ids_of, Xb)
        est.interaction_coefficients_ = base_gamma
        np.testing.assert_allclose((up - down) / (2 * eps), P1[:, e] * P2[:, e], rtol=1e-7, atol=1e-7)


def test_affine_search_is_newton_at_zero_tree():
    rng = np.random.default_rng(4)
    margin, contrib, a = rng.normal(size=37), rng.normal(size=37), rng.uniform(.5, 1.5, 37)
    y, w = rng.uniform(0, 1, 37), rng.uniform(0, 2, 37)
    g, h = grad_hess(y, margin - a * contrib, w)
    ga, ha = affine_grad_hess(y, w, margin, contrib, a)
    np.testing.assert_allclose(ga, g * a)
    np.testing.assert_allclose(ha, h * a**2)


def test_refit_step_lowers_training_loss_and_keeps_margin_exact():
    est, X, y = mixed_model()
    from bettertrees.sums._common import rebin
    Xb = rebin(X.astype(np.float64), est.bin_edges_)
    values, ids_of = est._values(Xb)
    margin = est._margin(values, ids_of, Xb)
    target, w = y.astype(float), np.ones(len(y))

    def loss(m):
        return np.mean(np.logaddexp(0, m) - target * m)

    before = loss(margin)
    est.learning_rate_, est.max_delta_step = .3, 1.0
    for _ in range(30):
        margin = est._refit_active(values, ids_of, Xb, target, w, margin)
        np.testing.assert_allclose(margin, est._margin(values, ids_of, Xb), atol=1e-10)
    assert loss(margin) < before


# ------------------------------------------------------------ fitted models

@pytest.mark.parametrize("form", [*FORMS, "all"])
@pytest.mark.parametrize("order", ["interleaved", "two_stage"])
def test_fit_margin_budget_and_heredity(form, order):
    X, y, soft = product_data()
    forms = FORMS if form == "all" else (form,)
    est = InteractingTreeClassifier(**rule(16), forms=forms, order=order).fit(X, y)
    assert est.n_units_ <= 16
    np.testing.assert_allclose(est.training_margin_, est.decision_function(X), atol=1e-12)
    np.testing.assert_allclose(est._incremental_margin_, est.training_margin_, atol=1e-9)
    np.testing.assert_allclose(est.base_margin_ + est.predict_contributions(X).sum(1),
                               est.decision_function(X), atol=1e-12)
    if order == "two_stage":
        assert est.n_splits_ <= 12 and est.n_interactions_ <= 4
    used_cuts = {(int(f), int(t)) for tr in est.trees_
                 for f, t, lft in zip(tr.feature, tr.threshold, tr.left) if lft != -1}
    for term in est.terms_:
        assert term["form"] in forms
        assert len({term["f1"], term["f2"]}) == 2
        for kind, a, b in (term["f1"], term["f2"]):
            if kind in (TREE, NODE):
                assert a < len(est.trees_)
            if kind == NODE:
                assert b < len(est.trees_[a].feature)
        if term["form"] == "M5":
            assert {term["f1"][1:], term["f2"][1:]} <= used_cuts
    json.dumps(est.to_dict())
    if est.n_interactions_:
        assert "term 1" in est.explain() and "term 1" in est.export_text()
        with pytest.raises(NotImplementedError):
            est.to_shap_model()


def test_interacting_data_selects_terms_of_every_form():
    X, y, soft = product_data()
    for form in FORMS:
        est = InteractingTreeClassifier(**rule(16), forms=(form,)).fit(X, y, y_soft=soft)
        assert est.n_interactions_ > 0, form


@pytest.mark.parametrize("form", [*FORMS, "all"])
def test_sql_matches_every_term_kind_with_missing_values(form):
    X, y, soft = product_data(n=4000)
    X[::11, 0] = np.nan
    X[::7, 2] = np.nan
    forms = FORMS if form == "all" else (form,)
    est = InteractingTreeClassifier(**rule(16), forms=forms).fit(X, y, y_soft=soft)
    if form == "all":
        assert est.n_interactions_ > 0
    check_sql(est, X)


def test_sql_matches_hand_model_with_every_term_kind():
    est, X, y = mixed_model()
    X = X.copy()
    X[::5, 0] = np.nan
    X[::9, 3] = np.nan
    est.nan_features_ = np.ones(X.shape[1], dtype=bool)  # the export writes NULL branches only then
    assert {t["f1"][0] for t in est.terms_} | {t["f2"][0] for t in est.terms_} == {TREE, NODE, CUT}
    check_sql(est, X)


def check_sql(est, X):
    cols = ", ".join(f"x{j} REAL" for j in range(X.shape[1]))
    with sqlite3.connect(":memory:") as db:
        db.create_function("EXP", 1, np.exp)
        db.execute(f"CREATE TABLE input ({cols})")
        db.executemany(f"INSERT INTO input VALUES ({', '.join('?' * X.shape[1])})",
                       [tuple(None if np.isnan(v) else float(v) for v in row) for row in X[:300]])
        rows = db.execute(est.to_sql(precision=17)).fetchall()
    out = np.asarray(rows, dtype=float)
    np.testing.assert_allclose(out[:, -2], est.decision_function(X[:300]), atol=1e-12)
    np.testing.assert_allclose(out[:, -1], est.predict_proba(X[:300])[:, 1], atol=1e-12)


def test_splitting_a_referenced_leaf_keeps_predictions():
    est, X, y = mixed_model()
    before = est.decision_function(X)
    for term in est.terms_:
        for kind, a, b in (term["f1"], term["f2"]):
            if kind == NODE and est.trees_[a].left[b] == -1:
                est.trees_[a].split(b, 2, 3)  # children inherit the leaf value
    assert any(est.trees_[t["f1"][1]].left[t["f1"][2]] != -1
               for t in est.terms_ if t["f1"][0] == NODE)
    np.testing.assert_array_equal(est.decision_function(X), before)
    for term in est.terms_:
        for kind, a, b in (term["f1"], term["f2"]):
            if kind == NODE:
                assert len(_subtree_leaves(est.trees_[a], b)) >= 1


def test_posthoc_freezes_partitions_and_does_not_mutate_source():
    X, y, soft = product_data()
    base = FIGSClassifier(max_splits=8, learning_rate=.3).fit(X, y)
    before = copy.deepcopy(base)
    est = InteractingTreeClassifier.from_fitted(base, X, y, max_interactions=2, y_soft=soft)
    assert est.n_splits_ == base.n_splits_
    assert est.n_interactions_ > 0
    for a, b in zip(base.trees_, est.trees_):
        for name in ("feature", "threshold", "left", "right"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert_same(base, before, X)


def test_exact_null_selects_nothing():
    X, y, _ = data()
    est = InteractingTreeClassifier(max_splits=8, min_gain=1e-10).fit(X, y, y_soft=np.full(len(y), .5))
    assert est.n_units_ == 0


def test_sklearn_clone_and_refit_clear_state():
    X, y, soft = product_data(n=2000)
    est = InteractingTreeClassifier(**rule(8))
    assert clone(est).get_params() == est.get_params()
    est.fit(X, y, y_soft=soft)
    est.set_params(max_interactions=0).fit(X, y, y_soft=soft)
    assert est.n_interactions_ == 0 and est.terms_ == []


def test_conditions_and_distinct_counts():
    est, X, y = mixed_model()
    base_cuts = {(int(f), int(t)) for tr in est.trees_
                 for f, t, lft in zip(tr.feature, tr.threshold, tr.left) if lft != -1}
    assert est.n_distinct_splits_ == len(base_cuts | {(3, 4)})
    assert est.conditions_read(X[:50]) >= 1


@pytest.mark.parametrize("params", [dict(max_splits=-1), dict(max_interactions=-1),
                                    dict(interaction_lam=0), dict(learning_rate=0),
                                    dict(lam=float('nan')), dict(max_delta_step=-1),
                                    dict(forms=("M9",)), dict(order="sideways"),
                                    dict(max_partners=0)])
def test_invalid_parameters(params):
    X, y, _ = data(n=100)
    with pytest.raises(ValueError):
        InteractingTreeClassifier(**params).fit(X, y)


def test_single_tree_from_nested_region_text():
    tree = SmallTree.from_nested((0, 3, None, (1, 2, None, None)))
    assert sorted(_subtree_leaves(tree, 2)) == [3, 4]
