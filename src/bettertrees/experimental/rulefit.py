"""RuleFit (Friedman & Popescu, 2008) with candidate rules from optimal depth-2 trees.

1. Candidates: every node (except the root) of the trees of a depth-2
   ``AdditiveTreeBooster`` becomes a rule of 1 or 2 conditions; duplicates and
   rules with support outside [``min_support``, 1 - ``min_support``] are dropped.
2. Lasso path (L1 logistic regression, columns scaled by 1/sd as in RuleFit)
   over the indicators: coordinate descent with warm starts in Numba
   (``solver="path"``, default), or independent liblinear fits with bisection
   on log C (``solver="liblinear"``, reference, about 20x slower).
3. For each budget b (in **conditions**, summed over rules), the least
   penalized point of the path with cost <= b; then a nearly unpenalized refit
   on the chosen rules only (relaxed lasso).

References
----------
Friedman, Popescu. "Predictive Learning via Rule Ensembles." Annals of
Applied Statistics, 2008.
Friedman, Hastie, Tibshirani. "Regularization Paths for Generalized Linear
Models via Coordinate Descent." Journal of Statistical Software, 2010.
"""

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from ..sums._common import as_float_matrix, bin_threshold, rebin
from ..sums.additive import AdditiveTreeBooster


def booster_rules(booster):
    """Rules (sorted tuples of (f, t, goes_left)) from the terms of a depth-1/2 booster."""
    rules = set()
    for (f1, t1, fl, tl, fr, tr), _ in booster.terms_:
        for side, (fc, tc) in ((True, (fl, tl)), (False, (fr, tr))):
            root = (int(f1), int(t1), side)
            rules.add((root,))
            if fc >= 0:
                for child_side in (True, False):
                    rules.add(tuple(sorted((root, (int(fc), int(tc), child_side)))))
    return sorted(rules)


def rule_matrix(Xb, rules):
    out = np.ones((len(Xb), len(rules)), dtype=bool)
    for k, rule in enumerate(rules):
        for f, t, left in rule:
            cond = Xb[:, f] <= t
            out[:, k] &= cond if left else ~cond
    return out


class RuleFitLasso(ClassifierMixin, BaseEstimator):
    """Scorecard of rules chosen by the lasso under budgets of conditions.

    ``fit`` computes the path once and fits one model per budget in
    ``budgets``; ``predict_proba(X, budget=b)`` uses the one with budget b
    (default: the largest). ``y_soft`` only changes the candidates (booster
    fitted on the soft target).
    """

    def __init__(self, *, budgets=(16,), n_rounds=300, learning_rate=0.1,
                 min_support=0.01, n_C=15, n_bisect=8, refit_C=10.0,
                 solver="path", n_lambdas=100, lambda_ratio=1e-4, refine_levels=3,
                 refine_points=8, random_state=0):
        self.budgets = budgets
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.min_support = min_support
        self.n_C = n_C
        self.n_bisect = n_bisect
        self.refit_C = refit_C
        self.solver = solver
        self.n_lambdas = n_lambdas
        self.lambda_ratio = lambda_ratio
        self.refine_levels = refine_levels
        self.refine_points = refine_points
        self.random_state = random_state

    def fit(self, X, y, y_soft=None):
        X = as_float_matrix(X)
        booster = AdditiveTreeBooster(depth=2, learning_rate=self.learning_rate,
                                      max_rounds=self.n_rounds, patience=30,
                                      random_state=self.random_state).fit(X, y, y_soft=y_soft)
        self.classes_ = booster.classes_
        yy = np.searchsorted(self.classes_, np.asarray(y))
        self.bin_edges_ = booster.bin_edges_
        self.prior_ = float(np.clip(yy.mean(), 1e-6, 1 - 1e-6))
        Xb = rebin(X, self.bin_edges_)
        rules = booster_rules(booster)
        R = rule_matrix(Xb, rules)
        support = R.mean(axis=0)
        keep = (support >= self.min_support) & (support <= 1 - self.min_support)
        R, rules, support = R[:, keep], [r for r, k in zip(rules, keep) if k], support[keep]
        # identical columns (same partition) count once
        _, first = np.unique(np.packbits(R, axis=0), axis=1, return_index=True)
        first = np.sort(first)
        R, rules, support = R[:, first], [rules[k] for k in first], support[first]
        self.candidate_rules_ = rules
        cost = np.array([len(r) for r in rules])
        # Lasso on the scaled min(R, 1 - R): the complement only flips the sign
        # of the coefficient (the intercept absorbs it), the selected set is the
        # same and the matrix stays sparse (support <= 50%).
        flip = support > 0.5
        scale = 1.0 / np.sqrt(support * (1 - support))
        if self.solver == "path":
            chosen = self._select_path(np.where(flip, ~R, R), scale, cost, yy)
        elif self.solver == "liblinear":
            Z = sparse.csr_matrix(np.where(flip, ~R, R).astype(np.float64) * scale)
            chosen = self._select_liblinear(Z, cost, yy)
        else:
            raise ValueError("solver must be 'path' or 'liblinear'.")
        self.models_ = {}
        for b, sel in chosen.items():
            if sel is None or not len(sel):
                self.models_[b] = ([], None)
                continue
            refit = LogisticRegression(C=self.refit_C, max_iter=1000).fit(R[:, sel], yy)
            self.models_[b] = ([rules[k] for k in sel], refit)
        self.n_features_in_ = X.shape[1]
        return self

    def _select_path(self, Z, scale, cost, yy):
        """Lasso path with warm starts (Numba): one point per lambda, lambda decreasing.

        For each budget, the point with the smallest lambda whose cost fits (the
        same criterion as the "largest C" of the liblinear version). The path is
        dense (``n_lambdas`` geometric lambdas down to lambda_max *
        ``lambda_ratio``) and stops when the cost exceeds the largest budget.
        """
        from ..sums._kernels import l1_logistic_path
        n, m = Z.shape
        cols = [np.flatnonzero(Z[:, j]) for j in range(m)]
        indptr = np.zeros(m + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(c) for c in cols])
        indices = (np.concatenate(cols) if m else np.zeros(0)).astype(np.int64)
        y = yy.astype(np.float64)
        grad = np.array([scale[j] * np.sum(y[cols[j]] - y.mean()) for j in range(m)])
        lam_max = np.abs(grad).max() / n if m else 1.0
        lambdas = lam_max * np.geomspace(1.0, self.lambda_ratio, self.n_lambdas)
        args = (indptr, indices, scale.astype(np.float64), y)
        extra = (cost.astype(np.float64), float(max(self.budgets)), 1e-7, 50, 1000)
        betas, b0s, done = l1_logistic_path(*args, lambdas, *extra, np.zeros(m), np.nan)
        # points (lambda, beta, b0, cost); the grid path may add several rules
        # at once and jump over a budget: refine that stretch with a fine
        # sub-path starting from the last point that fit (warm start)
        pts = [(lambdas[k], betas[k].copy(), b0s[k],
                int(cost[betas[k] != 0].sum())) for k in range(done)]
        for _ in range(self.refine_levels):
            added = []
            for b in self.budgets:
                fit = [i for i, p in enumerate(pts) if p[3] <= b]
                if not fit:
                    continue
                i = max(fit, key=lambda i: (pts[i][3], -pts[i][0]))
                over = [p for p in pts if p[3] > b and p[0] < pts[i][0]]
                if pts[i][3] == b or not over:
                    continue
                lam_hi, lam_lo = pts[i][0], max(p[0] for p in over)
                sub = np.geomspace(lam_hi, lam_lo, self.refine_points + 2)[1:-1]
                sb, s0, sd = l1_logistic_path(*args, sub, *extra, pts[i][1], pts[i][2])
                added += [(sub[k], sb[k].copy(), s0[k], int(cost[sb[k] != 0].sum()))
                          for k in range(sd)]
            if not added:
                break
            pts = sorted(pts + added, key=lambda p: -p[0])
        path = [(lam, np.flatnonzero(bt != 0), c) for lam, bt, _, c in pts]
        self.path_ = [(float(lam), len(sel), c) for lam, sel, c in path]
        out = {}
        for b in self.budgets:
            fits = [(lam, sel) for lam, sel, c in path if c <= b and len(sel)]
            out[b] = min(fits, key=lambda t: t[0])[1] if fits else None
        return out

    def _select_liblinear(self, Z, cost, yy):
        """Reference version: independent liblinear fits plus bisection on log C."""
        cache = {}

        def point(C):
            if C not in cache:
                lr = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=500)
                lr.fit(Z, yy)
                sel = np.flatnonzero(lr.coef_[0] != 0)
                cache[C] = (sel, int(cost[sel].sum()))
            return cache[C]

        grid = np.logspace(-5, 1, self.n_C)
        for C in grid:
            point(C)
        out = {}
        for b in self.budgets:
            fits = sorted((C, c) for C, (sel, c) in cache.items() if c <= b)
            lo = max([C for C, _ in fits if len(cache[C][0])], default=None)
            above = [C for C, (_, c) in cache.items() if c > b]
            hi = min([C for C in above if lo is None or lo < C], default=None)
            start = lo if lo is not None else max([C for C, _ in fits], default=grid[0])
            if hi is not None:
                a, z = np.log(start), np.log(hi)
                for _ in range(self.n_bisect):
                    mid = float(np.exp((a + z) / 2))
                    sel, c = point(mid)
                    if c <= b:
                        a = np.log(mid)
                        if len(sel) and (lo is None or mid > lo):
                            lo = mid
                    else:
                        z = np.log(mid)
            out[b] = None if lo is None else cache[lo][0]
        self.path_ = sorted((C, len(sel), c) for C, (sel, c) in cache.items())
        return out

    def _model(self, budget):
        check_is_fitted(self, "models_")
        return self.models_[max(self.budgets) if budget is None else budget]

    def decision_function(self, X, budget=None):
        rules, model = self._model(budget)
        if model is None:
            return np.full(len(X), np.log(self.prior_ / (1 - self.prior_)))
        Xb = rebin(as_float_matrix(X), self.bin_edges_)
        return model.decision_function(rule_matrix(Xb, rules))

    def predict_proba(self, X, budget=None):
        rules, model = self._model(budget)
        if model is None:  # no point of the path fits: intercept only
            p = np.full(len(X), self.prior_)
            return np.column_stack([1 - p, p])
        return model.predict_proba(rule_matrix(rebin(as_float_matrix(X), self.bin_edges_), rules))

    def predict(self, X, budget=None):
        return self.classes_[self.predict_proba(X, budget).argmax(axis=1)]

    def n_conditions(self, budget=None):
        return int(sum(len(r) for r in self._model(budget)[0]))

    def scorecard(self, budget=None, feature_names=None):
        rules, model = self._model(budget)
        name = (lambda j: feature_names[j]) if feature_names is not None else (lambda j: f"x{j}")
        rows = []
        for rule, coef in zip(rules, [] if model is None else model.coef_[0]):
            conds = [f"{name(f)} {'<=' if left else '>'} {bin_threshold(self.bin_edges_[f], t):.6g}"
                     for f, t, left in rule]
            rows.append((" & ".join(conds), float(coef)))
        return rows
