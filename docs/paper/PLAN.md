# Paper plan (not the paper)

Working title: **Sums of Shallow Optimal Trees: Interpretable Models on the
Size–Loss Frontier**

Status: plan. Numbers marked *(prelim)* come from the development benchmark
(20–25 binary datasets, 10k–100k rows, outer CV3/holdout) and will be replaced
by the final nested-CV5 run (runbook: `bench/docker/README.md` in the research repo).

## 1. Claim in one paragraph

With a budget of 4–16 cuts (the regime where a model can be read end to end), a
logit sum of a few shallow trees fitted with Newton steps and backfitting beats
every single-tree method of the same size — greedy CART (sklearn, rpart, ours),
CART with hierarchical shrinkage, one LightGBM tree, **and optimal trees
(ConTree)** — and also beats a LightGBM restricted to the same number of cuts.
The gain comes from the *shape* (additive effects are not replicated across
branches) and from optimizing the log-loss directly; exhaustive (optimal) search
inside each small tree adds a smaller, consistent gain from 16 cuts up. With free
capacity the same machinery is an additive model of depth-2 trees that matches a
LightGBM with depth-2 trees and trails a tuned LightGBM by ~1% log-loss, at a
fraction of the tuning time. It does not help when the target is a deep
interaction (hierarchical targets, non-stationary series).

## 2. Contributions

1. A Newton/logit version of FIGS with backfitting and a budget in cuts, plus
   `SumOfOptimalTrees`: sums of exhaustively searched depth-2 trees on histogram bins.
2. Evidence on the size–loss frontier (area under the size–loss curve, as
   recommended by van der Linden et al. 2024) against greedy, shrunk, optimal and
   boosted trees of the same size, paired per dataset.
3. A mechanism: the replication problem (Pagallo & Haussler 1990) measured on
   real data — CART spends 8–11 of 16 cuts copying additive effects
   (california, heloc), while the sum spends each cut once.
4. A fast single-tree engine (histogram + exact, hierarchical shrinkage, CV over
   leaves × shrinkage in one fit per fold) that matches tuned sklearn CART + HS
   and is faster *(to confirm: ≥ 2× sklearn)*.
5. An interpretation toolkit that is the model itself (scorecard, rules, shape
   functions, exact contributions, TreeSHAP export) and a reproducible benchmark.

## 3. Research questions and the evidence for each

| RQ | question | evidence (prelim) | still needed |
|---|---|---|---|
| RQ1 | At 4/8/16 cuts, does a sum of trees beat single trees of the same size? | FIGS vs greedy tree −1.5/−1.7/−1.9% log-loss, 17–18/20 wins, p ≤ 0.001; vs 1 LightGBM tree −2.8/−4.8/−4.7% | ConTree (running), rpart (running) |
| RQ2 | Is it "just a small LightGBM"? | vs LightGBM with ⌊b/3⌋ depth-2 trees, tuned: FIGS −4.0/−3.3/−1.4% (25/0, 23/2, 20/5 wins); tie at 32–64 cuts. Shape-free control (LightGBM, ≤ b cuts, leaves per tree tuned in 2..b+1): FIGS −1.3/−0.8/−0.8/−0.5% at 4/8/16/32 (20/0, 18/2, 14/6, 14/6), tie at 64; sum_d2 ties it (−0.1 to −0.5%) at every budget. Below 10k rows FIGS loses at 32–64 (full Newton step, no shrinkage: variance) | bagged structure selection at 32–64 (planned) |
| RQ3 | Does optimal search matter inside the sum? | optimal vs greedy d2 in the same sum: 0% at 4–8 cuts, −0.2/−0.3/−0.5% at 16/32/64 (p < 0.01) | — |
| RQ4 | Why? | replication counts + stumps-only GAM ablation (`ESTRUTURA_CART_VS_SOMA.md`) | figure |
| RQ5 | Free capacity vs boosting | additive d2 +1.2% log-loss, −0.24 AUC pts vs tuned LightGBM, 3 s vs 155 s; = LightGBM-d2 control (−0.1%, p = 0.57) | ratio at max capacity table |
| RQ6 | Single tree: speed and quality vs sklearn | wide-grid CV ties sklearn + HS tuned (+0.16%, p = 0.65) in 0.5 s | speed ratio table (≥ 2×?) |
| RQ7 | Where it fails | pol (hierarchical target), electricity (non-stationary interaction), Amazon (high-cardinality categoricals) | discussion |

Negative results to report: distillation from a LightGBM teacher (~0 at any
budget, 2× time), ratio vocabulary (niche: large gains on accounting/per-capita
data, 0 elsewhere), depth-3 optimal trees (slow, no gain over d2).

## 4. Experimental protocol

- Datasets: binary tasks from TabArena and Grinsztajn et al. (2022) suites, plus
  4 Kaggle pseudo-tasks; tiers by size (≤ 13k, 13.5k–50k, 50k–100k, > 150k).
- Outer: CV5 (final), holdout 80/20 above 50k rows; inner: CV3 for tuning
  (Optuna TPE, same trial budget per hyperparameter for every tuned method).
- Budgets: 4, 8, 16, 32, 64 cuts and free; a method may use *up to* b cuts.
- Metrics: log-loss (primary), AUC; area under the size–loss curve; fit time
  (whole procedure and final refit), all methods on the same machine.
- Statistics: per-dataset paired differences, Wilcoxon signed-rank, win/loss
  counts, median relative Δ (Demšar 2006); critical-difference plot per budget.
- Baselines: sklearn CART (raw and + HS), rpart (cost-complexity pruning),
  LightGBM single tree, ConTree (optimal, depths 2/3/4, 300 s limit, anytime),
  FIGS (imodels), LightGBM with ⌊b/3⌋ depth-2 trees, EBM; ceilings: LightGBM,
  XGBoost, CatBoost with early stopping and tuning.
- Fairness notes: ConTree optimizes 0-1 loss (leaf probabilities by m-estimate);
  the budget-matched LightGBM uses at most 3⌊b/3⌋ cuts (cannot mix stumps).

## 5. Figures and tables

1. Size–loss frontier: log-loss (relative to LightGBM) vs cuts, one line per
   method, median over datasets with IQR band; inset: area under the curve.
2. Loss–time frontier: log-loss vs total fit time (log scale), one point per
   method × budget.
3. Win-rate heatmap: method × budget, share of datasets beating the greedy tree.
4. Mechanism: CART vs FIGS on california at 16 cuts (replicated cuts highlighted)
   + scorecard.
5. Table: ratio at max capacity (our best / LightGBM) per dataset.
6. Table: speed of the single-tree engine vs sklearn by n and p.

## 6. Outline

1. Introduction — interpretability needs ≤ 16 cuts; single trees waste them.
2. Related work — greedy trees, optimal trees (DP/BnB: MurTree, ConTree, SPLIT;
   MIP: OCT), tree sums (FIGS, RuleFit, GA2M/EBM), shrinkage (HS), boosting.
3. Method — Newton gain on bins; optimal depth-2 search; FIGS in logit;
   backfitting; budgets in cuts; interpretation outputs.
4. Experiments — protocol, RQ1–RQ6.
5. Mechanism — replication, additive vs interaction datasets.
6. Limitations — deep interactions, categorical features, binary only (v1).
7. Conclusion.

## 7. Open items before writing

- [ ] Final nested-CV5 run in the cloud (< US$ 10, approval pending).
- [ ] ConTree and rpart cells (running locally).
- [ ] Speed table for the single-tree engine vs sklearn (≥ 2× claim).
- [ ] Area-under-size–loss metric in `eff_claims.py`.
- [ ] Next phase (separate paper?): Rashomon sets of sums + rule mutation.
