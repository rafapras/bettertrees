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


## Update 2026-09-28: 128 cuts, compact booster, editing API

- Budgets now go to 128; a budget b is *eligible* for a dataset only with
  n_train >= 50 b and minority events >= 10 b (EPV rule); below that, 64-128 cut
  comparisons mostly measure split-to-split variance.
- Controls: LightGBM and XGBoost with <= b cuts and tuned shape; LightGBM with
  early stopping and the same distinct-cut accounting (identical trees merged).
- Family chosen by inner CV (FIGS with learning rate and optional backfitting,
  CompactTreeBooster, sum of d2) vs shape-tuned LightGBM, 10k-100k rows:
  -1.4 / -1.0 / -0.9 / -0.4 / -0.4 % at 4-64 cuts (p <= 0.005), tie at 128.
  Vs XGBoost: significant to 64, -0.2 % at 128 (ns).
- Why 128 ties: on HR, bank, credit, credit_card, heloc our tuned models use 30-90
  cuts (more overfits); with backfitting after every cut the learning rate of FIGS
  is undone at large budgets (old leaves return to the full Newton step), so
  backfit_sweeps=0 is a tuning option. `pol` (hierarchical target) is the only
  large loss.
- Negative results: bagged cut vote, forest distillation and cut-mutation Rashomon
  search do not beat FIGS at 16-64 cuts; exhaustive depth-3 terms are 10-100x
  slower for mixed gains.
- Also tried for 128 cuts, all without a gain (paired with the same tuned
  LightGBM cell, 5 losing + 3 winning datasets, 2 splits): grow-then-prune
  (reduced-error pruning, `sums/prune.py`; +1.2 % median), leaf-wise
  LightGBM-shaped candidate trees in the compact booster (+0.4 to +1.1 %),
  no early stopping (+1.1 %). HR, heloc and credit_card lose in every shape,
  including LightGBM's own; the 128-cut claim is "tie with LightGBM and XGBoost".
- With the minimum leaf weight in the tuning (`figs_tuned2`, like LightGBM's
  min_child_samples), family selection by inner CV beats XGBoost with <= b cuts
  at every budget up to 128 (-0.29 % at 128, 11/5, p = 0.021, 16 datasets) and
  LightGBM up to 64; at 128 it ties LightGBM (+0.02 %, 7/9).
- No tuning at all (rule by budget: FIGS full step up to 8 cuts, FIGS with
  learning rate 0.3 up to 64, CompactTreeBooster above): vs the Optuna-tuned
  shape-free LightGBM, -1.0 / -0.8 / -1.0 / -0.5 % at 4/8/16/32 cuts
  (p <= 0.002), tie at 64 and 128, at 0.03-0.6x its total time; same picture vs
  XGBoost. (Rule chosen on split (0,0) of the same datasets: a mild selection bias.)
- LightGBM imported into bettertrees with the leaves refitted jointly
  (`LightGBMRefitClassifier`, refit_lam tuned) beats the same shape-tuned
  LightGBM at 4-64 cuts (-0.5 to -0.7 %, p <= 0.002) and ties it at 128
  (-0.02 %). With it in the family selection: vs LightGBM -1.3 / -1.0 / -0.7 /
  -0.5 / -0.5 % at 4-64 (p <= 0.003), -0.09 % at 128 (9/7, p = 0.21); vs XGBoost
  significant at every budget including 128 (-0.40 %, 12/4, p = 0.011).
