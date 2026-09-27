"""Benchmarks do baseline e do splitter de um único nó.

Execute da raiz: python -m arvore_rapida.benchmark --samples 200000 --repeats 5
Dados/splits são fixos; preparação é incluída em cada fit. O teste final é
reservado; métricas nesta fase usam somente a validação. Nenhum tuning.
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
from sklearn.datasets import fetch_covtype, make_classification
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .core import (
    Split,
    build_feature_histogram,
    build_feature_histogram_into,
    fit_bin_edges,
    scan_histogram_feature,
    transform_bins,
    validate_X,
)


def load_data(dataset, samples, classes, seed, data_home=None):
    """Gere dados controlados ou carregue Covertype, registrando a receita.

    Covertype usa subconjunto aleatório estratificado; samples=581012 inclui
    a base completa. O download só ocorre quando explicitamente selecionado.
    """
    if dataset == "synthetic":
        recipe = dict(n_samples=samples, n_features=50, n_informative=20,
                      n_redundant=10, n_classes=classes, n_clusters_per_class=2,
                      weights=None, flip_y=0.01, class_sep=1.0, random_state=seed)
        X, y = make_classification(**recipe)
    else:
        X, y = fetch_covtype(data_home=data_home, return_X_y=True)
        if samples > len(X):
            raise ValueError("samples excede as linhas do Covertype.")
        if samples < len(X):
            X, _, y, _ = train_test_split(X, y, train_size=samples, stratify=y, random_state=seed)
        recipe = dict(dataset="covtype", n_samples=samples, random_state=seed)
    return X, y, recipe


def measure_fit(X, y, X_val, y_val, params):
    """Meça conversão+fit e previsão completa, sem incluir cálculo de métricas."""
    start = perf_counter()
    prepared = np.ascontiguousarray(X, dtype=np.float32)
    prepared_at = perf_counter()
    model = DecisionTreeClassifier(**params).fit(prepared, y)
    fitted_at = perf_counter()
    probabilities = model.predict_proba(np.ascontiguousarray(X_val, dtype=np.float32))
    predicted_at = perf_counter()
    return dict(preparation_seconds=prepared_at-start,
                fit_complete_seconds=fitted_at-start,
                predict_proba_seconds=predicted_at-fitted_at,
                accuracy=float(accuracy_score(y_val, model.classes_[probabilities.argmax(axis=1)])),
                log_loss=float(log_loss(y_val, probabilities, labels=model.classes_)),
                leaves=int(model.get_n_leaves()), depth=int(model.get_depth()))


def evaluate_gate(baseline, candidate):
    """Avalie o portão do plano em resultados de MESMO dataset/split/orçamento.

    Entradas: median_fit_seconds, accuracy e log_loss (validação ou teste
    igualmente identificado para ambos). Redução de acurácia <= 0.005,
    log loss <= 1.01*baseline e velocidade >=1.5. Não substitui verificar
    protocolo, equivalência do experimento e repetibilidade antes de chamar.
    """
    speedup = baseline["median_fit_seconds"] / candidate["median_fit_seconds"]
    accuracy_drop = baseline["accuracy"] - candidate["accuracy"]
    loss_ok = candidate["log_loss"] <= 1.01 * baseline["log_loss"]
    return dict(speedup=speedup, accuracy_drop=accuracy_drop,
                log_loss_ok=loss_ok, passed=bool(speedup >= 1.5 and accuracy_drop <= 0.005 and loss_ok))


def _build_node_histograms(X_binned, y, weights, sample_indices, edges, n_classes):
    """Acumule um histograma por feature para o mesmo nó, sem buscar cortes."""
    histograms = []
    for feature in range(X_binned.shape[1]):
        histograms.append(build_feature_histogram(
            X_binned[:, feature], y, weights, sample_indices, 0,
            len(sample_indices), len(edges[feature]) + 2, n_classes))
    return tuple(histograms)


def _scan_prebuilt_histograms(histograms, edges, parent_mass, min_samples_leaf,
                              feature_order, stopping, bound_interval=1,
                              parent_gini=None):
    """Varra histogramas já acumulados; usado para isolar o custo da busca."""
    best = Split(-1, np.nan, False, -np.inf, 0, -1)
    evaluated = 0
    skipped = 0
    for feature in feature_order:
        mass, count = histograms[int(feature)]
        result = scan_histogram_feature(
            mass, count, edges[int(feature)], parent_mass,
            min_samples_leaf=min_samples_leaf, incumbent_gain=best.gain,
            search_stopping=stopping, gain_tolerance=0.0,
            bound_interval=bound_interval, parent_gini=parent_gini)
        bin_threshold, missing_left, gain, n_left, seen, omitted = result
        evaluated += seen
        skipped += omitted
        if bin_threshold < 0 or not np.isfinite(gain):
            continue
        n_finite_bins = len(edges[int(feature)]) + 1
        threshold = (np.inf if bin_threshold == n_finite_bins
                     else float(edges[int(feature)][bin_threshold - 1]))
        if gain > best.gain:
            best = Split(int(feature), threshold, bool(missing_left),
                         float(gain), int(n_left), int(bin_threshold))
    return best, int(evaluated), int(skipped)


def _run_reused_buffer_pipeline(X_binned, y, weights, sample_indices, edges,
                                parent_mass, min_samples_leaf, feature_order,
                                stopping, bound_interval, parent_gini):
    """Acumule e escaneie feature a feature usando um único scratch reutilizado."""
    max_bins = max((len(edge) + 2 for edge in edges), default=1)
    mass_buffer = np.empty((max_bins, len(parent_mass)), dtype=np.float64)
    count_buffer = np.empty(max_bins, dtype=np.int64)
    best = Split(-1, np.nan, False, -np.inf, 0, -1)
    evaluated = 0
    skipped = 0
    histogram_seconds = 0.0
    search_seconds = 0.0
    total_start = perf_counter()
    for feature in feature_order:
        feature = int(feature)
        n_bins = len(edges[feature]) + 2
        histogram_start = perf_counter()
        mass, count = build_feature_histogram_into(
            X_binned[:, feature], y, weights, sample_indices, 0,
            len(sample_indices), n_bins, len(parent_mass), mass_buffer, count_buffer)
        histogram_seconds += perf_counter() - histogram_start
        search_start = perf_counter()
        result = scan_histogram_feature(
            mass, count, edges[feature], parent_mass,
            min_samples_leaf=min_samples_leaf, incumbent_gain=best.gain,
            search_stopping=stopping, gain_tolerance=0.0,
            bound_interval=bound_interval, parent_gini=parent_gini)
        search_seconds += perf_counter() - search_start
        bin_threshold, missing_left, gain, n_left, seen, omitted = result
        evaluated += seen
        skipped += omitted
        if bin_threshold < 0 or not np.isfinite(gain):
            continue
        n_finite_bins = len(edges[feature]) + 1
        threshold = (np.inf if bin_threshold == n_finite_bins
                     else float(edges[feature][bin_threshold - 1]))
        if gain > best.gain:
            best = Split(feature, threshold, bool(missing_left), float(gain),
                         int(n_left), int(bin_threshold))
    return (float(perf_counter() - total_start), float(histogram_seconds),
            float(search_seconds), best, int(evaluated), int(skipped))


def _split_record(split):
    """Converta Split em JSON sem perder o marcador de ausência de corte."""
    return dict(feature=int(split.feature), threshold=float(split.threshold),
                missing_left=bool(split.missing_left), gain=float(split.gain),
                n_left=int(split.n_left), bin_threshold=int(split.bin_threshold))


def benchmark_one_node(X, y, *, max_bins=255, min_samples_leaf=20,
                       repeats=5, sample_size=None, random_state=42,
                       bound_interval=1, cache_parent_gini=False,
                       reuse_histograms=False):
    """Compare busca completa e bound no mesmo histograma de um único nó.

    A saída separa: (i) histograma, (ii) busca sobre histogramas já congelados
    e (iii) pipeline histograma+busca. O bound é comparado na mesma ordem de
    features e no mesmo conjunto de massas; nenhum dado de validação entra.
    """
    if repeats < 1:
        raise ValueError("repeats deve ser positivo.")
    prepared = validate_X(X)
    target = np.asarray(y)
    if target.ndim != 1 or len(target) != len(prepared):
        raise ValueError("y deve ter uma classe por linha de X.")
    classes, encoded = np.unique(target, return_inverse=True)
    encoded = np.ascontiguousarray(encoded, dtype=np.int32)
    rng = np.random.default_rng(random_state)
    if sample_size is not None:
        if sample_size < 1 or sample_size > len(prepared):
            raise ValueError("sample_size deve estar entre 1 e o tamanho de X.")
        selected = rng.permutation(len(prepared))[:sample_size]
        prepared = np.ascontiguousarray(prepared[selected])
        encoded = np.ascontiguousarray(encoded[selected], dtype=np.int32)
    n_samples, n_features = prepared.shape
    weights = np.ones(n_samples, dtype=np.float64)
    sample_indices = np.arange(n_samples, dtype=np.int64)
    feature_order = np.arange(n_features, dtype=np.int64)
    edges = fit_bin_edges(prepared, max_bins=max_bins)
    X_binned = transform_bins(prepared, edges)
    parent_mass = np.bincount(encoded, weights=weights,
                              minlength=len(classes)).astype(np.float64)

    # Warm up Numba and both search paths outside the recorded samples.
    warm_hist = _build_node_histograms(
        X_binned, encoded, weights, sample_indices, edges, len(classes))
    parent_gini = (float(1.0 - ((parent_mass / parent_mass.sum()) ** 2).sum())
                   if cache_parent_gini else None)
    _scan_prebuilt_histograms(warm_hist, edges, parent_mass, min_samples_leaf,
                              feature_order, "off", bound_interval, parent_gini)
    _scan_prebuilt_histograms(warm_hist, edges, parent_mass, min_samples_leaf,
                              feature_order, "bound", bound_interval, parent_gini)

    shared_hist_start = perf_counter()
    shared_hist = _build_node_histograms(
        X_binned, encoded, weights, sample_indices, edges, len(classes))
    shared_hist_seconds = perf_counter() - shared_hist_start
    search = {}
    pipeline = {}
    for stopping in ("off", "bound"):
        search_runs = []
        pipeline_runs = []
        for _ in range(repeats):
            search_start = perf_counter()
            split, evaluated, skipped = _scan_prebuilt_histograms(
                shared_hist, edges, parent_mass, min_samples_leaf,
                feature_order, stopping, bound_interval, parent_gini)
            search_runs.append(dict(seconds=perf_counter() - search_start,
                                    evaluated=evaluated, skipped=skipped,
                                    split=_split_record(split)))

            total_start = perf_counter()
            if reuse_histograms:
                (measured_total, hist_seconds, search_seconds, total_split,
                 total_evaluated, total_skipped) = _run_reused_buffer_pipeline(
                    X_binned, encoded, weights, sample_indices, edges,
                    parent_mass, min_samples_leaf, feature_order, stopping,
                    bound_interval, parent_gini)
            else:
                hist_start = perf_counter()
                histograms = _build_node_histograms(
                    X_binned, encoded, weights, sample_indices, edges, len(classes))
                hist_seconds = perf_counter() - hist_start
                search_start = perf_counter()
                total_split, total_evaluated, total_skipped = _scan_prebuilt_histograms(
                    histograms, edges, parent_mass, min_samples_leaf,
                    feature_order, stopping, bound_interval, parent_gini)
                search_seconds = perf_counter() - search_start
                measured_total = perf_counter() - total_start
            pipeline_runs.append(dict(seconds=measured_total,
                                      histogram_seconds=hist_seconds,
                                      search_seconds=search_seconds,
                                      evaluated=total_evaluated,
                                      skipped=total_skipped,
                                      split=_split_record(total_split)))
        search[stopping] = search_runs
        pipeline[stopping] = pipeline_runs

    off_split = search["off"][0]["split"]
    bound_split = search["bound"][0]["split"]
    same_split = (
        off_split["feature"] == bound_split["feature"]
        and off_split["bin_threshold"] == bound_split["bin_threshold"]
        and off_split["missing_left"] == bound_split["missing_left"]
        and np.isclose(off_split["gain"], bound_split["gain"])
    )
    return dict(
        samples=int(n_samples), features=int(n_features), classes=len(classes),
        max_bins=int(max_bins), min_samples_leaf=int(min_samples_leaf),
        bound_interval=int(bound_interval), cache_parent_gini=bool(cache_parent_gini),
        reuse_histograms=bool(reuse_histograms),
        edges_bins=[int(len(edge) + 1) for edge in edges],
        histogram_bytes=int(X_binned.nbytes),
        shared_histogram_seconds=float(shared_hist_seconds),
        search={mode: dict(
            median_seconds=float(np.median([run["seconds"] for run in runs])),
            evaluated=int(runs[0]["evaluated"]), skipped=int(runs[0]["skipped"]),
            runs=runs) for mode, runs in search.items()},
        pipeline={mode: dict(
            median_seconds=float(np.median([run["seconds"] for run in runs])),
            median_histogram_seconds=float(np.median([run["histogram_seconds"] for run in runs])),
            median_search_seconds=float(np.median([run["search_seconds"] for run in runs])),
            evaluated=int(runs[0]["evaluated"]), skipped=int(runs[0]["skipped"]),
            runs=runs) for mode, runs in pipeline.items()},
        same_best_split=bool(same_split),
    )


def benchmark_one_node_variants(X, y, *, max_bins=255, min_samples_leaf=20,
                                repeats=5, sample_size=None, random_state=42):
    """Execute a matriz das variantes de bound já implementadas.

    Cada entrada reutiliza os mesmos dados, seed e protocolo, mas a função
    mantém os resultados separados para não confundir uma combinação com o
    baseline. Variantes de layout, nós pequenos e subtração de histogramas
    entram na matriz quando forem implementadas.
    """
    variants = {
        "reference_every_bin": dict(bound_interval=1, cache_parent_gini=False),
        "cached_parent_gini": dict(bound_interval=1, cache_parent_gini=True),
        "block_8_cached_parent_gini": dict(bound_interval=8, cache_parent_gini=True),
        "block_16_cached_parent_gini": dict(bound_interval=16, cache_parent_gini=True),
        "block_8_without_parent_cache": dict(bound_interval=8, cache_parent_gini=False),
        "reused_histogram_buffers": dict(bound_interval=1, cache_parent_gini=False,
                                          reuse_histograms=True),
    }
    return {name: benchmark_one_node(
        X, y, max_bins=max_bins, min_samples_leaf=min_samples_leaf,
        repeats=repeats, sample_size=sample_size, random_state=random_state,
        **options) for name, options in variants.items()}


def main():
    """Salve versões, receita, cinco medições aquecidas e custo isolado de bins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "covtype"), default="synthetic")
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--max-leaf-nodes", type=int, default=256)
    parser.add_argument("--max-bins", type=int, default=255)
    parser.add_argument("--data-home", default="dados_originais/arvore_rapida")
    parser.add_argument("--output", default="resultados/arvore_rapida/baseline.json")
    parser.add_argument("--one-node", action="store_true",
                        help="meça busca off/bound em um único histograma")
    parser.add_argument("--node-samples", type=int, default=16384)
    parser.add_argument("--bound-interval", type=int, default=1)
    parser.add_argument("--cache-parent-gini", action="store_true")
    parser.add_argument("--reuse-histograms", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats deve ser positivo")
    X, y, recipe = load_data(args.dataset, args.samples, args.classes, args.seed, args.data_home)
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.4, random_state=args.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.5, random_state=args.seed+1, stratify=y_remaining)
    params = dict(criterion="gini", splitter="best", random_state=args.seed,
                  max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
                  max_leaf_nodes=args.max_leaf_nodes, min_impurity_decrease=0.0)
    with threadpool_limits(limits=1):
        cold = measure_fit(X_train, y_train, X_val, y_val, params)
        print(json.dumps({"first_call": cold}), flush=True)
        runs = []
        for i in range(args.repeats):
            result = measure_fit(X_train, y_train, X_val, y_val, params)
            runs.append(result)
            print(json.dumps({"run": i+1, **result}), flush=True)
        bin_runs = []
        for _ in range(args.repeats):
            start = perf_counter()
            prepared = np.ascontiguousarray(X_train, dtype=np.float32)
            edges = fit_bin_edges(prepared, args.max_bins)
            bins = transform_bins(prepared, edges)
            bin_runs.append(perf_counter()-start)
    median = float(np.median([run["fit_complete_seconds"] for run in runs]))
    process = psutil.Process()
    memory = process.memory_info()._asdict()
    one_node = None
    if args.one_node:
        one_node = benchmark_one_node(
            X_train, y_train, max_bins=args.max_bins,
            min_samples_leaf=args.min_samples_leaf, repeats=args.repeats,
            sample_size=min(args.node_samples, len(X_train)),
            random_state=args.seed, bound_interval=args.bound_interval,
            cache_parent_gini=args.cache_parent_gini,
            reuse_histograms=args.reuse_histograms)
        print(json.dumps({"one_node": {
            "samples": one_node["samples"],
            "off_search_seconds": one_node["search"]["off"]["median_seconds"],
            "bound_search_seconds": one_node["search"]["bound"]["median_seconds"],
            "same_best_split": one_node["same_best_split"]}}, ensure_ascii=False), flush=True)
    result = dict(status="baseline_only_candidate_not_measured",
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=platform.python_version(), os=platform.platform(),
                      cpu=platform.processor(), physical_cores=psutil.cpu_count(logical=False),
                      logical_cores=psutil.cpu_count(), threads_limit=1,
                      versions={name: version(name) for name in ("numpy", "numba", "scikit-learn", "psutil")}),
                  dataset=recipe, input_dtype=str(X_train.dtype), prepared_dtype="float32",
                  splits=dict(train=len(X_train), validation=len(X_val), test_reserved=len(X_test),
                              train_seed=args.seed, validation_test_seed=args.seed+1),
                  class_counts={str(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))},
                  params=params, first_call=cold, warm_runs=runs,
                  summary=dict(median_fit_seconds=median, accuracy=runs[-1]["accuracy"], log_loss=runs[-1]["log_loss"]),
                  binning=dict(max_bins=args.max_bins, raw_seconds=bin_runs,
                               median_seconds=float(np.median(bin_runs)), bytes=int(bins.nbytes),
                               target_total_fit_seconds=median/1.5),
                  memory=dict(process_end=memory,
                              scope="Process lifetime peak if supplied by OS; includes data generation; NOT isolated fit peak."),
                  jit_first_call_seconds=None, candidate=None, gate=None,
                  one_node=one_node)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"saved": str(output.resolve()), "median_fit_seconds": median,
                      "median_binning_seconds": result["binning"]["median_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
