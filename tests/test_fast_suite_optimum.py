"""Nenhuma árvore de profundidade <= d fica abaixo do ótimo 0-1 certificado.

Usa as referências ConTree em cache (``bench/build_contree_refs.py``). Se a
árvore "vencer" o ótimo, há bug no roteamento, nos dados ou na referência.
Também confere que a referência reproduz o próprio erro registrado.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from arvore_rapida import FastDecisionTreeClassifier

BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH))
from build_contree_refs import REF_PATH, route  # noqa: E402
from fast_suite_data import load  # noqa: E402

pytestmark = pytest.mark.skipif(not REF_PATH.exists(),
                                reason="referências ConTree não geradas")


def _certified():
    if not REF_PATH.exists():
        return []
    entries = json.loads(REF_PATH.read_text(encoding="utf-8"))["entries"]
    return [e for e in entries.values() if e["certified"]]


@pytest.mark.parametrize("entry", _certified(),
                         ids=lambda e: f"{e['dataset']}-d{e['depth']}")
def test_greedy_never_beats_certified_optimum(entry):
    X_train, y_train, _, _ = load(entry["dataset"])
    leaves = route(entry["paths"], X_train)
    mass = np.zeros((len(entry["paths"]), 2))
    np.add.at(mass, (leaves, y_train), 1.0)
    assert 1.0 - mass.max(axis=1).sum() / len(y_train) == pytest.approx(
        entry["train_error"], abs=1e-12)
    for splitter in ("hist", "exact"):
        model = FastDecisionTreeClassifier(
            splitter=splitter, max_depth=entry["depth"], min_samples_leaf=1,
            search_stopping="off", random_state=0).fit(X_train, y_train)
        train_error = np.mean(model.predict(X_train) != y_train)
        assert train_error >= entry["train_error"] - 1e-12
