"""Árvores rápidas e somas interpretáveis de árvores para classificação binária.

- ``FastDecisionTreeClassifier`` / ``...CV``: árvore única (motores exact e hist).
- ``SumOfOptimalTrees``: soma em logit de poucas árvores Newton ótimas (d1–d3).
- ``FIGSClassifier``: FIGS (Tan et al., 2022) em logit.
- ``AdditiveTreeBooster``: soma longa de árvores ótimas d1/d2 com early stopping.

As somas expõem ``rules()``, ``explain()``, ``to_dict()``,
``predict_contributions()``, ``plot_shapes()`` e ``plot_contributions()``.
O resto de ``arvore_rapida.capacidade`` é experimental.
"""

from .autotune import FastDecisionTreeClassifierCV
from .capacidade import AdditiveTreeBooster, FIGSClassifier, SumOfOptimalTrees
from .estimator import FastDecisionTreeClassifier
from .multilevel import fit_multilevel_tree

__all__ = ["AdditiveTreeBooster", "FIGSClassifier", "FastDecisionTreeClassifier",
           "FastDecisionTreeClassifierCV", "SumOfOptimalTrees", "fit_multilevel_tree"]
