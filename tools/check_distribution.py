"""Check built distributions without importing the checkout.

Run after ``python -m build`` from an environment with ``.[test]`` installed:
    python tools/check_distribution.py [dist]

The wheel gets its own environment with only its declared runtime dependencies.
The sdist's tests are collected using the calling environment's test dependencies.
All installations, caches and unpacked files live in a temporary directory.
"""

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

SMOKE = """\
import importlib.util
import numpy as np
import bettertrees
from bettertrees import (
    BudgetClassifier, CompactTreeBooster, FastDecisionTreeClassifier, SumOfOptimalTrees,
)

for optional in ('pandas', 'lightgbm', 'shap', 'matplotlib'):
    assert importlib.util.find_spec(optional) is None, optional

rng = np.random.default_rng(0)
X = rng.normal(size=(160, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X[::11, 2] = np.nan
for estimator in (
    BudgetClassifier(max_splits=4),
    FastDecisionTreeClassifier(max_depth=3),
    SumOfOptimalTrees(n_trees=1, depth=2),
    CompactTreeBooster(max_splits=3),
):
    estimator.fit(X, y)
    p = estimator.predict_proba(X[:7])
    assert p.shape == (7, 2)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert estimator.predict(X[:7]).shape == (7,)
    print(type(estimator).__name__ + ': OK')
print('Installed package:', bettertrees.__file__)
"""


def run(command, *, cwd, env, capture=False):
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            capture_output=capture, check=False)
    if result.returncode:
        if capture:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)
    return result


def check_distribution(dist, wheelhouse=None):
    wheels = list(dist.glob('*.whl'))
    sources = list(dist.glob('*.tar.gz'))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError('Expected exactly one wheel and one sdist; clean the output directory.')
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        if any('__pycache__' in name or name.endswith(('.nbc', '.nbi')) for name in names):
            raise ValueError('The wheel contains Python or Numba cache files.')
        if not any(name.endswith('/licenses/LICENSE') for name in names):
            raise ValueError('The wheel is missing its license.')
        for typed_file in ('bettertrees/py.typed', 'bettertrees/sums/budget.pyi'):
            if typed_file not in names:
                raise ValueError(f'The wheel is missing {typed_file}.')
    with tempfile.TemporaryDirectory(prefix='bettertrees-dist-') as scratch:
        scratch = Path(scratch)
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['NUMBA_CACHE_DIR'] = str(scratch / 'numba-cache')
        runtime = scratch / 'runtime'
        venv.EnvBuilder(with_pip=True).create(runtime)
        python = runtime / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        install = [str(python), '-m', 'pip', 'install', '--no-cache-dir']
        if wheelhouse is not None:
            install += ['--no-index', '--find-links', str(wheelhouse.resolve())]
        run([*install, str(wheels[0])], cwd=scratch, env=env)
        run([str(python), '-c', SMOKE], cwd=scratch, env=env)
        source_dir = scratch / 'source'
        with tarfile.open(sources[0]) as source:
            source.extractall(source_dir, filter='data')
        roots = list(source_dir.iterdir())
        if len(roots) != 1 or not (roots[0] / 'pyproject.toml').is_file():
            raise ValueError('The sdist does not contain one source project.')
        env['PYTHONPATH'] = str(roots[0] / 'src')
        env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
        run([sys.executable, '-m', 'pytest', '--collect-only', '-q', '-p', 'no:cacheprovider'],
            cwd=roots[0], env=env, capture=True)
        print('Sdist test collection: OK')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dist', nargs='?', default='dist', type=Path)
    parser.add_argument('--wheelhouse', type=Path, help='Optional offline dependency directory')
    args = parser.parse_args()
    check_distribution(args.dist.resolve(), args.wheelhouse)
