import contextlib
import io
import re
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
text = (root / 'README.md').read_text(encoding='utf-8')
blocks = re.findall(r'```python\n(.*?)\n```', text, re.S)
namespace = {}
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    exec(blocks[0], namespace)
    exec(blocks[1], namespace)
output = stream.getvalue().strip()
quickstart = re.search(r'```\n(8 cuts.*?)\n```', text, re.S).group(1)
explanation = re.search(r'```\n(logit P.*?)\n```', text, re.S).group(1)
assert output == quickstart + '\n' + explanation, output
model, X_test = namespace['model'], namespace['X_test']
np.testing.assert_allclose(model.base_margin_ + model.predict_contributions(X_test).sum(1),
                           model.decision_function(X_test), rtol=0, atol=1e-12)
for label, target in re.findall(r'!?\[([^\]]*)\]\(([^)]+)\)', text):
    for prefix in (
            'https://raw.githubusercontent.com/rafapras/bettertrees/main/',
            'https://github.com/rafapras/bettertrees/blob/main/',
            'https://github.com/rafapras/bettertrees/tree/main/'):
        if target.startswith(prefix):
            target = target[len(prefix):]
            break
    if target.startswith(('https:', '#')):
        continue
    assert (root / target).exists(), (label, target)
print('README: quickstart and full explanation match exactly; local links/assets exist.')
print(quickstart)
