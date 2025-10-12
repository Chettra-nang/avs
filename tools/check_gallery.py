import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
html = ROOT / 'data' / 'ambulance_plots_html' / 'index.html'
if not html.exists():
    print('index.html not found at', html)
    sys.exit(2)

text = html.read_text()
# find src="..."
import re
srcs = re.findall(r'src="([^"]+)"', text)
print('Found', len(srcs), 'image tags')

problems = []
checked = 0
for s in srcs[:200]:
    # support file:// URLs and absolute paths
    if s.startswith('file://'):
        img_path = Path(s[len('file://'):])
    elif os.path.isabs(s):
        img_path = Path(s)
    else:
        # resolve relative to html file
        img_path = (html.parent / s).resolve()
    exists = img_path.exists()
    size = img_path.stat().st_size if exists else 0
    info = {'src': s, 'path': str(img_path), 'exists': exists, 'size': size}
    if not exists:
        problems.append((info, 'missing'))
        print('MISSING:', info)
        continue
    try:
        im = Image.open(img_path)
        arr = np.array(im)
        mean = float(arr.mean())
        std = float(arr.std())
        info.update({'mean': mean, 'std': std, 'mode': im.mode, 'shape': arr.shape})
        print('OK:', s, 'size=', size, 'mode=', im.mode, 'shape=', arr.shape, 'mean={:.3f}'.format(mean))
        # flag near-black images
        if mean < 2.0:
            problems.append((info, 'dark'))
        checked += 1
    except Exception as e:
        problems.append((info, f'read-error: {e}'))
        print('READ ERROR', img_path, e)

print('\nSummary:')
print('checked images:', checked)
print('total issues:', len(problems))
for p in problems[:20]:
    print(p)

if problems:
    sys.exit(1)
else:
    sys.exit(0)
