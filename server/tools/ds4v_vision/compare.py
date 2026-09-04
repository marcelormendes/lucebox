#!/usr/bin/env python3
"""Compare native raster tensors against independently generated parent fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('reference', type=Path)
p.add_argument('native', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
manifest = json.loads((a.reference / 'manifest.json').read_text())
results = {}
for label, entry in manifest['images'].items():
    results[label] = {}
    patches = entry['patches']
    assert hashlib.sha256((a.reference / patches['file']).read_bytes()).hexdigest() == patches['sha256']
    for stage in ('features', 'embeddings'):
        meta = entry[stage]
        reference_path = a.reference / meta['file']
        assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == meta['sha256']
        ref = np.fromfile(reference_path, np.float32)
        actual = np.fromfile(a.native / f'{label}-{stage}.f32', np.float32)
        assert actual.size == ref.size == np.prod(meta['shape']), (label, stage, 'shape mismatch')
        finite = bool(np.isfinite(actual).all())
        delta = actual.astype(np.float64) - ref.astype(np.float64)
        cosine = np.dot(actual.astype(np.float64), ref.astype(np.float64)) / (np.linalg.norm(actual.astype(np.float64)) * np.linalg.norm(ref.astype(np.float64)))
        results[label][stage] = dict(shape=meta['shape'], finite=finite, max_abs=float(np.abs(delta).max()), rmse=float(np.sqrt(np.mean(delta ** 2))), cosine=float(cosine), exact_fraction=float(np.mean(actual==ref)))
a.output.write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
