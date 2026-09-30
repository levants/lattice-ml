"""Verify cached numerical claims, image splits, and position invariance."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from lattmc.vision.overcomplete_data_codexgen import load
from lattmc.vision.overcomplete_evaluate_codexgen import read_codes
from lattmc.vision.overcomplete_fetch_codexgen import ROOT


def verify():
    checked, closures, permuted = 0, 0, 0
    rng = np.random.default_rng(721)
    for path in sorted((ROOT / 'results').glob('*/*_codexgen.json')):
        result = json.loads(path.read_text())
        if 'closure_checks' not in result:
            continue
        name, dataset = result['model'], result['dataset']
        matrix, stats = read_codes(name, dataset)
        assert np.isfinite(matrix.data).all() and (matrix.data > 0).all()
        assert np.isfinite(stats['error']).all()
        assert (stats['error'] >= 0).all() and (stats['baseline'] > 0).all()
        _, records = load(dataset)
        sites = matrix.shape[0] // len(records)
        for query in result['queries']:
            z = matrix[:, query['features']].toarray()
            z = z.reshape(len(records), sites, 2)
            shuffled = np.stack([v[rng.permutation(sites)] for v in z])
            for operation in query['operations'].values():
                threshold = np.array(operation['query'])
                original = (z >= threshold).all(2).any(1)
                assert np.array_equal(original,
                                      (shuffled >= threshold).all(2).any(1))
                assert np.array_equal(z.max(1), shuffled.max(1))
                closed = np.array(operation['closed'])
                assert np.array_equal((z >= threshold).all(2),
                                      (z >= closed).all(2))
                permuted += 1
        closures += result['closure_checks']
        checked += 1
    fingerprints = {}
    for dataset in ['imagenette', 'imagewoof', 'pets', 'parts', 'dtd']:
        data, records = load(dataset)
        within = {}
        for i, record in enumerate(records):
            key = hashlib.sha256(data['images'][i].tobytes()).hexdigest()
            within.setdefault(key, set()).add(record['split'])
            fingerprints.setdefault(key, []).append([dataset, i])
        assert all(len(splits) == 1 for splits in within.values()), dataset
    duplicates = [v for v in fingerprints.values() if len(v) > 1]
    for path in Path(__file__).parent.glob('overcomplete*_codexgen.py'):
        source = path.read_text()
        ast.parse(source)
        assert all(len(line) <= 79 for line in source.splitlines()), path
    for name in ['imagenette', 'imagewoof', 'pets', 'parts', 'dtd', 'shapes']:
        path = ROOT / f'activations/prisma_transcoder/{name}'
        record = json.loads((path / 'verification_codexgen.json').read_text())
        assert record['native_vs_explicit_max_error'] < 1e-4
    pretrained = json.loads((ROOT / 'checkpoints/pretrained/'
                            'verification_codexgen.json').read_text())
    assert pretrained['code_max_error'] < 1e-4
    assert pretrained['reconstruction_max_error'] < 1e-4
    result = {'evaluated_model_dataset_pairs': checked,
              'verified_query_closures': closures,
              'independent_image_position_permutations': permuted,
              'within_dataset_exact_crop_split_leakage': False,
              'exact_duplicate_crop_groups': duplicates,
              'source_width_and_syntax': 'passed',
              'native_adapter_checks': 'passed'}
    (ROOT / 'results/verification_codexgen.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
