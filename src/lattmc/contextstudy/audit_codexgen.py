"""Independently check saved witnesses, exclusions, and perturbation effects.

This audit uses arrays and token IDs, without neural inference or downloads.
"""

from __future__ import annotations
from typing import Any

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

ROOT = Path(os.environ.get('MY_PAPERS_REPOSITORY', Path.cwd())).resolve()
DATA = ROOT / 'data/activation_studies/context_v1'


def digest(path: Path | str) -> str:
    """Compute the SHA-256 digest of a source or cached artifact."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> dict[str, Any]:
    """Verify the cached evidence and save a structured audit report."""
    conditions, pairs = [], []
    counts = dict(rows=0, fresh_members=0, prefix_trials=0,
                  changed_prefix_trials=0, material_changes=0,
                  witness_threshold_losses=0, suffix_controls=0)
    max_suffix = 0.0
    for kind in ('sae', 'tc'):
        for layer in (8, 11):
            stem = f'{kind}_layer{layer}'
            report = json.loads((DATA / f'{stem}.json').read_text())
            arrays = np.load(DATA / f'{stem}.npz')
            assert report['source_sha256'] == digest(
                Path(__file__).with_name('run_codexgen.py'),
            )
            for case in report['cases']:
                name = case['case']
                eligible = arrays[f'{name}_eligible']
                chosen = sorted(np.random.default_rng(report['seed']).choice(
                    eligible, min(3, len(eligible)), replace=False,
                ).tolist())
                assert chosen == [e['row'] for e in case['rows']]
                query = arrays[f'{name}_query']
                local = dict(prefix=0, changes=0, losses=0)
                for row in case['rows']:
                    counts['rows'] += 1
                    key = f"{name}_{row['row']}"
                    assert row['row'] != case['source_row']
                    assert not re.search(case['exclusion_pattern'],
                                         ''.join(row['pieces']), re.I)
                    active = arrays[f'{key}_active']
                    codes = arrays[f'{key}_codes']
                    witnesses = np.argmax(codes, axis=0)
                    assert np.array_equal(
                        witnesses, row['all_coordinate_witnesses'],
                    )
                    fresh = bool(np.all(codes.max(axis=0) >= query[active]))
                    assert fresh == row['fresh_member']
                    counts['fresh_members'] += int(fresh)
                    target = arrays[f'{key}_target']
                    j, p = row['feature'], row['position']
                    col = int(np.flatnonzero(active == j)[0])
                    assert int(np.argmax(codes[:, col])) == p
                    assert float(target[j]) == row['activation']
                    assert row['threshold'] == float(query[j])
                    for repeat, trial in enumerate(row['prefix']):
                        counts['prefix_trials'] += 1
                        slots, perm = trial['slots'], trial['permutation']
                        assert sorted(slots) == sorted(perm)
                        assert all(0 < s < p for s in slots)
                        assert all(row['tokens'][s] != 50256 for s in slots)
                        changed = sum(row['tokens'][a] != row['tokens'][b]
                                      for a, b in zip(slots, perm))
                        assert changed == trial['changed_tokens']
                        value = float(arrays[f'{key}_prefix{repeat}'][j])
                        assert value == trial['activation']
                        if changed:
                            counts['changed_prefix_trials'] += 1
                            local['prefix'] += 1
                            material = abs(value - target[j]) > (
                                0.001 + 0.001 * abs(target[j])
                            )
                            loss = target[j] >= query[j] > value
                            counts['material_changes'] += int(material)
                            counts['witness_threshold_losses'] += int(loss)
                            local['changes'] += int(material)
                            local['losses'] += int(loss)
                            pairs.append(dict(
                                model=kind, layer=layer, case=name,
                                row=row['row'], repeat=repeat,
                                original=float(target[j]), shuffled=value,
                                threshold=float(query[j]),
                            ))
                    suffix = arrays[f'{key}_suffix']
                    assert np.allclose(suffix, target, atol=1e-3, rtol=1e-3)
                    slots = row['suffix']['slots']
                    assert all(s > p for s in slots)
                    assert sorted(slots) == sorted(
                        row['suffix']['permutation'],
                    )
                    max_suffix = max(max_suffix,
                                     float(abs(suffix - target).max()))
                    counts['suffix_controls'] += 1
                conditions.append(dict(
                    kind=kind, layer=layer, case=name,
                    extent=case['extent_count'],
                    eligible=case['eligible_count'],
                    sampled=len(case['rows']), **local,
                ))
    result = dict(
        checks_passed=True, counts=counts, conditions=conditions,
        maximum_suffix_error=max_suffix, pairs=pairs,
        interpretation='Descriptive dependent-sample counts, not accuracy.',
        hashes={p.name: digest(p) for p in sorted(DATA.glob('*'))
                if p.suffix in ('.npz', '.json') and p.name != 'audit.json'},
    )
    (DATA / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(counts, indent=2))
    print('Maximum suffix-control error:', max_suffix)
    return result


if __name__ == '__main__':
    main()
