"""Inspect every measured query coordinate without changing retrieval rules."""

from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from lattmc.fca.lattice_utils import join_all, le, upper_mask
from lattmc.activationstudy.common_codexgen import sha256, save_json
from .depth_codexgen import ROOT, OUT, OLD

DEST = ROOT / 'data/activation_studies/witnesses_v1'


def classify(
    trace: ArrayLike,
    query: ArrayLike,
    alpha: float = 1.,
    positions: ArrayLike | None = None,
) -> dict[str, Any]:
    """Use one scaled threshold for item and whole-token satisfaction."""
    trace = np.asarray(trace, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    assert trace.ndim == 2 and trace.shape[1] == len(query)
    assert len(trace) and np.isfinite(trace).all() and (trace >= 0).all()
    assert np.isfinite(query).all() and (query >= 0).all()
    assert np.isfinite(alpha) and alpha > 0
    positions = (np.arange(len(trace)) if positions is None else
                 np.asarray(positions, dtype=int))
    assert len(positions) == len(trace) and len(set(positions)) == len(trace)
    required = alpha * query
    active = query > 0
    trace, query, required = trace[:, active], query[active], required[active]
    if not len(query):
        return dict(status='N', member=True, whole=positions.tolist(),
                    partial=[], witnesses=[], score=None, requirements=[])
    pooled = join_all(trace)
    member = bool(le(required, pooled))
    whole = upper_mask(required, trace)
    matches = trace >= required
    maxima = np.argmax(trace, axis=0)
    status = 'S' if whole.any() else 'D' if member else 'R'
    assert not whole.any() or member
    return dict(
        status=status, member=member,
        whole=positions[whole].tolist(),
        partial=positions[matches.any(axis=1)].tolist(),
        matches=[np.flatnonzero(row).tolist() for row in matches],
        score=float(np.min(pooled / query)),
        requirements=required.tolist(),
        witnesses=[dict(position=int(positions[p]), value=float(trace[p, j]),
                        requirement=float(required[j]),
                        meets=bool(matches[p, j]))
                   for j, p in enumerate(maxima)],
    )


def contexts() -> list[dict[str, Any]]:
    """Collect coordinate witnesses from contextual replay artifacts."""
    directory = ROOT / 'data/activation_studies/context_v1'
    records = []
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    for path in sorted(directory.glob('*layer*.json')):
        data = json.loads(path.read_text())
        arrays = np.load(path.with_suffix('.npz'))
        for case in data['cases']:
            for row in case['rows']:
                key = f"{case['case']}_{row['row']}"
                indices = arrays[key + '_active']
                query = arrays[case['case'] + '_query'][indices]
                values = arrays[key + '_codes']
                result = classify(values, query)
                assert result['member'] == row['fresh_member']
                records.append(dict(
                    id=f"{data['kind']}{data['layer']}-{key}",
                    group='context', family=data['kind'], layer=data['layer'],
                    case=case['case'], row=row['row'],
                    sources=[case['source_row']],
                    source_positions=case['source_positions'],
                    source_kind='token', alpha=1.,
                    coordinates=indices.tolist(), query=query.tolist(),
                    tokens=row['tokens'], pieces=row['pieces'],
                    decoded_text=tokenizer.decode(row['tokens'],
                        clean_up_tokenization_spaces=False),
                    valid_positions=list(range(len(values))),
                    diagnostic=row['position'],
                    trace_file=str(path.with_suffix('.npz').relative_to(ROOT)),
                    trace_key=key + '_codes', trace_sha256=sha256(
                        path.with_suffix('.npz')),
                    original_manifest_sha256=sha256(path), **result,
                ))
    return records


def families() -> list[dict[str, Any]]:
    """Collect coordinate witnesses from cross-family retrieval artifacts."""
    records = []
    design = json.loads((OLD / 'dbpedia_14/design.json').read_text())
    from lattmc.activationstudy.families_config_codexgen import FAMILIES
    for path in sorted(OUT.glob('*_gallery.json')):
        data = json.loads(path.read_text())
        arrays = np.load(path.with_suffix('.npz'))
        family = data['records'][0]['family']
        model = FAMILIES[family]['model']
        tokenizer = AutoTokenizer.from_pretrained(
            model, revision=design['pinned'][model]['revision'],
            local_files_only=True)
        for row in data['records']:
            key = f"{row['task_label']}_{row['rank']}"
            positions = row['valid_positions']
            result = classify(arrays[key][positions], row['query'],
                              row['calibrated_threshold'], positions)
            assert result['member'] == row['accepted']
            assert np.isclose(result['score'], row['score'], rtol=1e-6)
            records.append(dict(
                id=f'{family}-{key}', group='family', family=family,
                layer=row['layer'], case=row['task_label'], row=row['row'],
                rank=row['rank'], sources=row['positive'],
                source_kind='document', alpha=row['calibrated_threshold'],
                coordinates=row['selected_coordinates'], query=row['query'],
                tokens=row['token_ids'], valid_positions=positions,
                decoded_text=tokenizer.decode(row['token_ids'],
                    clean_up_tokenization_spaces=False),
                pieces=[tokenizer.decode([i],
                    clean_up_tokenization_spaces=False)
                    for i in row['token_ids']],
                dataset_identity=row['dataset_identity'],
                source_identities=[design['rows'][i] for i in row['positive']],
                diagnostic=row['position'],
                trace_file=str(path.with_suffix('.npz').relative_to(ROOT)),
                trace_key=key, trace_sha256=sha256(path.with_suffix('.npz')),
                original_manifest_sha256=sha256(path), **result,
            ))
    return records


def main() -> None:
    """Inspect every measured query coordinate without changing retrieval
    rules.
    """
    DEST.mkdir(parents=True, exist_ok=True)
    records = contexts() + families()
    assert len(records) == 72
    counts = {group: {s: sum(r['group'] == group and r['status'] == s
                            for r in records) for s in ('S', 'D', 'R', 'N')}
              for group in ('context', 'family')}
    save_json(DEST / 'records.json', dict(
        source_sha256=sha256(__file__), records=records, counts=counts,
        interpretation='Activation satisfaction, not semantic correctness.',
        thresholds='Exact float64 comparison; original alpha and query.',
    ))
    print(counts)


if __name__ == '__main__':
    main()
