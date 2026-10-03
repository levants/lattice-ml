"""Shared manuscript table names, verified condition labels, and notes."""

from __future__ import annotations
from typing import Any
from collections.abc import Callable

import json
import os
from pathlib import Path
import re
import textwrap

ROOT = Path(os.environ.get('MY_PAPERS_REPOSITORY',
                           Path(__file__).resolve().parents[3])).resolve()
PAPER = Path(os.environ.get('LATTCONFERENCE_PAPER',
                           ROOT / 'texs/sparsesurrs/lattconference'))

NAMES = {
    'sae': 'GPT-2-small / ReLU SAE',
    'tc': 'GPT-2-small / ReLU transcoder',
    'pythia_topk': 'Pythia-70M / TopK SAE',
    'smol_topk': 'SmolLM2-135M / TopK SAE',
    'qwen_transcoder': 'Qwen3-0.6B / ReLU transcoder',
    'gemma_matryoshka': 'Gemma-2-2B / Matryoshka JumpReLU SAE',
}
SITES = {
    'sae': 'residual-pre', 'tc': 'MLP input to output',
    'pythia_topk': 'residual-post', 'smol_topk': 'MLP output',
    'qwen_transcoder': 'MLP input to output',
    'gemma_matryoshka': 'residual-post',
}
BLOCKS = dict(pythia_topk=3, smol_topk=15, qwen_transcoder=14,
              gemma_matryoshka=8)

TITLES = {
    'example-infomorphism-contexts':
        'OpenWebText source items for canonical concept maps',
    'example-dbpedia-contexts':
        'DBpedia source items for canonical concept maps',
    'common-features-layer-8-11':
        'OpenWebText common document features: GPT-2-small blocks 8 and 11',
    'common-features-dbpedia':
        'DBpedia common document features: Pythia-70M and SmolLM2-135M',
    'city-layer0-deagg-min-act':
        'GPT-2-small City support-floor token probe at block 0',
    'nyc-highlighted-excerpts':
        'GPT-2-small New York City support-floor witnesses at blocks 0 and 11',
    'animals-highlighted-excerpts':
        'GPT-2-small Cat/dog support-floor witnesses at blocks 8 and 11',
    'witness-ledger': 'Coordinate witnesses for complete-query satisfaction',
    'graded-coincidence':
        'GPT-2-small cross-surrogate coincidence of token-meet extents',
    'graded-layer-coincidence':
        'GPT-2-small cross-block coincidence of token-meet extents',
    'checkpoint-identities':
        'Backbones, sparse surrogates, and activation sites',
    'context-highlighted':
        'OpenWebText token-meet witnesses: paired GPT-2 views',
    'depth-highlighted':
        'DBpedia document-meet witnesses: four surrogate families',
    'depth-sources':
        'DBpedia source documents for the calibrated meet galleries',
    'context-counts':
        'OpenWebText token-meet retrieval and context perturbations',
    'external-main': 'External-corpus retrieval: average precision',
    'external-checkpoints':
        'External-corpus sparse-code activity and reconstruction',
    'families-checkpoints':
        'Additional-family sparse-code activity and reconstruction',
    'depth-main': 'SmolLM2-135M document retrieval across MLP-output blocks',
    'depth-coverage':
        'SmolLM2-135M document coverage and lexical-overlap controls',
    'depth-granularity':
        'SmolLM2-135M retrieval at two DBpedia category resolutions',
    'depth-readouts':
        'SmolLM2-135M contextual readouts across MLP-output blocks',
}

WITNESS_NOTE = (
    r'Blocks, cached rows, and token positions are zero-based; '
    r'$r@(p,\ldots)$ identifies source row $r$ and token positions. '
    r'Train/test item IDs refer to the original dataset split. '
    r'$d$ counts positive query coordinates; $\alpha$ scales the query; '
    r'$s=\min_{j:u_{j}>0}f(T)_{j}/u_{j}$, so membership means '
    r'$s\geq\alpha$. Activations are unnormalized checkpoint codes; '
    r'rounding to three decimals affects display only. '
    r'Exact unrounded comparisons use no membership tolerance.'
)
GPT_NOTE = (
    r'Both conditions use GPT-2-small. SAE denotes the ReLU residual-pre '
    r'autoencoder; TC denotes the ReLU transcoder from layer-normalized '
    r'MLP input to MLP output. The block index $\ell$ is zero-based. '
    r'Checkpoint identifiers are in \appcref{tab:checkpoint-identities}.'
)
EXTERNAL_NOTE = (
    r'All conditions use block 8 (zero-based). GPT-2 residual and MLP are '
    r'GPT-2-small ReLU SAEs at residual-pre and MLP output, respectively. '
    r'Gemma 37/301 are Gemma-2-2B residual-post JumpReLU SAEs; the numbers '
    r'are release identifiers, not measured activity. '
    r'Exact checkpoints: \appcref{tab:checkpoint-identities}.'
)
FAMILY_NOTE = (
    r'Pythia: Pythia-70M TopK SAE, block 3, residual-post; '
    r'SmolLM2: SmolLM2-135M TopK SAE, block 15, MLP output; '
    r'Qwen TC: Qwen3-0.6B ReLU transcoder, block 14, MLP input to output; '
    r'Mat: Gemma-2-2B Matryoshka JumpReLU SAE, block 8, residual-post. '
    r'Mat-512/2048 are prefixes of Mat-32k, not separate checkpoints. '
    r'All blocks are zero-based; see \appcref{tab:checkpoint-identities}.'
)


def panel(family: str, block: int | None = None) -> str:
    """Format a checkpoint family, block, and activation-site label."""
    block = BLOCKS[family] if block is None else block
    return NAMES[family] + r'\newline ' + f'block {block}; {SITES[family]}'


def group(text: str, start: int) -> tuple[str, int]:
    """Read a balanced TeX argument, respecting escaped braces."""
    assert text[start] == '{'
    depth = 1
    for i in range(start + 1, len(text)):
        if text[i] in '{}' and text[i - 1] != '\\':
            depth += 1 if text[i] == '{' else -1
            if not depth:
                return text[start + 1:i], i + 1
    raise ValueError('Unclosed TeX argument')


def title_for(key: str, old: str, body: str) -> str:
    """Select an informative table title from its label and content."""
    if key in TITLES:
        return TITLES[key]
    if key.startswith('context-layer'):
        block = re.search(r'layer(\d+)', key)[1]
        case = re.search(r'Complete (.*?) sample', body)
        case = case[1] if case else old.split(': ')[-1].split(' at ')[0]
        return f'OpenWebText token-meet witnesses: {case} at block {block}'
    if key.startswith('depth-gallery'):
        kind = 'NaturalPlace' if 'geographic' in key else 'Animal'
        part = ('SmolLM2 and Qwen3' if 'continued' in key
                else 'Gemma-2 and Pythia')
        return f'DBpedia {kind} document-meet witnesses: {part}'
    if re.fullmatch(r'layer(8|11)-comparison(-min-act)?', key):
        block = re.search(r'layer(\d+)', key)[1]
        rule = 'support-floor' if 'min-act' in key else 'exact'
        return f'GPT-2-small Cat/dog {rule} meet at block {block}'
    if old.startswith(('Layer ', 'Block ')):
        match = re.match(r'(?:Layer|Block) (\d+) (.*)', old)
        block, description = match.groups()
        return (f'GPT-2-small {description} at block {block}')
    # Retain the experiment's established descriptive first sentence.
    return old.rstrip('.')


def notes_for(key: str, text: str) -> str:
    """Select the explanatory notes required by a table's experiment."""
    if key in ('nyc-highlighted-excerpts', 'animals-highlighted-excerpts'):
        return (r'Both columns use OpenWebText and zero-based block indices. '
                r'The transcoder encodes layer-normalized MLP input and '
                r'predicts MLP output; the SAE encodes the residual stream '
                r'before that block. Exact checkpoint identifiers are in '
                r'\appcref{tab:checkpoint-identities}.')
    if key == 'conference-tasks':
        return ''
    if key == 'witness-ledger':
        return (
            r'Item is the stable gallery ID (W: cached gallery; L: historical '
            r'replay). Status is S (whole-query token), D (distributed), or '
            r'R (rejected); yellow D in the galleries is not semantic '
            r'approval. '
            r'$j$ is a zero-based dictionary coordinate local to that '
            r'checkpoint, $q_{j}=\alpha u_{j}$ its required activation, '
            r'$p$ the zero-based first maximizing valid token position, and '
            r'$a=z_{p,j}$ its activation. Pass tests $a\geq q_{j}$. '
            r'All positive coordinates are listed. Six significant digits '
            r'are shown; exact comparisons use unrounded codes and no '
            r'tolerance. U rows have no measured ledger entries.'
        )
    if key.startswith(('context-layer', 'context-highlighted',
                       'depth-gallery', 'depth-highlighted',
                       'layer8-comparison', 'layer11-comparison')):
        return WITNESS_NOTE
    if key in ('depth-main', 'depth-readouts', 'depth-coverage',
               'depth-granularity'):
        return (r'AG denotes AG News and DB denotes DBpedia. Depth uses '
                r'SmolLM2-135M TopK ($k=32$) MLP-output SAEs; block '
                r'indices are zero-based. AP is average precision; cos. '
                r'denotes cosine similarity, Single is the calibrated '
                r'best single feature, and TF-IDF is the lexical baseline.')
    if key.startswith('external-') and key != 'external-tasks':
        return EXTERNAL_NOTE + diagnostics(key)
    if key.startswith('families-'):
        return FAMILY_NOTE + diagnostics(key)
    if (key.startswith(('graded-', 'conference-')) or
            key in ('context-counts', 'animals-highlighted-excerpts')):
        return GPT_NOTE
    return ''


def diagnostics(key: str) -> str:
    """Explain diagnostic columns for the selected table family."""
    if not key.endswith('checkpoints'):
        return ''
    return (
        r' Tokens counts valid test-token positions; Mean $L_{0}$ is the '
        r'average number of positive coordinates per such token. Width is '
        r'the dictionary dimension. NMSE is total squared reconstruction '
        r'error divided by total squared target magnitude (no centering); '
        r'for a transcoder the target is MLP output. Values are rounded '
        r'for display, without changing the saved measurements.'
    )


def annotate(text: str) -> str:
    """Idempotently name tables; preserve numerical cells and stable labels."""
    pattern = r'\\begin\{(table\*?|longtable)\}.*?\\end\{\1\}'

    def replace(match: re.Match[str]) -> str:
        """Rewrite one matched table while preserving its body."""
        block = match[0]
        if any(r'\label{tab:' + key + '}' in block for key in (
                'nyc-highlighted-excerpts', 'animals-highlighted-excerpts')):
            header = (r'\toprule' + '\n' + r'\textbf{Block} & ' +
                      r'\textbf{' + NAMES['tc'] + r'}\newline ' +
                      SITES['tc'] + ' &\n' + r'\textbf{' + NAMES['sae'] +
                      r'}\newline ' + SITES['sae'] + r' \\' + '\n' +
                      r'\midrule')
            block = re.sub(r'\\toprule.*?\\midrule',
                           lambda _: header, block, count=1, flags=re.S)
        label = re.search(r'\\label\{tab:([^}]+)\}', block)
        cap = re.search(r'\\caption(?:\[([^]]*)\])?\s*\{', block)
        if not label or not cap:
            return block
        key = label[1]
        body, end = group(block, cap.end() - 1)
        note = body.find(r'\TableNote{')
        if note >= 0:
            _, stop = group(body, note + len(r'\TableNote'))
            body = body[:note] + body[stop:]
        old = re.sub(r'\s+', ' ', cap[1]).strip() if cap[1] else None
        body = body.strip()
        if body.startswith(r'\textbf{'):
            title, pos = group(body, len(r'\textbf'))
            old = old or re.sub(r'\s+', ' ', title).rstrip('.')
            body = body[pos:].strip()
        old = old or re.split(
            r'\.\s', re.sub(r'\s+', ' ', body), maxsplit=1)[0]
        title = title_for(key, old, body)
        if key == 'witness-ledger':
            body = ''
        pattern = re.escape(title).replace(r'\ ', r'\s+') + r'\.\s*'
        duplicate = re.match(pattern, body)
        if duplicate:
            body = body[duplicate.end():].strip()
        if key == 'graded-coincidence':
            body = body.replace('Cross-model', 'Cross-surrogate')
        if key == 'graded-layer-coincidence':
            body = body.replace('Within-model', 'Within-surrogate')
        notes = notes_for(key, block)
        caption = (r'\caption[' + title + ']{' + r'\textbf{' + title + '.} '
                   + body
                   + (' ' + r'\TableNote{' + notes + '}' if notes else '')
                   + '}')
        return block[:cap.start()] + caption + block[end:]

    text = text.replace('Model/layer', 'Surrogate/block')
    text = re.sub(r'\bLayers(?=\s*&)', 'Blocks', text)
    text = re.sub(r'\bLayer(?=\s*&)', 'Block', text)
    text = re.sub(r'\bModel(?=\s*&)', 'Surrogate', text)
    result = re.sub(pattern, replace, text, flags=re.S)
    return result.replace('}\\label{', '}\n\\label{')


def wrap(text: str) -> str:
    """Wrap LaTeX source lines without breaking words or commands."""
    lines = []
    for line in text.splitlines():
        lines.extend(textwrap.wrap(line, 79, break_long_words=False,
                                   break_on_hyphens=False) or [''])
    assert all(len(line) <= 79 for line in lines)
    return '\n'.join(lines) + '\n'


def main(paper: Path = PAPER) -> list[dict[str, Any]]:
    """Update paper captions and return the table title inventory."""
    inventory = []
    for path in sorted((paper / 'tables').glob('*.tex')):
        content = annotate(path.read_text())
        path.write_text(wrap(content))
        for match in re.finditer(r'\\caption\[([^]]+)\].*?'
                                 r'\\label\{([^}]+)\}', content, re.S):
            inventory.append(dict(label=match[2], title=match[1],
                                  file=str(path.relative_to(paper))))
    dest = ROOT / 'experiments/activation_studies/tablecontexts_v1'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'table_titles.json').write_text(
        json.dumps(inventory, indent=2) + '\n')
    return inventory


if __name__ == '__main__':
    main()


def named_table(function: Callable[..., str]) -> Callable[..., str]:
    """Apply the shared presentation to independently generated tables."""
    def wrapped(*args: Any, **kwargs: Any) -> str:
        """Render a table and normalize its caption, notes, and source
        wrapping.
        """
        return wrap(annotate(function(*args, **kwargs))).rstrip()
    return wrapped
