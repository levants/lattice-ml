"""Render the fixed DBpedia context illustration and checkpoint identities."""

from __future__ import annotations

import json
from pathlib import Path

from lattmc.activationstudy.common_codexgen import sha256, save_json
from lattmc.activationstudy.families_config_codexgen import FAMILIES
from lattmc.latex.naming_codexgen import panel, PAPER, ROOT
from lattmc.latex.texttables_codexgen import paired, tex, write
from .commonexample_codexgen import DEST
from .witnessrender_codexgen import entry, LEGEND


def examples(paper: Path) -> None:
    """Write the illustrative context comparison tables."""
    data = json.loads((DEST / 'records.json').read_text())
    lines = [r'\begin{table}[htbp]', r'\centering\SampleTableSetup',
             r'\begin{tabularx}{\linewidth}{llX}', r'\toprule',
             r'Cached row & DBpedia train ID & Source excerpt \\',
             r'\midrule']
    for i, identity, text in zip(data['sources'], data['identities'],
                                 data['texts']):
        excerpt = text if len(text) <= 210 else text[:210].rsplit(' ', 1)[0]
        if excerpt != text:
            excerpt += ' ...'
        lines += [f"{i} & {identity['source_id']} & {tex(excerpt)}" + r' \\']
    lines += [r'\bottomrule\end{tabularx}',
              r'\caption{The first five NaturalPlace training items in '
              r'the fixed DBpedia cache, without activation-based selection. '
              r'Cached row is the zero-based row in the 2,100-item design; '
              r'train ID is the zero-based original dataset row. Text is '
              r'shortened for identification, not used as a lexical query. '
              r'Each backbone encodes its own first 128 token positions; '
              r'padding and position 0 are excluded from the tokenwise join. '
              r'The separate full document meets define the canonical '
              r'concepts in \cref{tab:common-features-dbpedia}; no '
              r'cross-model infomorphism is claimed.}',
              r'\label{tab:example-dbpedia-contexts}', r'\end{table}']
    write(paper / 'tables/example-dbpedia-contexts.tex', lines)
    cells = []
    for family in ('pythia_topk', 'smol_topk'):
        rows = data['families'][family]['rows'][:3]
        cells.append([entry(row) for row in rows])
    caption = (
        r'Exact intents $u=F(A)=\bigwedge_{T\in A}f(T)$ for the same '
        r'five documents in \cref{tab:example-dbpedia-contexts}. Each '
        r'$f(T)$ is the tokenwise join in its own dictionary. Pythia has '
        r'119 positive coordinates and SmolLM2 has 139; no coordinate '
        r'selection, support-floor substitution, or scaling is applied. '
        r'Retrieval tests $f(T)\geq u$ over all 2,100 cached items; both '
        r'extents contain only the five sources. The first three are shown '
        r'in source order, with every valid token position. All five are '
        r'distributed in both contexts: the whole query exceeds the '
        r'per-token TopK budgets 20 and 32. Rose marks every token reaching '
        r'any positive coordinate threshold, and bold would mark a token '
        r'reaching all of them. Yellow D denotes collective satisfaction, '
        r'not semantic correctness. Coordinates and amplitudes are not '
        r'aligned across models. Here $d$ is the positive-coordinate count, '
        r'$\alpha=1$, and $s=\min_{j:u_{j}>0}f(T)_{j}/u_{j}$; '
        r'membership is $s\geq1$. Scores are rounded to three decimals '
        r'only for display; comparisons use unrounded codes without tolerance.'
    )
    write(paper / 'tables/common-features-dbpedia.tex', paired([
        (panel('pythia_topk'), panel('smol_topk'), *cells)], caption,
        'tab:common-features-dbpedia'))


def checkpoints(paper: Path) -> None:
    """Write checkpoint identities and activation-site documentation."""
    records = []
    for family, cfg in FAMILIES.items():
        path = ROOT / 'data/activation_studies/families_v2/dbpedia_14'
        path /= family + '_extraction.json'
        rec = json.loads(path.read_text())
        assert rec['configuration'] == cfg
        metadata = rec['sae_config']['metadata']
        hook = metadata['hook_name']
        assert f".{cfg['layer']}." in hook
        if cfg['hook'] == 'residual_post':
            assert hook.endswith('hook_resid_post')
        elif cfg['hook'] == 'mlp_output':
            assert hook.endswith('hook_mlp_out')
        else:
            assert hook.endswith('mlp.hook_in')
            assert metadata['hook_name_out'].endswith('hook_mlp_out')
        records.append(dict(
            key=family, configuration=cfg, sae_config=rec['sae_config'],
            evidence=str(path.relative_to(ROOT)), evidence_sha256=sha256(path),
            weights=rec['sae_files'], backbone=rec['backbone_files'],
        ))
    rows = []
    for rec in records:
        key, cfg, sae = rec['key'], rec['configuration'], rec['sae_config']
        mechanism = sae['architecture']
        if key == 'qwen_transcoder':
            mechanism = 'ReLU transcoder'
        elif key == 'gemma_matryoshka':
            assert mechanism == 'jumprelu'
            mechanism = 'Matryoshka JumpReLU SAE'
        else:
            assert mechanism == 'topk'
            mechanism = f"TopK SAE ($k={sae['k']}$)"
        from lattmc.latex.naming_codexgen import SITES
        rows.append((r'\path{' + cfg['model'] + r'}\newline ' + mechanism,
                     f"{cfg['layer']}; {SITES[key]}" + r'\newline '
                     + f"{sae['d_sae']:,} coordinates",
                     cfg['release'], cfg['sae_id']))
    external = ROOT / 'data/activation_studies/external_v1/dbpedia_14'
    for name, site, expected in [
            ('gpt2_res8', 'residual-pre', 'hook_resid_pre'),
            ('gpt2_mlp8', 'MLP output', 'hook_mlp_out'),
            ('gemma2_l0_37', 'residual-post', 'hook_resid_post'),
            ('gemma2_l0_301', 'residual-post', 'hook_resid_post')]:
        path = external / (name + '_extraction.json')
        rec = json.loads(path.read_text())
        assert rec['hook'] == 'blocks.8.' + expected
        label = ('GPT-2-small / ReLU SAE' if name.startswith('gpt2') else
                 'Gemma-2-2B / JumpReLU SAE')
        width = '24,576' if name.startswith('gpt2') else '16,384'
        rows.append((label, f'8; {site}' + r'\newline '
                     + width + ' coordinates',
                     rec['release'], rec['sae_id']))
        records.append(dict(key=name, evidence=str(path.relative_to(ROOT)),
                            evidence_sha256=sha256(path), extraction=rec))
    rows.extend([
        ('GPT-2-small / ReLU transcoder',
         r'0, 8, 11; MLP input to output\newline 24,576 coordinates',
         'pchlenski/gpt2-transcoders',
         'final_sparse_autoencoder_gpt2-small_blocks.{b}.'
         'ln2.hook_normalized_24576.pt'),
        ('GPT-2-small / ReLU SAE (historical)',
         r'0, 11; residual-pre\newline 24,576 coordinates',
         'gpt2-small-res-jb', 'blocks.{b}.hook_resid_pre'),
    ])
    lines = [r'\begin{table}[htbp]', r'\centering\scriptsize',
             r'\begin{tabularx}{\linewidth}{',
             r'>{\raggedright\arraybackslash}p{0.26\linewidth}',
             r'>{\raggedright\arraybackslash}p{0.23\linewidth}',
             r'>{\raggedright\arraybackslash}X}', r'\toprule',
             r'Backbone / surrogate & Block; site; width & '
             r'Release / identifier',
             r'\\\midrule']
    for name, detail, release, identifier in rows:
        # Break only at genuine path components, not inside an identifier.
        if '{b}' in identifier:
            first, last = identifier.split('{b}')
            shown = (r'\path{' + first + '}%' + '\n' + r'\texttt{b}%' +
                     '\n' + r'\path{' + last + '}')
        else:
            shown = r'\path{' + identifier + '}'
        lines += [name + ' & ' + detail + ' &', r'\path{' + release + '}',
                  r'\newline ' + shown + r' \\']
    lines += [r'\bottomrule\end{tabularx}',
              r'\caption{Checkpoint identities used by the displayed '
              r'experiments. Blocks are zero-based; residual-pre is before '
              r'the block and residual-post is after it. MLP output is the '
              r'feed-forward contribution before residual addition. '
              r'Transcoders encode the normalized MLP input and predict its '
              r'output; MLP-output SAEs reconstruct that output. Width is '
              r'the dictionary dimension; TopK $k$ is the per-token budget. '
              r'GPT-2-small denotes the 124M backbone. Pythia is the deduped '
              r'70M checkpoint. Matryoshka prefixes of 512 and 2,048 '
              r'coordinates use the same Gemma checkpoint. SmolLM2 depth '
              r'tables additionally use the same release at blocks 3 and 27. '
              r'In historical identifiers, b is the displayed block. '
              r'Exact revisions and file hashes are retained in the '
              r'extraction records and the companion identity manifest.}',
              r'\label{tab:checkpoint-identities}', r'\end{table}']
    write(paper / 'tables/checkpoint-identities.tex', lines)
    save_json(DEST / 'identities.json', records)


def main(paper: Path = PAPER) -> None:
    """Generate context examples and checkpoint tables for the paper."""
    examples(paper)
    checkpoints(paper)


if __name__ == '__main__':
    main()
