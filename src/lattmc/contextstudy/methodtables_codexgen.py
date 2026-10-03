"""Summarize lattice-operation experiments and generate manuscript tables.

Bootstrap draws resample source groups within each class. Undefined
precision is excluded explicitly, never replaced by a successful outcome.
Rendering preserves the manuscript's rose/yellow witness conventions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import numpy as np

from lattmc.contextstudy.operations_codexgen import Record, save

NAMES = dict(smol3='SmolLM2-135M / TopK SAE, block 3',
             smol15='SmolLM2-135M / TopK SAE, block 15',
             smol27='SmolLM2-135M / TopK SAE, block 27',
             pythia_topk='Pythia-70M / TopK SAE, block 3',
             qwen_transcoder='Qwen3-0.6B / ReLU TC, block 14',
             gemma_matryoshka='Gemma-2-2B / Matryoshka SAE, block 8')


def mean(values: list[float | None]) -> float | None:
    """Average defined values only, returning None for an empty sample."""
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def bootstrap(values: list[float | None]) -> Record:
    """Estimate a four-class macro mean with stratified source bootstrap.

    Input consists of five source-group values for each of four classes.
    Undefined class means remain missing; report their count explicitly.
    """
    converted = [np.nan if v is None else v for v in values]
    array = np.array(converted).reshape(4, 5)
    rng = np.random.default_rng(20261003)
    estimates = []
    with np.errstate(invalid='ignore'):
        for _ in range(2000):
            chosen = np.array([row[rng.integers(0, 5, 5)] for row in array])
            classes = [mean([None if np.isnan(v) else v for v in row])
                       for row in chosen]
            value = mean(classes)
            if value is not None:
                estimates.append(value)
    classes = [mean([None if np.isnan(v) else v for v in row])
               for row in array]
    return dict(mean=mean(classes), interval=(np.quantile(
        estimates, [.025, .975]).tolist() if estimates else None),
        defined_groups=int(np.isfinite(array).sum()),
        defined_classes=sum(v is not None for v in classes))


def summarize(records: list[Record], pair_type: str) -> Record:
    """Summarize the fixed selected grid without choosing favorable ranks."""
    precision, recall, delta, strict, empty, clean_p = [], [], [], [], [], []
    same = count = joint_nonempty = 0
    for record in records:
        rows = [r for r in record['selected'] if r['pair_type'] == pair_type
                and 'skipped' not in r]
        prec, rec, dif, clean = [], [], [], []
        for row in rows:
            result = row['results']
            p = result['join']['test']['precision']
            prec.append(p)
            rec.append(result['join']['test']['recall'])
            if p is not None:
                dif.append(p - max(result['left']['test']['precision'],
                                   result['right']['test']['precision']))
                joint_nonempty += 1
            if 'name_excluded' in result['join']:
                clean.append(result['join']['name_excluded']['precision'])
            same += int(bool(row['common_coordinates']))
            count += 1
        precision.append(mean(prec))
        recall.append(mean(rec))
        delta.append(mean(dif))
        clean_p.append(mean(clean))
        rates = [float(r['relation']['strict_both']) for r in rows]
        strict.append(mean(rates))
        empty.append(mean([float(r['results']['join']['test']['n'] == 0)
                           for r in rows]))
    return dict(precision=bootstrap(precision), recall=bootstrap(recall),
                paired_delta=bootstrap(delta), strict=bootstrap(strict),
                empty=bootstrap(empty), name_excluded=bootstrap(clean_p),
                count=count, nonempty=joint_nonempty,
                overlapping_coordinate_cases=same)


def number(value: float | None, percent: bool = True) -> str:
    """Format a defined statistic to one decimal, or show an em dash."""
    return '--' if value is None else f'{value * (100 if percent else 1):.1f}'


def write_tex(path: Path, lines: list[str]) -> None:
    """Wrap generated TeX without breaking commands or token adjacency."""
    output = []
    for line in '\n'.join(lines).splitlines():
        output.extend(textwrap.wrap(line, width=79, break_long_words=False,
                                    break_on_hyphens=False) or [''])
    if any(len(line) > 79 for line in output):
        raise ValueError(f'Overlong TeX token in {path}')
    path.write_text('\n'.join(output) + '\n')


def table(path: Path, label: str, title: str, columns: str, header: str,
          rows: list[str], note: str) -> None:
    """Write a compact table with a descriptive short and full caption."""
    if columns.startswith('ll'):
        flexible = r'>{\raggedright\arraybackslash}X'
        columns = ('l' + flexible + columns[2:] if 'sources' not in label
                   else flexible + columns[1:])
    if label == 'tab:lattice-token-summary':
        columns = r'll>{\raggedright\arraybackslash}Xrrrrrr'
    write_tex(path, [r'\begin{table}[htbp]', r'\centering\SampleTableSetup',
                    r'\begin{tabularx}{\linewidth}{' + columns + '}',
                    r'\toprule',
                    header + r' \\', r'\midrule', *rows, r'\bottomrule',
                    r'\end{tabularx}', r'\caption[' + title + ']{' + title +
                    '. ' + note + '}', r'\label{' + label + '}',
                    r'\end{table}'])


def report(out: Path, paper: Path) -> None:
    """Generate summary JSON and all numeric tables from executed records."""
    summaries = {}
    rows, controls, transports = [], [], []
    for dataset, short in [('ag_news', 'News'), ('dbpedia_14', 'DBpedia')]:
        for condition, name in NAMES.items():
            path = out / f'{dataset}_{condition}.json'
            data = json.loads(path.read_text())
            records = data['records']
            related = summarize(records, 'related')
            unrelated = summarize(records, 'unrelated')
            summaries[f'{dataset}/{condition}'] = dict(
                related=related, unrelated=unrelated,
                full_pair_meet_nonempty=sum(
                    r['full']['pair_meet']['test']['n'] > 0 for r in records),
                full_triple_meet_nonempty=sum(
                    r['full']['triple_meet']['test']['n'] > 0
                    for r in records))
            s = related
            rows.append(f'{short} & {name} & {s["nonempty"]}/220 & '
                        + number(s['precision']['mean']) + ' & '
                        + number(s['recall']['mean']) + ' & '
                        + number(s['strict']['mean']) + r' \\')
            interval = s['paired_delta']['interval']
            ci = '--' if interval is None else '[' + ', '.join(
                number(v) for v in interval) + ']'
            controls.append(f'{short} & {name} & '
                + number(s['paired_delta']['mean']) + ' & ' + ci + ' & '
                + number(unrelated['precision']['mean']) + ' & '
                + number(s['name_excluded']['mean']) + r' \\')
        data = json.loads((out / f'{dataset}_transport.json').read_text())
        for a, b in ((3, 15), (15, 27), (3, 27)):
            selected = [r for r in data['records'] if r['start'] == a
                        and r['stop'] == b and
                        r['query_kind'] == 'selected_join']
            nonempty = [r for r in selected if not r['empty_reference']]
            before = mean([r['results']['before']['test']['precision']
                           for r in nonempty])
            after = mean([r['results']['after']['test']['precision']
                          for r in nonempty])
            added = sum(r['results']['extension']['test']['n']
                        for r in nonempty)
            correct = sum(r['results']['extension']['test']['tp']
                          for r in nonempty)
            transports.append(
                short + f' & ${a}\\to {b}$ & {len(nonempty)}/20 & '
                + str(sum(r['heldout_lost'] for r in nonempty)) + ' & '
                + str(added) + ' & ' + number(
                    correct / added if added else None) + r' \\')
            summaries[f'{dataset}/transport/{a}-{b}'] = dict(
                eligible=len(nonempty), before_precision=before,
                after_precision=after, added=added, correct_added=correct,
                lost=sum(r['heldout_lost'] for r in nonempty),
                zero=sum(r['zero_transport'] for r in nonempty))
    save(out / 'summary.json', summaries)
    folder = paper / 'tables'
    table(folder / 'lattice-selected.tex', 'tab:lattice-selected',
          'Selected-component joins across sparse dictionaries', 'llrrrr',
          'Data & Backbone / surrogate / block & Nonempty & P & R & Both',
          rows, r'P and R are category precision and recall (\%). Both is '
          r'the percentage restricting both constituent extents. Each row '
          r'contains 20 source groups and 11 fixed rank pairs. Precision '
          r'averages nonempty queries within groups, then groups within '
          r'classes; recall includes empty queries. SmolLM2 reads MLP output, '
          r'Pythia and Gemma residual-post; TC reads MLP input and predicts '
          r'output. Gemma uses the full Matryoshka dictionary with JumpReLU '
          r'inference. Blocks are zero-based; no threshold calibration.')
    table(folder / 'lattice-controls.tex', 'tab:lattice-controls',
          'Constituent controls and lexical exclusion', 'llrrrr',
          r'Data & Backbone / surrogate / block & $\Delta$P & 95\% CI & '
          r'Unrelated P & Name-free P', controls,
          r'$\Delta$P subtracts the stronger constituent precision from '
          r'join precision on each nonempty joined query (percentage '
          r'points). Intervals use 2{,}000 source-group bootstrap draws '
          r'within classes, conditional on this corpus and grid. Unrelated '
          r'P uses a different-label source matched on training covariates. '
          r'Name-free P excludes source-title words of at least four '
          r'characters on DBpedia; -- means unavailable. All precision '
          r'columns omit empty extents and therefore need the counts in '
          r'\maincref{tab:lattice-selected}; they are not deployment '
          r'accuracy.')
    table(folder / 'lattice-transport.tex', 'tab:lattice-transport',
          'Training-fitted description transport across SmolLM2 blocks',
          'llrrrr', 'Data & Blocks & Fitted & Lost & Added & Added P',
          transports, r'Rank-1/rank-1 joins, TopK SAEs on MLP outputs. '
          r'Fitted counts nonempty training extents among 20 queries. Lost '
          r'and Added count held-out query--item memberships relative to '
          r'the original query; repeated items count for each query. '
          r'Added P is the pooled category precision (\%) of those added '
          r'memberships. Inclusion holds on the training reference set, '
          r'not necessarily on held-out items. Empty reference cases are '
          r'excluded here and retained in the records.')


def galleries(out: Path, paper: Path) -> None:
    """Render measured token snippets with the established shared macros."""
    from lattmc.contextstudy.witnessrender_codexgen import snippet
    from lattmc.latex.texttables_codexgen import paired, tex

    lines = []
    for kind in ('sae', 'tc'):
        for layer in (8, 11):
            data = json.loads((out / f'gpt2_{kind}{layer}.json').read_text())
            entries = []
            for row in data['gallery']:
                if row.get('empty'):
                    entries.append(r'\item No non-source member.')
                    continue
                row['valid_positions'] = list(range(128))
                row['diagnostic'] = (row['partial'] or [1])[0]
                badge = (r'\DistributedMark{D}' if row['status'] == 'D'
                         else row['status'])
                query = row['query']
                description = ', '.join(f'{j}:{a:.4g}' for j, a in zip(
                    query['coordinates'], query['values'])) or 'zero'
                entries.append(r'\item ' + badge + ' ' + row['query_name']
                    + f', row {row["row"]}: ``' + snippet(row, short=True)
                    + "''" + r'\par\emph{' + description + '}')
            name = 'ReLU SAE; residual-pre' if kind == 'sae' else (
                'ReLU transcoder; MLP input to output')
            panels = [(f'GPT-2-small / {name}; block {layer}',
                       'Combined requirements and retained mismatch',
                       entries[:2], entries[2:])]
            surrogate = 'SAE' if kind == 'sae' else 'transcoder'
            title = ('Selected New/de components in the GPT-2-small '
                     f'{surrogate} at block {layer}')
            label = f'tab:lattice-token-{kind}{layer}'
            content = paired(panels, title + '. Left and right use New '
                'rank 2 and de rank 3; meet and join use those same vectors. '
                'Each operation displays its first non-source member by row '
                'ID; nonmember is the first join rejection. Rose marks any '
                'positive requirement, bold rose a whole-query token; yellow '
                'D marks distributed satisfaction, N a zero query, '
                'R rejection. '
                'Requirements j:amplitude are rounded to four significant '
                'digits only for display. Windows show up to 64 raw tokens '
                'around the first partial witness. No semantic filtering.',
                label)
            write_tex(paper / 'tables' / f'lattice-token-{kind}{layer}.tex',
                      [line.replace(r'\caption{',
                       r'\caption[' + title + ']{') for line in content])
            lines.append(r'\input{tables/lattice-token-' + kind +
                         str(layer) + '}')
    write_tex(paper / 'tables/lattice-token-gallery.tex', lines)
    data = json.loads((out / 'document_gallery.json').read_text())
    panels = []
    for operation in ('left', 'right', 'join', 'meet', 'nonmember',
                      'join_category'):
        cells = []
        for family in ('pythia_topk', 'smol_topk'):
            row = next(r for r in data['records']
                       if r['family'] == family and
                       r['query_name'] == operation)
            if row.get('empty'):
                cells.append([r'\item Empty held-out extent.'])
                continue
            row['valid_positions'] = row['valid']
            row['diagnostic'] = (row['partial'] or row['valid'])[0]
            badge = (r'\DistributedMark{D}' if row['status'] == 'D'
                     else row['status'])
            cells.append([r'\item ' + badge + f', row {row["row"]}: ``'
                + snippet(row, short=True) + "''" + r'\par\emph{'
                + f'label {row["label"]}; ' + tex(str(row['identity'][
                    'source_split'])) + ':' + str(row['identity']['source_id'])
                + '}'])
        panels.append(('Pythia-70M / TopK SAE; block 3, residual-post',
                       'SmolLM2-135M / TopK SAE; block 15, MLP output',
                       *cells))
        titles = dict(left='First-source components',
                      right='Second-source components',
                      join='Joined source components',
                      meet='Shared source components',
                      nonmember='Rejected joined-component examples',
                      join_category='Category-matching joined components')
        title = titles[operation] + ' in Pythia and SmolLM2'
        content = paired([panels[-1]], title +
                '. First NaturalPlace source group; '
                'left and right are rank-1 components of its first two '
                'document summaries. The same components define join and '
                'meet. Each panel shows the first held-out member by row ID, '
                'or the first join rejection for nonmember. The join category '
                'panel instead selects the first category-correct joined '
                'member and is an outcome-conditioned illustration. '
                'Source IDs '
                'and '
                'complete requirements appear in '
                r'\appcref{app:lattice-methods}. Label 7 is NaturalPlace and '
                'label 0 is Company. Rose and bold rose mark partial '
                'and whole-query token witnesses; yellow D is distributed '
                'satisfaction and N denotes a zero query. BOS/padding are '
                'excluded. Snippets retain original decoded tokens.',
                f'tab:lattice-document-{operation}')
        if operation != 'join_category':
            note = ('The join category panel instead selects the first '
                    'category-correct joined member and is an '
                    'outcome-conditioned illustration. ')
            content = [line.replace(note, '') for line in content]
        content = [line.replace(r'\caption{',
                   r'\caption[' + title + ']{') for line in content]
        write_tex(paper / 'tables' / f'lattice-document-{operation}.tex',
                  content)


def extra_tables(out: Path, paper: Path) -> None:
    """Write fixed token-case counts and complete document query metadata."""
    rows = []
    for kind in ('sae', 'tc'):
        for layer in (0, 8, 11):
            data = json.loads((out / f'gpt2_{kind}{layer}.json').read_text())
            for pair in data['records']:
                if (pair['left_name'], pair['right_name']) not in (
                        ('Rio', 'Janeiro'), ('New', 'de')):
                    continue
                full = pair['full']
                selected = next(r for r in pair['selected']
                    if r['left_ranks'] == [2] and r['right_ranks'] == [3])
                relation = selected['relation']
                rows.append(f'{kind.upper()} & {layer} & '
                    + pair['left_name'] + '/' + pair['right_name'] + ' & '
                    + f'{pair["meet_positive"]} & {full["meet"]} & '
                    + f'{full["extension"]} & {full["join"]} & '
                    + f'{relation["both"]} & '
                    + number(relation['rho_left']) + '/' +
                    number(relation['rho_right']) + r' \\')
    table(paper / 'tables/lattice-token-summary.tex',
          'tab:lattice-token-summary',
          'Full and selected contextual operations in GPT-2-small',
          'lllrrrrrr',
          r'Model & Block & Pair & $d_{\wedge}$ & $|G(u\wedge v)|$ & '
          r'$|E|$ & $|G(u\vee v)|$ & Selected & $\rho_{p}/\rho_{r}$',
          rows, r'SAE is the ReLU residual-pre surrogate; TC is the ReLU '
          r'MLP transcoder. $d_{\wedge}$ counts positive coordinates in '
          r'the full meet. E excludes both constituent full extents. '
          r'Selected counts the rank-2/rank-3 join; its retention ratios '
          r'are percentages relative to the selected constituents. Counts '
          r'cover all 25{,}600 cached rows, including sources. No lexical '
          r'or semantic label is used. Zero denominators appear as --.')
    data = json.loads((out / 'document_gallery.json').read_text())
    sources = json.loads((out / 'dbpedia_14_design.json').read_text())
    rows = []
    for family in ('pythia_topk', 'smol_topk'):
        for operation in ('left', 'right'):
            row = next(r for r in data['records'] if r['family'] == family
                       and r['query_name'] == operation)
            source = row['sources'][0 if operation == 'left' else 1]
            identity = sources['rows'][source]
            j, = row['query']['coordinates']
            value, = row['query']['values']
            name = NAMES['smol15' if family == 'smol_topk' else family]
            rows.append(name + ' & ' + operation + ' & '
                + f'{source} & {identity["source_id"]} & {j} & '
                + f'{value:.6g}' + r' \\')
    table(paper / 'tables/lattice-document-sources.tex',
          'tab:lattice-document-sources',
          'Sources and requirements for the selected document components',
          'llrrrr', 'Backbone / surrogate / block & Component & Row & '
          'DBpedia ID & Coordinate & Amplitude', rows,
          r'Both source documents belong to NaturalPlace and the original '
          r'DBpedia training split. Row is the pooled-cache index; DBpedia '
          r'ID is the index within the original training split. Coordinate '
          r'indices are zero-based within each separate dictionary. The '
          r'largest positive stored amplitude is retained without scaling; '
          r'values are displayed to six significant digits. SmolLM2 reads '
          r'MLP output and Pythia residual-post. Source titles and full '
          r'texts remain recoverable from recorded IDs and the pinned data.')


def main() -> None:
    """Produce numeric tables and, when replay is complete, text galleries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--galleries', action='store_true')
    args = parser.parse_args()
    report(args.out, args.paper)
    if args.galleries:
        galleries(args.out, args.paper)
        extra_tables(args.out, args.paper)


if __name__ == '__main__':
    main()
