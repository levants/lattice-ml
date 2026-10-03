"""Shared paired text tables following the original notebook highlights."""

from __future__ import annotations
from collections.abc import Sequence

import html
from pathlib import Path
import re
import textwrap
import unicodedata

from lattmc.tc.tokenization_utils import PASTEL_BG_RGBS
from lattmc.fca.visualization_utils import with_background

RGB = PASTEL_BG_RGBS[0]
HEX = ''.join(f'{value:02X}' for value in RGB)


def terminal_highlight(text: str) -> str:
    """Add the shared background color to terminal text."""
    return with_background(text, '48;2;' + ';'.join(map(str, RGB)))


def html_highlight(text: str) -> str:
    """Escape and highlight text with an HTML mark element."""
    return f'<mark style="background:#{HEX}">{html.escape(text)}</mark>'


def tex(text: str) -> str:
    """Normalize whitespace and escape LaTeX special characters."""
    for old, new in {'\u2019': "'", '\u2018': "'", '\u201c': '"',
                     '\u201d': '"', '\u2013': '-', '\u2014': '--',
                     '\u2026': '...'}.items():
        text = text.replace(old, new)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = ''.join(c if ord(c) < 128 else f'[U+{ord(c):04X}] ' for c in text)
    escapes = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
               '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
               '}': r'\}', '~': r'\textasciitilde{}',
               '^': r'\textasciicircum{}'}
    return ''.join(escapes.get(c, c) for c in re.sub(r'\s+', ' ', text))


def highlighted(before: str, token: str, after: str) -> str:
    """Build an escaped excerpt with a highlighted token and ellipses."""
    word = tex(token)
    leading = ' ' if word.startswith(' ') else ''
    trailing = ' ' if word.endswith(' ') else ''
    word = word.strip() or r'\textvisiblespace{}'
    return (r'\ldots{}' + tex(before) + leading + r'\TokenHighlight{' +
            word + '}' + trailing + tex(after) + r'\ldots{}')


def item(snippet: str, metadata: str) -> str:
    """Format an excerpt and its metadata as a LaTeX list item."""
    return r'\item ``' + snippet + "''" + r'\par\emph{' + metadata + '}'


def cell(items: list[str]) -> list[str]:
    """Wrap excerpt items in a compact LaTeX itemize environment."""
    return ([r'\begin{itemize}[nosep,leftmargin=*]'] + items +
            [r'\end{itemize}'])


def paired(
    panels: Sequence[tuple[str, str, list[str], list[str]]],
    caption: str,
    label: str,
) -> list[str]:
    """Assemble paired excerpt panels into a captioned LaTeX table."""
    lines = [r'\begin{table}[htbp]', r'\centering\TokenTableSetup',
             r'\begin{tabularx}{\linewidth}{',
             r'  >{\raggedright\arraybackslash}X',
             r'  >{\raggedright\arraybackslash}X}', r'\toprule']
    for n, (left, right, left_items, right_items) in enumerate(panels):
        if n:
            lines += [r'\midrule']
        lines += [r'\textbf{' + left + '} & ' + r'\textbf{' + right + '}'
                  + r' \\', r'\midrule']
        lines += cell(left_items) + ['&'] + cell(right_items) + [r'\\']
    return lines + [r'\bottomrule', r'\end{tabularx}',
                    r'\caption{' + caption + '}',
                    r'\label{' + label + '}', r'\end{table}']


def write(path: Path, lines: Sequence[str]) -> None:
    """Write LaTeX source lines with consistent wrapping."""
    from .naming_codexgen import annotate
    text = '\n'.join(lines)
    if Path(path).parent.name == 'tables':
        text = annotate(text)
    lines = text.splitlines()
    output = [part for line in lines for part in textwrap.wrap(
        line, 79, break_long_words=False, break_on_hyphens=False) or ['']]
    assert all(len(line) <= 79 for line in output), path
    Path(path).write_text('\n'.join(output) + '\n')


def styles(paper: Path) -> None:
    """Write the shared token-highlight and table styles for a paper."""
    text = r'''% Finite-shrink backport for TeX Live 2025 longtable.
% https://github.com/latex3/latex2e/issues/1907
\makeatletter
\patchcmd{\LT@output}{\copy\LT@foot\vss}
{\copy\LT@foot\vskip 0pt plus \maxdimen minus \normalbaselineskip}{}{}
\patchcmd{\LT@output}{\copy\LT@foot\vss}
{\copy\LT@foot\vskip 0pt plus \maxdimen minus \normalbaselineskip}{}{}
\setlength{\@fptop}{0pt}
\setlength{\@fpsep}{14pt}
\setlength{\@fpbot}{0pt plus 1fil}
\makeatother
% Shared style for historical and replayed token examples.
\definecolor{TokenRose}{HTML}{COLOR}
\definecolor{DistributedYellow}{HTML}{FFEF9F}
\newcommand{\DistributedMark}[1]{%
  \colorbox{DistributedYellow}{\textbf{#1}}}
\newcommand{\TokenHighlight}[1]{\colorbox{TokenRose}{#1}}
\newcommand{\FeatureHighlight}[2]{\colorbox[HTML]{#1}{#2}}
\newcommand{\TableNote}[1]{\space #1}
\newcommand{\SampleTableSetup}{%
  \footnotesize
  \setlength{\tabcolsep}{4pt}%
  \renewcommand{\arraystretch}{1.06}%
}
\newcommand{\TokenTableSetup}{%
  \scriptsize
  \sloppy
  \setlength{\tabcolsep}{3pt}%
  \setlength{\fboxsep}{0.35pt}%
  \renewcommand{\arraystretch}{0.9}%
}
\newcommand{\ModelComparisonHeader}{%
  \toprule
  \textbf{GPT-2-small / ReLU transcoder}\newline MLP input to output &
  \textbf{GPT-2-small / ReLU SAE}\newline residual-pre \\
  \midrule
}
'''.replace('COLOR', HEX)
    write(Path(paper) / 'config/table_styles.tex', text.splitlines())
