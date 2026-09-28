"""Utilities for LaTeX tables and figures."""
import logging
import re
from typing import Dict

from tqdm import tqdm

from src.lattmc.tc.tokenization_utils import (  # noqa: E402
    BG_CODES as _BG_CODES,
    PASTEL_BG_RGBS as _PASTEL_BG_RGBS,
)

# LaTeX \colorbox colors corresponding to tokenization_utils.BG_CODES indices.
# Values are xcolor HTML specs matching the truecolor ANSI pastel palette.
LATEX_BG_CODES = {
    idx: f"{red:02X}{green:02X}{blue:02X}"
    for idx, (red, green, blue) in enumerate(_PASTEL_BG_RGBS)
}

# ANSI reset and background escape pattern: \033[<code>m...\033[0m
_ANSI_BG_RE = re.compile(r"\033\[(\d+(?:;\d+)*)m(.*?)\033\[0m", re.DOTALL)
_HTML_COLOR_RE = re.compile(r"[0-9A-Fa-f]{6}")

# Build a reverse map from ANSI code string -> LATEX_BG_CODES index.
_ANSI_TO_INDEX: dict[str, int] = {}
for _idx, _code in _BG_CODES.items():
    _ANSI_TO_INDEX[str(_code)] = _idx


def replace_bg_codes_with_latex(text: str) -> str:
    """Replace ANSI background-colour escapes in *text* with LaTeX 
        \\colorbox commands.

    Each ``\\033[<code>m...\\033[0m`` span produced by
    ``tokenization_utils.with_background`` is converted to
    ``\\colorbox[HTML]{<color>}{...}`` for the default palette.  Spans
    whose ANSI code is not in :data:`LATEX_BG_CODES` are left unchanged.

    Args:
        text: String that may contain ANSI background-colour escapes.

    Returns:
        String with ANSI escapes replaced by LaTeX ``\\colorbox`` commands.
    """
    def _replace(m: re.Match) -> str:
        code_str = m.group(1)
        content = m.group(2)
        idx = _ANSI_TO_INDEX.get(code_str)
        if idx is None:
            return m.group(0)
        latex_color = LATEX_BG_CODES[idx]
        if _HTML_COLOR_RE.fullmatch(latex_color):
            return rf"\colorbox[HTML]{{{latex_color}}}{{{content}}}"
        return rf"\colorbox{{{latex_color}}}{{{content}}}"

    return _ANSI_BG_RE.sub(_replace, text)


def truncate_latex_text(
    text: str,
    max_len: int = 70,
    cont_indent: str = "  "
) -> str:
    """Wrap *text* so that each source line is at most *max_len* 
        characters.

    No content is removed.  Whitespace-delimited tokens (including whole
    ``\\colorbox{color}{content}`` commands) are placed on the current line
    when they fit; otherwise a new line starting with *cont_indent* is opened.
    LaTeX ignores extra whitespace in the output document, so wrapping only
    affects source readability.

    Args:
        text: LaTeX string as produced by
            :func:`replace_bg_codes_with_latex`.
        max_len: Maximum source-line length (default 70).
        cont_indent: Prefix added to every continuation line 
            (default two spaces).

    Returns:
        Multi-line LaTeX string with lines no longer than *max_len* 
            characters.
    """
    tokens = text.split()
    if not tokens:
        return text

    lines: list[str] = []
    current = tokens[0]

    for token in tokens[1:]:
        if len(current) + 1 + len(token) <= max_len:
            current += " " + token
        else:
            lines.append(current)
            current = cont_indent + token

    lines.append(current)
    return "\n".join(lines)


def print_latex_text(multitexts: Dict[int, str]):
    """Print the texts in LaTeX format.

    Args:
        multitexts (Dict[int, str]): A dictionary of texts.
    """
    pref_item = '\\item{ID '
    suff_item = ':} "\\ldots'
    text_suff = ' \\ldots"'
    with tqdm(multitexts.items(), desc='Printing texts') as ptext_items:
        for text_id, tex_text in ptext_items:
            tex_text = replace_bg_codes_with_latex(tex_text)
            cln_text = tex_text.replace(
                '<|endoftext|>,',
                ''
            ).replace(
                '<|endoftext|>',
                ''
            )
            text_prefix = f'{pref_item}{text_id}{suff_item}'
            cln_text = f'{text_prefix}{cln_text}{text_suff}'
            output_text = truncate_latex_text(cln_text)
            print(output_text)
