"""Visualization of detected tokens background for 
Lattice-theoretic Formal Concept Analysis (FCA) output."""

import re
import unicodedata
from typing import Union

REPL = "\uFFFD"                      # '' replacement character
CTRL = re.compile(r"[\x00-\x1F\x7F-\x9F]")


def green(s: str) -> str:
    """
    Wrap `s` in ANSI escape codes for green text.
    Resets styling at the end.

    Args:
        s (str): The text to wrap.
    Returns:
        str: The wrapped text.
    """
    return f"\033[32m{s}\033[0m"


def with_background(text: str, bg_code: Union[int, str]) -> str:
    """
    Wrap `text` in ANSI escape codes for background color `bg_code`.
    Resets styling at the end.

    Args:
        text (str): The text to wrap.
        bg_code (Union[int, str]): ANSI background code — int for
            standard (e.g. 41), str for 256-color
            (e.g. "48;5;224").
    Returns:
        str: The wrapped text.
    """
    return f"\033[{bg_code}m{text}\033[0m"


def with_texgraph(text: str, bg_code: str = 'hi') -> str:
    """
    Wrap `text` in ANSI escape codes for background color `bg_code`.
    Resets styling at the end.

    Args:
        text (str): The text to wrap.
        bg_code (str): The background color code. 
            Default is 'hi'.
    Returns:
        str: The wrapped text.
    """
    return f"{{{bg_code}}}{text}"


def clean_str_tokens(
    str_toks,
    tokenizer=None,                  # model.tokenizer
    quote_style="curly",             # "curly" or "ascii"
    drop_special=True,
    normalize="NFKC",
    remove_control=True,
):
    """
    Clean the string tokens.

    Args:
        str_toks (List[str]): The string tokens.
        tokenizer (Any): The tokenizer. Default is None.
        quote_style (str): The quote style. Default is "curly".
        drop_special (bool): Whether to drop special tokens.
            Default is True.
        normalize (str): The normalization method. 
            Default is "NFKC".
        remove_control (bool): Whether to remove control characters.
            Default is True.
    Returns:
        List[str]: The cleaned string tokens.
    """
    if quote_style not in ("curly", "ascii"):
        raise ValueError("quote_style must be 'curly' or 'ascii'")

    def q_open(): return "“" if quote_style == "curly" else '"'
    def q_close(): return "”" if quote_style == "curly" else '"'
    def apos(): return "’" if quote_style == "curly" else "'"

    specials = set()
    if tokenizer is not None:
        specials |= set(getattr(tokenizer, "all_special_tokens", []) or [])
    # common GPT-2 / TransformerLens default BOS/EOS/PAD string
    specials |= {"<|endoftext|>"}

    out = []
    i = 0
    n = len(str_toks)

    def last_char_of_output():
        for t in reversed(out):
            if t:
                return t[-1]
        return ""

    while i < n:
        t = str_toks[i]

        # drop special tokens
        if drop_special and t in specials:
            i += 1
            continue

        # normalize + remove control chars
        if normalize:
            t = unicodedata.normalize(normalize, t)
        if remove_control:
            t = CTRL.sub("", t)

        # handle runs of pure '' tokens/strings
        if t and all(ch == REPL for ch in t):
            j = i
            while j < n and str_toks[j] and all(
                ch == REPL for ch in str_toks[j]
            ):
                j += 1

            prev_c = last_char_of_output()
            nxt = str_toks[j] if j < n else ""

            # choose based on context
            # ignore leading spaces (GPT-2 tokens often include them)
            nxt_first = nxt.lstrip()[:1]
            if nxt_first == "s" and prev_c.isalnum():
                # de Blasio   s  -> de Blasio’s
                out.append(apos())
            else:
                # opening quote if at start or after whitespace/opening
                # bracket; else closing
                if (not prev_c) or prev_c.isspace() or prev_c in "([{\n\t":
                    out.append(q_open())
                else:
                    out.append(q_close())

            i = j
            continue

        # remove any inline replacement chars inside token
        if REPL in t:
            t = t.replace(REPL, "")

        if t:
            out.append(t)

        i += 1

    return out
