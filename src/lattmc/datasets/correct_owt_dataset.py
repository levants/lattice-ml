"""Correct textual representation of OpenWebText token dataset.

Fixes token conversion and detokenization issues:
1. Fixes CSV header malformation ('i,d,c,s' / 't,e,x,t' -> 'id,text').
2. Restores split multi-byte UTF-8 boundary tokens (where byte-level
   BPE slicing across 128-token windows produced '\\ufffd').
3. Resolves authentic original values from the local OpenWebText archive.
"""

from pathlib import Path
import csv
import io
import re
import time
import zipfile

DATA_DIR = Path("/Users/ltsinadze/git/my_papers")
DEFAULT_CSV = DATA_DIR / "owt_tokens" / "dataset.csv"
DEFAULT_ZIP = (
    DATA_DIR
    / "notebooks"
    / "data"
    / "owt_tokens"
    / "dataset_shuffled.zip"
)
DEFAULT_OUT = DATA_DIR / "owt_tokens" / "dataset_corrected.csv"
DELIM_PAT = re.compile(r'[\s\?\.\!\,\;\:\"\”\“\(\)\[\]]+')


def _find_char_before(clean_text: str, source_text: str) -> str | None:
    """Find the original character preceding clean_text in source_text."""
    for length in (35, 25, 20, 15, 10):
        anchor = clean_text[:length]
        pos = source_text.find(anchor)
        if pos > 0:
            return source_text[pos - 1]
    tokens = [w for w in DELIM_PAT.split(clean_text) if w]
    if tokens:
        for w in (tokens[0],):
            if len(w) >= 4:
                pos = source_text.find(w)
                if pos > 0:
                    return source_text[pos - 1]
    return None


def _find_char_after(clean_text: str, source_text: str) -> str | None:
    """Find the original character following clean_text in source_text."""
    for length in (35, 25, 20, 15, 10):
        anchor = clean_text[-length:]
        pos = source_text.find(anchor)
        if pos != -1 and pos + len(anchor) < len(source_text):
            return source_text[pos + len(anchor)]
    tokens = [w for w in DELIM_PAT.split(clean_text) if w]
    if tokens:
        last_tok = tokens[-1]
        for w in (last_tok,):
            if len(w) >= 4:
                pos = source_text.find(w)
                if pos != -1 and pos + len(w) < len(source_text):
                    return source_text[pos + len(w)]
    return None


def correct_owt_dataset(
    dataset_csv: Path | str = DEFAULT_CSV,
    zip_archive: Path | str = DEFAULT_ZIP,
    output_csv: Path | str = DEFAULT_OUT,
    preload_bytes: int = 150 * 1024 * 1024,
):
    """Correct the OWT dataset CSV."""
    t0 = time.time()
    dataset_csv = Path(dataset_csv)
    zip_archive = Path(zip_archive)
    output_csv = Path(output_csv)

    print(f"Loading reference text from {zip_archive.name}...")
    with zipfile.ZipFile(zip_archive, "r") as z:
        with z.open("skylion007_openwebtext_204800_shuffled.csv") as f:
            raw = f.read(preload_bytes)
            text_io = io.StringIO(raw.decode("utf-8", errors="replace"))
            reader = csv.reader(text_io)
            _ = next(reader)
            docs = [r[1] for r in reader if len(r) > 1]

    all_text = "\n<|endoftext|>\n".join(docs)
    elapsed_load = time.time() - t0
    print(
        f"Loaded {len(docs)} docs ({len(all_text):,} chars) in "
        f"{elapsed_load:.2f}s"
    )

    print(f"Reading corrupted dataset from {dataset_csv}...")
    df_rows = []
    with open(dataset_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Skip the corrupted 2-line header: 'i,d,c,s' and 't,e,x,t'
        next(reader)
        next(reader)
        for r in reader:
            df_rows.append(r)

    print(f"Read {len(df_rows)} rows.")

    fixed_starts = 0
    fixed_ends = 0
    corrected_rows = []

    for idx, r in enumerate(df_rows):
        row_id = idx
        txt = r[1]

        # Fix broken continuation byte at start of sequence
        if txt.startswith("\ufffd"):
            clean = txt.lstrip("\ufffd")
            char_before = _find_char_before(clean, all_text)
            if char_before:
                txt = char_before + clean
                fixed_starts += 1

        # Fix broken multi-byte prefix at end of sequence
        if txt.endswith("\ufffd"):
            clean = txt.rstrip("\ufffd")
            char_after = _find_char_after(clean, all_text)
            if char_after:
                txt = clean + char_after
                fixed_ends += 1

        corrected_rows.append([row_id, txt])

    print(f"Corrected starts: {fixed_starts}, Corrected ends: {fixed_ends}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text"])
        writer.writerows(corrected_rows)

    total_time = time.time() - t0
    print(
        f"Wrote {len(corrected_rows)} rows to {output_csv} in "
        f"{total_time:.2f}s"
    )


if __name__ == "__main__":
    correct_owt_dataset()
