"""GPT2 Detokenizer for converting token IDs to human-readable text."""

from pathlib import Path
from typing import Union

import einops
import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer


class GPT2Detokenizer:
    """Detokenizer for GPT2 models that converts token IDs back to text.

    Handles GPT2's byte-level BPE tokenization and produces properly formatted
    text with correct spacing and punctuation.
    """

    def __init__(self, model_name: str = "gpt2"):
        """Initialize the detokenizer with a GPT2 tokenizer.

        Args:
            model_name: Name of the GPT2 model to load tokenizer from.
                       Options: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.
                       Default is 'gpt2'.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    @property
    def bos_token_id(self) -> int:
        """Get the BOS token ID (same as EOS for GPT2: 50256)."""
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        """Get the EOS token ID (50256)."""
        return self.tokenizer.eos_token_id

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        add_bos_token: bool = False,
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: str | None = None,
    ) -> list[int] | torch.Tensor:
        """Convert text to token IDs.

        Args:
            text (str): Input text string to tokenize.
            add_special_tokens (bool): Whether to add special tokens 
                (e.g., BOS/EOS). Default is True.
            add_bos_token (bool): Whether to prepend BOS token 
                (for compatibility with data_utils.tokenize_and_concatenate 
                which adds BOS). Default False.
            max_length (int | None): Target sequence length. Used with 
                padding/truncation. Default is None.
            padding (bool): If True, pad to max_length with pad_token_id.
                Default is False.
            truncation (bool): If True, truncate to max_length. 
                Default is False.
            return_tensors (str | None): If 'pt', returns a PyTorch tensor. 
                None returns a list. Default is None.

        Returns:
            Token IDs as a list of integers or torch tensor.
        """
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )

        if add_bos_token:
            tokens = [self.bos_token_id] + tokens

        # Truncate if needed
        if truncation and max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]

        # Pad if needed
        if padding and max_length is not None and len(tokens) < max_length:
            pad_length = max_length - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * pad_length

        if return_tensors == "pt":
            tokens = torch.tensor([tokens])

        return tokens

    def encode_concatenated(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
    ) -> list[int] | torch.Tensor:
        """Concatenate multiple texts with EOS tokens and encode.

        This matches the tokenization strategy used in 
        data_utils.tokenize_and_concatenate,
        where texts are joined with EOS tokens before encoding.

        Args:
            texts (list[str]): List of text strings to concatenate and tokenize.
            add_special_tokens (bool): Whether to add special tokens.
                Default is True.
            return_tensors (str | None): If 'pt', returns a PyTorch tensor. 
                None returns a list. Default is None.

        Returns:
            Token IDs as a list of integers or torch tensor.
        """
        full_text = self.tokenizer.eos_token.join(texts)
        return self.tokenizer.encode(
            full_text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> dict | list[list[int]]:
        """Convert a batch of text strings to token IDs.

        Args:
            texts (list[str]): List of input text strings to tokenize.
            add_special_tokens (bool): Whether to add special tokens.
                Default is True.
            padding (bool | str): Padding strategy. True/'longest' pads to 
                longest in batch, 'max_length' pads to max_length, 
                False for no padding. Default is False.
            truncation (bool): Whether to truncate sequences exceeding
                max_length. Default is False.
            max_length (int | None): Maximum sequence length (required if 
                truncation=True). Default is None.
            return_tensors (str | None): If 'pt', returns PyTorch tensors. 
                None returns lists. Default is None.

        Returns:
            If return_tensors is set, returns a dict with 'input_ids' and
            'attention_mask'. Otherwise returns a list of token ID lists.
        """
        if return_tensors or padding:
            return self.tokenizer(
                texts,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
            )

        return [
            self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            for text in texts
        ]

    def decode(
        self,
        token_ids: Union[list[int], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Convert token IDs to text.

        Args:
            token_ids (Union[list[int], torch.Tensor]): Token IDs as a list of 
                integers or torch tensor.
            skip_special_tokens (bool): Whether to remove special tokens from 
                output. Default is True.
            clean_up_tokenization_spaces (bool): Whether to clean up extra 
                spaces. Default is True.

        Returns:
            Decoded text string with proper formatting.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if isinstance(token_ids[0], list):
            return [
                self.tokenizer.decode(
                    ids,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
                for ids in token_ids
            ]

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def decode_batch(
        self,
        batch_token_ids: Union[list[list[int]], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> list[str]:
        """Convert a batch of token ID sequences to text.

        Args:
            batch_token_ids (Union[list[list[int]], torch.Tensor]): Batch of 
                token ID sequences.
            skip_special_tokens (bool): Whether to remove special tokens from 
                output. Default is True.
            clean_up_tokenization_spaces (bool): Whether to clean up extra 
                spaces. Default is True.

        Returns:
            List of decoded text strings.
        """
        if isinstance(batch_token_ids, torch.Tensor):
            batch_token_ids = batch_token_ids.tolist()

        return self.tokenizer.batch_decode(
            batch_token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def token_to_string(self, token_id: int) -> str:
        """Convert a single token ID to its string representation.

        Args:
            token_id (int): A single token ID.

        Returns:
            String representation of the token (may include special 
            chars like 'Ġ').
        """
        return self.tokenizer.convert_ids_to_tokens(token_id)

    def tokens_to_strings(
        self,
        token_ids: Union[list[int], torch.Tensor],
    ) -> list[str]:
        """Convert token IDs to their individual string representations.

        Useful for inspecting individual tokens without merging.

        Args:
            token_ids (Union[list[int], torch.Tensor]): Token IDs as a list or
                tensor.

        Returns:
            List of string representations for each token.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return len(self.tokenizer)

    def tokenize_openwebtext(
        self,
        num_sequences: int = 25600,
        max_length: int = 128,
        seed: int = 42,
        buffer_size: int = 10_000,
        add_bos_token: bool = True,
        canonicalize: bool = True,
        save_dir: Path | str | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        """Tokenize OpenWebText and return both tokens and source texts.

        This matches the tokenization in data_utils.tokenize_and_concatenate
        but also preserves the original source texts for each token sequence.

        Args:
            num_sequences: Number of token sequences to generate. 
                Default is 25600.
            max_length: Length of each token sequence. Default is 128.
            seed: Random seed for shuffling. Default is 42.
            buffer_size: Buffer size for streaming shuffle. Default is 10_000.
            add_bos_token: Whether to prepend BOS token to each sequence.
                Default is True.
            canonicalize: If True, ensures 100% round-trip by re-encoding
                decoded text. This fixes BPE tokenization ambiguities but may
                slightly alter tokens. Default is True.
            save_dir: If provided, save tokens.pt and texts.txt to this
                directory. Default is None.

        Returns:
            Tuple of (tokens tensor, list of source texts).
            The source texts are the concatenated texts that produced each
            row.
        """
        dataset = load_dataset("Skylion007/openwebtext",
                               split="train", streaming=True)
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

        seq_len = max_length - 1 if add_bos_token else max_length
        tokens_needed = num_sequences * seq_len

        all_tokens = []

        for doc in dataset:
            text = doc["text"]
            doc_tokens = self.tokenizer.encode(text)

            all_tokens.extend(doc_tokens)
            all_tokens.append(self.eos_token_id)  # EOS separator

            if len(all_tokens) >= tokens_needed:
                break

        # Truncate to exact size needed
        all_tokens = all_tokens[:tokens_needed]

        # Reshape into sequences
        tokens_array = np.array(all_tokens).reshape(-1, seq_len)

        if add_bos_token:
            bos_column = np.full((tokens_array.shape[0], 1), self.bos_token_id)
            tokens_array = np.concatenate([bos_column, tokens_array], axis=1)

        tokens_tensor = torch.from_numpy(tokens_array).long()

        # Generate source text for each sequence by decoding
        source_texts = []
        canonical_tokens = []

        for i in range(len(tokens_tensor)):
            text = self.decode(
                tokens_tensor[i].tolist(), skip_special_tokens=False)
            source_texts.append(text)

            if canonicalize:
                # Re-encode to get canonical tokenization that guarantees
                # round-trip
                reencoded = self.encode(
                    text, max_length=max_length, padding=True, truncation=True)
                canonical_tokens.append(reencoded)

        if canonicalize:
            tokens_tensor = torch.tensor(canonical_tokens).long()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(tokens_tensor, save_dir / "tokens.pt")
            with open(save_dir / "texts.txt", "w", encoding="utf-8") as f:
                for text in source_texts:
                    # Replace newlines with special marker for line-by-line
                    # storage
                    f.write(text.replace("\n", "\\n") + "\n")

        return tokens_tensor, source_texts

    def load_tokens_with_texts(
        self,
        tokens_path: Path | str,
        texts_path: Path | str | None = None,
    ) -> tuple[torch.Tensor, list[str] | None]:
        """Load tokens and optionally their source texts.

        Args:
            tokens_path: Path to tokens .pt file.
            texts_path: Path to texts .txt file. If None, 
                tries tokens_path parent/texts.txt.

        Returns:
            Tuple of (tokens tensor, list of source texts or None if not
            found).
        """
        tokens_path = Path(tokens_path)
        tokens = torch.load(tokens_path, weights_only=True)

        if texts_path is None:
            texts_path = tokens_path.parent / "texts.txt"
        else:
            texts_path = Path(texts_path)

        texts = None
        if texts_path.exists():
            with open(texts_path, "r", encoding="utf-8") as f:
                texts = [line.rstrip("\n").replace("\\n", "\n") for line in f]

        return tokens, texts

    def find_source_documents(
        self,
        snippets: list[str],
        max_docs: int | None = None,
        pattern_len: int = 20,
        save_path: Path | str | None = None,
        checkpoint_every: int = 100000,
        resume_from: Path | str | None = None,
        verbose: bool = True,
    ) -> dict[int, str]:
        """Find original OpenWebText documents containing the given text
        snippets.

        Searches through OpenWebText for documents that contain the decoded
        text snippets. Uses short unique patterns from each snippet for
        matching.

        Note: This searches the raw (unshuffled) dataset since streaming
        shuffle order is not reproducible. Snippets may be found anywhere in
        the 8M+ docs.

        Args:
            snippets: List of decoded text snippets to search for.
            max_docs: Maximum documents to search. Default is None = search
                entire dataset.
            pattern_len: Length of pattern to extract from each snippet.
                Shorter = faster but more false positives. Default is 20.
            save_path: If provided, save results as joblib file. 
                Default is None.
            checkpoint_every: Save checkpoint every N documents searched.
                Default is 100000.
            resume_from: Path to checkpoint file to resume from. 
                Default is None.
            verbose: Print progress updates. Default is True.

        Returns:
            Dict mapping snippet index to the full source document text.
        """
        import joblib

        # Resume from checkpoint if provided
        found: dict[int, str] = {}
        start_doc = 0
        if resume_from is not None:
            resume_path = Path(resume_from)
            if resume_path.exists():
                checkpoint = joblib.load(resume_path)
                found = checkpoint.get("found", {})
                start_doc = checkpoint.get("doc_count", 0)
                if verbose:
                    print(
                        f"Resuming from doc {start_doc}, "
                        f"already found {len(found)}")

        # Create search patterns - use middle portion of each snippet
        search_patterns = {}
        for i, snippet in enumerate(snippets):
            if i in found:
                continue  # Already found
            clean = snippet.replace("<|endoftext|>", "").strip()
            if len(clean) > pattern_len * 2:
                mid = len(clean) // 2
                half = pattern_len // 2
                pattern = clean[mid - half: mid + half]
                search_patterns[i] = pattern

        if verbose:
            print(
                f"Searching for {len(search_patterns)} snippets in "
                f"OpenWebText...")
            print(f"Dataset has ~8M documents, this may take a while.")

        dataset = load_dataset("Skylion007/openwebtext",
                               split="train", streaming=True)

        doc_count = 0
        for doc in dataset:
            if doc_count < start_doc:
                doc_count += 1
                continue

            text = doc["text"]

            # Check all unfound patterns
            for idx, pattern in list(search_patterns.items()):
                if pattern in text:
                    found[idx] = text
                    del search_patterns[idx]
                    if verbose:
                        print(
                            f"  Found snippet {idx} at doc {doc_count} "
                            f"({len(found)}/{len(snippets)})")

            doc_count += 1

            # Checkpoint
            if save_path and doc_count % checkpoint_every == 0:
                checkpoint = {"found": found, "doc_count": doc_count}
                joblib.dump(checkpoint, Path(
                    save_path).with_suffix(".checkpoint.joblib"))
                if verbose:
                    print(
                        f"  Checkpoint at doc {doc_count}, found {len(found)}")

            if max_docs and doc_count >= max_docs:
                break

            if not search_patterns:  # All found
                break

            if verbose and doc_count % 100000 == 0:
                print(
                    f"  Searched {doc_count} docs, found {len(found)}/"
                    f"{len(snippets)}")

        if verbose:
            print(f"Found {len(found)} / {len(snippets)} source documents")

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(found, save_path)
            if verbose:
                print(f"Saved to {save_path}")

        return found

    def match_snippets_to_sources(
        self,
        snippets_path: Path | str,
        max_docs: int | None = None,
        save_path: Path | str | None = None,
        resume: bool = True,
        verbose: bool = True,
    ) -> dict[int, str]:
        """Load snippets from joblib and find their source documents.

        Convenience method that loads snippets from a joblib file and searches
        OpenWebText for the original source documents.

        Args:
            snippets_path: Path to joblib file containing list of text 
                snippets.
            max_docs: Maximum documents to search. Default is None = full 
                dataset (~8M docs).
            save_path: If provided, save results as joblib file. 
                Default is None.
            resume: If True, resume from checkpoint if available. 
                Default is True.
            verbose: Print progress updates. Default is True.

        Returns:
            Dict mapping snippet index to the full source document text.
        """
        import joblib

        snippets_path = Path(snippets_path)
        snippets = joblib.load(snippets_path)

        if verbose:
            print(f"Loaded {len(snippets)} snippets from {snippets_path}")

        if save_path is None:
            save_path = snippets_path.parent / "source_documents.joblib"

        resume_from = None
        if resume:
            checkpoint_path = Path(save_path).with_suffix(".checkpoint.joblib")
            if checkpoint_path.exists():
                resume_from = checkpoint_path

        return self.find_source_documents(
            snippets=snippets,
            max_docs=max_docs,
            save_path=save_path,
            resume_from=resume_from,
            verbose=verbose,
        )

    def __call__(
        self,
        token_ids: Union[list[int], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> Union[str, list[str]]:
        """Shorthand for decode method.

        Args:
            token_ids: (Union[list[int], torch.Tensor]) The token ids.
            skip_special_tokens: (bool) Whether to skip special tokens. 
                Default is True.
            clean_up_tokenization_spaces: (bool) Whether to clean up
                tokenization spaces. Default is True.

        Returns:
            (Union[str, list[str]]) The decoded text.
        """
        return self.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
