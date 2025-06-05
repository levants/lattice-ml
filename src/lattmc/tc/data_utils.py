# This function was stolen from one of Neel Nanda's exploratory notebooks
# Thanks, Neel!
import csv
from pathlib import Path
from typing import List, Union

import einops
import joblib
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from src.lattmc.tc.transcoders_utils import Transcoder


def tokenize_and_concatenate(
    dataset,
    tokenizer,
    streaming=False,
    max_length=1024,
    column_name="text",
    add_bos_token=True,
):
    """Helper function to tokenizer and concatenate a dataset of text. 
        This converts the text to tokens, concatenates them 
        (separated by EOS tokens) and then reshapes them into 
        a 2D array of shape (____, sequence_length), dropping the last batch. 
        Tokenizers are much faster if parallelised, so we chop the string 
        into 20, 
        feed it into the tokenizer, in parallel with padding, 
        then remove padding at the end.

    This tokenization is useful for training language models, 
    as it allows us to efficiently train on a large corpus of 
    text of varying lengths (without, eg, a lot of truncation or padding). 
    Further, for models with absolute positional encodings, 
    this avoids privileging early tokens 
    (eg, news articles often begin with CNN, 
    and models may learn to use early positional encodings to predict these)

    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a 
            HuggingFace text dataset.
        tokenizer (AutoTokenizer): The tokenizer. 
            Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. 
            If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window 
            of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column 
            in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.

    Returns:
        Dataset: Returns the tokenized dataset, as a dataset of 
                tensors, with a single column called "tokens"

    Note: There is a bug when inputting very small datasets 
    (eg, <1 batch per process) where it just outputs nothing. 
    I'm not super sure why
    """
    for key in dataset.features:
        if key != column_name:
            dataset = dataset.remove_columns(key)

    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer.
        # This will be removed before inputting tokens to the model,
        # so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Define the length to chop things up into - leaving space
    # for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    def tokenize_function(examples):
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.eos_token.join(text)
        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [
            full_text[i * chunk_length: (i + 1) * chunk_length]
            for i in range(num_chunks)
        ]
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace
        # map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors='np', padding=True)[
            'input_ids'
        ].flatten()
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)
        num_batches = num_tokens // (seq_len)
        # Drop the final tokens if not enough to make a full sequence
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[column_name],
    )
    # tokenized_dataset.set_format(type="torch", columns=["tokens"])

    return tokenized_dataset


def init_dataset(
    transcoder: Transcoder,
    seed: int = 42,
    buffer_size: int = 10_000,
    max_length: int = 128,
    streaming: bool = True,
    num_tokens: int = 12800*2
):
    """Initialize the tokens for the model.
    Args:
        transcoder (Transcoder): The transcoder to use.
        seed (int, optional): The seed to use for the random number generator. 
            Defaults to 42.
        buffer_size (int, optional): The buffer size to use for the dataset. 
            Defaults to 10_000.
        max_length (int, optional): The maximum length of the tokens. 
            Defaults to 128.
        streaming (bool, optional): Whether to stream the dataset. 
            Defaults to True.
        num_tokens (int, optional): The number of tokens to use. 
            Defaults to 12800*2.
    Returns:
        Dataset: The tokenized dataset.
    """
    dataset = load_dataset(
        'Skylion007/openwebtext',
        split='train',
        streaming=True
    )
    dataset = dataset.shuffle(
        seed=seed, buffer_size=buffer_size)  # type:ignore
    tokenized_owt = tokenize_and_concatenate(
        dataset, transcoder.model.tokenizer,
        max_length=max_length,
        streaming=streaming
    )
    tokenized_owt = tokenized_owt.shuffle(seed)
    tokenized_owt = tokenized_owt.take(num_tokens)  # type: ignore

    return tokenized_owt


def init_tokens(
    transcoder: Transcoder,
    seed: int = 42,
    buffer_size: int = 10_000,
    max_length: int = 128,
    streaming: bool = True,
    num_tokens: int = 12800*2,
    device: torch.device = torch.device('cpu')
):
    """Initialize the tokens for the model.
    Args:
        transcoder (Transcoder): The transcoder to use.
        seed (int, optional): The seed to use for the random number generator. 
            Defaults to 42.
        buffer_size (int, optional): The buffer size to use for the dataset. 
            Defaults to 10_000.
        max_length (int, optional): The maximum length of the tokens. 
            Defaults to 128.
        streaming (bool, optional): Whether to stream the dataset. 
            Defaults to True.
        num_tokens (int, optional): The number of tokens to use. 
            Defaults to 12800*2.
        device (torch.device, optional): The device to use.
            Defaults to torch.device('cpu').
    Returns:
        Dataset: The tokenized dataset.
    """
    tokenized_owt = init_dataset(
        transcoder,
        seed=seed,
        buffer_size=buffer_size,
        max_length=max_length,
        streaming=streaming,
        num_tokens=num_tokens
    )
    owt_tokens = np.stack([x['tokens'] for x in tokenized_owt])
    owt_tokens_torch = torch.from_numpy(owt_tokens).to(device)

    return owt_tokens_torch


def load_pt_tokens(
        transcoder: Transcoder,
        tokens_path: Union[Path, str],
        seed: int = 42,
        buffer_size: int = 10_000,
        max_length: int = 128,
        streaming: bool = True,
        num_tokens: int = 12800*2,
        device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """Load the tokens from the given path."""
    if tokens_path.exists():
        owt_tokens_torch = torch.load(
            tokens_path,
            weights_only=True,
            map_location=device
        ) if tokens_path.suffix == '.pt' else joblib.load(
            tokens_path
        )
        # Check if the tokens are on the same device as the transcoder
        # owt_tokens_torch = owt_tokens_torch.to(device)
    else:
        owt_tokens_torch = init_tokens(
            transcoder,
            seed=seed,
            buffer_size=buffer_size,
            max_length=max_length,
            streaming=streaming,
            num_tokens=num_tokens,
            device=device
        )
        torch.save(owt_tokens_torch, tokens_path)

    return owt_tokens_torch


def load_csv_tokens(
    csv_path: Union[Path, str],
    transcoder: Transcoder,
    columns: list = None,
    tokens_path: Union[Path, str] = None
) -> np.ndarray:
    """Load a CSV file and return its contents as a NumPy array."""
    df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
    tokens_list = list()
    for index, row in df.iterrows():
        for column in columns:
            x = row[column]
            t = transcoder.tokenize(f' {x}')[0].cpu()
            tokens_list.append(t)
    tokens = np.array(tokens_list, dtype=object)
    joblib.dump(tokens, tokens_path)

    return tokens


def load_tokens(
        transcoder: Transcoder,
        tokens_path: Union[Path, str],
        seed: int = 42,
        buffer_size: int = 10_000,
        max_length: int = 128,
        streaming: bool = True,
        num_tokens: int = 12800*2,
        device: torch.device = torch.device('cpu'),
        csv_path: Union[Path, str] = None,
        columns: list = None
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Load the tokens from the given path."""
    if Path(tokens_path).suffix == '.joblib' or csv_path is not None:
        if not isinstance(csv_path, Path):
            csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f'CSV file {csv_path} does not exist')
        if columns is None:
            columns = ['text']
        owt_tokens_torch = load_csv_tokens(
            csv_path,
            transcoder,
            columns=columns,
            tokens_path=tokens_path
        )
    else:
        owt_tokens_torch = load_pt_tokens(
            transcoder,
            tokens_path,
            seed=seed,
            buffer_size=buffer_size,
            max_length=max_length,
            streaming=streaming,
            num_tokens=num_tokens,
            device=device
        )

    return owt_tokens_torch
