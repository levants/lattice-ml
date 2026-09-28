"""Tokenizer de-tokenizer and text processing utilities."""

import colorsys
import logging
import re
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizer

from src.lattmc.fca.utils import in_any, not_empty
from src.lattmc.fca.visualization_utils import (clean_str_tokens,
                                                with_background, with_texgraph)
from src.lattmc.models.detokenizer import GPT2Detokenizer
from src.lattmc.tc.model_utils import ModelUtils

logger = logging.getLogger(name=__file__)

_PASTEL_HUE_COUNT = 27
_PASTEL_HUE_STEP = 10
_PASTEL_TONES = (
    (0.54, 0.88),
    (0.60, 0.84),
    (0.48, 0.86),
    (0.66, 0.82),
    (0.52, 0.82),
    (0.62, 0.88),
    (0.56, 0.80),
    (0.46, 0.90),
    (0.68, 0.80),
    (0.50, 0.84),
    (0.58, 0.86),
)


def _pastel_rgb(
    hue_degree: float,
    saturation: float,
    lightness: float,
) -> Tuple[int, int, int]:
    """Return a soft but visible RGB color for terminal/LaTeX highlights.

    Args:
        hue_degree (float): The hue degree.
        saturation (float): The saturation.
        lightness (float): The lightness.

    Returns:
        Tuple[int, int, int]: The RGB color.
    """
    red, green, blue = colorsys.hls_to_rgb(
        hue_degree / 360.0,
        lightness,
        saturation,
    )
    return tuple(round(channel * 255) for channel in (red, green, blue))


def _generate_pastel_bg_rgbs() -> Tuple[Tuple[int, int, int], ...]:
    """Generate pastel colors with adjacent indices spread by hue.

    Returns:
        Tuple[Tuple[int, int, int], ...]: The RGB colors.
    """
    return tuple(
        _pastel_rgb(
            hue_degree=(
                ((hue_idx * _PASTEL_HUE_STEP) % _PASTEL_HUE_COUNT)
                * 360.0
                / _PASTEL_HUE_COUNT
            ),
            saturation=saturation,
            lightness=lightness,
        )
        for saturation, lightness in _PASTEL_TONES
        for hue_idx in range(_PASTEL_HUE_COUNT)
    )


PASTEL_BG_RGBS = _generate_pastel_bg_rgbs()

BG_CODES = {
    idx: f"48;2;{red};{green};{blue}"
    for idx, (red, green, blue) in enumerate(PASTEL_BG_RGBS)
}


class TokenizationUtils(ModelUtils):
    """Tokenizer de-tokenizer and text processing utilities.

    Attributes:
        model (HookedTransformer): model
        device (torch.device): device
        background_dets (int): background detection code
        texgraph_dets (str): texgraph detection code
    """

    def __init__(
        self,
        model: HookedTransformer,
        device: torch.device = torch.device('cpu'),
        background_dets: int = None,
        texgraph_dets: str = None,
    ):
        super().__init__(model, device)
        self._detokenizer = GPT2Detokenizer()
        self._background_dets = background_dets
        self._texgraph_dets = texgraph_dets
        self._bg_codes = BG_CODES

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer.
        Returns:
            PreTrainedTokenizer: tokenizer
        """
        return self.model.tokenizer

    @property
    def detokenizer(self) -> GPT2Detokenizer:
        """Get the detokenizer.
        Returns:
            GPT2Detokenizer: detokenizer
        """
        return self._detokenizer

    @property
    def background_dets(self) -> int:
        """Get the background detection code.
        Returns:
            int: background detection code
        """
        return self._background_dets

    @background_dets.setter
    def background_dets(self, bg_code: int):
        """Set the background detection code.
        Args:
            bg_code (int): background detection code
        """
        if bg_code is None or isinstance(bg_code, int):
            self._background_dets = bg_code
        else:
            raise ValueError(
                f'Background detection code must be an integer, '
                f'got {bg_code} of type {type(bg_code)}.'
            )

    @property
    def texgraph_dets(self) -> str:
        """Get the texgraph detection code."""
        return self._texgraph_dets

    @texgraph_dets.setter
    def texgraph_dets(self, bg_code: str):
        """Set the texgraph detection code.
        Args:
            bg_code (str): texgraph detection code
        """
        if bg_code is None or isinstance(bg_code, str):
            self._texgraph_dets = bg_code
        else:
            raise ValueError(
                f'Texgraph detection code must be a string, '
                f'got {bg_code} of type {type(bg_code)}.'
            )

    @property
    def bg_codes(self) -> Dict[int, Union[int, str]]:
        """Get the background detection codes.
        Returns:
            Dict[int, Union[int, str]]: background detection codes
        """
        return self._bg_codes

    @bg_codes.setter
    def bg_codes(self, bg_codes: Dict[int, str]):
        """Set the background detection codes.
        Args:
            bg_codes (Dict[int, str]): background detection codes
        """
        if bg_codes is None or isinstance(bg_codes, dict):
            self._bg_codes = bg_codes
        else:
            raise ValueError(
                f'Background detection codes must be a dictionary, '
                f'got {bg_codes} of type {type(bg_codes)}.'
            )

    def _add_padding_token(self):
        """Add a padding token to the tokenizer."""
        if self.tokenizer.pad_token is None:
            # We add a padding token, purely to implement the tokenizer.
            # This will be removed before inputting tokens to the model,
            # so we do not need to increment d_vocab in the model.
            self.tokenizer.add_special_tokens(  # type: ignore
                {"pad_token": "<PAD>"}
            )

    def tokenize(
        self,
        prompt: str,
        return_tensors: str = 'pt',
        padding: bool = True
    ) -> torch.Tensor:
        """Tokenize the input prompt.
        Args:
            prompt (str): input prompt
            return_tensors (str): return tensors type. Default is 'pt'.
            padding (bool): padding. Default is True.
        Returns:
            tokens (torch.Tensor): tokenized prompt
        """
        # Check if the tokenizer has a padding token
        self._add_padding_token()
        # Tokenize the input prompt
        text = f'{self.tokenizer.eos_token}{prompt}'  # type: ignore
        # Tokenize the text and return the tensor
        tokens = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding
        ).to(self.device)  # type: ignore

        input_ids = tokens['input_ids']

        return input_ids

    def _check_and_tokenize(
        self,
        prompt: Union[str, torch.Tensor],
        return_tensors: str = 'pt',
        padding: bool = True
    ) -> torch.Tensor:
        """Check the type of the input prompt and tokenize it.
        Args:
            prompt (Union[str, torch.Tensor]): input prompt
            return_tensors (str): return tensors type. Default is 'pt'.
            padding (bool): padding. Default is True.
        Returns:
            tokens (torch.Tensor): tokenized prompt
        """
        if isinstance(prompt, str):
            tokens = self.tokenize(prompt, return_tensors, padding)
        elif isinstance(prompt, torch.Tensor):
            tokens = prompt.to(self.device)
        else:
            raise ValueError(
                f'Invalid prompt type: {type(prompt)}. '
                'Expected str or torch.Tensor.'
            )

        return tokens

    def to_string(self, prompt: torch.Tensor) -> str:
        """Convert a prompt to a string.
        Args:
            prompt (torch.Tensor): prompt tokens
        Returns:
            str: decoded string
        """
        return self.model.to_string(prompt)  # type: ignore

    def to_single_str_token(self, int_token: int) -> str:
        """Convert a single integer token to a string.
        Args:
            int_token (int): integer token
        Returns:
            str: string token
        """
        return self.model.to_single_str_token(int_token)

    def to_str_tokens(
        self,
        prompt: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Union[List[str], List[List[str]]]:
        """Convert tokens to a list of token strings.

        Args:
            prompt (Union[torch.Tensor, List[torch.Tensor]]): 
                1D tensor [pos], 2D tensor [batch, pos], or list of 1D tensors.
        Returns:
            Union[List[str], List[List[str]]]: List[str] for 1D input,
                List[List[str]] for 2D or list input.
        """
        if isinstance(prompt, list):
            str_tokens = [self.to_str_tokens(t) for t in prompt]
        elif prompt.dim() == 1:
            str_tokens = self.tokenizer.convert_ids_to_tokens(prompt.tolist())
        elif prompt.dim() == 2:
            str_tokens = [
                self.tokenizer.convert_ids_to_tokens(row.tolist())
                for row in prompt
            ]
        else:
            raise ValueError(
                f"Expected 1D or 2D tensor, got shape {prompt.shape}"
            )

        return str_tokens

    def clean_token_strings(self, str_token: str) -> str:
        """Clean a list of strings.
        Args:
            str_token (str): list of strings to clean
        Returns:
            str: cleaned list of strings
        """
        return clean_str_tokens(
            str_token,
            tokenizer=self.tokenizer,
            quote_style="curly",
            drop_special=True,
            normalize="NFKC",
            remove_control=True,
        )

    def clean_token_string(self, str_token: str) -> str:
        """Clean a string.
        Args:
            str_token (str): string to clean
        Returns:
            str: cleaned string
        """
        cln_token = self.clean_token_strings([str_token])
        res_token = cln_token[0] if not_empty(cln_token) else ''

        return res_token

    def decode(
        self,
        prompt: torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        """Decode a prompt tokens to a string.
        Args:
            prompt (torch.Tensor): prompt tokens
            skip_special_tokens (bool): whether to skip special tokens
                Default is False.
            clean_up_tokenization_spaces (bool): whether to clean 
                up tokenization spaces. Default is False.
        Returns:
            str: decoded string
        """
        return self.tokenizer.decode(
            prompt,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def detokenize(
        self,
        prompt: torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        """Detokenize a prompt tokens to a string.
        Args:
            prompt (torch.Tensor): prompt tokens
            skip_special_tokens (bool): whether to skip special tokens
                Default is True.
            clean_up_tokenization_spaces (bool): whether to clean 
                up tokenization spaces. Default is False.
        Returns:
            str: decoded string
        """
        return self.detokenizer.decode(
            prompt,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def _clean_each_token(self, token: str) -> str:
        """Clean each token.
        Args:
            token (str): token to clean
        Returns:
            str: cleaned token
        """
        text = self.tokenizer.convert_tokens_to_string([token])
        # text = self.tokenizer.clean_up_tokenization(text)
        text = re.sub(r'\uFFFD+', "'", text)

        return text

    def to_clean_string(self, token: str) -> str:
        """Convert a token to a clean string.
        Args:
            token (str): token to clean
        Returns:
            str: cleaned token
        """
        ctokn = self._clean_each_token(token)
        stokn = ctokn[1:] if ctokn.startswith(' ') else ctokn
        ctokn = f' {stokn}'

        return ctokn

    def to_space_string(self, token: str) -> str:
        """Convert a token to a space and string.
        Args:
            token (str): token to convert
        Returns:
            str: space and string
        """
        ctokn = self._clean_each_token(token)
        return (' ', ctokn[1:]) if ctokn.startswith(' ') else ('', ctokn)

    def to_background_string(self, token: str, bg_code: Union[int, str]) -> str:
        """Convert a token to a background string.
        Args:
            token (str): token to convert
            bg_code (Union[int, str]): background code (int for standard
                ANSI, str for 256-color or truecolor e.g.
                "48;5;224" or "48;2;253;213;220")
        Returns:
            str: background string
        """
        space, ctokn = self.to_space_string(token)
        btokn = with_background(ctokn, bg_code=bg_code)
        btext = f'{space}{btokn}'

        return btext

    def to_texgraph_string(self, token: str, bg_code: str) -> str:
        """Convert a token to a texgraph string.
        Args:
            token (str): token to convert
            bg_code (str): background code
        Returns:
            str: texgraph string
        """
        space, ctokn = self.to_space_string(token)
        btokn = with_texgraph(ctokn, bg_code=bg_code)
        btext = f'{space}{btokn}'

        return btext

    def text_tokens(self, prompt: str) -> List[Tuple[torch.Tensor, str]]:
        """Get the text tokens for the input prompt.
        Args:
            prompt (str): input prompt
        Returns:
            List[Tuple[torch.Tensor, str]]: list of text tokens
        """
        tokens = self.tokenize(prompt)[0] if isinstance(
            prompt,
            str
        ) else prompt
        text_tokens = [(tk, self.to_string(tk)) for tk in tokens]

        return text_tokens

    def print_text_tokens(self, prompt: Union[str, torch.Tensor]):
        """Print the text tokens for the input prompt.
        Args:
            prompt (Union[str, torch.Tensor]): input prompt
        """
        text_tokens = self.text_tokens(prompt)
        prompt_text = self.decode(prompt) if isinstance(
            prompt,
            torch.Tensor
        ) else prompt
        logger.info(f'Text tokens for prompt: {prompt_text}')
        for idx, (tk, t) in enumerate(text_tokens):
            logger.info(f'{idx} {tk}: {t}')

    def _generate_text_with_background(
        self,
        str_tokens: List[str],
        idx: int,
        bg_code: Union[int, str] = None,
    ) -> str:
        """Generate text with background.
        Args:
            str_tokens (List[str]): list of text tokens
            idx (int): index of the token
            bg_code (Union[int, str]): background code. Default is None.
                If None, use the self.background_dets or self.texgraph_dets.
        Returns:
            str: text with background
        """
        bg_cd = self.background_dets if bg_code is None else bg_code
        return with_background(
            self.clean_token_string(str_tokens[idx]),
            bg_code=bg_cd
        ) if bg_cd else with_texgraph(
            self.clean_token_string(str_tokens[idx]),
            bg_code=self.texgraph_dets
        ) if self.texgraph_dets else self.clean_token_string(str_tokens[idx])

    def assemble_text(
        self,
        prompt: torch.Tensor,
        idxs: np.ndarray,
        color_idx: int = 35,
    ) -> str:
        """Assemble the text from the tokens.

        Args:
            prompt (torch.Tensor): input prompt
            idxs (np.ndarray): indices of the tokens
            color_idx (int): index of the color code in self.bg_codes.
                Default is 35.
        Returns:
            text_tokens (str): assembled text tokens
        """
        str_tokens = self.to_str_tokens(prompt)
        decoded_tokens = []
        for idx in range(len(str_tokens)):
            token = str_tokens[idx]
            if in_any(idx, idxs):
                decoded_tokens.append(
                    self.to_background_string(
                        token,
                        bg_code=self.bg_codes[color_idx]
                    )
                )
            else:
                space, ctokn = self.to_space_string(token)
                decoded_tokens.append(f'{space}{ctokn}')
        text_tokens = ''.join(decoded_tokens)

        return text_tokens

    def match_tokens(
        self,
        prompt: torch.Tensor,
        tokens: Union[
            np.ndarray,
            List[int],
            Set[int],
            List[torch.Tensor],
            Set[torch.Tensor],
            Tuple[np.ndarray],
            Tuple[List[int]],
            Tuple[Set[int]],
            Tuple[List[torch.Tensor]],
            Tuple[Set[torch.Tensor]]
        ]
    ) -> np.ndarray:
        """Match the tokens.

        Args:
            prompt (torch.Tensor): input prompt
            tokens (Union[np.ndarray, List[int], Set[int],
                List[torch.Tensor], Set[torch.Tensor],
                Tuple[np.ndarray], Tuple[List[int]], Tuple[Set[int]],
                Tuple[List[torch.Tensor]], Tuple[Set[torch.Tensor]]]):
                tokens to match
        Returns:
            np.ndarray: indices of the matched tokens
        """
        tokens_tns = torch.tensor(list(tokens)) if isinstance(
            tokens,
            (set, tuple)
        ) else torch.tensor(tokens)
        tokens_tns = tokens_tns.to(prompt.device)
        mask = torch.isin(prompt, tokens_tns)
        idcs = torch.nonzero(mask).squeeze()

        return idcs

    def assembe_texts(
        self,
        prompts: List[torch.Tensor],
        idc_list: List[Union[np.ndarray, List[int]]],
    ) -> List[str]:
        """Assemble the text from the tokens for a batch of prompts
            with highlighted tokens on the given indices with background.
        Args:
            prompts (List[torch.Tensor]): list of input prompts
            idc_list (List[np.ndarray]): list of indices of the tokens
        Returns:
            text_tokens (List[str]): assembled text tokens
        """
        texts = []
        for prompt, idcs in zip(prompts, idc_list):
            texts.append(self.assemble_text(prompt, idcs))

        return texts

    def assemble_multitext(
        self,
        prompt: torch.Tensor,
        idxs_list: List[np.ndarray],
        color_codes: List[str],
    ) -> str:
        """Assemble the text from the tokens.

        Args:
            prompt (torch.Tensor): input prompt
            idxs_list (List[np.ndarray]): list of indices of the tokens
            color_codes (List[str]): list of colour codes for detected tokens
        Returns:
            text_tokens (str): assembled text tokens
        """
        str_tokens = self.to_str_tokens(prompt)
        decoded_tokens = list()
        not_detected = True
        with tqdm(
            list(range(len(str_tokens))),
            desc='Assembling text',
        ) as p_tokens:
            for idx in p_tokens:
                token = str_tokens[idx]
                for idxs, color_code in zip(idxs_list, color_codes):
                    if not_detected and in_any(idx, idxs):
                        decoded_tokens.append(
                            self.to_background_string(
                                token,
                                bg_code=color_code,
                            )
                        )
                        not_detected = False
                        break
                if not_detected:
                    space, ctokn = self.to_space_string(token)
                    decoded_tokens.append(f'{space}{ctokn}')
                not_detected = True
        text_tokens = ''.join(decoded_tokens)

        return text_tokens

    def print_tokens_with_text(self, tokens: torch.Tensor):
        """Print the tokens with their text.

        Args:
            tokens (torch.Tensor): input tokens
        """
        for tk in tokens:
            logger.info(f'{tk} - "{self.decode(tk)}"')
