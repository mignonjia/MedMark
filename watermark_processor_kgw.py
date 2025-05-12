from __future__ import annotations
import collections
from math import sqrt

import scipy.stats

import torch
from torch import nn
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from nltk.util import ngrams

from utils.normalizers import normalization_strategy_lookup

class KGWWatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        context_window: int = 1,  
        hash_key: int = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
        gamma: float = 0.5,
        delta: float = 2.0,
        device: torch.device = None,
    ):
        assert device, "Must pass device"
        # watermarking parameters
        self.vocab = vocab
        self.vocab_size = len(vocab) #50272
        self.rng = None
        self.hash_key = hash_key
        self.device = device
        self.context_window = context_window
        self.seed_positional = torch.arange(1, self.context_window + 1).to(device)

        self.gamma = torch.tensor([gamma]).to(device)
        self.delta = torch.tensor([delta]).to(device)

        self.gamma_list = torch.empty(0, dtype=torch.float).to(device)
        self.delta_list = torch.empty(0, dtype=torch.float).to(device)

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        assert input_ids.shape[-1] >= self.context_window, f"Compute hash seed requires at least a {self.context_window} token prefix sequence to seed rng"
        seed = (self.hash_key * (input_ids[-self.context_window:] * self.seed_positional).sum()).item()
        self.rng.manual_seed(seed) 
        return

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        # seed the rng using the previous tokens/prefix 
        self._seed_rng(input_ids)
        
        delta = self.delta
        gamma = self.gamma
        
        self.gamma_list = torch.cat([self.gamma_list, gamma])
        self.delta_list = torch.cat([self.delta_list, delta])

        greenlist_size = int(self.vocab_size * gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)
        
        greenlist_ids = vocab_permutation[:greenlist_size] 
        
        return greenlist_ids, gamma, delta


class KGWWatermarkLogitsProcessor(KGWWatermarkBase, LogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, logits: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        green_tokens_mask = torch.zeros_like(logits)
        green_tokens_mask[greenlist_token_ids] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        if self.rng is None:
            self.rng = torch.Generator(device=self.device)

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids, gamma, delta = self._get_greenlist_ids(input_ids[b_idx])
            green_tokens_mask = self._calc_greenlist_mask(logits=scores[b_idx], greenlist_token_ids=greenlist_ids)
            scores[b_idx][green_tokens_mask] = scores[b_idx][green_tokens_mask] + delta.half()
        
        return scores


class KGWWatermarkDetector(KGWWatermarkBase):
    def __init__(
        self,
        *args,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_bigrams: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)
        
        self.min_prefix_len = self.context_window

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))
        
        self.ignore_repeated_bigrams = ignore_repeated_bigrams

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        var = torch.sum(self.gamma_list * (1 - self.gamma_list))
        mean = torch.sum(self.gamma_list)
        z = (observed_count - mean)/torch.sqrt(var)  
        self.gamma_list = torch.empty(0, dtype=torch.float).to(self.device)         
        self.delta_list = torch.empty(0, dtype=torch.float).to(self.device)       
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z.cpu())
        return p_value

    def _score_sequence(
        self,
        input_ids: Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ):
        if self.ignore_repeated_bigrams:
            # Method that only counts a green/red hit once per unique bigram.
            # New num total tokens scored (T) becomes the number unique bigrams.
            # We iterate over all unqiue token bigrams in the input, computing the greenlist
            # induced by the first token in each, and then checking whether the second
            # token falls in that greenlist.
            assert return_green_token_mask is False, "Can't return the green/red mask when ignoring repeats."
            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())
            for idx, bigram in enumerate(freq.keys()):
                prefix = torch.tensor([bigram[0]], device=self.device)  # expects a 1-d prefix tensor on the randperm device
                greenlist_ids, gamma, delta = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = True if bigram[1] in greenlist_ids else False
            green_token_count = sum(bigram_table.values())
        else:
            num_tokens_scored = len(input_ids) - self.min_prefix_len
            if num_tokens_scored < 1:
                raise ValueError(
                    (
                        f"Must have at least {1} token to score after "
                        f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
                    )
                )
            # Standard method.
            # Since we generally need at least 1 token (for the simplest scheme)
            # we start the iteration over the token sequence with a minimum
            # num tokens as the first prefix for the seeding scheme,
            # and at each step, compute the greenlist induced by the
            # current prefix and check if the current token falls in the greenlist.
            green_token_count, green_token_mask = 0, []
            for idx in range(self.min_prefix_len, len(input_ids)):
                curr_token = input_ids[idx]
                greenlist_ids, gamma, delta = self._get_greenlist_ids(input_ids[:idx])
                # print(input_ids, curr_token, greenlist_ids)
                # exit()
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)

        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))

        return score_dict

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        **kwargs,
    ) -> dict:

        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = True  # to return the "confidence":=1-p of positive detections

        # run optional normalizers on text
        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            # try to remove the bos_tok at beginning if it's there
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        # call score method
        output_dict = {}
        score_dict = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict