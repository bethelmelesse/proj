"""Greedy autoregressive generation for the KV-caching decoder-only model."""

import torch
from transformers import AutoTokenizer

from src.gpt_with_kv_caching.model import DecoderOnlyTransformer


class Generator:
    """Greedy autoregressive generator for a decoder-only transformer."""

    def __init__(
        self,
        model_args: dict,
        tokenizer_args: dict,
        device: str | torch.device = "cpu",
        use_kv_cache: bool = True,
    ):
        """Initialize the generator.

        Args:
            model_args (dict): Model config — ``d_model``, ``d_ff``, ``n_heads``,
                ``n_layers``, ``max_seq_len``, ``dropout``.
            tokenizer_args (dict): Tokenizer config — ``tokenizer_path``.
            device (str | torch.device): Device to run the model on.
                Defaults to "cpu".
            use_kv_cache (bool): Whether to use key-value caching during
                generation. When True, the prompt prefills the cache and each
                step feeds only the newest token; when False, the full growing
                sequence is re-run each step. Defaults to True.
        """
        self.device = device
        self.use_kv_cache = use_kv_cache

        # Initialize tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_args["tokenizer_path"])

        # Get EOS token id.
        self.eos_token_id = self.tokenizer.eos_token_id

        # GPT-2-style tokenizers have no pad token; reuse EOS.
        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.eos_token_id
        )

        # Build model and put it in eval mode.
        self.model = DecoderOnlyTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=model_args["d_model"],
            d_ff=model_args["d_ff"],
            n_heads=model_args["n_heads"],
            n_layers=model_args["n_layers"],
            max_seq_len=model_args["max_seq_len"],
            dropout=model_args["dropout"],
            use_kv_cache=self.use_kv_cache,
        )
        self.model.to(device)
        self.model.eval()

    def __call__(self, input: str | list[str], max_new_tokens: int) -> list[str]:
        """Alias for `generate`."""
        return self.generate(input, max_new_tokens)

    def generate(
        self, input: str | list[str], max_new_tokens: int
    ) -> tuple[torch.Tensor, list[str]]:
        """Greedily generate continuations for one or more prompts.

        Args:
            input (str | list[str]): Prompt or batch of prompts to continue.
            max_new_tokens (int): Maximum number of tokens to generate per prompt.

        Returns:
            tuple[torch.Tensor, list[str]]: The generated token ids of shape
                (batch_size, seq_len), and the decoded sequences (prompt +
                generated continuation), one per input prompt, with special
                tokens stripped.
        """
        # Tokenize inputs with left padding so all prompts end at the same column.
        tokenized_input = self.tokenizer(
            input,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            padding_side="left",
        )
        input_ids = tokenized_input["input_ids"].to(self.device)
        attention_mask = tokenized_input["attention_mask"].to(self.device)

        if self.use_kv_cache:
            output_ids = self._generate_cached(
                input_ids, attention_mask, max_new_tokens
            )
        else:
            output_ids = self._generate_uncached(
                input_ids, attention_mask, max_new_tokens
            )

        decoded_output = self._decode(output_ids)
        return output_ids, decoded_output

    def _decode(self, output: torch.Tensor) -> list[str]:
        """Decode generated token ids back into strings.

        Args:
            output (torch.Tensor): Token ids of shape (batch_size, seq_len).

        Returns:
            list[str]: One decoded sequence per row, with special tokens stripped.
        """
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)

    def _generate_cached(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        """Generate with KV caching: prefill the prompt, then feed one token per step.

        Args:
            input_ids (torch.Tensor): Prompt token ids of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Mask of shape (batch_size, seq_len).
            max_new_tokens (int): Maximum number of tokens to generate per prompt.

        Returns:
            torch.Tensor: The full token sequences (prompt + continuation).
        """
        # Track finished sequences
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=self.device)

        # Prefill: full prompt seeds the cache and produces the first new token.
        next_token, kv_cache = self._generate_next(input_ids, attention_mask, None)

        #  Flag any whose first token is EOS.
        finished = torch.logical_or(
            finished, next_token.squeeze(-1) == self.eos_token_id
        )
        # A column of ones marking the new token as a real (non-pad) position.
        ones_col = torch.ones(
            (input_ids.size(0), 1), dtype=attention_mask.dtype, device=self.device
        )

        # Extend the mask for the new token and append it to the prompt output.
        attention_mask = torch.cat([attention_mask, ones_col], dim=-1)
        output = torch.cat((input_ids, next_token), dim=-1)

        # Decode: feed only the new token each step.
        for _ in range(max_new_tokens - 1):
            if finished.all():
                break
            next_token, kv_cache = self._generate_next(
                next_token, attention_mask, kv_cache
            )
            next_token[finished] = self.pad_id
            finished = torch.logical_or(
                finished, next_token.squeeze(-1) == self.eos_token_id
            )
            output = torch.cat((output, next_token), dim=-1)
            attention_mask = torch.cat([attention_mask, ones_col], dim=-1)

        return output

    def _generate_uncached(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Generate without caching: re-run the full growing sequence each step.

        Args:
            input_ids (torch.Tensor): Prompt token ids of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Mask of shape (batch_size, seq_len).
            max_new_tokens (int): Maximum number of tokens to generate per prompt.

        Returns:
            torch.Tensor: The full token sequences (prompt + continuation).
        """
        ones_col = torch.ones(
            (input_ids.size(0), 1), dtype=attention_mask.dtype, device=self.device
        )
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=self.device)

        output = input_ids
        for _ in range(max_new_tokens):
            if finished.all():
                break
            next_token, _ = self._generate_next(output, attention_mask, None)
            next_token[finished] = self.pad_id
            finished = torch.logical_or(
                finished, next_token.squeeze(-1) == self.eos_token_id
            )
            output = torch.cat((output, next_token), dim=-1)
            attention_mask = torch.cat([attention_mask, ones_col], dim=-1)

        return output

    def _generate_next(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: dict | None,
    ) -> tuple[torch.Tensor, dict | None]:
        """Run one forward pass and greedily pick the next token.

        Args:
            input_ids (torch.Tensor): Token ids of shape (batch_size, seq_len).
                During cached decode this is just the latest token.
            attention_mask (torch.Tensor): Attention mask of shape
                (batch_size, total_seq_len); 1 for real tokens, 0 for padding.
            kv_cache (dict | None): Per-layer key/value cache, or None to start fresh.

        Returns:
            tuple[torch.Tensor, dict | None]: The greedily selected next token of
                shape (batch_size, 1) and the updated key/value cache (None when
                caching is disabled).
        """
        with torch.no_grad():
            if self.use_kv_cache:
                logits, kv_cache = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                )
            else:
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

        next_logit = logits[:, -1:]
        next_token = torch.argmax(next_logit, dim=-1)
        return next_token, kv_cache


if __name__ == "__main__":
    tokenizer_args = {"tokenizer_path": "t5-small"}

    model_args = {
        "d_model": 768,
        "d_ff": 3072,
        "n_heads": 12,
        "n_layers": 2,
        "max_seq_len": 512,
        "dropout": 0.1,
    }

    generator = Generator(
        model_args=model_args,
        tokenizer_args=tokenizer_args,
        device="cpu",
        use_kv_cache=True,
    )

    seq_list = [
        "Hello, my name",
        "The president of US",
        "Once upon a time there was a frog",
        "The brown",
        "Bethel",
    ]

    def test_w_string():
        token_ids, predictions = generator(seq_list[0], max_new_tokens=4)

        for src, pred in zip(seq_list, predictions):
            print(f"Starting sequence: {src}")
            print(f"Final Prediction: {pred}\n")
        return predictions

    def test_w_list_strings():
        token_ids, predictions = generator(seq_list, max_new_tokens=4)

        for src, pred in zip(seq_list, predictions):
            print(f"Starting sequence: {src}")
            print(f"Final Prediction: {pred}\n")
        return predictions

    pred_string = test_w_string()
    pred_list = test_w_list_strings()

    def test_match():
        # The same prompt should yield the same continuation whether passed alone or
        # batched with others. This only holds if attention masking and positional
        # encoding correctly ignore the left padding added during batching.
        assert pred_string[0] == pred_list[0], (
            "Single-string vs batched prediction mismatch:\n"
            f"  single: {pred_string[0]!r}\n"
            f"  batched: {pred_list[0]!r}"
        )
        print("PASS: single-string and batched predictions match.")

    test_match()
