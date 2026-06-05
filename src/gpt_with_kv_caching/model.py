"""GPT-style (decoder-only) Transformer language model with key-value caching."""

import torch
import torch.nn as nn

from src.gpt_with_kv_caching.decoder import DecoderStack
from src.utils.positional_encoder import PositionalEncoder


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only Transformer with tied embeddings and optional key-value caching."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float = 0.0,
        use_kv_cache: bool = False,
    ):
        """Initialize the decoder only transformer.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed forward layer.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of decoder layers.
            max_seq_len (int): Maximum sequence length.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            use_kv_cache (bool, optional): Whether to maintain and return a
                per-layer key/value cache during the forward pass. Defaults to False.
        """
        super().__init__()
        # Positional Embedding Layers
        self.embedding_layer = PositionalEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Decoder Stack
        self.decoder = DecoderStack(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_kv_cache=use_kv_cache,
        )

        # Final output projection layer
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding_layer.token_embedder.weight

        self.use_kv_cache = use_kv_cache

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Forward pass for the decoder only transformer.

        Args:
            input_ids (torch.Tensor): Token ids of shape (batch_size, seq_len).
                During cached decoding this may be a single new token.
            attention_mask (torch.Tensor): Attention mask of shape
                (batch_size, seq_len); 1 for real tokens, 0 for padding.
            kv_cache (dict[int, tuple[torch.Tensor, torch.Tensor]], optional):
                Per-layer cached (key, value) tensors from previous steps.
                Defaults to None.

        Returns:
            torch.Tensor | tuple[torch.Tensor, dict]: Output logits of shape
                (batch_size, seq_len, vocab_size). When ``use_kv_cache`` is True,
                also returns the updated per-layer key/value cache.
        """
        # Convert attention masks to padding masks and reshape to (B, 1, 1, K)
        # so they broadcast over (B, n_heads, Q, K) attention scores.
        padding_mask = (~attention_mask.bool()).unsqueeze(1).unsqueeze(2)

        # Embed the input tokens
        input_embed = self.embedding_layer(input_ids, attention_mask)

        # Decode the input tokens
        decoder_output, kv_cache = self.decoder(
            decoder_input=input_embed,
            decoder_padding_mask=padding_mask,
            kv_cache=kv_cache,
        )

        # Project the decoder output to the output vocabulary
        output = self.output_proj(decoder_output)

        if self.use_kv_cache:
            return output, kv_cache
        else:
            return output


if __name__ == "__main__":
    vocab_size = 40000
    d_model = 768
    d_ff = 3072
    n_heads = 12
    n_layers = 12
    max_seq_len = 512
    dropout = 0.1

    # create a encoder decoder transformer
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total:,} total ({trainable:,} trainable)")

    original_gpt_param_count = 116_130_816
    if trainable == original_gpt_param_count:
        print("Congratulations!!")

    # create a test tensor
    input_ids = torch.randint(0, vocab_size, (1, max_seq_len))
    attention_mask = torch.ones((input_ids.size(0), 1), dtype=input_ids.dtype)

    # forward pass
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    print(output.shape)
    print(output)
