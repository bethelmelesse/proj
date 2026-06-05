"""Encoder-decoder Transformer model (Vaswani et al., 2017)."""

import torch
import torch.nn as nn

from src.transformer.decoder import DecoderStack
from src.transformer.encoder import EncoderStack
from src.utils.positional_encoder import PositionalEncoder


class EncoderDecoderTransformer(nn.Module):
    """Encoder-decoder Transformer with tied input/output embeddings."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        """Initialize the encoder decoder transformer.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed forward layer.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of encoder and decoder layers.
            max_seq_len (int): Maximum sequence length.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        # Positional Embedding Layers
        self.embedding_layer = PositionalEncoder(
            vocab_size, d_model, max_seq_len, dropout
        )

        # Encoder and Decoder Stacks
        self.encoder = EncoderStack(d_model, d_ff, n_heads, n_layers, dropout)
        self.decoder = DecoderStack(d_model, d_ff, n_heads, n_layers, dropout)

        # Final output projection layer
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding_layer.token_embedder.weight

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        source_attention_mask: torch.Tensor | None = None,
        target_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for the encoder decoder transformer.

        Args:
            source_tokens (torch.Tensor): Encoder token ids of shape
                (batch_size, source_seq_len).
            target_tokens (torch.Tensor): Decoder token ids of shape
                (batch_size, target_seq_len).
            source_attention_mask (torch.Tensor, optional): Attention mask for the
                encoder input; 1 for real tokens, 0 for padding. Defaults to None.
            target_attention_mask (torch.Tensor, optional): Attention mask for the
                decoder input; 1 for real tokens, 0 for padding. Defaults to None.

        Returns:
            torch.Tensor: Output logits of shape
                (batch_size, target_seq_len, vocab_size).
        """
        # Convert attention masks to padding masks and reshape to (B, 1, 1, K)
        # so they broadcast over (B, n_heads, Q, K) attention scores.
        source_padding_mask = (
            (~source_attention_mask.bool()).unsqueeze(1).unsqueeze(2)
            if source_attention_mask is not None
            else None
        )

        target_padding_mask = (
            (~target_attention_mask.bool()).unsqueeze(1).unsqueeze(2)
            if target_attention_mask is not None
            else None
        )

        # Replace -100 labels with 0 to avoid embedding lookup errors
        if (target_tokens == -100).any():
            target_tokens = target_tokens.clone()
            target_tokens[target_tokens == -100] = 0

        # Embed the source and target tokens
        source_embed = self.embedding_layer(source_tokens)
        target_embed = self.embedding_layer(target_tokens)

        # Encode the source tokens
        encoder_output = self.encoder(
            encoder_input=source_embed, encoder_padding_mask=source_padding_mask
        )

        # Decode the target tokens
        decoder_output = self.decoder(
            encoder_output=encoder_output,
            decoder_input=target_embed,
            encoder_padding_mask=source_padding_mask,
            decoder_padding_mask=target_padding_mask,
        )

        # Project the decoder output to the output vocabulary
        output = self.output_proj(decoder_output)
        return output


if __name__ == "__main__":
    vocab_size = 37000
    d_model = 512
    d_ff = 2048
    n_heads = 8
    n_layers = 6
    max_seq_len = 512
    dropout = 0.1

    # create a encoder decoder transformer
    model = EncoderDecoderTransformer(
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

    # Base Transformer (Vaswani 2017): sinusoidal pos encoding, tied embeddings, no
    # LM-head bias.
    original_transformer_param_count = 63_045_632
    if trainable != original_transformer_param_count:
        diff = trainable - original_transformer_param_count
        print(
            f"Mismatch: actual {trainable:,} vs expected "
            f"{original_transformer_param_count:,} "
            f"(diff {diff:+,})"
        )

        # Expected per-module counts for the original paper config.
        # Embedding: token table only (sinusoidal pos = 0 params, output proj tied = 0).
        expected_embedding = vocab_size * d_model
        # Per encoder layer: 4 attn projections (Q/K/V/O, no bias) + 2 LayerNorms + FFN
        # (with bias).
        expected_encoder_layer = (
            4 * d_model * d_model
            + 2 * (2 * d_model)
            + (d_model * d_ff + d_ff)
            + (d_ff * d_model + d_model)
        )
        # Per decoder layer: 8 attn projections (self Q/K/V/O + cross Q/K/V/O) + 3
        # LayerNorms + FFN.
        expected_decoder_layer = (
            8 * d_model * d_model
            + 3 * (2 * d_model)
            + (d_model * d_ff + d_ff)
            + (d_ff * d_model + d_model)
        )
        # All encoder layers are identical, same for decoder layers — compare just one
        # of each.
        expected = {
            "embedding_layer": expected_embedding,
            "encoder.layers.0": expected_encoder_layer,
            "decoder.layers.0": expected_decoder_layer,
        }

        actual = {
            "embedding_layer": sum(
                p.numel() for p in model.embedding_layer.parameters()
            ),
            "encoder.layers.0": sum(
                p.numel() for p in model.encoder.layers[0].parameters()
            ),
            "decoder.layers.0": sum(
                p.numel() for p in model.decoder.layers[0].parameters()
            ),
        }

        print(f"\n{'Module':30s} {'Actual':>15} {'Expected':>15} {'Diff':>12}  Match")
        for key, exp in expected.items():
            act = actual[key]
            d = act - exp
            marker = "OK" if d == 0 else "DIFF"
            print(f"  {key:28s} {act:>15,} {exp:>15,} {d:>+12,}  {marker}")

    # create a test tensor
    source_tokens = torch.randint(0, vocab_size, (1, max_seq_len))
    target_tokens = torch.randint(0, vocab_size, (1, max_seq_len))

    # forward pass

    output = model(source_tokens=source_tokens, target_tokens=target_tokens)
    print(output.shape)
    print(output)

    print()
