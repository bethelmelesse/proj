"""GPT-style (decoder-only) Transformer decoder layer and stack."""

import torch
import torch.nn as nn

from src.attention_utils.multihead_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    """A single GPT-style decoder block (causal self-attention + feed-forward)."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.0):
        """Initialize the decoder layer.

        Args:
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed forward layer.
            n_heads (int): Number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model)

        # Projection Layers for decoder input
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Multi-Head attention Layer
        self.masked_self_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout, is_causal=True
        )

        # Output Projection Layer
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

        self.layer_norm2 = nn.LayerNorm(d_model)

        # Feed Forward Layer
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        # Dropout Layer
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, decoder_input: torch.Tensor, decoder_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the decoder layer.

        Input shapes: (batch_size, seq_len, d_model)

        Args:
            decoder_input (torch.Tensor): Input tensor to the decoder layer.
            decoder_padding_mask (torch.Tensor): Padding mask for the decoder input.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Projection Layers for decoder input
        decoder_query = self.q_proj(decoder_input)
        decoder_key = self.k_proj(decoder_input)
        decoder_value = self.v_proj(decoder_input)

        # Multi-Head attention
        attention_output = self.masked_self_attention(
            query=decoder_query,
            key=decoder_key,
            value=decoder_value,
            attn_mask=decoder_padding_mask,
        )

        # Output projection layer
        attention_output = self.output_proj(attention_output)

        # Dropout, Residual, Layer Norm
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm1((attention_output + decoder_input))

        # Feed Forward
        ff_output = self.feed_forward(attention_output)

        # Dropout, Residual, Layer Norm
        ff_output = self.dropout(ff_output)
        ff_output = self.layer_norm2(ff_output + attention_output)

        return ff_output


class DecoderStack(nn.Module):
    """A stack of GPT-style decoder layers applied sequentially."""

    def __init__(
        self, d_model: int, d_ff: int, n_heads: int, n_layers: int, dropout: float = 0.0
    ):
        """Initialize the decoder stack.

        Args:
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed forward layer.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of decoder layers.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)]
        )

    def forward(
        self, decoder_input: torch.Tensor, decoder_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the decoder stack.

        Input shapes: (batch_size, seq_len, d_model)

        Args:
            decoder_input (torch.Tensor): Input tensor to the decoder stack.
            decoder_padding_mask (torch.Tensor): Padding mask for the decoder input.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        for layer in self.layers:
            decoder_input = layer(
                decoder_input=decoder_input, decoder_padding_mask=decoder_padding_mask
            )

        return decoder_input


if __name__ == "__main__":
    batch_size, seq_len, d_dim = 1, 3, 5
    decoder_input = torch.randn(batch_size, seq_len, d_dim)

    decoder_stack = DecoderStack(
        d_model=d_dim, d_ff=d_dim, n_heads=1, n_layers=1, dropout=0.0
    )
    decoder_output = decoder_stack(
        decoder_input=decoder_input, decoder_padding_mask=None
    )
    print(decoder_output)
    print(decoder_output.shape)
