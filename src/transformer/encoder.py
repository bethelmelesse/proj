"""Encoder Layer and Stack for the Transformer model."""

import torch
import torch.nn as nn

from src.attention_utils.multihead_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """A single Transformer encoder block (self-attention + feed-forward)."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.0):
        """Initialize the encoder layer.

        Args:
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed forward layer.
            n_heads (int): Number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        # Projection Layers for query, key, value
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Multi-Head attention Layer
        self.self_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout, is_causal=False
        )

        # Output Projection Layer
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

        self.layer_norm1 = nn.LayerNorm(d_model)

        # Feed Forward Layer
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.layer_norm2 = nn.LayerNorm(d_model)

        # Dropout Layer
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, encoder_input: torch.Tensor, encoder_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the encoder layer.

        Input shapes: (batch_size, seq_len, d_model)

        Args:
            encoder_input (torch.Tensor): Input tensor to the encoder layer.
            encoder_padding_mask (torch.Tensor): Padding mask for the encoder input.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Projection Layers
        query = self.q_proj(encoder_input)
        key = self.k_proj(encoder_input)
        value = self.v_proj(encoder_input)

        # Multi-Head attention
        attention_output = self.self_attention(
            query, key, value, attn_mask=encoder_padding_mask
        )

        # Output projection layer
        attention_output = self.output_proj(attention_output)

        # Dropout, Residual, Layer Norm
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm1((attention_output + encoder_input))

        # Feed Forward
        ff_output = self.feed_forward(attention_output)

        # Dropout, Residual, Layer Norm
        ff_output = self.dropout(ff_output)
        ff_output = self.layer_norm2((ff_output + attention_output))

        return ff_output


class EncoderStack(nn.Module):
    """A stack of Transformer encoder layers applied sequentially."""

    def __init__(
        self, d_model: int, d_ff: int, n_heads: int, n_layers: int, dropout: float = 0.0
    ):
        """Initialize the encoder stack.

        Args:
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed forward layer.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of encoder layers.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)]
        )

    def forward(
        self, encoder_input: torch.Tensor, encoder_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the encoder stack.

        Input shapes: (batch_size, seq_len, d_model)

        Args:
            encoder_input (torch.Tensor): Input tensor to the encoder stack.
            encoder_padding_mask (torch.Tensor): Padding mask for the encoder input.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        for layer in self.layers:
            encoder_input = layer(encoder_input, encoder_padding_mask)

        return encoder_input


if __name__ == "__main__":
    batch_size, seq_len, d_dim = 1, 3, 5
    encoder_input = torch.randn(batch_size, seq_len, d_dim)
    encoder_padding_mask = None

    encoder_stack = EncoderStack(
        d_model=d_dim, d_ff=d_dim, n_heads=1, n_layers=1, dropout=0.0
    )
    encoder_output = encoder_stack(encoder_input, encoder_padding_mask)
    print(encoder_output)
    print(encoder_output.shape)
