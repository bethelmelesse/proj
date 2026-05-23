"""Decoder Layer and Stack for the Transformer model."""

import torch.nn as nn
from src.attention_utils.multihead_attention import MultiHeadAttention
import torch


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.0):
        """Initialize the decoder layer.

        Args:
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed forward layer.
            n_heads (int): Number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()

        # Projection Layers for decoder input
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Projection Layers for cross-attention
        # Q comes from the decoder side; K/V come from the encoder output.
        self.q_proj_cross = nn.Linear(d_model, d_model, bias=False)
        self.k_proj_encoder = nn.Linear(d_model, d_model, bias=False)
        self.v_proj_encoder = nn.Linear(d_model, d_model, bias=False)

        # Masked Self-Attention Layer
        self.masked_self_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout, is_causal=True
        )

        self.layer_norm1 = nn.LayerNorm(d_model)

        # Cross-Attention Layer
        self.cross_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout, is_causal=False
        )

        self.layer_norm2 = nn.LayerNorm(d_model)

        # Feed Forward Layer
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.layer_norm3 = nn.LayerNorm(d_model)

        # Dropout Layer
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        encoder_output: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        decoder_padding_mask: torch.Tensor,
    ):
        """Forward pass for the decoder layer.

        Input shapes: batch_size, seq_len, d_dim

        Args:
            encoder_output (torch.Tensor): Output tensor from the encoder layer.
            decoder_input (torch.Tensor): Input tensor to the decoder layer.
            encoder_padding_mask (torch.Tensor): Padding mask for the encoder input (used in cross-attention).
            decoder_padding_mask (torch.Tensor): Padding mask for the decoder input.

        Returns:
            torch.Tensor: Output tensor from the decoder layer.
        """
        # Projection Layers for decoder input for masked self-attention
        decoder_query = self.q_proj(decoder_input)
        decoder_key = self.k_proj(decoder_input)
        decoder_value = self.v_proj(decoder_input)

        # Masked Self-Attention
        masked_attention_output = self.masked_self_attention(
            query=decoder_query,
            key=decoder_key,
            value=decoder_value,
            attn_mask=decoder_padding_mask,
        )

        # Dropout, Residual, Layer Norm
        masked_attention_output = self.dropout(masked_attention_output)
        masked_attention_output = self.layer_norm1(
            (masked_attention_output + decoder_input)
        )

        # Cross-Attention: Q from decoder, K/V from encoder
        decoder_cross_query = self.q_proj_cross(masked_attention_output)
        encoder_key = self.k_proj_encoder(encoder_output)
        encoder_value = self.v_proj_encoder(encoder_output)
        cross_attention_output = self.cross_attention(
            query=decoder_cross_query,
            key=encoder_key,
            value=encoder_value,
            attn_mask=encoder_padding_mask,
        )

        # Dropout, Residual, Layer Norm
        cross_attention_output = self.dropout(cross_attention_output)
        cross_attention_output = self.layer_norm2(
            (cross_attention_output + masked_attention_output)
        )

        # Feed Forward
        ff_output = self.feed_forward(cross_attention_output)

        # Dropout, Residual, Layer Norm
        ff_output = self.dropout(ff_output)
        ff_output = self.layer_norm3((ff_output + cross_attention_output))

        return ff_output


class DecoderStack(nn.Module):
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
        self,
        encoder_output: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        decoder_padding_mask: torch.Tensor,
    ):
        """Forward pass for the decoder stack.

        Input shapes: batch_size, seq_len, d_dim

        Args:
            encoder_output (torch.Tensor): Output tensor from the encoder layer.
            decoder_input (torch.Tensor): Input tensor to the decoder layer.
            encoder_padding_mask (torch.Tensor): Padding mask for the encoder input (used in cross-attention).
            decoder_padding_mask (torch.Tensor): Padding mask for the decoder input.

        Returns:
            torch.Tensor: Output tensor from the decoder layer.
        """
        for layer in self.layers:
            decoder_input = layer(
                encoder_output=encoder_output,
                decoder_input=decoder_input,
                encoder_padding_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
            )

        return decoder_input


if __name__ == "__main__":
    batch_size, seq_len, d_dim = 1, 3, 5
    encoder_output = torch.randn(batch_size, seq_len, d_dim)
    decoder_input = torch.randn(batch_size, seq_len, d_dim)
    decoder_padding_mask = None
    encoder_padding_mask = None

    decoder_stack = DecoderStack(
        d_model=d_dim, d_ff=d_dim, n_heads=1, n_layers=1, dropout=0.0
    )
    decoder_output = decoder_stack(
        encoder_output=encoder_output,
        decoder_input=decoder_input,
        encoder_padding_mask=encoder_padding_mask,
        decoder_padding_mask=decoder_padding_mask,
    )
    print(decoder_output)
    print(decoder_output.shape)
