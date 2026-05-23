from __future__ import annotations

import torch
import torch.nn as nn

from src.attention_utils.scaled_dot_product_attention import (
    scaled_dot_product_attention,
)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.0, is_causal: bool = False
    ):
        """Initialize the Multi-Head Attention Layer

        Args:
            d_model (int): The dimension of the model.
            n_heads (int, optional): The number of attention heads.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            is_causal (bool, optional): Check if the attention is causal.
                This is used to mask future tokens in the sequence for decoder. Defaults to False.
        """
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.is_causal = is_causal

        self.head_dim = d_model // n_heads

        # Output Projection Layer
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, attn_mask: torch.Tensor | None = None):
        """Scaled Dot Product Attention.

        Shape of query, key, value: (batch_size, seq_len, d_dim)
            batch_size: number of samples in the batch
            seq_len: length of the sequence
            d_dim: dimension of the tensor

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
                This is used to mask the padding tokens in the sequence.

        Returns:
            torch.Tensor: Attention output tensor.
                Shape: (batch_size, seq_len, d_dim)
        """
        # Get Size
        batch_size, q_len, _ = query.size()
        kv_len = key.size(1)

        # Split heads - (batch_size, seq_len, n_heads, head_dim)
        query = query.view(batch_size, q_len, self.n_heads, self.head_dim)
        key = key.view(batch_size, kv_len, self.n_heads, self.head_dim)
        value = value.view(batch_size, kv_len, self.n_heads, self.head_dim)

        # Transpose for multi-head attention - (batch_size, n_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply attention - (batch_size, n_heads, q_len, head_dim)
        dropout = self.dropout if self.training else 0.0
        attention_output = scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=self.is_causal,
            attn_mask=attn_mask,
            dropout=dropout,
        )

        # Reshape for output - (batch_size, q_len, d_model)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, q_len, self.d_model)
        )

        # Output projection layer
        attention_output = self.output_proj(attention_output)

        return attention_output


if __name__ == "__main__":
    batch_size, seq_len, d_dim = 1, 3, 5
    query = torch.randn(batch_size, seq_len, d_dim)
    key = torch.randn(batch_size, seq_len, d_dim)
    value = torch.randn(batch_size, seq_len, d_dim)

    is_causal = True
    attn_mask = None

    multihead_attention = MultiHeadAttention(
        d_model=d_dim, n_heads=1, dropout=0.0, is_causal=is_causal
    )
    attention_output = multihead_attention(query, key, value, attn_mask)
    print(attention_output)
    print(attention_output.shape)
