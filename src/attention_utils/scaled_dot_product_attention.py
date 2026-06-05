"""Scaled-dot product attention."""

import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    attn_mask: torch.Tensor = None,
    dropout: float = 0.0,
) -> torch.Tensor:
    """Scaled Dot Product Attention.

    Shape of query, key, value: (batch_size, n_heads, seq_len, d_dim)
        batch_size: number of samples in the batch
        n_heads: number of attention heads
        seq_len: length of the sequence
        d_dim: dimension of the tensor

    Args:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.
        is_causal (bool, optional): Check if the attention is causal. Defaults to False.
            This is used to mask future tokens in the sequence for decoder.
        attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            This is used to mask the padding tokens in the sequence.
        dropout (float, optional): Dropout probability applied to the attention
            weights after softmax. Defaults to 0.0 (no dropout).

    Returns:
        torch.Tensor: Attention output tensor.
            Shape: (batch_size, n_heads, seq_len, d_dim)
    """
    # Get shape
    _, _, seq_len, d_dim = query.size()

    # Transpose Key to batch_size, n_heads, seq_len, d_dim
    key_transpose = torch.transpose(key, dim0=2, dim1=3)

    # Calculate attention weights - (Qk^T)/sqrt(dk)
    # resulting shape: batch_size, n_heads, seq_len, seq_len
    qk_score = query @ key_transpose
    scaled_score = qk_score / math.sqrt(d_dim)

    # Apply padding mask - fill with -inf
    if attn_mask is not None:
        scaled_score = scaled_score.masked_fill(attn_mask, float("-inf"))

    # Apply causal mask for decoder
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=query.device, dtype=bool), diagonal=1
        )
        scaled_score = scaled_score.masked_fill(causal_mask, float("-inf"))

    # Softmax
    attention_weights = F.softmax(scaled_score, dim=-1)

    # Dropout
    attention_weights = F.dropout(attention_weights, p=dropout)

    # compute weighted sum of value - (batch_size, n_heads, seq_len, head_dim)
    attention_output = attention_weights @ value

    return attention_output


if __name__ == "__main__":
    batch_size, n_heads, seq_len, d_dim = 1, 1, 3, 5
    query = torch.randn(batch_size, n_heads, seq_len, d_dim)
    key = torch.randn(batch_size, n_heads, seq_len, d_dim)
    value = torch.randn(batch_size, n_heads, seq_len, d_dim)

    attention_output = scaled_dot_product_attention(
        query, key, value, is_causal=True, attn_mask=None
    )
    print(attention_output)
    print(attention_output.shape)
