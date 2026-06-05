"""Positional Encoder."""

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """Token embedding with scaling plus learned absolute positional embedding."""

    def __init__(
        self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.0
    ):
        """Initialize the positional encoder.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model.
            max_seq_len (int): Maximum sequence length.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Token Embedding Layer
        self.token_embedder = nn.Embedding(vocab_size, d_model)

        # Learned Positional Embedding Layer
        self.pos_embedder = nn.Embedding(max_seq_len, d_model)

        # Scale token embeddings by sqrt(d_model), as in Vaswani et al. (2017)
        self.scale = d_model**0.5

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for the positional encoder.

        Args:
            input_ids (torch.Tensor): Input tensor.
                Shape: (batch_size, seq_len)
            attention_mask (torch.Tensor, optional): Mask of shape
                (batch_size, total_seq_len); 1 for real tokens, 0 for padding.
                Used to derive positions so left padding doesn't shift real
                tokens. When None, positions default to 0..seq_len-1 (no padding
                assumed). Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
                Shape: (batch_size, seq_len, d_model)
        """
        # Sequence length of the input
        _, seq_len = input_ids.size()

        # Look up token embeddings and scale - (batch_size, seq_len, d_model)
        token_embeddings = self.token_embedder(input_ids) * self.scale

        if attention_mask is None:
            # No mask: assume no padding, positions are just 0..seq_len-1 - (1, seq_len)
            positional_ids = torch.arange(
                0, seq_len, device=input_ids.device
            ).unsqueeze(0)
        else:
            # Mask-aware positions so left padding doesn't shift real tokens:
            # cumsum-1 makes the first real token position 0; clamp keeps pads >= 0.
            positional_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
            # During cached decode the mask spans the full sequence but the input
            # is a single new token, so keep only the newest seq_len positions.
            positional_ids = positional_ids[:, -seq_len:]

        # Look up positional embeddings - (1, seq_len, d_model)
        pos_embeddings = self.pos_embedder(positional_ids)

        # Add token and positional embeddings (broadcasts over the batch)
        embeddings = token_embeddings + pos_embeddings

        # Apply dropout
        embeddings = self.dropout(embeddings)

        return embeddings  # (batch_size, seq_len, d_model)


if __name__ == "__main__":
    vocab_size, d_model, max_seq_len = 10, 4, 12
    # create a positional encoder
    positional_encoder = PositionalEncoder(
        vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len
    )

    # create a test tensor
    tokens = torch.randint(0, vocab_size, (1, max_seq_len))

    # forward pass
    embeddings = positional_encoder(tokens)
    print(embeddings.shape)
    print(embeddings)
    print()
