import torch.nn as nn
from src.encoder_decoder_transformer.encoder import EncoderStack
from src.encoder_decoder_transformer.decoder import DecoderStack
from src.utils.positional_encoder import PositionalEncoder
import torch


class EncoderDecoderTransformer(nn.Module):
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
        self.encoder_embed_layer = PositionalEncoder(
            vocab_size, d_model, max_seq_len, dropout
        )
        self.decoder_embed_layer = PositionalEncoder(
            vocab_size, d_model, max_seq_len, dropout
        )

        # Encoder and Decoder Stacks
        self.encoder = EncoderStack(d_model, d_ff, n_heads, n_layers, dropout)
        self.decoder = DecoderStack(d_model, d_ff, n_heads, n_layers, dropout)

        # Final output projection layer
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        source_tokens,
        target_tokens,
        source_attention_mask=None,
        target_attention_mask=None,
    ):
        """Forward pass for the encoder decoder transformer.

        Args:
            source_tokens (torch.Tensor): Input tensor to the encoder layer.
            target_tokens (torch.Tensor): Input tensor to the decoder layer.
            source_attention_mask (torch.Tensor): Padding mask for the encoder input.
            target_attention_mask (torch.Tensor): Padding mask for the decoder input.
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
        source_embed = self.encoder_embed_layer(source_tokens)
        target_embed = self.decoder_embed_layer(target_tokens)

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
    vocab_size, d_model, d_ff, n_heads, n_layers, max_seq_len, dropout = (
        10,
        5,
        10,
        1,
        2,
        12,
        0.1,
    )
    # create a encoder decoder transformer
    encoder_decoder_transformer = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

    # create a test tensor
    source_tokens = torch.randint(0, vocab_size, (1, max_seq_len))
    target_tokens = torch.randint(0, vocab_size, (1, max_seq_len))

    # forward pass
    output = encoder_decoder_transformer(
        source_tokens=source_tokens, target_tokens=target_tokens
    )
    print(output.shape)
    print(output)
