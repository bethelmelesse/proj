from src.gpt.model import DecoderOnlyTransformer
from transformers import AutoTokenizer
import torch


def generate_next(input, model):
    with torch.no_grad():
        logits = model(input_tokens=input)

    next_logit = logits[:, -1:]
    next_token = torch.argmax(next_logit, dim=-1)
    return next_token


def generate(input, model_args, tokenizer_args, max_new_tokens):
    # Tokenizer
    tokenizer_path = tokenizer_args["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # Model
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=model_args["d_model"],
        d_ff=model_args["d_ff"],
        n_heads=model_args["n_heads"],
        n_layers=model_args["n_layers"],
        max_seq_len=model_args["max_seq_len"],
        dropout=model_args["dropout"],
    )

    model.eval()
    tokenized_input = tokenizer(
        input,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
        padding_side="left",
    )
    input_ids = tokenized_input["input_ids"]

    eos_token_id = tokenizer.eos_token_id

    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    )
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool)

    output = input_ids
    for _ in range(max_new_tokens):
        next_token = generate_next(input=output, model=model)  # (B, 1)
        next_token = next_token.masked_fill(finished.unsqueeze(1), pad_id)
        output = torch.cat((output, next_token), dim=-1)
        finished |= next_token.squeeze(1) == eos_token_id
        if finished.all():
            break

    decoded_seq = tokenizer.batch_decode(output, skip_special_tokens=True)

    inputs = input if isinstance(input, list) else [input]
    for src, pred in zip(inputs, decoded_seq):
        print(f"Starting sequence: {src}")
        print(f"Final Prediction: {pred}\n")

    return decoded_seq


if __name__ == "__main__":
    tokenizer_args = {"tokenizer_path": "t5-small"}

    model_args = {
        "d_model": 768,
        "d_ff": 3072,
        "n_heads": 12,
        "n_layers": 12,
        "max_seq_len": 512,
        "dropout": 0.1,
    }

    seq_list = [
        "Hello, my name",
        "The president of US",
        "Once upon a time there was a frog",
        "The brown",
        "Bethel",
    ]

    generate(
        input=seq_list,
        model_args=model_args,
        tokenizer_args=tokenizer_args,
        max_new_tokens=4,
    )
