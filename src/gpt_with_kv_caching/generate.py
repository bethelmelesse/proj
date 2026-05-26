from transformers import AutoTokenizer
import torch
from src.gpt_with_kv_caching.model import DecoderOnlyTransformer


def generate_next(input_ids, attention_mask, model, kv_cache, use_kv_cache):
    with torch.no_grad():
        if use_kv_cache:
            logits, kv_cache = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

    next_logit = logits[:, -1:]
    next_token = torch.argmax(next_logit, dim=-1)
    return next_token, kv_cache


def generate(input, model_args, tokenizer_args, max_new_tokens, device):
    # Initialize Tokenizer
    tokenizer_path = tokenizer_args["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    eos_token_id = tokenizer.eos_token_id
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    )

    # Build Model
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=model_args["d_model"],
        d_ff=model_args["d_ff"],
        n_heads=model_args["n_heads"],
        n_layers=model_args["n_layers"],
        max_seq_len=model_args["max_seq_len"],
        dropout=model_args["dropout"],
        use_kv_cache=model_args["use_kv_cache"],
    )
    model.to(device)
    model.eval()

    # Tokenize inputs
    tokenized_input = tokenizer(
        input,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
        padding_side="left",
    )
    # Get model inputs
    input_ids = tokenized_input["input_ids"]
    attention_mask = tokenized_input["attention_mask"]

    ones_col = torch.ones((input_ids.size(0), 1), dtype=attention_mask.dtype)

    use_kv_cache = model_args["use_kv_cache"]
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=device)

    if use_kv_cache:
        # Prefill: full prompt seeds the cache and produces the first new token.
        next_token, kv_cache = generate_next(
            input_ids=input_ids,
            attention_mask=attention_mask,
            model=model,
            kv_cache=None,
            use_kv_cache=True,
        )
        finished = torch.logical_or(finished, next_token.squeeze(-1) == eos_token_id)
        attention_mask = torch.cat([attention_mask, ones_col], dim=-1)
        output = torch.cat((input_ids, next_token), dim=-1)
        # Decode: feed only the new token each step.
        for _ in range(max_new_tokens - 1):
            if finished.all():
                break
            next_token, kv_cache = generate_next(
                input_ids=next_token,
                attention_mask=attention_mask,
                model=model,
                kv_cache=kv_cache,
                use_kv_cache=True,
            )
            next_token[finished] = pad_id
            finished = torch.logical_or(
                finished, next_token.squeeze(-1) == eos_token_id
            )
            output = torch.cat((output, next_token), dim=-1)
            attention_mask = torch.cat([attention_mask, ones_col], dim=-1)
    else:
        # No cache: re-run the full growing sequence each step.
        output = input_ids
        for _ in range(max_new_tokens):
            if finished.all():
                break
            next_token, _ = generate_next(
                input_ids=output,
                attention_mask=attention_mask,
                model=model,
                kv_cache=None,
                use_kv_cache=False,
            )
            next_token[finished] = pad_id
            finished = torch.logical_or(
                finished, next_token.squeeze(-1) == eos_token_id
            )
            output = torch.cat((output, next_token), dim=-1)
            attention_mask = torch.cat([attention_mask, ones_col], dim=-1)

    decoded_seq = tokenizer.batch_decode(output, skip_special_tokens=True)

    output = input if isinstance(input, list) else [input]
    for src, pred in zip(output, decoded_seq):
        print(f"Starting sequence: {src}")
        print(f"Final Prediction: {pred}\n")
    return output


if __name__ == "__main__":
    tokenizer_args = {"tokenizer_path": "t5-small"}

    model_args = {
        "d_model": 768,
        "d_ff": 3072,
        "n_heads": 12,
        "n_layers": 2,
        "max_seq_len": 512,
        "dropout": 0.1,
        "use_kv_cache": True,
    }

    seq_list = [
        "Hello, my name",
        # "The president of US",
        "Once upon a time there was a frog",
        # "The brown",
        # "Bethel",
    ]

    generate(
        input=seq_list,
        model_args=model_args,
        tokenizer_args=tokenizer_args,
        max_new_tokens=4,
        device="cpu",
    )
