"""End-to-end training pipeline for the decoder-only (GPT) transformer."""

from typing import Any

import torch
from datasets import Dataset, DatasetDict
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mlops.gpt.train_args import cli_args_parser
from mlops.gpt.trainer import Trainer
from src.gpt_with_kv_caching.model import DecoderOnlyTransformer
from src.utils.utils import print_header


def load_dataset(dataset_path: str, seed: int = 42) -> DatasetDict:
    """Load a line-delimited text file and split 70/20/10 into train/valid/test."""
    print_header(text="Loading and splitting dataset (Train/Valid/Test)")
    print(f"Source: {dataset_path}")
    with open(dataset_path, "r") as f:
        text = f.read().splitlines()

    dataset = Dataset.from_list([{"text": sentence} for sentence in text])

    # 70% train, 30% held out
    train_heldout = dataset.train_test_split(test_size=0.3, seed=seed)

    # Split the 30% so that valid is 20% of total and test is 10% of total.
    # test_size = 10/30 = 1/3 of the held-out portion.
    valid_test = train_heldout["test"].train_test_split(test_size=1 / 3, seed=seed)

    split_dataset_dict = DatasetDict(
        {
            "train": train_heldout["train"],
            "valid": valid_test["train"],
            "test": valid_test["test"],
        }
    )

    print(
        f"[Split sizes] Train: {len(split_dataset_dict['train'])}, "
        f"Valid: {len(split_dataset_dict['valid'])}, "
        f"Test: {len(split_dataset_dict['test'])}"
    )

    return split_dataset_dict


def tokenize_dataset(
    dataset_dict: DatasetDict, tokenizer_args: dict
) -> tuple[Dataset, Dataset, Dataset, PreTrainedTokenizerBase]:
    """Tokenize each split, returning torch-formatted HF Datasets that
    expose `input_ids` and `attention_mask` per row."""
    print_header(text="Tokenizing splits (Train/Valid/Test)")

    tokenizer_path = tokenizer_args["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def _tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            padding=tokenizer_args["padding"],
            max_length=tokenizer_args["max_length"],
            truncation=True,
        )

    tokenized_sets = {}
    for split in ["train", "valid", "test"]:
        tokenized = dataset_dict[split].map(
            _tokenize_batch,
            batched=True,
            batch_size=tokenizer_args["batch_size"],
        )
        tokenized.set_format("torch", columns=["input_ids", "attention_mask"])
        tokenized_sets[split] = tokenized

    print(f"Vocab size: {tokenizer.vocab_size:,}")
    print(f"Columns after tokenization: {tokenized_sets['train'].column_names}")
    return (
        tokenized_sets["train"],
        tokenized_sets["valid"],
        tokenized_sets["test"],
        tokenizer,
    )


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack tokenized rows into a batch with teacher-forcing labels.

    Returns:
        input_tokens:    [B, L]   full token sequence fed to the model.
        attention_mask:  [B, L]   1 for real tokens, 0 for pad.
        labels:          [B, L-1] next-token targets (input shifted left
            by one); pairs with `logits[:, :-1, :]` in the loss.
    """
    input_ids = torch.stack([row["input_ids"] for row in batch])
    attention_mask = torch.stack([row["attention_mask"] for row in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids[:, 1:],
    }


def create_dataloader(
    train_set: Dataset,
    valid_set: Dataset,
    test_set: Dataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap each split in a DataLoader with the teacher-forcing
    `collate_fn`. Train is shuffled; valid/test are not."""
    print_header(text="Building DataLoaders (Train/Valid/Test)")
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    print(
        f"Batch size: {batch_size} | "
        f"Train batches: {len(train_loader)}, "
        f"Valid batches: {len(valid_loader)}, "
        f"Test batches: {len(test_loader)}"
    )
    # Peek at one batch so the shapes are visible before model init.
    batch = next(iter(train_loader))
    print("First training batch (shape sanity check):")
    for key, value in batch.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else f"len={len(value)}"
        print(f"{key}: {shape}")

    return train_loader, valid_loader, test_loader


def build_model(tokenizer: AutoTokenizer, model_args: dict) -> DecoderOnlyTransformer:
    """Build the decoder-only transformer (GPT) and report its parameter count."""
    print_header(text="Initializing model")
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=model_args["d_model"],
        d_ff=model_args["d_ff"],
        n_heads=model_args["n_heads"],
        n_layers=model_args["n_layers"],
        max_seq_len=model_args["max_seq_len"],
        dropout=model_args["dropout"],
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model: d_model={model_args['d_model']}, d_ff={model_args['d_ff']}, "
        f"heads={model_args['n_heads']}, layers={model_args['n_layers']}, "
        f"max_seq_len={model_args['max_seq_len']}, dropout={model_args['dropout']}"
    )
    print(f"Params: {total:,} total ({trainable:,} trainable)")
    return model


def initialize_optimizer(model: nn.Module, optim_args: dict) -> optim.AdamW:
    """Build an AdamW optimizer over `model.parameters()` from `optim_args`."""
    print_header(text="Initializing optimizer")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=optim_args["lr"],
        betas=optim_args["betas"],
        eps=optim_args["eps"],
        weight_decay=optim_args["weight_decay"],
    )
    print(
        f"Optimizer: AdamW(lr={optim_args['lr']}, betas={optim_args['betas']}, "
        f"eps={optim_args['eps']}, weight_decay={optim_args['weight_decay']})"
    )
    return optimizer


def initialize_criterion(
    tokenizer: AutoTokenizer, criterion_args: dict
) -> nn.CrossEntropyLoss:
    """Build CrossEntropyLoss with `ignore_index=pad_id` so padded
    positions are excluded from the loss, and label smoothing from
    `criterion_args`."""
    print_header(text="Initializing criterion")
    # pad/eos collision check — `ignore_index=pad_id` would also mask EOS
    # tokens in targets if the two share an id (GPT-2-style tokenizers).
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id == eos_id:
        raise ValueError(
            f"pad_token_id ({pad_id}) equals eos_token_id ({eos_id}); "
            "ignore_index=pad_id would also mask real EOS tokens in targets. "
            "Configure a distinct pad token for this tokenizer."
        )
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
        label_smoothing=criterion_args["label_smoothing"],
    )
    print(
        f"Criterion: CrossEntropyLoss(ignore_index={pad_id}, "
        f"label_smoothing={criterion_args['label_smoothing']})"
    )
    return criterion


def train(args) -> None:
    """End-to-end training entry point."""
    # Load dataset
    dataset_dict = load_dataset(dataset_path=args["data_args"]["dataset_path"])

    # Tokenize dataset
    train_set, valid_set, test_set, tokenizer = tokenize_dataset(
        dataset_dict=dataset_dict, tokenizer_args=args["tokenizer_args"]
    )
    # build dataloader
    train_loader, valid_loader, test_loader = create_dataloader(
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        batch_size=args["dataloader_args"]["batch_size"],
    )

    # Build model, optimizer and criterion
    model = build_model(tokenizer=tokenizer, model_args=args["model_args"])
    optimizer = initialize_optimizer(model=model, optim_args=args["optim_args"])
    criterion = initialize_criterion(
        tokenizer=tokenizer, criterion_args=args["criterion_args"]
    )

    print_header(text="Initial Scheduler")
    scheduler = ExponentialLR(optimizer, gamma=args["scheduler_args"]["gamma"])

    device = args["training_args"]["device"]
    model.to(device)

    # Keys here must match what Trainer.__init__ reads (note: "val_loader").
    trainer_args = {
        "train_loader": train_loader,
        "val_loader": valid_loader,
        "test_loader": test_loader,
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
        "batch_size": args["dataloader_args"]["batch_size"],
        **args["training_args"],
        **args["dataloader_args"],
        **args["optim_args"],
        **args["mlflow_args"],
    }

    trainer = Trainer(trainer_args)
    trainer.train()


if __name__ == "__main__":
    args = cli_args_parser()

    train(args)

    print()
