import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from src.data_utils.tokenizer_utils import Tokenizer
from src.gpt.model import DecoderOnlyTransformer
from torch import optim, nn
from tqdm import tqdm
from src.utils.utils import print_header


def load_dataset(data_args: dict, seed: int = 42) -> DatasetDict:
    """Load a line-delimited text file and split 70/20/10 into train/valid/test."""
    print_header(text="Loading and splitting dataset (Train/Valid/Test)")
    print(f"Source: {data_args['dataset_path']}")
    with open(data_args["dataset_path"], "r") as f:
        text = f.read().splitlines()
    dataset = Dataset.from_list([{"text": sentence} for sentence in text])

    # 70% train, 30% held out
    train_heldout = dataset.train_test_split(test_size=0.3, seed=seed)

    # Split the 30% so that valid is 20% of total and test is 10% of total.
    # test_size = 10/30 = 1/3 of the held-out portion.
    valid_test = train_heldout["test"].train_test_split(test_size=1 / 3, seed=seed)

    split_dataset = DatasetDict(
        {
            "train": train_heldout["train"],
            "valid": valid_test["train"],
            "test": valid_test["test"],
        }
    )

    print(
        f"[Split sizes] Train: {len(split_dataset['train'])}, "
        f"Valid: {len(split_dataset['valid'])}, Test: {len(split_dataset['test'])}"
    )

    return split_dataset


def tokenize_dataset(
    dataset: DatasetDict, tokenizer: Tokenizer
) -> tuple[Dataset, Dataset, Dataset]:
    print_header(text="Tokenizing splits (Train/Valid/Test)")
    train_set = tokenizer(dataset["train"])
    valid_set = tokenizer(dataset["valid"])
    test_set = tokenizer(dataset["test"])
    print(f"Vocab size: {tokenizer.get_vocab_size():,}")
    print(f"Columns after tokenization: {train_set.column_names}")
    return train_set, valid_set, test_set


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([row["input_ids"] for row in batch])
    attention_mask = torch.stack([row["attention_mask"] for row in batch])

    return {
        "input_tokens": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids[:, 1:],
    }


def create_dataloader(
    train_set: Dataset,
    valid_set: Dataset,
    test_set: Dataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
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


def initialize_model(tokenizer: Tokenizer, model_args: dict) -> DecoderOnlyTransformer:
    print_header(text="Initializing model")
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.get_vocab_size(),
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
    tokenizer: Tokenizer, criterion_args: dict
) -> nn.CrossEntropyLoss:
    print_header(text="Initializing criterion")
    # pad/eos collision check — `ignore_index=pad_id` would also mask EOS
    # tokens in targets if the two share an id (GPT-2-style tokenizers).
    pad_id = tokenizer.tokenizer.pad_token_id
    eos_id = tokenizer.tokenizer.eos_token_id
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


def train_epoch(
    epoch: int,
    loader: DataLoader,
    model: nn.Module,
    optimizer: optim,
    criterion: nn.Module,
    device: str,
) -> None:
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch} [Train]"):
        # Forward pass
        logits = model(
            input_tokens=batch["input_tokens"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        labels = batch["labels"].to(device)

        # Reshape logit and labels
        logits_shifted = logits[:, 1:, :]
        batch_size, seq_len, vocab_size = logits_shifted.shape
        logits_shifted = logits_shifted.reshape(batch_size * seq_len, vocab_size)
        labels = labels.reshape(batch_size * seq_len)

        # Calculate loss
        loss = criterion(logits_shifted, labels)
        epoch_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_batches += 1

    avg_epoch_loss = epoch_loss / num_batches
    return avg_epoch_loss


def eval_epoch(
    epoch: int,
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: str,
    mode: str,
) -> None:
    model.eval()
    epoch_loss = 0.0
    num_batches = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch} [{mode}]"):
        # Forward pass
        logits = model(
            input_tokens=batch["input_tokens"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        labels = batch["labels"].to(device)

        # Reshape logit and labels
        logits_shifted = logits[:, 1:, :]
        batch_size, seq_len, vocab_size = logits_shifted.shape
        logits_shifted = logits_shifted.reshape(batch_size * seq_len, vocab_size)
        labels = labels.reshape(batch_size * seq_len)

        # Calculate loss
        loss = criterion(logits_shifted, labels)
        epoch_loss += loss

        num_batches += 1

    avg_epoch_loss = epoch_loss / num_batches
    return avg_epoch_loss


def train(
    data_args: dict,
    tokenizer_args: dict,
    dataloader_args: dict,
    model_args: dict,
    optim_args: dict,
    criterion_args: dict,
    training_args: dict,
) -> None:
    dataset = load_dataset(data_args)
    tokenizer = Tokenizer(
        tokenizer_path=tokenizer_args["tokenizer_path"],
        batch_size=tokenizer_args["batch_size"],
        padding=tokenizer_args["padding"],
        max_length=tokenizer_args["max_length"],
    )
    train_set, valid_set, test_set = tokenize_dataset(dataset, tokenizer)
    train_loader, valid_loader, test_loader = create_dataloader(
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        batch_size=dataloader_args["batch_size"],
    )

    model = initialize_model(tokenizer=tokenizer, model_args=model_args)
    optimizer = initialize_optimizer(model=model, optim_args=optim_args)
    criterion = initialize_criterion(tokenizer=tokenizer, criterion_args=criterion_args)

    epochs = training_args["epochs"]
    device = training_args["device"]
    model.to(device)

    best_model = model
    best_val_loss = float("inf")
    best_epoch = 0
    epoch_one_loss = None

    # Initial Validation
    print_header(text="Initital Validation")
    init_loss = eval_epoch(
        epoch=0,
        loader=test_loader,
        model=model,
        criterion=criterion,
        device=device,
        mode="Init",
    )
    print(f"Init Loss: {init_loss:.4f}")

    print_header(text="Started Training")
    for epoch in range(1, epochs + 1):
        # Training
        train_loss = train_epoch(
            epoch=epoch,
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        epoch_one_loss = train_loss if epoch == 1 else None

        # Validation
        val_loss = eval_epoch(
            epoch=epoch,
            loader=valid_loader,
            model=model,
            criterion=criterion,
            device=device,
            mode="Val",
        )
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss <= best_val_loss:
            best_model = model
            best_epoch = epoch
            best_val_loss = val_loss

    print(
        f"Initial Loss [epoch 0]={init_loss:.4f}"
        f"\nLoss [epoch 1]={epoch_one_loss}"
        f"\nFinal Loss [epoch {epoch}]={val_loss:.4f}"
    )
    print(f"Best model: loss={best_val_loss:.4f} at Epoch {best_epoch}")

    # Final test
    print_header(text="Final Validation")
    test_loss = eval_epoch(
        epoch=best_epoch,
        loader=test_loader,
        model=best_model,
        criterion=criterion,
        device=device,
        mode="Test",
    )
    print(f"Test Loss: {test_loss:.4}")


if __name__ == "__main__":
    data_args = {"dataset_path": "data/dataset.txt"}
    tokenizer_args = {
        "tokenizer_path": "t5-small",
        "batch_size": 100,
        "padding": "max_length",
        "max_length": 128,
    }
    dataloader_args = {"batch_size": 2}
    model_args = {
        "d_model": 512,
        "d_ff": 2048,
        "n_heads": 2,
        "n_layers": 6,
        "max_seq_len": 128,
        "dropout": 0.0,
    }  # EncoderDecoderTransformer Model
    optim_args = {
        "lr": 1e-4,
        "betas": (0.9, 0.98),
        "eps": 1e-9,
        "weight_decay": 0.0,
    }  # AdamW
    # betas=(0.9, 0.98) and eps=1e-9 are from "Attention Is All You Need".
    # weight_decay=0 keeps AdamW equivalent to paper-Adam; raise to 0.01 to
    # match common modern practice.

    criterion_args = {"label_smoothing": 0.1}

    training_args = {"epochs": 10, "device": "cpu"}

    train(
        data_args=data_args,
        tokenizer_args=tokenizer_args,
        dataloader_args=dataloader_args,
        model_args=model_args,
        optim_args=optim_args,
        criterion_args=criterion_args,
        training_args=training_args,
    )

    print()
