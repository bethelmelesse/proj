import copy
import os
from typing import Any, Literal

import mlflow
import torch
from datasets import Dataset, DatasetDict
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mlops.utils.train_utils import compute_grad_norm
from src.gpt_with_kv_caching.model import DecoderOnlyTransformer
from src.utils.utils import print_header


def load_dataset(data_args: dict[str, Any], seed: int = 42) -> DatasetDict:
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


def train_epoch(
    epoch: int,
    loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer | None,
    criterion: nn.Module,
    device: str,
    global_step: int | None,
    mode: Literal["Train", "Val", "Init", "Test"],
) -> tuple[float, int | None]:
    """Run one pass over `loader` in the given mode.

    Backward + optimizer step happen only when mode == "Train".
    Step-level metrics are logged for "Train" and "Val"; "Init" and
    "Test" log only the epoch-level loss. `global_step` advances only
    in "Train" mode.

    Args:
        optimizer:   Required when mode=="Train"; pass None otherwise.
        global_step: Training-step counter. Pass None for "Init"/"Test".

    Returns:
        `(avg_epoch_loss, updated_global_step)`. The counter is returned
        unchanged for non-"Train" modes.
    """
    model.train() if mode == "Train" else model.eval()

    epoch_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch} [{mode}]"):
        if mode == "Train":
            optimizer.zero_grad()

        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        labels = batch["labels"].to(device)

        # Drop the last logit (no next-token target) so logits[:, t] is
        # paired with the token at position t+1, which `labels` already holds.
        logits_shifted = logits[:, :-1, :]
        batch_size, seq_len, vocab_size = logits_shifted.shape
        logits_shifted = logits_shifted.reshape(batch_size * seq_len, vocab_size)
        labels = labels.reshape(batch_size * seq_len)

        loss = criterion(logits_shifted, labels)
        epoch_loss += loss.item()

        if mode == "Train":
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            )
            grad_norm2 = compute_grad_norm(model=model)
            optimizer.step()

        # Step-level logging: per-batch loss for Train/Val; LR + step
        # increment only for Train so the x-axis tracks gradient updates.
        if mode in ("Train", "Val"):
            mlflow.log_metrics(
                {f"step_level/{mode}_loss": loss.item()}, step=global_step
            )
        if mode == "Train":
            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metrics(
                {"step_level/lr": current_lr, "step_level/grad_norm": grad_norm},
                step=global_step,
            )
            global_step += 1

        num_batches += 1

    avg_epoch_loss = epoch_loss / num_batches

    print(f"{mode} Loss: {avg_epoch_loss:.4f}")
    mlflow.log_metrics({f"epoch_level/{mode}_loss": avg_epoch_loss}, step=epoch)
    return avg_epoch_loss, global_step


def train(
    data_args: dict,
    tokenizer_args: dict,
    dataloader_args: dict,
    model_args: dict,
    optim_args: dict,
    criterion_args: dict,
    scheduler_args: dict,
    training_args: dict,
    mlflow_args: dict,
) -> None:
    """End-to-end training entry point."""
    dataset_dict = load_dataset(data_args=data_args)

    train_set, valid_set, test_set, tokenizer = tokenize_dataset(
        dataset_dict=dataset_dict, tokenizer_args=tokenizer_args
    )
    train_loader, valid_loader, test_loader = create_dataloader(
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        batch_size=dataloader_args["batch_size"],
    )

    model = build_model(tokenizer=tokenizer, model_args=model_args)
    optimizer = initialize_optimizer(model=model, optim_args=optim_args)
    criterion = initialize_criterion(tokenizer=tokenizer, criterion_args=criterion_args)

    print_header(text="Initial Scheduler")
    scheduler = ExponentialLR(optimizer, gamma=scheduler_args["gamma"])

    epochs = training_args["epochs"]
    device = training_args["device"]
    checkpoint_dir = training_args["checkpoint_dir"]
    model.to(device)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Training with MLflow logging
    print_header(text="Setting up mlflow")
    mlflow.set_tracking_uri(mlflow_args["tracker_url"])
    mlflow.set_experiment(mlflow_args["experiment_name"])
    with mlflow.start_run(run_name=mlflow_args["run_name"]):
        # Enable system metrics logging
        mlflow.enable_system_metrics_logging()
        mlflow.log_params(
            {
                "epochs": training_args["epochs"],
                "batch_size": dataloader_args["batch_size"],
                "lr": optim_args["lr"],
            }
        )

        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")
        best_epoch = 0
        epoch_one_loss = None
        global_step = 0

        # Initial Validation
        print_header(text="Initial Validation")
        with torch.no_grad():
            init_loss, _ = train_epoch(
                epoch=0,
                loader=valid_loader,
                model=model,
                criterion=criterion,
                device=device,
                mode="Init",
                global_step=None,
                optimizer=None,
            )

        print_header(text="Started Training")
        for epoch in range(1, epochs + 1):
            # Training
            train_loss, global_step = train_epoch(
                epoch=epoch,
                loader=train_loader,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                global_step=global_step,
                mode="Train",
            )
            if epoch == 1:
                epoch_one_loss = train_loss

            # Validation
            with torch.no_grad():
                val_loss, _ = train_epoch(
                    epoch=epoch,
                    loader=valid_loader,
                    model=model,
                    criterion=criterion,
                    device=device,
                    mode="Val",
                    global_step=global_step,
                    optimizer=None,
                )

            scheduler.step()

            if val_loss <= best_val_loss:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_val_loss = val_loss

                if checkpoint_dir:
                    ckpt_path = os.path.join(checkpoint_dir, "best.pt")
                    torch.save(
                        {
                            "epoch": best_epoch,
                            "model_state_dict": best_state,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": best_val_loss,
                        },
                        ckpt_path,
                    )
                    print(f"Saved best checkpoint to {ckpt_path}")
                    mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

        print(
            f"\nInitial Loss [epoch 0]={init_loss:.4f}"
            f"\nLoss [epoch 1]={epoch_one_loss}"
            f"\nFinal Loss [epoch {epoch}]={val_loss:.4f}"
            f"\nBest Loss [epoch {best_epoch}]={best_val_loss:.4f}"
        )

        # Final test — restore the best validation checkpoint first.
        model.load_state_dict(best_state)
        print_header(text="Final Validation")
        with torch.no_grad():
            test_loss, _ = train_epoch(
                epoch=best_epoch,
                loader=test_loader,
                model=model,
                criterion=criterion,
                device=device,
                mode="Test",
                global_step=None,
                optimizer=None,
            )


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
    }  # DecoderOnlyTransformer Model
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

    exp_scheduler_args = {"gamma": 0.9}
    scheduler_args = {**exp_scheduler_args}

    training_args = {"epochs": 10, "device": "cpu", "checkpoint_dir": None}

    mlflow_args = {
        "experiment_name": "GPT (Decoder Only)",
        "run_name": "test: grad norm",
        "tracker_url": "http://localhost:5000",
    }

    train(
        data_args=data_args,
        tokenizer_args=tokenizer_args,
        dataloader_args=dataloader_args,
        model_args=model_args,
        optim_args=optim_args,
        criterion_args=criterion_args,
        scheduler_args=scheduler_args,
        training_args=training_args,
        mlflow_args=mlflow_args,
    )

    print()
