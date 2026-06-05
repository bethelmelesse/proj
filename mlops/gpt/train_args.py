import argparse
from typing import Any


def cli_args_parser() -> dict[str, dict[str, Any]]:
    """Parse CLI args for GPT training and return them as the grouped
    arg dicts consumed by ``train()``.

    Returns:
        A dict mapping each ``*_args`` group name (``data_args``,
        ``tokenizer_args``, ``dataloader_args``, ``model_args``,
        ``optim_args``, ``criterion_args``, ``scheduler_args``,
        ``training_args``, ``mlflow_args``) to its dict of values.
    """
    parser = argparse.ArgumentParser(
        description="Train the decoder-only (GPT) transformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--dataset-path", type=str, default="data/dataset.txt")

    tokenizer = parser.add_argument_group("tokenizer")
    tokenizer.add_argument("--tokenizer-path", type=str, default="t5-small")
    tokenizer.add_argument("--tokenizer-batch-size", type=int, default=100)
    tokenizer.add_argument("--padding", type=str, default="max_length")
    tokenizer.add_argument("--max-length", type=int, default=128)

    dataloader = parser.add_argument_group("dataloader")
    dataloader.add_argument("--batch-size", type=int, default=2)

    model = parser.add_argument_group("model")
    model.add_argument("--d-model", type=int, default=512)
    model.add_argument("--d-ff", type=int, default=2048)
    model.add_argument("--n-heads", type=int, default=2)
    model.add_argument("--n-layers", type=int, default=6)
    model.add_argument("--max-seq-len", type=int, default=128)
    model.add_argument("--dropout", type=float, default=0.0)

    optim = parser.add_argument_group("optim")
    optim.add_argument("--lr", type=float, default=1e-4)
    # betas=(0.9, 0.98) and eps=1e-9 are from "Attention Is All You Need".
    optim.add_argument(
        "--betas", type=float, nargs=2, default=(0.9, 0.98), metavar=("BETA1", "BETA2")
    )
    optim.add_argument("--eps", type=float, default=1e-9)
    # weight_decay=0 keeps AdamW equivalent to paper-Adam; raise to 0.01 to
    # match common modern practice.
    optim.add_argument("--weight-decay", type=float, default=0.0)

    criterion = parser.add_argument_group("criterion")
    criterion.add_argument("--label-smoothing", type=float, default=0.1)

    scheduler = parser.add_argument_group("scheduler")
    scheduler.add_argument("--gamma", type=float, default=0.9)

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--device", type=str, default="cpu")
    training.add_argument("--checkpoint-dir", type=str, default=None)

    mlflow = parser.add_argument_group("mlflow")
    mlflow.add_argument("--experiment-name", type=str, default="GPT (Decoder Only)")
    mlflow.add_argument("--run-name", type=str, default="test: grad norm")
    mlflow.add_argument("--tracker-url", type=str, default="http://localhost:5000")

    parsed = parser.parse_args()

    return {
        "data_args": {"dataset_path": parsed.dataset_path},
        "tokenizer_args": {
            "tokenizer_path": parsed.tokenizer_path,
            "batch_size": parsed.tokenizer_batch_size,
            "padding": parsed.padding,
            "max_length": parsed.max_length,
        },
        "dataloader_args": {"batch_size": parsed.batch_size},
        "model_args": {
            "d_model": parsed.d_model,
            "d_ff": parsed.d_ff,
            "n_heads": parsed.n_heads,
            "n_layers": parsed.n_layers,
            "max_seq_len": parsed.max_seq_len,
            "dropout": parsed.dropout,
        },
        "optim_args": {
            "lr": parsed.lr,
            "betas": tuple(parsed.betas),
            "eps": parsed.eps,
            "weight_decay": parsed.weight_decay,
        },
        "criterion_args": {"label_smoothing": parsed.label_smoothing},
        "scheduler_args": {"gamma": parsed.gamma},
        "training_args": {
            "epochs": parsed.epochs,
            "device": parsed.device,
            "checkpoint_dir": parsed.checkpoint_dir,
        },
        "mlflow_args": {
            "experiment_name": parsed.experiment_name,
            "run_name": parsed.run_name,
            "tracker_url": parsed.tracker_url,
        },
    }
