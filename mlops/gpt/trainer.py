"""Generic training loop with MLflow tracking, checkpointing and grad-norm logging."""

import copy
import os
from typing import Literal

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlops.utils.train_utils import compute_grad_norm
from src.utils.utils import print_header


class Trainer:
    """Runs train/val/test epochs, logging metrics and saving the best checkpoint."""

    def __init__(self, args: dict):
        """Unpack the flat `args` dict into the trainer's attributes."""
        # dataloader
        self.batch_size = args["batch_size"]
        self.train_loader = args["train_loader"]
        self.valid_loader = args["val_loader"]
        self.test_loader = args["test_loader"]
        # model
        self.model = args["model"]
        # optim
        self.lr = args["lr"]
        self.optimizer = args["optimizer"]
        # loss
        self.criterion = args["criterion"]
        # scheduler
        self.scheduler = args["scheduler"]
        # training
        self.epochs = args["epochs"]
        self.device = args["device"]
        self.checkpoint_dir = args["checkpoint_dir"]
        # mlflow
        self.run_name = args["run_name"]
        self.experiment_name = args["experiment_name"]
        self.tracker_url = args["tracker_url"]

    def train_epoch(
        self,
        epoch: int,
        loader: DataLoader,
        mode: Literal["Train", "Val", "Init", "Test"],
        global_step: int | None,
    ) -> tuple[float, int | None]:
        """Run one pass over `loader`; return average loss and updated global step."""
        # Train mode enables dropout/grad; every other mode runs in eval.
        self.model.train() if mode == "Train" else self.model.eval()
        epoch_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch} [{mode}]"):
            if mode == "Train":
                self.optimizer.zero_grad()

            logits = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
            )
            labels = batch["labels"].to(self.device)

            # Drop the last logit (no next-token target) so logits[:, t] is
            # paired with the token at position t+1, which `labels` already holds.
            logits_shifted = logits[:, :-1, :]
            batch_size, seq_len, vocab_size = logits_shifted.shape
            logits_shifted = logits_shifted.reshape(batch_size * seq_len, vocab_size)
            labels = labels.reshape(batch_size * seq_len)

            loss = self.criterion(logits_shifted, labels)
            epoch_loss += loss.item()

            if mode == "Train":
                loss.backward()
                # max_norm=inf measures the global grad norm without clipping.
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=float("inf")
                )
                # Manual L2 norm kept as a cross-check against clip_grad_norm_.
                grad_norm2 = compute_grad_norm(model=self.model)
                self.optimizer.step()

            # Step-level logging: per-batch loss for Train/Val; LR + step
            # increment only for Train so the x-axis tracks gradient updates.
            if mode in ("Train", "Val"):
                mlflow.log_metrics(
                    {f"step_level/{mode}_loss": loss.item()}, step=global_step
                )
            if mode == "Train":
                current_lr = self.optimizer.param_groups[0]["lr"]
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

    def train(self) -> None:
        """Run initial validation, the checkpointing epoch loop, then a final test."""
        # Training with MLflow logging
        print_header(text="Setting up mlflow")
        mlflow.set_tracking_uri(self.tracker_url)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.run_name):
            # Enable system metrics logging
            mlflow.enable_system_metrics_logging()
            mlflow.log_params(
                {"epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr}
            )

            # Track the best-by-val-loss weights so the final test uses them.
            best_state = copy.deepcopy(self.model.state_dict())
            best_val_loss = float("inf")
            best_epoch = 0
            epoch_one_loss = None
            global_step = 0

            # Initial Validation
            print_header(text="Initial Validation")
            with torch.no_grad():
                init_loss, _ = self.train_epoch(
                    epoch=0,
                    loader=self.valid_loader,
                    mode="Init",
                    global_step=None,
                )

            print_header(text="Started Training")
            for epoch in range(1, self.epochs + 1):
                # Training
                train_loss, global_step = self.train_epoch(
                    epoch=epoch,
                    loader=self.train_loader,
                    global_step=global_step,
                    mode="Train",
                )
                if epoch == 1:
                    epoch_one_loss = train_loss

                # Validation
                with torch.no_grad():
                    val_loss, _ = self.train_epoch(
                        epoch=epoch,
                        loader=self.valid_loader,
                        mode="Val",
                        global_step=global_step,
                    )

                self.scheduler.step()

                if val_loss <= best_val_loss:
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    best_val_loss = val_loss

                    if self.checkpoint_dir:
                        ckpt_path = os.path.join(self.checkpoint_dir, "best.pt")
                        torch.save(
                            {
                                "epoch": best_epoch,
                                "model_state_dict": best_state,
                                "optimizer_state_dict": self.optimizer.state_dict(),
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
            self.model.load_state_dict(best_state)
            print_header(text="Final Test")
            with torch.no_grad():
                test_loss, _ = self.train_epoch(
                    epoch=best_epoch,
                    loader=self.test_loader,
                    mode="Test",
                    global_step=None,
                )
