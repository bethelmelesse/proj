from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from src.utils.utils import print_header


def compute_grad_norm(model: nn.Module):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None]

    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)  # L2 norm
        total_norm += param_norm.item() ** 2

    total_norm = total_norm**0.5
    return total_norm


def adamw_optimizer(
    model: nn.Module, lr: float, betas: tuple, eps: float, weight_decay: float
) -> optim.AdamW:
    """Build an AdamW optimizer over `model.parameters()` from `optim_args`."""
    print_header(text="Initializing optimizer - AdamW")
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )
    print(
        f"Optimizer: AdamW(lr={lr}, betas={betas}, "
        f"eps={eps}, weight_decay={weight_decay})"
    )
    return optimizer


def cosine_annealing_lr_scheduler(
    optimizer: optim.Optimizer, steps: int, eta_min: float
) -> CosineAnnealingLR:
    """Build a CosineAnnealingLR that decays `lr` from its initial value to
    `eta_min` over `steps` scheduler ticks (one per optimizer step in the trainer)."""
    print_header(text="Initializing Scheduler - CosineAnnealingLR")
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=eta_min)
    print(f"Set CosineAnnealingLR: Tmax value: {steps}, eta_min: {eta_min}")

    return scheduler


def linear_lr_scheduler(optimizer: optim.Optimizer):
    print_header(text="Initializing Scheduler - CosineAnnealingLR")
    scheduler = LinearLR(optimizer, start_factor=0.05, total_iters=40)

    return scheduler
