from torch import nn


def compute_grad_norm(model: nn.Module):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None]

    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)  # L2 norm
    total_norm += param_norm.item() ** 2

    total_norm = total_norm**0.5
    return total_norm
