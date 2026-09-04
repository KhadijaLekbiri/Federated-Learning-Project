"""DLG/iDLG baseline and reconstruction scoring."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LeakRecord:
    """Only attacker-visible values; ground truth belongs to evaluation code."""
    round_id: int
    client_id: int
    theta_used: dict[str, torch.Tensor]
    gradient: dict[str, torch.Tensor]


def capture_leaks(clients, theta_used, round_id):
    return [LeakRecord(round_id, c.client_id,
            {k: v.detach().clone() for k, v in theta_used.items()},
            {k: v.detach().clone() for k, v in c.last_gradient.items()}) for c in clients]


def score_reconstruction(x_hat: torch.Tensor, x_true: torch.Tensor,
                         dataset: torch.Tensor, rng, eps: float = 1e-8) -> dict:
    rel_err = torch.abs(x_hat - x_true) / (torch.abs(x_true) + eps)
    idx = rng.integers(0, dataset.shape[0], size=x_true.shape[0])
    baseline_err = torch.abs(dataset[idx] - x_true) / (torch.abs(x_true) + eps)
    return {"attack_err": rel_err.mean(dim=0), "baseline_err": baseline_err.mean(dim=0)}


def infer_label(gradient):
    return torch.argmin(gradient["net.4.bias"]).long()


def dlg_attack(theta_used, gradient, input_dim, num_iters=300, lr=0.1, device="cpu"):
    x_hat, _, _ = dlg_attack_with_metrics(theta_used, gradient, input_dim, num_iters, lr, device)
    return x_hat


def dlg_attack_with_metrics(theta_used, gradient, input_dim, num_iters=300,
                            lr=0.1, device="cpu"):
    """Run iDLG without accepting ground-truth values through the attack API."""
    from model import NetflowClassifier
    model = NetflowClassifier(input_dim).to(device)
    model.load_state_dict(theta_used)
    y_fixed = infer_label(gradient).reshape(1)
    x_hat = torch.randn(1, input_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([x_hat], lr=lr)
    names = [name for name, _ in model.named_parameters()]
    distance = torch.tensor(float("nan"))
    for _ in range(num_iters):
        optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(model(x_hat), y_fixed)
        dummy = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        distance = sum((dg - gradient[name]).pow(2).sum() for name, dg in zip(names, dummy))
        distance.backward()
        optimizer.step()
    return x_hat.detach(), y_fixed.detach(), float(distance.detach())
