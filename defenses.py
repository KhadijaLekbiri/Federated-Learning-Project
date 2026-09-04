"""Client-side experimental defenses and consistency simulation."""
from __future__ import annotations

import hashlib
import json
import struct
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from attacks import gradient_norms
from experiment_types import DefenseDecision
from model import NetflowClassifier


def gradient_statistics(gradient: Mapping[str, torch.Tensor]) -> dict[str, float]:
    norms = gradient_norms(gradient)
    total_sq = sum(float(t.detach().pow(2).sum()) for t in gradient.values())
    final_sq = float(gradient["net.4.bias"].detach().pow(2).sum())
    stats = {f"norm_{key}": value for key, value in norms.items()}
    stats["total_gradient_norm"] = total_sq ** 0.5
    stats["final_bias_energy_fraction"] = final_sq / (total_sq + 1e-30)
    return stats


def zero_gradient_check(gradient: Mapping[str, torch.Tensor], threshold: float) -> DefenseDecision:
    stats = gradient_statistics(gradient)
    data_dependent = [stats[f"norm_{name}"] for name in
                      ("net.0.weight", "net.0.bias", "net.2.weight")]
    suppressed = max(data_dependent) <= threshold and stats["final_bias_energy_fraction"] > 0.9
    return DefenseDecision(not suppressed,
        "early-layer gradients suppressed with energy concentrated in output bias" if suppressed else "gradient profile accepted",
        stats)


def model_digest(round_id: int, architecture_version: str,
                 state: Mapping[str, torch.Tensor], learning_rate: float,
                 batch_size: int, local_epochs: int, loss_identifier: str,
                 participants: Sequence[int]) -> str:
    """Deterministic SHA-256 transcript digest (not an authenticated protocol)."""
    digest = hashlib.sha256()
    metadata = (round_id, architecture_version, learning_rate, batch_size,
                local_epochs, loss_identifier, tuple(sorted(participants)))
    digest.update(json.dumps(metadata, separators=(",", ":")).encode())
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(struct.pack("!I", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("!Q", dimension))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class ModelConsistencyChecker:
    """Simulation of authenticated client digest comparison; no signatures."""
    def __init__(self) -> None:
        self.received: dict[int, str] = {}

    def register(self, client_id: int, digest: str) -> None:
        self.received[client_id] = digest

    def decision(self, expected_clients: Sequence[int]) -> DefenseDecision:
        groups: dict[str, list[int]] = defaultdict(list)
        for client_id, digest in self.received.items():
            groups[digest].append(client_id)
        consistent = set(self.received) == set(expected_clients) and len(groups) == 1
        return DefenseDecision(consistent, "all model digests match" if consistent else "model inconsistency detected",
                               {"digests": dict(self.received), "digest_groups": dict(groups),
                                "extra_bytes_estimate": len(expected_clients) * 32})


def secret_probe_check(state: Mapping[str, torch.Tensor], input_dim: int,
                       private_seed: int, round_id: int, probe_count: int,
                       dead_threshold: float, gradient_threshold: float) -> DefenseDecision:
    """Inspect synthetic private probes without touching client training data."""
    generator = torch.Generator(device="cpu").manual_seed(private_seed + round_id)
    probes = torch.randn(probe_count, input_dim, generator=generator)
    labels = torch.randint(0, 2, (probe_count,), generator=generator)
    model = NetflowClassifier(input_dim)
    model.load_state_dict({k: v.detach().cpu().clone() for k, v in state.items()})
    activations: list[torch.Tensor] = []
    handles = [model.net[index].register_forward_hook(
        lambda module, inputs, output: activations.append(output.detach().clone())) for index in (1, 3)]
    try:
        model.zero_grad()
        nn.CrossEntropyLoss()(model(probes), labels).backward()
    finally:
        for handle in handles:
            handle.remove()
    gradient = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    stats = gradient_statistics(gradient)
    stats["first_relu_dead_ratio"] = float((activations[0] == 0).float().mean())
    stats["second_relu_dead_ratio"] = float((activations[1] == 0).float().mean())
    early = max(stats[f"norm_{name}"] for name in
                ("net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"))
    suppressed = (stats["first_relu_dead_ratio"] >= dead_threshold and
                  stats["second_relu_dead_ratio"] >= dead_threshold and early <= gradient_threshold)
    return DefenseDecision(not suppressed, "synthetic probes detected dead-ReLU suppression" if suppressed else "synthetic probe accepted", stats)
