"""Model-distribution strategies, including the Pasquini-style simulation."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch


State = Mapping[str, torch.Tensor]


def clone_state(state: State) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state.items()}


class ModelDistribution(ABC):
    @abstractmethod
    def model_for_client(self, client_id: int, target_client_id: int,
                         honest_state: State, round_id: int) -> dict[str, torch.Tensor]:
        """Return a detached model state for one client."""


class HonestModelDistribution(ModelDistribution):
    def model_for_client(self, client_id: int, target_client_id: int,
                         honest_state: State, round_id: int) -> dict[str, torch.Tensor]:
        return clone_state(honest_state)


class GradientSuppressionDistribution(ModelDistribution):
    """Send non-targets a deliberately dead first ReLU model.

    This is an experimental model-inconsistency attack. It does not guarantee
    every gradient is zero: in particular, the output bias can remain nonzero.
    Suppression strength is measured by the runner rather than assumed.
    """

    def __init__(self, negative_bias: float = -100.0) -> None:
        self.negative_bias = negative_bias

    def model_for_client(self, client_id: int, target_client_id: int,
                         honest_state: State, round_id: int) -> dict[str, torch.Tensor]:
        state = clone_state(honest_state)
        if client_id != target_client_id:
            state["net.0.weight"].zero_()
            state["net.0.bias"].fill_(self.negative_bias)
            # The first dead layer already removes data dependence. Making the
            # next bias negative creates the explicit test fixture needed by
            # the probe heuristic; output-bias gradients can still survive.
            state["net.2.bias"].fill_(self.negative_bias)
        return state


def gradient_norms(gradient: State) -> dict[str, float]:
    return {name: float(tensor.detach().norm()) for name, tensor in gradient.items()}


def flattened(update: State) -> torch.Tensor:
    return torch.cat([update[name].detach().reshape(-1).cpu() for name in sorted(update)])
