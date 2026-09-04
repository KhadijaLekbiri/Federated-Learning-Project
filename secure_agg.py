"""Secure-aggregation boundaries used by the FL simulation.

``IdealSecureAggregation`` models the ideal functionality from the paper: clients
can submit private updates, while the server can only obtain their weighted sum.
It is not a cryptographic protocol; pairwise masking is a later milestone.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


TensorUpdate = Mapping[str, torch.Tensor]


def _validate_compatible(reference: TensorUpdate, update: TensorUpdate) -> None:
    if set(update) != set(reference):
        raise ValueError("all updates must contain the same parameter names")

    for name, tensor in update.items():
        expected = reference[name]
        if tensor.shape != expected.shape:
            raise ValueError(f"shape mismatch for parameter {name!r}")
        if tensor.dtype != expected.dtype:
            raise ValueError(f"dtype mismatch for parameter {name!r}")
        if tensor.device != expected.device:
            raise ValueError(f"device mismatch for parameter {name!r}")


def ideal_secure_aggregate(
    updates: list[TensorUpdate], weights: list[float] | None = None
) -> dict[str, torch.Tensor]:
    """Return a detached, weighted sum of compatible tensor dictionaries.

    The default is an unweighted sum, matching the ideal SA functionality
    ``f_sa(v_1, ..., v_n) = sum(v_i)``. Normalized weights can be supplied by
    the FL protocol to obtain a weighted average without exposing inputs.
    """
    if not updates:
        raise ValueError("secure aggregation requires at least one update")
    if weights is None:
        weights = [1.0] * len(updates)
    if len(weights) != len(updates):
        raise ValueError("weights and updates must have the same length")

    reference = updates[0]
    for update in updates[1:]:
        _validate_compatible(reference, update)

    aggregate = {name: torch.zeros_like(tensor) for name, tensor in reference.items()}
    with torch.no_grad():
        for update, weight in zip(updates, weights):
            for name, tensor in update.items():
                aggregate[name].add_(tensor.detach(), alpha=float(weight))
    return aggregate


class IdealSecureAggregation:
    """One-round ideal-SA session.

    Clients submit into this object. Its public API intentionally provides no
    method for retrieving an individual contribution.
    """

    def __init__(self) -> None:
        self.__updates: list[dict[str, torch.Tensor]] = []
        self.__weights: list[float] = []
        self.__client_ids: set[int] = set()
        self.__finalized = False

    @property
    def num_submissions(self) -> int:
        return len(self.__client_ids)

    def submit(self, client_id: int, update: TensorUpdate, weight: float = 1.0) -> None:
        if self.__finalized:
            raise RuntimeError("cannot submit after aggregation was finalized")
        if client_id in self.__client_ids:
            raise ValueError(f"client {client_id} submitted more than once")
        if weight < 0:
            raise ValueError("aggregation weights must be non-negative")

        # Clone at the trust boundary so callers cannot mutate a submitted value.
        private_copy = {name: tensor.detach().clone() for name, tensor in update.items()}
        if self.__updates:
            _validate_compatible(self.__updates[0], private_copy)
        self.__updates.append(private_copy)
        self.__weights.append(float(weight))
        self.__client_ids.add(client_id)

    def finalize(self) -> dict[str, torch.Tensor]:
        if self.__finalized:
            raise RuntimeError("aggregation session was already finalized")
        self.__finalized = True
        aggregate = ideal_secure_aggregate(self.__updates, self.__weights)
        # Release private contributions as soon as the aggregate is produced.
        self.__updates.clear()
        self.__weights.clear()
        self.__client_ids.clear()
        return aggregate
