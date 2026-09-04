"""Typed configuration and immutable result records for FL experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class AggregationMode(str, Enum):
    PLAIN = "plain"
    IDEAL_SA = "ideal_sa"


class AttackMode(str, Enum):
    HONEST = "honest"
    GRADIENT_SUPPRESSION = "gradient_suppression"


class DefenseMode(str, Enum):
    NONE = "none"
    ZERO_GRADIENT_ABORT = "zero_gradient_abort"
    MODEL_HASH_CONSISTENCY = "model_hash_consistency"
    SECRET_PROBE = "secret_probe"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    num_rounds: int = 2
    batch_size: int = 1
    target_client_id: int = 0
    attack_rounds: tuple[int, ...] = (0,)
    aggregation_mode: AggregationMode = AggregationMode.IDEAL_SA
    attack_mode: AttackMode = AttackMode.HONEST
    defense_mode: DefenseMode = DefenseMode.NONE
    dlg_iterations: int = 300
    dlg_learning_rate: float = 0.1
    dlg_rounds: tuple[int, ...] = (0,)
    attack_success_threshold: float = 1.0
    zero_gradient_threshold: float = 1e-8
    probe_count: int = 16
    probe_dead_relu_threshold: float = 0.95
    probe_gradient_threshold: float = 1e-8
    output_dir: Path = Path("results")
    min_cohort_size: int = 2
    suppression_bias: float = -100.0
    learning_rate: float = 0.01
    local_epochs: int = 1
    architecture_version: str = "NetflowClassifier-v1"
    loss_identifier: str = "CrossEntropyLoss"
    scenario_name: str = "experiment"
    run_dlg: bool = True

    def json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass(frozen=True)
class RoundPlanEntry:
    round_id: int
    participant_ids: tuple[int, ...]
    sample_indices: dict[int, tuple[int, ...]]
    target_client_id: int
    attacked: bool


@dataclass(frozen=True)
class RoundPlan:
    rounds: tuple[RoundPlanEntry, ...]


@dataclass
class DefenseDecision:
    accepted: bool
    reason: str
    statistics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackEvaluation:
    attempted: bool = False
    reconstruction_error: float | None = None
    random_baseline_error: float | None = None
    error_ratio: float | None = None
    inferred_label_correct: bool | None = None
    final_gradient_matching_loss: float | None = None
    normalized_by_target_weight: bool = False
    success: bool | None = None


@dataclass
class RoundResult:
    scenario_name: str
    seed: int
    round_id: int
    selected_clients: str
    target_client: int
    aggregation_mode: str
    attack_mode: str
    defense_mode: str
    attack_detected: bool
    round_aborted: bool
    number_submitted_clients: int
    number_individual_leak_records: int
    total_gradient_norm: float | None
    per_layer_gradient_norms: str
    relative_isolation_error: float | None
    aggregate_target_cosine_similarity: float | None
    dlg_reconstruction_error: float | None
    random_baseline_error: float | None
    inferred_label_correct: bool | None
    accuracy: float
    precision: float
    recall: float
    f1: float
    round_runtime_ms: float
    defense_runtime_ms: float
    hash_extra_bytes_estimate: int = 0
    hash_digests: str = ""
    hash_inconsistency_detected: bool = False
    attack_error_ratio: float | None = None
    final_gradient_matching_loss: float | None = None
    aggregate_normalized_by_target_weight: bool = False
    attack_success: bool | None = None


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    round_plan: RoundPlan
    rounds: list[RoundResult]
    final_state: dict[str, Any]
