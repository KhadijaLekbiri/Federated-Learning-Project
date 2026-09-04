"""Reproducible orchestration for the six FL scenarios."""
from __future__ import annotations

import copy, json, random, time
from collections.abc import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset

from attacker import dlg_attack_with_metrics, score_reconstruction
from attacks import GradientSuppressionDistribution, HonestModelDistribution, flattened, gradient_norms
from client import Client
from defenses import ModelConsistencyChecker, gradient_statistics, model_digest, secret_probe_check, zero_gradient_check
from evaluate import evaluate
from experiment_types import AggregationMode, AttackEvaluation, AttackMode, DefenseDecision, DefenseMode, ExperimentConfig, ExperimentResult, RoundPlan, RoundPlanEntry, RoundResult
from model import NetflowClassifier
from secure_agg import IdealSecureAggregation


def reset_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_round_plan(config: ExperimentConfig, datasets: Sequence[Dataset], participants_per_round: int | None = None) -> RoundPlan:
    """Choose all clients and local examples before any scenario runs."""
    rng, count = np.random.default_rng(config.seed), len(datasets)
    take = participants_per_round or count
    entries = []
    for rid in range(config.num_rounds):
        participants = list(map(int, rng.permutation(count)[:take]))
        if config.target_client_id not in participants:
            participants[-1] = config.target_client_id
        participants = tuple(dict.fromkeys(participants))
        indices = {cid: tuple(map(int, rng.choice(len(datasets[cid]), min(config.batch_size, len(datasets[cid])), replace=False))) for cid in participants}
        entries.append(RoundPlanEntry(rid, participants, indices, config.target_client_id, rid in config.attack_rounds))
    return RoundPlan(tuple(entries))


def _weighted_sum(updates, weights):
    result = {k: torch.zeros_like(v) for k, v in updates[0].items()}
    for update, weight in zip(updates, weights):
        for name in result: result[name].add_(update[name], alpha=weight)
    return result


def _evaluate_attack(config, entry, aggregate, target_weight, target_gradient, target_state, truth, baseline):
    normalized = config.aggregation_mode == AggregationMode.IDEAL_SA
    result = AttackEvaluation(normalized_by_target_weight=normalized)
    if not config.run_dlg or entry.round_id not in config.dlg_rounds or truth is None: return result
    estimate = {k: v / target_weight for k, v in aggregate.items()} if normalized else target_gradient
    x_hat, label, loss = dlg_attack_with_metrics(target_state, estimate, truth[0].shape[1], config.dlg_iterations, config.dlg_learning_rate)
    scores = score_reconstruction(x_hat, truth[0].cpu(), baseline.cpu(), np.random.default_rng(config.seed + entry.round_id))
    result.attempted = True
    result.reconstruction_error = float(scores["attack_err"].mean())
    result.random_baseline_error = float(scores["baseline_err"].mean())
    result.error_ratio = result.reconstruction_error / (result.random_baseline_error + 1e-12)
    result.inferred_label_correct = bool(label.item() == truth[1].item())
    result.final_gradient_matching_loss = loss
    result.success = result.error_ratio <= config.attack_success_threshold
    return result


def run_experiment(config: ExperimentConfig, datasets: Sequence[Dataset], X_test: torch.Tensor,
                   y_test: torch.Tensor, initial_state=None, round_plan: RoundPlan | None = None) -> ExperimentResult:
    """Run one scenario; ideal SA and hash comparison are explicit simulations."""
    reset_seeds(config.seed)
    input_dim = int(datasets[0][0][0].numel())
    model = NetflowClassifier(input_dim)
    if initial_state is not None: model.load_state_dict(copy.deepcopy(initial_state))
    plan = round_plan or build_round_plan(config, datasets)
    clients = [Client(i, ds, input_dim, config.learning_rate, config.local_epochs, config.batch_size) for i, ds in enumerate(datasets)]
    probe_seeds = {i: config.seed * 100003 + i * 7919 for i in range(len(clients))}  # never stored on Server
    round_results = []
    baseline = torch.cat([torch.stack([ds[i][0] for i in range(len(ds))]) for ds in datasets])
    for entry in plan.rounds:
        start, defense_ms = time.perf_counter(), 0.0
        honest_state = copy.deepcopy(model.state_dict())
        distribution = GradientSuppressionDistribution(config.suppression_bias) if entry.attacked and config.attack_mode == AttackMode.GRADIENT_SUPPRESSION else HonestModelDistribution()
        records, decisions = [], {}
        checker = ModelConsistencyChecker()
        target_gradient = target_state = target_truth = None
        distributed_states = {cid: distribution.model_for_client(cid, entry.target_client_id, honest_state, entry.round_id) for cid in entry.participant_ids}
        hash_decision = None
        if config.defense_mode == DefenseMode.MODEL_HASH_CONSISTENCY:
            tick = time.perf_counter()
            for cid, state in distributed_states.items():
                checker.register(cid, model_digest(entry.round_id, config.architecture_version, state, config.learning_rate, config.batch_size, config.local_epochs, config.loss_identifier, entry.participant_ids))
            hash_decision = checker.decision(entry.participant_ids)
            defense_ms += (time.perf_counter() - tick) * 1000
        training_clients = entry.participant_ids if not hash_decision or hash_decision.accepted else ()
        for cid in training_clients:
            client = clients[cid]
            state = distributed_states[cid]
            if cid == entry.target_client_id: target_state = copy.deepcopy(state)
            client.receive_model(state)
            tick = time.perf_counter()
            if config.defense_mode == DefenseMode.SECRET_PROBE:
                decision = secret_probe_check(state, input_dim, probe_seeds[cid], entry.round_id, config.probe_count, config.probe_dead_relu_threshold, config.probe_gradient_threshold)
                decisions[cid] = decision
                defense_ms += (time.perf_counter() - tick) * 1000
                if not decision.accepted: continue  # real dataset has not been read
            gradient, _, batch = client._compute_fedsgd_update(entry.sample_indices[cid])
            if config.defense_mode == DefenseMode.ZERO_GRADIENT_ABORT:
                decision = zero_gradient_check(gradient, config.zero_gradient_threshold)
                decisions[cid] = decision
                defense_ms += (time.perf_counter() - tick) * 1000
                if not decision.accepted: continue
            if cid == entry.target_client_id: target_gradient, target_truth = gradient, batch
            records.append((cid, gradient, batch))
        aborted = len(records) < config.min_cohort_size or bool(hash_decision and not hash_decision.accepted)
        aggregate, weights = None, []
        if not aborted:
            sizes = [len(entry.sample_indices[cid]) for cid, *_ in records]; total = sum(sizes); weights = [n / total for n in sizes]
            if config.aggregation_mode == AggregationMode.IDEAL_SA:
                session = IdealSecureAggregation()
                for (cid, gradient, _), weight in zip(records, weights): session.submit(cid, gradient, weight)
                aggregate = session.finalize()
                for client in clients: client.last_gradient = client.last_batch = None
            else:
                aggregate = _weighted_sum([r[1] for r in records], weights)
                for cid, gradient, batch in records:
                    clients[cid].last_gradient = {k: v.detach().clone() for k, v in gradient.items()}; clients[cid].last_batch = tuple(v.detach().clone() for v in batch)
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate); optimizer.zero_grad()
            for name, parameter in model.named_parameters(): parameter.grad = aggregate[name].detach().clone()
            optimizer.step()
        metrics = evaluate(model, X_test, y_test)
        isolation = cosine = target_weight = None
        if aggregate is not None and target_gradient is not None:
            target_weight = weights[[r[0] for r in records].index(entry.target_client_id)]
            target_part = {k: v * target_weight for k, v in target_gradient.items()}
            av, tv = flattened(aggregate), flattened(target_part)
            isolation = float((av-tv).norm()/(tv.norm()+1e-12)); cosine = float(torch.nn.functional.cosine_similarity(av[None], tv[None]))
        attack_eval = AttackEvaluation()
        if aggregate is not None and target_gradient is not None and config.batch_size == 1:
            attack_eval = _evaluate_attack(config, entry, aggregate, target_weight, target_gradient, target_state, target_truth, baseline)
        aggregate_norms = gradient_norms(aggregate) if aggregate is not None else {}
        # Preserve every submitted client's per-parameter norms so suppression
        # (including a surviving output-bias gradient) is measured, not assumed.
        norms = {"aggregate": aggregate_norms, "clients": {str(cid): gradient_norms(gradient) for cid, gradient, _ in records}}
        stats = gradient_statistics(aggregate) if aggregate is not None else {}
        detected = any(not d.accepted for d in decisions.values()) or bool(hash_decision and not hash_decision.accepted)
        hs = hash_decision.statistics if hash_decision else {}
        round_results.append(RoundResult(config.scenario_name, config.seed, entry.round_id, json.dumps(entry.participant_ids), entry.target_client_id,
            config.aggregation_mode.value, config.attack_mode.value, config.defense_mode.value, detected, aborted, len(records),
            len(records) if config.aggregation_mode == AggregationMode.PLAIN else 0, stats.get("total_gradient_norm"), json.dumps(norms, sort_keys=True),
            isolation, cosine, attack_eval.reconstruction_error, attack_eval.random_baseline_error, attack_eval.inferred_label_correct,
            metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], (time.perf_counter()-start)*1000, defense_ms,
            hs.get("extra_bytes_estimate", 0), json.dumps(hs.get("digests", {}), sort_keys=True), bool(hash_decision and not hash_decision.accepted),
            attack_eval.error_ratio, attack_eval.final_gradient_matching_loss, attack_eval.normalized_by_target_weight,
            attack_eval.success))
    return ExperimentResult(config, plan, round_results, {k: v.detach().clone() for k, v in model.state_dict().items()})
