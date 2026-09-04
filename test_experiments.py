import csv
import inspect
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from attacker import dlg_attack
from attacks import GradientSuppressionDistribution, HonestModelDistribution, gradient_norms
from defenses import ModelConsistencyChecker, model_digest, secret_probe_check, zero_gradient_check
from experiment_runner import build_round_plan, run_experiment
from experiment_types import AggregationMode, AttackMode, DefenseMode, ExperimentConfig
from model import NetflowClassifier
from reporting import REQUIRED_COLUMNS, write_results


class ExperimentFrameworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5); self.dim = 4
        x = torch.randn(24, self.dim); y = (x[:, 0] > 0).long()
        self.datasets = [TensorDataset(x[i::3], y[i::3]) for i in range(3)]
        self.xt, self.yt = x[:8], y[:8]
        self.initial = NetflowClassifier(self.dim).state_dict()
        self.base = ExperimentConfig(num_rounds=1, batch_size=1, run_dlg=False, min_cohort_size=2)
        self.plan = build_round_plan(self.base, self.datasets)

    def test_plain_leaks_and_sa_does_not(self):
        plain = run_experiment(replace(self.base, aggregation_mode=AggregationMode.PLAIN), self.datasets, self.xt, self.yt, self.initial, self.plan)
        secure = run_experiment(replace(self.base, aggregation_mode=AggregationMode.IDEAL_SA), self.datasets, self.xt, self.yt, self.initial, self.plan)
        self.assertGreater(plain.rounds[0].number_individual_leak_records, 0)
        self.assertEqual(secure.rounds[0].number_individual_leak_records, 0)
        for key in plain.final_state: torch.testing.assert_close(plain.final_state[key], secure.final_state[key])

    def test_distribution_target_and_non_target(self):
        attack = GradientSuppressionDistribution(-100)
        target = attack.model_for_client(0, 0, self.initial, 0)
        other = attack.model_for_client(1, 0, self.initial, 0)
        for key in target: torch.testing.assert_close(target[key], self.initial[key])
        self.assertEqual(float(other["net.0.weight"].abs().max()), 0)
        self.assertLess(float(other["net.0.bias"].max()), -90)

    def test_suppression_norms_and_zero_defense(self):
        state = GradientSuppressionDistribution(-100).model_for_client(1, 0, self.initial, 0)
        model = NetflowClassifier(self.dim); model.load_state_dict(state); model.zero_grad()
        torch.nn.CrossEntropyLoss()(model(torch.randn(2, self.dim)), torch.tensor([0, 1])).backward()
        gradient = {n: p.grad for n, p in model.named_parameters()}; norms = gradient_norms(gradient)
        self.assertLessEqual(norms["net.0.weight"], 1e-8)
        self.assertFalse(zero_gradient_check(gradient, 1e-8).accepted)

    def test_hash_accepts_honest_and_rejects_attack(self):
        checker = ModelConsistencyChecker()
        for cid in (0, 1): checker.register(cid, model_digest(0, "v1", self.initial, .01, 1, 1, "CE", (0, 1)))
        self.assertTrue(checker.decision((0, 1)).accepted)
        checker = ModelConsistencyChecker(); checker.register(0, "a"); checker.register(1, "b")
        self.assertFalse(checker.decision((0, 1)).accepted)

    def test_probe_rejects_suppression_accepts_normal(self):
        bad = GradientSuppressionDistribution(-100).model_for_client(1, 0, self.initial, 0)
        self.assertFalse(secret_probe_check(bad, self.dim, 10, 0, 16, .95, 1e-8).accepted)
        self.assertTrue(secret_probe_check(self.initial, self.dim, 10, 0, 16, .95, 1e-8).accepted)

    def test_no_truth_in_attacker_api(self):
        parameters = inspect.signature(dlg_attack).parameters
        self.assertNotIn("x_true", parameters); self.assertNotIn("y_true", parameters)

    def test_attack_isolation_and_hash_abort(self):
        attack = replace(self.base, attack_mode=AttackMode.GRADIENT_SUPPRESSION, aggregation_mode=AggregationMode.IDEAL_SA)
        unprotected = run_experiment(attack, self.datasets, self.xt, self.yt, self.initial, self.plan)
        self.assertLess(unprotected.rounds[0].relative_isolation_error, 0.1)
        protected = run_experiment(replace(attack, defense_mode=DefenseMode.MODEL_HASH_CONSISTENCY), self.datasets, self.xt, self.yt, self.initial, self.plan)
        self.assertTrue(protected.rounds[0].round_aborted); self.assertTrue(protected.rounds[0].hash_inconsistency_detected)

    def test_output_columns(self):
        result = run_experiment(self.base, self.datasets, self.xt, self.yt, self.initial, self.plan)
        with tempfile.TemporaryDirectory() as directory:
            write_results([result], Path(directory))
            with (Path(directory) / "round_results.csv").open(newline="") as handle:
                self.assertEqual(csv.DictReader(handle).fieldnames, REQUIRED_COLUMNS)
            self.assertTrue((Path(directory) / "experiment_summary.csv").exists())


if __name__ == "__main__": unittest.main()
