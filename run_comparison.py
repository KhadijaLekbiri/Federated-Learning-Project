"""Run all six reproducible malicious-server comparison scenarios."""
from __future__ import annotations
import argparse, copy, json
from dataclasses import replace
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from data import load_and_preprocess, make_client_datasets, make_train_test_split
from experiment_runner import build_round_plan, reset_seeds, run_experiment
from experiment_types import AggregationMode, AttackMode, DefenseMode, ExperimentConfig
from model import NetflowClassifier
from reporting import console_table, write_results


SCENARIOS = [
    ("A_plain_honest", AggregationMode.PLAIN, AttackMode.HONEST, DefenseMode.NONE),
    ("B_sa_honest", AggregationMode.IDEAL_SA, AttackMode.HONEST, DefenseMode.NONE),
    ("C_sa_attack", AggregationMode.IDEAL_SA, AttackMode.GRADIENT_SUPPRESSION, DefenseMode.NONE),
    ("D_sa_attack_zero_check", AggregationMode.IDEAL_SA, AttackMode.GRADIENT_SUPPRESSION, DefenseMode.ZERO_GRADIENT_ABORT),
    ("E_sa_attack_hash", AggregationMode.IDEAL_SA, AttackMode.GRADIENT_SUPPRESSION, DefenseMode.MODEL_HASH_CONSISTENCY),
    ("F_sa_attack_probe", AggregationMode.IDEAL_SA, AttackMode.GRADIENT_SUPPRESSION, DefenseMode.SECRET_PROBE),
]


def _data(seed: int, quick: bool):
    path = Path("NF-UNSW-NB15-V2.parquet")
    if path.exists():
        X, y, dim = load_and_preprocess(path); Xtr, Xte, ytr, yte = make_train_test_split(X, y, seed=seed)
        limit = 400 if quick else len(Xtr); Xtr, ytr = Xtr[:limit], ytr[:limit]
        return make_client_datasets(Xtr, ytr, 4, seed), Xte[:200], yte[:200], dim
    generator = torch.Generator().manual_seed(seed); dim = 8; X = torch.randn(160, dim, generator=generator)
    y = (X[:, 0] + .3 * X[:, 1] > 0).long()
    datasets = [TensorDataset(X[i::4][:30], y[i::4][:30]) for i in range(4)]
    return datasets, X[120:], y[120:], dim


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42]); parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results")); args = parser.parse_args(argv)
    all_results, configurations = [], []
    for seed in args.seeds:
        datasets, Xte, yte, dim = _data(seed, args.quick)
        base = ExperimentConfig(seed=seed, num_rounds=args.rounds, output_dir=args.output_dir,
            dlg_iterations=5 if args.quick else 300, run_dlg=True, min_cohort_size=2)
        plan = build_round_plan(base, datasets)
        reset_seeds(seed); initial = copy.deepcopy(NetflowClassifier(dim).state_dict())
        seed_results = []
        for name, aggregation, attack, defense in SCENARIOS:
            config = replace(base, scenario_name=name, aggregation_mode=aggregation, attack_mode=attack, defense_mode=defense)
            configurations.append(config.json_dict()); result = run_experiment(config, datasets, Xte, yte, initial, plan)
            all_results.append(result); seed_results.append(result)
        plain, honest_sa = seed_results[:2]
        for key in initial: torch.testing.assert_close(plain.final_state[key], honest_sa.final_state[key], rtol=1e-6, atol=1e-7)
    summary = write_results(all_results, args.output_dir)
    (args.output_dir / "config.json").write_text(json.dumps(configurations, indent=2), encoding="utf-8")
    print(console_table(summary))


if __name__ == "__main__": main()
