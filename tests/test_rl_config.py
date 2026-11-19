import json
import tempfile
import unittest
from contextlib import ExitStack, closing
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.agents.rl.models.sac import SAC
from src.agents.rl.training.constants import DEFAULT_DISTRIBUTION_PATH
from src.simulation.environment import ActionRanges, PushEnv
from src.simulation.environment.config import load_action_ranges
from src.simulation.modelling.sampling import load_object_sampler


class TestRLConfig(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_checkpoint_action_bounds(self) -> None:
        for config, expected in [
            ({"f_max": 15}, ActionRanges(force=(0, 15))),
            (
                {
                    "action_ranges": asdict(
                        ActionRanges(height_fraction=(0.2, 0.8), force=(0, 25))
                    )
                },
                ActionRanges(height_fraction=(0.2, 0.8), force=(0, 25)),
            ),
        ]:
            (self.tmp_path / "config.json").write_text(json.dumps(config))
            assert load_action_ranges(self.tmp_path / "sac_final.pt") == expected

    def test_checkpoint_requires_action_metadata(self) -> None:
        tmp_path = self.tmp_path
        with self.assertRaises(FileNotFoundError):
            load_action_ranges(tmp_path / "sac_final.pt")

    def test_training_and_evaluation_share_physical_action_bounds(self) -> None:
        self.check_bounds(1)
        self.check_bounds(2)

    def check_bounds(self, n_envs: int) -> None:
        from src.agents.rl.training.train_sac import _make_evaluator, make_train_vec

        bounds = ActionRanges(force=(0, 25))
        sampler = load_object_sampler(DEFAULT_DISTRIBUTION_PATH)
        with ExitStack() as resources:
            training = resources.enter_context(
                closing(
                    make_train_vec(
                        n_envs, seed=0, action_ranges=bounds, distribution=sampler
                    )
                )
            )
            env = resources.enter_context(
                closing(PushEnv(action_ranges=bounds, object_sampler=sampler))
            )
            training.reset(seed=0)
            _, _, terminated, truncated, infos = training.step([[0, 1, -1]] * n_envs)
            for index in range(n_envs):
                info = (
                    infos["final_info"][index]
                    if terminated[index] or truncated[index]
                    else {"executed": infos["executed"][index]}
                )
                assert info["executed"][1] == 25
            evaluator = _make_evaluator(SimpleNamespace(seed=1, n_eval_episodes=1), env)
            evaluator.env.reset(seed=1)
            assert evaluator.env.step(np.array([0, 1, -1]))[-1]["executed"][1] == 25

    def test_training_saves_loadable_final_and_best_checkpoints(self) -> None:
        self.check_training_metadata("uniform_excluding_evaluation_regions.json")

    def test_training_without_exclusions_records_empty_list(self) -> None:
        self.check_training_metadata("uniform_without_exclusions.json")

    def check_training_metadata(self, filename: str) -> None:
        tmp_path = self.tmp_path / "run"
        from src.agents.rl.training.train_sac import train

        distribution_path = self.tmp_path / "distribution.json"
        distribution_path.write_text(
            DEFAULT_DISTRIBUTION_PATH.with_name(filename).read_text()
        )
        distribution = load_object_sampler(distribution_path)
        args = SimpleNamespace(
            out_dir=tmp_path,
            distribution_config=distribution_path,
            tb_dir=tmp_path / "tb",
            f_max=25.0,
            n_envs=1,
            seed=42,
            lr=3e-4,
            buffer_size=100,
            learning_starts=1,
            batch_size=8,
            gradient_steps=1,
            eval_freq=2,
            n_eval_episodes=1,
            total_steps=4,
            run_name="test",
        )
        train(args)
        for directory, filename in (
            (tmp_path, "sac_final.pt"),
            (tmp_path / "best_model", "best_model.pt"),
        ):
            bounds = load_action_ranges(directory / filename)
            assert bounds.force == (0, 25)
            config = json.loads((directory / "config.json").read_text())
            assert config["reward"]["alpha"] == 10
            assert "f_max" not in config
            assert config["training_distribution"] == json.loads(
                json.dumps(asdict(distribution))
            )
            assert (
                config["training_distribution_sha256"]
                == sha256(
                    json.dumps(
                        config["training_distribution"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            )
            assert config["run_id"]
            model = SAC.load(directory / filename)
            assert model.training_metadata == config
            self.assertAlmostEqual(model.normalizer.count, model.num_timesteps + 1.0001)
            env = PushEnv(action_ranges=bounds, object_sampler=distribution)
            try:
                obs, _ = env.reset(seed=42)
                action = model.predict(obs)
                assert np.all(np.abs(action) <= 1)
                env.step(action)
            finally:
                env.close()

        # The copied weights retain their metadata without an adjacent JSON file.
        copied = self.tmp_path / "copied.pt"
        copied.write_bytes((tmp_path / "sac_final.pt").read_bytes())
        distribution_path.write_text("{}")
        assert SAC.load(copied, inference_only=True).training_metadata == config

    def test_training_does_not_overwrite_existing_output(self) -> None:
        from src.agents.rl.training.train_sac import train

        marker = self.tmp_path / "existing-model.pt"
        marker.write_bytes(b"previous run")
        with self.assertRaises(FileExistsError):
            train(SimpleNamespace(out_dir=self.tmp_path))
        assert marker.read_bytes() == b"previous run"


def test_embedded_action_bounds_override_conflicting_sidecar(tmp_path: Path) -> None:
    from src.agents.rl.models.sac import SACConfig

    model = SAC(SACConfig(7, 3, hidden_dim=16, buffer_size=16, batch_size=8))
    expected = ActionRanges(force=(0, 25))
    model.training_metadata = {"action_ranges": asdict(expected)}
    path = tmp_path / "portable.pt"
    model.save(path)
    sidecar = tmp_path / "config.json"
    sidecar.write_text(json.dumps({"f_max": 15}))
    assert load_action_ranges(path) == expected
    sidecar.unlink()
    assert load_action_ranges(path) == expected
    # Embedded metadata is authoritative; do not merge in conflicting sidecars.
    model.training_metadata = {"run_id": "missing_action_semantics"}
    model.save(path)
    sidecar.write_text(json.dumps({"f_max": 15}))
    with pytest.raises(ValueError, match="no action_ranges"):
        load_action_ranges(path)


def test_legacy_weights_read_sidecar_action_bounds(tmp_path: Path) -> None:
    from src.agents.rl.models.sac import SACConfig

    model = SAC(SACConfig(7, 3, hidden_dim=16, buffer_size=16, batch_size=8))
    path = tmp_path / "legacy.pt"
    model.save(path)
    (tmp_path / "config.json").write_text(json.dumps({"f_max": 15}))
    assert load_action_ranges(path).force == (0, 15)


def test_legacy_evaluation_reads_weights_only_once(tmp_path: Path) -> None:
    from unittest.mock import patch

    import torch

    from src.agents.rl.models.sac import SACConfig
    from src.evaluation.rl import Evaluator
    from src.evaluation.rl.evaluation_sets import (
        DEFAULT_EVALUATION_SET_PATH,
        EvaluationSet,
    )

    model = SAC(SACConfig(7, 3, hidden_dim=16, buffer_size=16, batch_size=8))
    path = tmp_path / "legacy.pt"
    model.save(path)
    (tmp_path / "config.json").write_text(json.dumps({"f_max": 15}))
    cases = EvaluationSet.load(DEFAULT_EVALUATION_SET_PATH, "in_distribution").cases
    with patch.object(torch, "load", wraps=torch.load) as read:
        result = Evaluator.evaluate_checkpoint(path, cases[0].spec, 1, seed=42)
    assert read.call_count == 1
    assert result.metadata["action_ranges"]["force"] == (0, 15)
