"""Read collector-shaped v3 Parquet data through the real LeRobot loader."""

import dataclasses
import json

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.models import model
from openpi.models import pi0_config
from openpi.policies import franka_ee_policy
from openpi.policies import policy
from openpi.shared import normalize
from openpi.training import config
from openpi.training import data_loader
from scripts import compute_norm_stats
from scripts import train


@pytest.fixture
def panda_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    monkeypatch.delenv("OPENPI_DATASET_ROOT", raising=False)
    repo_id = "openpi/panda_v3_test"
    root = tmp_path / "panda_v3_test"
    features = {
        key: {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channel"]}
        for key in ("observation.exterior_image", "observation.wrist_image_left")
    }
    features.update(
        {
            key: {"dtype": "float32", "shape": (dimension,), "names": [f"value_{i}" for i in range(dimension)]}
            for key, dimension in {
                "observation.ee_pose": 6,
                "observation.joint_position": 7,
                "observation.gripper_position": 1,
                "action.ee_pose": 6,
                "action.joint_position": 7,
                "action.gripper_position": 1,
            }.items()
        }
    )
    dataset = LeRobotDataset.create(repo_id=repo_id, root=root, fps=30, features=features, robot_type="panda")
    for episode, length in enumerate((3, 2)):
        for index in range(length):
            value = episode * 10 + index
            target = episode * 10 + min(index + 1, length - 1)
            dataset.add_frame(
                {
                    "observation.exterior_image": np.full((8, 8, 3), index, np.uint8),
                    "observation.wrist_image_left": np.full((8, 8, 3), index + 1, np.uint8),
                    "observation.ee_pose": np.full(6, value, np.float32),
                    "observation.joint_position": np.full(7, value, np.float32),
                    "observation.gripper_position": np.ones(1, np.float32),
                    "action.ee_pose": np.full(6, target, np.float32),
                    "action.joint_position": np.full(7, target, np.float32),
                    "action.gripper_position": np.full(1, 0.25, np.float32),
                    "task": f"task {episode}",
                }
            )
        dataset.save_episode()
        (root / f"episode_{episode:06d}.sync.json").write_text(
            json.dumps(
                {
                    "episode_index": episode,
                    "action_semantics": "next_camera_observation_measured_state",
                }
            )
        )
    dataset.finalize()
    return repo_id, root


def test_v3_split_actions_tasks_and_episode_padding(panda_dataset):
    repo_id, root = panda_dataset
    assert json.loads((root / "meta/info.json").read_text())["codebase_version"] == "v3.0"
    assert (root / "meta/tasks.parquet").is_file()
    data_config = config.DataConfig(
        repo_id=repo_id,
        prompt_from_task=True,
        action_sequence_keys=("action.ee_pose", "action.gripper_position"),
    )
    dataset = data_loader.create_torch_dataset(data_config, 4, pi0_config.Pi0Config())
    first, last, next_episode = dataset[0], dataset[2], dataset[3]
    assert len(dataset) == 5
    assert first["prompt"] == "task 0"
    assert next_episode["prompt"] == "task 1"
    np.testing.assert_array_equal(first["action.ee_pose"][:, 0], [1, 2, 2, 2])
    np.testing.assert_array_equal(last["action.ee_pose"][:, 0], [2, 2, 2, 2])
    np.testing.assert_array_equal(next_episode["action.ee_pose"][:, 0], [11, 11, 11, 11])
    inputs = franka_ee_policy.FrankaEEInputs(model.ModelType.PI0)(first)
    assert inputs["state"].shape == (7,)
    assert inputs["actions"].shape == (4, 7)
    np.testing.assert_array_equal(inputs["actions"][:, -1], 0.25)


@pytest.mark.parametrize("config_name", ["pi0_base_ee_ros_30hz", "pi0_rlt"])
def test_v3_collector_metadata_is_accepted(panda_dataset, config_name):
    repo_id, _ = panda_dataset
    train_config = config.get_config(config_name)
    train_config = dataclasses.replace(train_config, data=dataclasses.replace(train_config.data, repo_id=repo_id))
    train.validate_ros_dataset(train_config)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_v3_data_can_compute_ee_normalization(panda_dataset, num_workers):
    repo_id, _ = panda_dataset
    data_config = config.DataConfig(
        repo_id=repo_id,
        prompt_from_task=True,
        action_sequence_keys=("action.ee_pose", "action.gripper_position"),
        data_transforms=transforms.Group(
            inputs=[
                franka_ee_policy.FrankaEEInputs(model.ModelType.PI0),
                franka_ee_policy.EEDeltaActions(),
            ]
        ),
    )
    loader, num_batches = compute_norm_stats.create_torch_dataloader(
        data_config,
        action_horizon=4,
        batch_size=2,
        model_config=pi0_config.Pi0Config(),
        num_workers=num_workers,
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    assert num_batches == 2
    for batch in loader:
        for key, accumulator in stats.items():
            accumulator.update(np.asarray(batch[key]))
    assert stats["state"].get_statistics().mean.shape == (7,)
    np.testing.assert_allclose(stats["actions"].get_statistics().mean[-1], 0.25)


def test_explicit_root_keeps_the_hub_repo_id(panda_dataset, monkeypatch):
    repo_id, root = panda_dataset
    monkeypatch.setenv("OPENPI_DATASET_ROOT", str(root))
    monkeypatch.setenv("HF_LEROBOT_HOME", str(root / "unused"))
    data_config = config.DataConfig(repo_id=repo_id, action_sequence_keys=("action.ee_pose", "action.gripper_position"))
    dataset = data_loader.create_torch_dataset(data_config, 1, pi0_config.Pi0Config())
    assert dataset.root == root
    assert dataset.repo_id == repo_id


def test_ee_policy_accepts_collector_wire_keys():
    class EEModel(torch.nn.Module):
        def sample_actions(self, observation):
            assert observation.state.shape == (1, 32)
            actions = torch.zeros(1, 16, 32)
            actions[..., 6] = 0.25
            return actions

    ee_policy = policy.Policy(
        EEModel(),
        is_pytorch=True,
        transforms=[
            franka_ee_policy.FrankaEEInputs(model.ModelType.PI0),
            franka_ee_policy.EEDeltaActions(),
            transforms.PadStatesAndActions(32),
        ],
        output_transforms=[franka_ee_policy.EEAbsoluteActions(), franka_ee_policy.FrankaEEOutputs()],
    )
    pose = np.array([0.3, 0.1, 0.4, 0.2, -0.1, 0.5], np.float32)
    result = ee_policy.infer(
        {
            "observation.exterior_image": np.zeros((224, 224, 3), np.uint8),
            "observation.wrist_image_left": np.zeros((224, 224, 3), np.uint8),
            "observation.ee_pose": pose,
            "observation.gripper_position": np.ones(1, np.float32),
        }
    )
    assert result["actions"].shape == (16, 7)
    np.testing.assert_allclose(result["actions"][:, :6], np.tile(pose, (16, 1)), atol=1e-6)
    np.testing.assert_array_equal(result["actions"][:, 6], 0.25)
