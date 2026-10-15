"""Read collector-shaped v3 Parquet data through the real LeRobot loader."""

import dataclasses
import json

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.models import model
from openpi.models import pi0_config
from openpi.policies import franka_ee_policy
from openpi.policies import franka_policy
from openpi.policies import policy
from openpi.shared import normalize
from openpi.training import config
from openpi.training import data_loader
from openpi.training.validation import validate_dataset
from scripts import compute_norm_stats


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


@pytest.mark.parametrize("config_name", ["openpi", "openpi_rlt"])
def test_v3_collector_metadata_is_accepted(panda_dataset, config_name):
    repo_id, _ = panda_dataset
    train_config = config.get_config(config_name)
    train_config = dataclasses.replace(train_config, data=dataclasses.replace(train_config.data, repo_id=repo_id))
    validate_dataset(train_config)


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


@pytest.fixture
def legacy_mp4_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    monkeypatch.delenv("OPENPI_DATASET_ROOT", raising=False)
    repo_id = "openpi/panda_v3_mp4"
    root = tmp_path / "panda_v3_mp4"
    features = {
        "exterior_image_1_left": {"dtype": "video", "shape": (8, 8, 3), "names": ["height", "width", "channel"]},
        "exterior_image_2_left": {"dtype": "video", "shape": (8, 8, 3), "names": ["height", "width", "channel"]},
        "wrist_image_left": {"dtype": "video", "shape": (8, 8, 3), "names": ["height", "width", "channel"]},
        "joint_position": {"dtype": "float32", "shape": (7,), "names": [f"joint_{i}" for i in range(7)]},
        "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "actions": {"dtype": "float32", "shape": (8,), "names": [f"action_{i}" for i in range(8)]},
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=20,
        features=features,
        robot_type="panda",
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, g=2, preset="fast"),
        image_writer_processes=0,
        image_writer_threads=0,
        metadata_buffer_size=1,
        encoder_threads=1,
    )
    for episode in range(2):
        for index in range(3):
            joint = np.full(7, episode * 10 + index, np.float32)
            nxt = np.full(7, episode * 10 + index + 1, np.float32)
            dataset.add_frame(
                {
                    "exterior_image_1_left": np.full((8, 8, 3), 40 + episode * 20, np.uint8),
                    "exterior_image_2_left": np.zeros((8, 8, 3), np.uint8),
                    "wrist_image_left": np.full((8, 8, 3), 80 + episode * 20, np.uint8),
                    "joint_position": joint,
                    "gripper_position": np.ones(1, np.float32),
                    "actions": np.concatenate((nxt, np.full(1, 0.25, np.float32))),
                    "task": "place the object",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    (root / "audio").mkdir()
    for episode in range(2):
        (root / f"audio/episode_{episode:06d}.sync.json").write_text(
            json.dumps({"episode_index": episode, "frame_records": [{"action_joint_position": [0] * 7}]})
        )
    return repo_id, root


def test_legacy_mp4_frames_decode_into_joint_targets(legacy_mp4_dataset, tmp_path):
    repo_id, root = legacy_mp4_dataset
    info = json.loads((root / "meta/info.json").read_text())
    assert info["features"]["exterior_image_1_left"]["dtype"] == "video"
    assert (root / "videos/exterior_image_1_left").is_dir()

    data_config = config.LeRobotFrankaDataConfig(repo_id=repo_id, action_space="joint", action_dim=8).create(
        tmp_path, pi0_config.Pi0Config()
    )
    data_config = dataclasses.replace(data_config, prompt_from_task=True)
    loaded = data_loader.create_torch_dataset(data_config, 2, pi0_config.Pi0Config())
    sample = loaded[0]
    assert sample["observation.exterior_image"].dtype == torch.uint8
    parsed = franka_policy.FrankaInputs(model.ModelType.PI0, franka_policy.ActionLayout("joint", 8, 7, 1))(sample)
    assert parsed["prompt"] == "place the object"
    assert parsed["image"]["base_0_rgb"].shape == (8, 8, 3)
    assert parsed["image"]["base_0_rgb"].dtype == np.uint8
    np.testing.assert_allclose(parsed["image"]["base_0_rgb"], 40, atol=2)
    np.testing.assert_allclose(parsed["image"]["left_wrist_0_rgb"], 80, atol=2)
    np.testing.assert_allclose(parsed["actions"][:, :7], [[1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2]])
    np.testing.assert_array_equal(parsed["actions"][:, 7], 0.25)
    terminal = loaded[2]
    np.testing.assert_array_equal(terminal["action.joint_position"], np.full((2, 7), 3))
    np.testing.assert_array_equal(terminal["action.joint_position_is_pad"], [False, True])
    np.testing.assert_array_equal(terminal["action.gripper_position_is_pad"], [False, True])
    next_episode = loaded[3]
    np.testing.assert_array_equal(next_episode["observation.joint_position"], np.full(7, 10))
    np.testing.assert_allclose(next_episode["observation.exterior_image"], 60, atol=2)

    train_config = config.get_config("openpi")
    train_config = dataclasses.replace(
        train_config,
        data=dataclasses.replace(train_config.data, repo_id=repo_id, action_space="joint", action_dim=8),
        policy_metadata={"action_fps": 20.0},
    )
    validate_dataset(train_config)
    ee_config = dataclasses.replace(
        train_config,
        data=dataclasses.replace(train_config.data, action_space="ee_delta", action_dim=7),
    )
    with pytest.raises(ValueError, match="joint"):
        validate_dataset(ee_config)
    with pytest.raises(ValueError, match="joint targets"):
        data_loader.create_torch_dataset(
            config.DataConfig(repo_id=repo_id, action_sequence_keys=("action.ee_pose", "action.gripper_position")),
            2,
            pi0_config.Pi0Config(),
        )


@pytest.mark.parametrize("num_workers", [0, 2])
def test_mp4_joint_normalization_and_training_batches(legacy_mp4_dataset, tmp_path, num_workers):
    repo_id, _ = legacy_mp4_dataset
    model_config = pi0_config.Pi0Config(action_horizon=4)
    data_config = config.LeRobotFrankaDataConfig(repo_id=repo_id, action_space="joint", action_dim=8).create(
        tmp_path, model_config
    )
    data_config = dataclasses.replace(data_config, prompt_from_task=True)
    loader, num_batches = compute_norm_stats.create_torch_dataloader(
        data_config, model_config.action_horizon, 2, model_config, num_workers
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    assert num_batches == 3
    for batch in loader:
        for key, accumulator in stats.items():
            accumulator.update(np.asarray(batch[key]))
    norm_stats = {key: accumulator.get_statistics() for key, accumulator in stats.items()}
    assert norm_stats["state"].mean.shape == (8,)
    np.testing.assert_allclose(norm_stats["actions"].mean[-1], 0.25)
    train_loader = data_loader.create_torch_data_loader(
        dataclasses.replace(data_config, norm_stats=norm_stats),
        model_config,
        model_config.action_horizon,
        2,
        num_workers=num_workers,
        num_batches=1,
    )
    observation, actions = next(iter(train_loader))
    assert observation.state.shape == (2, 32)
    assert actions.shape == (2, 4, 32)
    assert torch.isfinite(actions).all()


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
