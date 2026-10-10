import numpy as np
import pytest

from openpi.models import model
from openpi.models import pi0_config
from openpi.policies.franka_ee_policy import FrankaEEInputs
from openpi.policies.panda_policy import PandaJointInputs
from openpi.training.config import LeRobotFrankaEEDataConfig
from openpi.training.config import LeRobotFrankaJointDataConfig
from openpi.transforms import compose


@pytest.mark.parametrize("space", ["joint", "ee"])
def test_split_targets_are_combined_only_at_model_input(space):
    target = "joint_position" if space == "joint" else "ee_pose"
    dimension = 7 if space == "joint" else 6
    obs = {
        "observation.exterior_image": np.zeros((8, 8, 3), np.uint8),
        "observation.wrist_image_left": np.zeros((8, 8, 3), np.uint8),
        f"observation.{target}": np.arange(dimension, dtype=np.float32),
        "observation.gripper_position": np.ones(1, np.float32),
        "prompt": "test",
    }
    inputs = PandaJointInputs(model.ModelType.PI0) if space == "joint" else FrankaEEInputs(model.ModelType.PI0)
    assert "actions" not in inputs(obs)
    obs.update(
        {
            f"action.{target}": np.ones((4, dimension), np.float32),
            "action.gripper_position": np.full((4, 1), 0.25, np.float32),
        }
    )
    result = inputs(obs)
    assert result["actions"].shape == (4, dimension + 1)
    np.testing.assert_array_equal(
        inputs({**obs, "action.gripper_position": np.full(4, 0.25, np.float32)})["actions"],
        result["actions"],
    )
    np.testing.assert_array_equal(result["actions"][:, -1], 0.25)


@pytest.mark.parametrize(
    ("factory", "target"),
    [
        (LeRobotFrankaJointDataConfig, "joint_position"),
        (LeRobotFrankaEEDataConfig, "ee_pose"),
    ],
)
def test_training_config_reads_canonical_fields_without_dataset_yaml(tmp_path, factory, target):
    cfg = factory(repo_id="local/panda").create(tmp_path, pi0_config.Pi0Config())
    assert tuple(cfg.action_sequence_keys) == (f"action.{target}", "action.gripper_position")
    keys = cfg.repack_transforms.inputs[0].structure
    assert "observation.exterior_image" in keys
    assert "actions" not in keys
    assert "observation.exterior_image_2_left" not in keys
    dimension = 7 if target == "joint_position" else 6
    data = {key: np.zeros(1) for key in keys}
    data.update(
        {
            "observation.exterior_image": np.zeros((8, 8, 3), np.uint8),
            "observation.wrist_image_left": np.zeros((8, 8, 3), np.uint8),
            f"observation.{target}": np.zeros(dimension, np.float32),
            "observation.gripper_position": np.ones(1, np.float32),
            f"action.{target}": np.zeros((4, dimension), np.float32),
            "action.gripper_position": np.ones((4, 1), np.float32),
            "prompt": "test",
        }
    )
    result = compose((*cfg.repack_transforms.inputs, *cfg.data_transforms.inputs))(data)
    assert result["actions"].shape[-1] >= dimension + 1
