import numpy as np

from openpi.policies import forcevla_policy


def test_forcevla_inputs_keep_temporal_gripper_images():
    transform = forcevla_policy.Forcevla_inputs(action_dim=32)
    outputs = transform(
        {
            "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/gripper_image_left": np.zeros((4, 3, 224, 224), dtype=np.float32),
            "observation/gripper_image_right": np.zeros((4, 224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros((7,), dtype=np.float32),
            "observation/gripper_position": np.ones((1,), dtype=np.float32),
            "actions": np.zeros((50, 8), dtype=np.float32),
            "prompt": "test",
        }
    )

    assert outputs["state"].shape == (32,)
    assert outputs["actions"].shape == (50, 32)
    assert outputs["image"]["gripper_left_0_rgb"].shape == (4, 224, 224, 3)
    assert outputs["image"]["gripper_right_0_rgb"].shape == (4, 224, 224, 3)
    parsed = forcevla_policy.Forcevla_outputs()({"actions": np.zeros((50, 32), dtype=np.float32)})
    assert parsed["actions"].shape == (50, 8)
