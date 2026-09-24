import json

import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_copy_present_audio_fields():
    inputs = _transforms.copy_present_audio_fields(
        {
            "observation/audio": np.array([1.0, 2.0], dtype=np.float32),
            "observation/audio_mask": np.True_,
            "frame_index": np.int32(7),
        },
        {"state": np.array([0.0], dtype=np.float32)},
    )

    assert np.allclose(inputs["ambient_audio"], np.array([1.0, 2.0], dtype=np.float32))
    assert inputs["ambient_audio_mask"] == np.True_
    assert np.allclose(inputs["audio"], np.array([1.0, 2.0], dtype=np.float32))
    assert inputs["audio_mask"] == np.True_
    assert inputs["frame_index"] == np.int32(7)


def test_extract_audio_window_from_raw_audio():
    transform = _transforms.ExtractAudioWindow(window_num_samples=6, target_sample_rate=16_000)

    output = transform(
        {
            "audio": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "audio_sample_rate": np.int32(16_000),
        }
    )

    assert output["audio"].shape == (6,)
    assert output["ambient_audio"].shape == (6,)
    assert np.allclose(output["audio"], np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(output["ambient_audio"], np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    assert output["audio_mask"] == np.True_
    assert output["ambient_audio_mask"] == np.True_


def test_extract_instruction_audio_window_from_raw_audio():
    transform = _transforms.ExtractInstructionAudioWindow(window_num_samples=6, target_sample_rate=16_000)

    output = transform(
        {
            "instruction_audio": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "instruction_audio_sample_rate": np.int32(16_000),
        }
    )

    assert output["instruction_audio"].shape == (6,)
    assert np.allclose(
        output["instruction_audio"],
        np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    )
    assert output["instruction_audio_mask"] == np.True_


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def test_extract_route_label_from_direct_label():
    output = _transforms.ExtractRouteLabel()({"label": "detour"})

    assert output["route_label"] == np.int32(_transforms.ROUTE_LABEL_TO_ID["detour"])
    assert output["route_label_mask"] == np.True_


def test_extract_route_label_from_sync_sidecar(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sync_path = audio_dir / "episode_000003.sync.json"
    sync_path.write_text(
        json.dumps({"label": "straight", "instruction_audio_window": {"start_sec": 0.5, "end_sec": 0.6}}),
        encoding="utf-8",
    )

    output = _transforms.ExtractRouteLabel(dataset_root=str(tmp_path))(
        {"episode_index": np.int64(3), "frame_index": np.int64(10)}
    )

    assert output["route_label"] == np.int32(_transforms.ROUTE_LABEL_TO_ID["straight"])
    assert output["route_label_mask"] == np.True_


def test_extract_route_label_from_sync_sidecar_outside_window_is_masked(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sync_path = audio_dir / "episode_000003.sync.json"
    sync_path.write_text(
        json.dumps({"label": "detour", "instruction_audio_window": {"start_sec": 0.5, "end_sec": 0.6}}),
        encoding="utf-8",
    )

    output = _transforms.ExtractRouteLabel(dataset_root=str(tmp_path))(
        {"episode_index": np.int64(3), "frame_index": np.int64(20)}
    )

    assert output["route_label"] == np.int32(_transforms.ROUTE_LABEL_TO_ID["detour"])
    assert output["route_label_mask"] == np.False_


def test_extract_route_label_missing_is_masked():
    output = _transforms.ExtractRouteLabel()({})

    assert output["route_label"] == np.int32(_transforms.INVALID_ROUTE_LABEL)
    assert output["route_label_mask"] == np.False_


def _audio_vla_conditioning_sample(frame_index: int = 0) -> dict:
    return {
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
        "audio_mask": np.True_,
        "ambient_audio_mask": np.True_,
        "instruction_audio_mask": np.True_,
        "frame_index": np.int32(frame_index),
    }


def test_audio_vla_conditioning_audio_action_masks_images():
    output = _transforms.AudioVLAConditioning(mode="audio_action")(_audio_vla_conditioning_sample())

    assert all(mask == np.False_ for mask in output["image_mask"].values())
    assert output["audio_mask"] == np.True_
    assert output["ambient_audio_mask"] == np.True_
    assert output["instruction_audio_mask"] == np.True_


def test_audio_vla_conditioning_image_action_nulls_audio():
    output = _transforms.AudioVLAConditioning(mode="image_action")(_audio_vla_conditioning_sample())

    assert all(mask == np.True_ for mask in output["image_mask"].values())
    assert output["audio_mask"] == np.False_
    assert output["ambient_audio_mask"] == np.False_
    assert output["instruction_audio_mask"] == np.False_


def test_audio_vla_conditioning_mixed_uses_stable_frame_index():
    transform = _transforms.AudioVLAConditioning(
        mode="mixed_aa_avla",
        mixed_modes=("audio_action", "full_avla"),
        stable_index_keys=("frame_index",),
    )

    audio_action = transform(_audio_vla_conditioning_sample(frame_index=0))
    full_avla = transform(_audio_vla_conditioning_sample(frame_index=1))

    assert all(mask == np.False_ for mask in audio_action["image_mask"].values())
    assert all(mask == np.True_ for mask in full_avla["image_mask"].values())
