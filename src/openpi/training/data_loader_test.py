import dataclasses
import importlib.util
import pathlib
import sys

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _load_train_stages_module():
    module_path = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "train_stages.py"
    spec = importlib.util.spec_from_file_location("train_stages_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_stages = _load_train_stages_module()


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_fake_audio_dataset():
    config = _config.get_config("debug_pi0_audio")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for observation, actions in batches:
        assert observation.audio is not None
        assert observation.audio.shape == (config.batch_size, config.model.ambient_audio_num_samples)
        assert observation.audio_mask is not None
        assert observation.audio_mask.shape == (config.batch_size,)
        assert observation.ambient_audio is not None
        assert observation.ambient_audio.shape == (config.batch_size, config.model.ambient_audio_num_samples)
        assert observation.ambient_audio_mask is not None
        assert observation.ambient_audio_mask.shape == (config.batch_size,)
        assert observation.instruction_audio is not None
        assert observation.instruction_audio.shape == (config.batch_size, config.model.instruction_audio_num_samples)
        assert observation.instruction_audio_mask is not None
        assert observation.instruction_audio_mask.shape == (config.batch_size,)
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


AUDIO_CURRICULUM_CONFIGS = (
    "pi0_base_audio_joint_route_warmup_lora",
    "pi0_base_audio_joint_audio_action_lora",
    "pi0_base_audio_joint_mixed_aa_avla_lora",
    "pi0_base_audio_joint_full_avla_lora",
)


def test_audio_curriculum_configs_are_registered_with_expected_modes():
    route, aa, mixed, full = [_config.get_config(name) for name in AUDIO_CURRICULUM_CONFIGS]

    assert route.model.action_loss_weight == 0.0
    assert route.model.route_loss_weight == 1.0
    assert route.data.conditioning_mode == "audio_action"
    assert aa.data.conditioning_mode == "audio_action"
    assert mixed.data.conditioning_mode == "mixed_aa_avla"
    assert mixed.data.mixed_conditioning_modes == ("audio_action", "full_avla")
    assert full.data.conditioning_mode == "full_avla"
    assert full.lr_schedule.peak_lr == 5e-6
    assert full.lr_schedule.decay_lr == 5e-7


def test_train_stages_default_configs_match_audio_curriculum():
    assert train_stages.StagedTrainArgs(exp_name="test").configs == AUDIO_CURRICULUM_CONFIGS


def test_audio_curriculum_freeze_filters_match_stage_roles():
    param = nnx.Param(jnp.array(0.0))
    route = _config.get_config("pi0_base_audio_joint_route_warmup_lora")
    aa = _config.get_config("pi0_base_audio_joint_audio_action_lora")
    full = _config.get_config("pi0_base_audio_joint_full_avla_lora")

    assert route.trainable_filter(("route_head", "kernel"), param)
    assert route.trainable_filter(("instruction_audio_resampler", "query", "kernel"), param)
    assert not route.trainable_filter(("action_out_proj", "kernel"), param)
    assert aa.trainable_filter(("action_out_proj", "kernel"), param)
    assert aa.trainable_filter(("PaliGemma", "llm", "layers_1", "lora", "kernel"), param)
    assert full.trainable_filter(("PaliGemma", "llm", "layers", "attn", "kernel"), param)


def test_qwen_override_preserves_audio_curriculum_freeze_filter():
    param = nnx.Param(jnp.array(0.0))
    route = _config.enable_qwen_encoder(_config.get_config("pi0_base_audio_joint_route_warmup_lora"))

    assert route.model.use_qwen
    assert not route.model.use_whisper
    assert route.trainable_filter(("route_head", "kernel"), param)
    assert route.trainable_filter(("instruction_audio_resampler", "query", "kernel"), param)
    assert not route.trainable_filter(("action_out_proj", "kernel"), param)


def test_usad_override_preserves_audio_curriculum_freeze_filter():
    param = nnx.Param(jnp.array(0.0))
    route = _config.enable_usad_encoder(_config.get_config("pi0_base_audio_joint_route_warmup_lora"))

    assert route.model.use_usad
    assert not route.model.use_qwen
    assert not route.model.use_whisper
    assert route.trainable_filter(("route_head", "kernel"), param)
    assert route.trainable_filter(("instruction_audio_resampler", "query", "kernel"), param)
    assert not route.trainable_filter(("action_out_proj", "kernel"), param)
