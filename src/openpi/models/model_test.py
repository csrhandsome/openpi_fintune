import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import model as _model
from openpi.models import pi0_audio
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_audio_model():
    key = jax.random.key(0)
    config = pi0_audio.Pi0AudioConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_audio_whisper_inputs_spec_uses_feature_dim():
    config = pi0_audio.Pi0AudioConfig(use_whisper=True, whisper_model_name="openai/whisper-tiny")
    obs_spec, _ = config.inputs_spec(batch_size=2)

    assert config.audio_feature_dim == 384
    assert config.ambient_audio_num_frames == 50
    assert config.instruction_audio_num_frames == 50
    assert obs_spec.audio.shape == (2, 50, 384)
    assert obs_spec.ambient_audio.shape == (2, 50, 384)
    assert obs_spec.instruction_audio.shape == (2, 50, 384)


def test_pi0_audio_qwen_inputs_spec_uses_feature_dim():
    config = pi0_audio.Pi0AudioConfig(
        use_qwen=True,
        qwen_model_name="Qwen/Qwen3-ASR-0.6B",
        instruction_audio_num_samples=32_000,
    )
    obs_spec, _ = config.inputs_spec(batch_size=2)

    assert config.audio_feature_dim == 1024
    assert config.ambient_audio_num_frames == 13
    assert config.instruction_audio_num_frames == 26
    assert obs_spec.audio.shape == (2, 13, 1024)
    assert obs_spec.ambient_audio.shape == (2, 13, 1024)
    assert obs_spec.instruction_audio.shape == (2, 26, 1024)


def test_pi0_audio_whisper_baseline_inputs_spec_uses_pooled_feature_dim():
    config = pi0_audio.Pi0AudioConfig(
        use_whisper=True,
        audio_tokenizer_type="pooled_baseline",
        whisper_model_name="openai/whisper-tiny",
    )
    obs_spec, _ = config.inputs_spec(batch_size=2)

    assert config.audio_feature_dim == 384
    assert obs_spec.audio.shape == (2, 384)
    assert obs_spec.ambient_audio.shape == (2, 384)
    assert obs_spec.instruction_audio.shape == (2, 384)


def test_pi0_audio_guidance_config():
    config = pi0_audio.Pi0AudioConfig(
        use_audio_guidance=True,
        audio_guidance_lambda=2.0,
    )

    assert config.use_audio_guidance is True
    assert config.audio_guidance_lambda == 2.0

    with pytest.raises(ValueError, match="audio_guidance_lambda must be >= 0"):
        pi0_audio.Pi0AudioConfig(audio_guidance_lambda=-0.1)


def test_pi0_audio_route_head_config():
    config = pi0_audio.Pi0AudioConfig(use_route_head=True, route_loss_weight=0.1)
    obs_spec, _ = config.inputs_spec(batch_size=2)

    assert obs_spec.route_label.shape == (2,)
    assert obs_spec.route_label_mask.shape == (2,)
    assert config.route_label_names == ("straight", "detour", "none")
    assert config.action_loss_weight == 1.0

    with pytest.raises(ValueError, match="route_loss_weight requires use_route_head=True"):
        pi0_audio.Pi0AudioConfig(route_loss_weight=0.1)
    with pytest.raises(ValueError, match="action_loss_weight must be >= 0"):
        pi0_audio.Pi0AudioConfig(action_loss_weight=-0.1)


def test_pi0_audio_route_head_dummy_model():
    key = jax.random.key(0)
    config = pi0_audio.Pi0AudioConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        audio_projector_dim=8,
        ambient_audio_num_tokens=2,
        instruction_audio_num_tokens=2,
        audio_resampler_heads=2,
        use_route_head=True,
        route_loss_weight=0.1,
    )
    model = config.create(key)
    obs, act = config.fake_obs(2), config.fake_act(2)

    logits = nnx_utils.module_jit(model.predict_route_logits)(obs)
    assert logits.shape == (2, 3)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (2, config.action_horizon)


def test_pi0_audio_action_loss_mask_dummy_model():
    key = jax.random.key(1)
    config = pi0_audio.Pi0AudioConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        audio_projector_dim=8,
        ambient_audio_num_tokens=2,
        instruction_audio_num_tokens=2,
        audio_resampler_heads=2,
    )
    model = config.create(key)
    obs, act = config.fake_obs(2), config.fake_act(2)
    obs = dataclasses.replace(obs, action_loss_mask=jnp.array([True, False]))

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)

    assert loss.shape == (2, config.action_horizon)
    assert jnp.all(loss[1] == 0)
    assert jnp.any(loss[0] != 0)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
