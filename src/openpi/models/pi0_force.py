import dataclasses
import logging
from typing import cast

import einops
import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.image_process_model as _cbam
import openpi.models.gemma as _gemma
import openpi.models.moe as _moe
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@at.typecheck
class ForceAdaMoEFusion(nn.Module):
    """force_embs + prefix_embs -> LN+SelfAttn+Res -> LN+AdaMoE+Res -> Linear."""

    width: int
    out_dim: int
    num_heads: int
    mlp_dim: int
    num_experts: int = 4
    top_k: int = 1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        prefix_embs: jnp.ndarray,
        force_embs: jnp.ndarray,
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        x = jnp.concatenate([prefix_embs, force_embs], axis=1)

        attn_in = nn.LayerNorm(dtype=self.dtype, name="pre_attn_norm")(x)
        attn_out = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            num_heads=self.num_heads,
            qkv_features=self.width,
            out_features=self.width,
            force_fp32_for_softmax=True,
            name="self_attn",
        )(attn_in, attn_in)
        x = x + attn_out

        moe_in = nn.LayerNorm(dtype=self.dtype, name="pre_moe_norm")(x)
        moe_out, moe_info = _moe.AdaMoEFeedForward(
            expert_dim=self.width,
            hidden_dim=self.mlp_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            name="adamoe_ffn",
        )(moe_in)
        x = x + moe_out

        x = nn.Dense(
            features=self.out_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="out_proj",
        )(x)
        return x, moe_info


@dataclasses.dataclass(frozen=True)
class Pi0_GuidanceConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0_Guidance":
        return Pi0_Guidance(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = cast(
                _model.Observation,
                _model.Observation(  # pyright: ignore[reportArgumentType]
                    images={
                        "base_0_rgb": image_spec,
                        "left_wrist_0_rgb": image_spec,
                        "right_wrist_0_rgb": image_spec,
                    },
                    image_masks={
                        "base_0_rgb": image_mask_spec,
                        "left_wrist_0_rgb": image_mask_spec,
                        "right_wrist_0_rgb": image_mask_spec,
                    },
                    state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                    tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                    tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                ),
            )
        action_spec = cast(
            _model.Actions,
            jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32),
        )
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)
            has_lora = True

        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0_Guidance(_model.BaseModel):
    def __init__(self, config: Pi0_GuidanceConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.force_cbam = nnx_bridge.ToNNX(_cbam.CBAM(dtype=jnp.float32))
        self.force_cbam.lazy_init(
            jnp.zeros((1, *_model.IMAGE_RESOLUTION, 6), dtype=jnp.float32),
            rngs=rngs,
        )
        force_gru_hidden = min(512, paligemma_config.width)
        self.force_temporal_gru = nnx_bridge.ToNNX(
            _cbam.TemporalGRUEncoder(hidden_dim=force_gru_hidden, dtype=jnp.float32)
        )
        self.force_temporal_gru.lazy_init(
            jnp.zeros((1, 1, 6), dtype=jnp.float32),
            rngs=rngs,
        )
        self.force_proj = nnx.Linear(force_gru_hidden, paligemma_config.width, rngs=rngs)

        self.force_fusion = nnx_bridge.ToNNX(
            ForceAdaMoEFusion(
                width=paligemma_config.width,
                out_dim=action_expert_config.width,
                mlp_dim=paligemma_config.width,
                num_heads=paligemma_config.num_heads,
                num_experts=4,
                top_k=1,
                dtype=jnp.float32,
            )
        )
        self.force_fusion.lazy_init(
            jnp.zeros((1, 32, paligemma_config.width), dtype=jnp.float32),
            jnp.zeros((1, 1, paligemma_config.width), dtype=jnp.float32),
            True,
            rngs=rngs,
        )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        observations = jnp.zeros_like(obs.state)
        observations = observations.at[:, :7].set(obs.state[:, :7])
        state_token = self.state_proj(observations)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        ar_mask += [True]

        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _embed_force_images(self, obs: _model.Observation) -> at.Float[at.Array, "b 1 d"]:
        left_wrist = obs.images["left_wrist_0_rgb"]
        right_wrist = obs.images["right_wrist_0_rgb"]

        if left_wrist.ndim == 4 and right_wrist.ndim == 4:
            # Default path: single frame per wrist, converted to a sequence with T=1.
            force_images = jnp.concatenate([left_wrist, right_wrist], axis=-1)[:, None, ...]
        elif left_wrist.ndim == 5 and right_wrist.ndim == 5:
            # Optional temporal path: [B, T, H, W, C] from each wrist camera.
            force_images = jnp.concatenate([left_wrist, right_wrist], axis=-1)
        else:
            raise ValueError(
                "left/right wrist image ranks must match and be either 4D or 5D. "
                f"Got left={left_wrist.shape}, right={right_wrist.shape}."
            )

        force_images = force_images.astype(jnp.float32)
        batch_size, seq_len, height, width, channels = force_images.shape
        cbam_features = self.force_cbam(jnp.reshape(force_images, (batch_size * seq_len, height, width, channels)))
        pooled = jnp.reshape(jnp.mean(cbam_features, axis=(1, 2)), (batch_size, seq_len, channels))
        temporal_features = self.force_temporal_gru(pooled)
        return self.force_proj(temporal_features)[:, None, :]

    def _fused_action_features(
        self,
        prefix_embs: at.Float[at.Array, "b s d"],
        force_embs: at.Float[at.Array, "b f d"],
        *,
        deterministic: bool,
    ) -> at.Float[at.Array, "b ah d"]:
        fused_features, _ = self.force_fusion(prefix_embs, force_embs, deterministic)
        fused_features = fused_features.astype(prefix_embs.dtype)

        if fused_features.shape[1] >= self.action_horizon:
            return fused_features[:, -self.action_horizon :]

        pooled = jnp.mean(fused_features, axis=1, keepdims=True)
        return jnp.tile(pooled, (1, self.action_horizon, 1))

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions)

        force_embs = self._embed_force_images(observation)
        fvl_features = self._fused_action_features(prefix_out, force_embs, deterministic=not train)
        suffix_action_features = suffix_out[:, -self.action_horizon :]
        v_t = self.action_out_proj(fvl_features + suffix_action_features)
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        force_embs = self._embed_force_images(observation)
        fvl_features = self._fused_action_features(prefix_out, force_embs, deterministic=True)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_step, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_step is None
            v_t = self.action_out_proj(fvl_features + suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
