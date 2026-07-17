import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.pi0_config as pi0_config
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.shared import audio_tools as _audio_tools

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"],
    embedding_dim: int,
    min_period: float,
    max_period: float,
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


@dataclasses.dataclass(frozen=True)
class Pi0AudioConfig(pi0_config.Pi0Config):
    ambient_audio_num_samples: int = 16_000
    instruction_audio_num_samples: int = 16_000
    audio_sample_rate: int = 16_000
    audio_projector_dim: int = 512
    ambient_audio_num_tokens: int = 32
    instruction_audio_num_tokens: int = 24
    audio_resampler_heads: int = 8
    use_whisper: bool = False
    use_qwen: bool = False
    use_usad: bool = False
    audio_tokenizer_type: str = "sequence_resampler"
    whisper_model_name: str = "openai/whisper-tiny"
    whisper_output_dim: int | None = None
    whisper_device: str = "cpu"
    qwen_model_name: str = "Qwen/Qwen3-ASR-0.6B"
    qwen_output_dim: int | None = None
    qwen_device: str = "auto"
    usad_model_name: str = "MIT-SLS/USAD2-Large"
    usad_output_dim: int | None = None
    usad_device: str = "auto"
    use_audio_limoe: bool = False
    audio_limoe_num_experts: int = 4
    audio_limoe_top_k: int = 1
    rtc_train_max_delay: int = 0
    use_route_head: bool = False
    route_loss_weight: float = 0.0
    route_loss_requires_instruction_audio: bool = True
    action_loss_weight: float = 1.0
    route_label_names: tuple[str, ...] = ("straight", "detour", "none")
    use_audio_adaln_gate: bool = False
    # Audio-CAG: v = v_no_audio + lambda * (v_audio - v_no_audio).
    # lambda=1.0 keeps the original audio-conditioned velocity; >1.0 amplifies audio.
    use_audio_guidance: bool = False
    audio_guidance_lambda: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.ambient_audio_num_samples <= 0:
            raise ValueError("ambient_audio_num_samples must be > 0")
        if self.instruction_audio_num_samples <= 0:
            raise ValueError("instruction_audio_num_samples must be > 0")
        if self.audio_sample_rate <= 0:
            raise ValueError("audio_sample_rate must be > 0")
        if self.audio_projector_dim <= 0:
            raise ValueError("audio_projector_dim must be > 0")
        if self.ambient_audio_num_tokens <= 0:
            raise ValueError("ambient_audio_num_tokens must be > 0")
        if self.instruction_audio_num_tokens <= 0:
            raise ValueError("instruction_audio_num_tokens must be > 0")
        if self.audio_resampler_heads <= 0:
            raise ValueError("audio_resampler_heads must be > 0")
        if self.audio_tokenizer_type not in ("sequence_resampler", "pooled_baseline"):
            raise ValueError("audio_tokenizer_type must be 'sequence_resampler' or 'pooled_baseline'")
        if sum((self.use_whisper, self.use_qwen, self.use_usad)) > 1:
            raise ValueError("Only one of use_whisper, use_qwen, and use_usad can be True")
        if self.audio_limoe_num_experts <= 0:
            raise ValueError("audio_limoe_num_experts must be > 0")
        if self.audio_limoe_top_k <= 0:
            raise ValueError("audio_limoe_top_k must be > 0")
        if self.whisper_output_dim is not None and self.whisper_output_dim <= 0:
            raise ValueError("whisper_output_dim must be > 0")
        if self.qwen_output_dim is not None and self.qwen_output_dim <= 0:
            raise ValueError("qwen_output_dim must be > 0")
        if self.usad_output_dim is not None and self.usad_output_dim <= 0:
            raise ValueError("usad_output_dim must be > 0")
        if self.rtc_train_max_delay < 0:
            raise ValueError("rtc_train_max_delay must be >= 0")
        if self.rtc_train_max_delay >= self.action_horizon:
            raise ValueError("rtc_train_max_delay must be smaller than action_horizon")
        if len(self.route_label_names) < 2:
            raise ValueError("route_label_names must contain at least two labels")
        if len(set(self.route_label_names)) != len(self.route_label_names):
            raise ValueError("route_label_names must be unique")
        if self.route_loss_weight < 0:
            raise ValueError("route_loss_weight must be >= 0")
        if self.action_loss_weight < 0:
            raise ValueError("action_loss_weight must be >= 0")
        if self.use_audio_adaln_gate and self.audio_tokenizer_type != "sequence_resampler":
            raise ValueError("use_audio_adaln_gate currently requires audio_tokenizer_type='sequence_resampler'")
        if self.route_loss_weight > 0 and not self.use_route_head:
            raise ValueError("route_loss_weight requires use_route_head=True")
        if self.audio_guidance_lambda < 0:
            raise ValueError("audio_guidance_lambda must be >= 0")

    @property
    def audio_feature_dim(self) -> int:
        if self.use_whisper:
            if self.whisper_output_dim is not None:
                return self.whisper_output_dim
            return _audio_tools.infer_whisper_hidden_size(self.whisper_model_name)
        if self.use_qwen:
            if self.qwen_output_dim is not None:
                return self.qwen_output_dim
            return _audio_tools.infer_qwen_asr_hidden_size(self.qwen_model_name)
        if self.use_usad:
            if self.usad_output_dim is not None:
                return self.usad_output_dim
            return _audio_tools.infer_usad_hidden_size(self.usad_model_name)
        return 1

    @property
    def ambient_audio_input_shape(self) -> tuple[int, ...]:
        if self.use_whisper or self.use_qwen or self.use_usad:
            if self.audio_tokenizer_type == "pooled_baseline":
                return (self.audio_feature_dim,)
            return (self.ambient_audio_num_frames, self.audio_feature_dim)
        return (self.ambient_audio_num_samples,)

    @property
    def instruction_audio_input_shape(self) -> tuple[int, ...]:
        if self.use_whisper or self.use_qwen or self.use_usad:
            if self.audio_tokenizer_type == "pooled_baseline":
                return (self.audio_feature_dim,)
            return (self.instruction_audio_num_frames, self.audio_feature_dim)
        return (self.instruction_audio_num_samples,)

    @property
    def audio_input_shape(self) -> tuple[int, ...]:
        return self.ambient_audio_input_shape

    @property
    def ambient_audio_num_frames(self) -> int:
        if self.use_qwen:
            return _audio_tools.qwen_asr_num_frames_for_duration(
                self.ambient_audio_num_samples,
                sample_rate=self.audio_sample_rate,
            )
        if self.use_usad:
            return _audio_tools.usad_num_frames_for_duration(
                self.ambient_audio_num_samples,
                sample_rate=self.audio_sample_rate,
                model_name=self.usad_model_name,
            )
        return _audio_tools.whisper_num_frames_for_duration(
            self.ambient_audio_num_samples,
            sample_rate=self.audio_sample_rate,
        )

    @property
    def instruction_audio_num_frames(self) -> int:
        if self.use_qwen:
            return _audio_tools.qwen_asr_num_frames_for_duration(
                self.instruction_audio_num_samples,
                sample_rate=self.audio_sample_rate,
            )
        if self.use_usad:
            return _audio_tools.usad_num_frames_for_duration(
                self.instruction_audio_num_samples,
                sample_rate=self.audio_sample_rate,
                model_name=self.usad_model_name,
            )
        return _audio_tools.whisper_num_frames_for_duration(
            self.instruction_audio_num_samples,
            sample_rate=self.audio_sample_rate,
        )

    @property
    def audio_num_samples(self) -> int:
        return self.ambient_audio_num_samples

    @property
    def audio_num_tokens(self) -> int:
        return self.ambient_audio_num_tokens

    @property
    def audio_num_frames(self) -> int:
        return self.ambient_audio_num_frames

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_AUDIO

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Audio":
        return Pi0Audio(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
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
                audio=jax.ShapeDtypeStruct([batch_size, *self.audio_input_shape], jnp.float32),
                audio_mask=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                ambient_audio=jax.ShapeDtypeStruct([batch_size, *self.ambient_audio_input_shape], jnp.float32),
                ambient_audio_mask=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                instruction_audio=jax.ShapeDtypeStruct([batch_size, *self.instruction_audio_input_shape], jnp.float32),
                instruction_audio_mask=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                route_label=(jax.ShapeDtypeStruct([batch_size], jnp.int32) if self.use_route_head else None),
                route_label_mask=(jax.ShapeDtypeStruct([batch_size], jnp.bool_) if self.use_route_head else None),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def load_pytorch(self, train_config, weight_path: str):
        raise NotImplementedError("Pi0AudioConfig currently supports JAX checkpoints only.")


class Pi0Audio(_model.BaseModel):
    def __init__(self, config: Pi0AudioConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.use_audio_limoe = config.use_audio_limoe
        self.audio_limoe_num_experts = config.audio_limoe_num_experts
        self.ambient_audio_num_tokens = config.ambient_audio_num_tokens
        self.instruction_audio_num_tokens = config.instruction_audio_num_tokens
        self.audio_tokenizer_type = config.audio_tokenizer_type
        self.rtc_train_max_delay = config.rtc_train_max_delay
        self.use_route_head = config.use_route_head
        self.route_loss_weight = config.route_loss_weight
        self.route_loss_requires_instruction_audio = config.route_loss_requires_instruction_audio
        self.action_loss_weight = config.action_loss_weight
        self.route_label_names = tuple(config.route_label_names)
        self.route_num_classes = len(config.route_label_names)
        self.use_audio_adaln_gate = config.use_audio_adaln_gate
        self.use_audio_guidance = config.use_audio_guidance
        self.audio_guidance_lambda = config.audio_guidance_lambda

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True] if config.pi05 or config.use_audio_adaln_gate else [False, False],
        )
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
        self.ambient_audio_in_proj = nnx.Linear(config.audio_feature_dim, config.audio_projector_dim, rngs=rngs)
        self.instruction_audio_in_proj = nnx.Linear(config.audio_feature_dim, config.audio_projector_dim, rngs=rngs)
        if self.audio_tokenizer_type == "sequence_resampler":
            self.ambient_audio_resampler = nnx.MultiHeadAttention(
                num_heads=config.audio_resampler_heads,
                in_features=config.audio_projector_dim,
                qkv_features=config.audio_projector_dim,
                out_features=config.audio_projector_dim,
                decode=False,
                rngs=rngs,
            )
            self.instruction_audio_resampler = nnx.MultiHeadAttention(
                num_heads=config.audio_resampler_heads,
                in_features=config.audio_projector_dim,
                qkv_features=config.audio_projector_dim,
                out_features=config.audio_projector_dim,
                decode=False,
                rngs=rngs,
            )
            self.ambient_audio_query_tokens = nnx.Param(
                jax.random.normal(
                    rngs.params(),
                    (config.ambient_audio_num_tokens, config.audio_projector_dim),
                    dtype=jnp.float32,
                )
                * 0.02
            )
            self.instruction_audio_query_tokens = nnx.Param(
                jax.random.normal(
                    rngs.params(),
                    (config.instruction_audio_num_tokens, config.audio_projector_dim),
                    dtype=jnp.float32,
                )
                * 0.02
            )
            self.ambient_audio_query_norm = nnx.LayerNorm(config.audio_projector_dim, rngs=rngs)
            self.ambient_audio_kv_norm = nnx.LayerNorm(config.audio_projector_dim, rngs=rngs)
            self.instruction_audio_query_norm = nnx.LayerNorm(config.audio_projector_dim, rngs=rngs)
            self.instruction_audio_kv_norm = nnx.LayerNorm(config.audio_projector_dim, rngs=rngs)
        else:
            self.ambient_audio_resampler = None
            self.instruction_audio_resampler = None
            self.ambient_audio_query_tokens = None
            self.instruction_audio_query_tokens = None
            self.ambient_audio_query_norm = None
            self.ambient_audio_kv_norm = None
            self.instruction_audio_query_norm = None
            self.instruction_audio_kv_norm = None
        self.ambient_audio_out_proj = nnx.Linear(config.audio_projector_dim, paligemma_config.width, rngs=rngs)
        self.instruction_audio_out_proj = nnx.Linear(config.audio_projector_dim, paligemma_config.width, rngs=rngs)
        self.instruction_audio_pool_dim = paligemma_config.width
        self.audio_adaln_mlp_in = (
            nnx.Linear(paligemma_config.width, action_expert_config.width, rngs=rngs)
            if self.use_audio_adaln_gate
            else None
        )
        self.audio_adaln_mlp_out = (
            nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            if self.use_audio_adaln_gate
            else None
        )
        self.route_head = (
            nnx.Linear(paligemma_config.width, self.route_num_classes, rngs=rngs) if self.use_route_head else None
        )
        if config.use_audio_limoe:
            try:
                from openpi.models import limoe_simple
            except ImportError as exc:
                raise ImportError(
                    "use_audio_limoe=True requires the dependencies for openpi.models.limoe_simple "
                    "(for example flaxformer) to be installed."
                ) from exc

            audio_limoe = nnx_bridge.ToNNX(
                limoe_simple.LIMoEBlock(
                    mlp_dim=paligemma_config.width,
                    num_heads=paligemma_config.num_heads,
                    out_dim=paligemma_config.width,
                    dtype=jnp.float32,
                    num_experts=config.audio_limoe_num_experts,
                    num_top_k=config.audio_limoe_top_k,
                )
            )
            audio_limoe.lazy_init(
                jnp.zeros(
                    (1, config.audio_limoe_num_experts, paligemma_config.width),
                    dtype=jnp.float32,
                ),
                deterministic=True,
                rngs=rngs,
            )
            self.audio_limoe = audio_limoe
        else:
            self.audio_limoe = None

        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.deterministic = True

    def _prepare_audio_sequence(
        self,
        audio_value,
        audio_mask_value,
    ) -> tuple[at.Float[at.Array, "b t d"] | None, at.Bool[at.Array, "b t"] | None]:
        if audio_value is None:
            return None, None

        audio = jnp.asarray(audio_value, dtype=jnp.float32)
        original_ndim = audio.ndim
        if original_ndim == 2:
            audio = audio[..., None]
        elif original_ndim != 3:
            raise ValueError(f"Audio input must have rank 2 or 3, got shape {audio.shape}")

        if audio_mask_value is None:
            audio_mask = jnp.ones((audio.shape[0],), dtype=jnp.bool_)
        else:
            audio_mask = jnp.asarray(audio_mask_value, dtype=jnp.bool_)

        if audio.shape[1] == 0:
            return None, None

        if original_ndim == 2:
            frame_mask = jnp.broadcast_to(audio_mask[:, None], audio.shape[:2])
        else:
            frame_energy = jnp.max(jnp.abs(audio), axis=-1)
            frame_mask = frame_energy > 0
            frame_mask = jnp.where(audio_mask[:, None], frame_mask, jnp.zeros_like(frame_mask))
        return audio, frame_mask

    def _embed_audio_prefix(
        self,
        audio_value,
        audio_mask_value,
        *,
        in_proj,
        resampler,
        query_tokens_param,
        query_norm,
        kv_norm,
        out_proj,
        num_tokens: int,
    ) -> tuple[at.Float[at.Array, "b s emb"] | None, at.Bool[at.Array, "b s"] | None]:
        if self.audio_tokenizer_type == "pooled_baseline":
            return self._embed_audio_prefix_baseline(
                audio_value,
                audio_mask_value,
                in_proj=in_proj,
                out_proj=out_proj,
            )

        audio, frame_mask = self._prepare_audio_sequence(audio_value, audio_mask_value)
        if audio is None or frame_mask is None:
            return None, None

        audio_hidden = in_proj(audio)
        audio_hidden = nnx.swish(audio_hidden)
        audio_hidden = kv_norm(audio_hidden)

        batch_size = audio_hidden.shape[0]
        query_tokens = jnp.broadcast_to(
            query_tokens_param.value[None, :, :],
            (
                batch_size,
                num_tokens,
                query_tokens_param.value.shape[-1],
            ),
        )
        query_tokens = query_norm(query_tokens)

        attn_mask = frame_mask[:, None, None, :]
        audio_tokens = resampler(
            query_tokens,
            audio_hidden,
            mask=attn_mask,
            decode=False,
        )
        audio_tokens = nnx.swish(audio_tokens)
        audio_tokens = out_proj(audio_tokens)

        if audio_mask_value is None:
            audio_mask = jnp.ones((batch_size, num_tokens), dtype=jnp.bool_)
        else:
            audio_mask = jnp.broadcast_to(
                jnp.asarray(audio_mask_value, dtype=jnp.bool_)[:, None],
                (batch_size, num_tokens),
            )
        audio_tokens = jnp.where(audio_mask[..., None], audio_tokens, 0.0)
        return audio_tokens, audio_mask

    def _embed_audio_prefix_baseline(
        self,
        audio_value,
        audio_mask_value,
        *,
        in_proj,
        out_proj,
    ) -> tuple[at.Float[at.Array, "b 1 emb"] | None, at.Bool[at.Array, "b 1"] | None]:
        if audio_value is None:
            return None, None

        audio = jnp.asarray(audio_value, dtype=jnp.float32)
        if audio.ndim != 2:
            raise ValueError(f"pooled_baseline expects audio features with shape [B, D]. Got shape {audio.shape}")

        audio_hidden = in_proj(audio)
        audio_hidden = nnx.swish(audio_hidden)
        audio_token = out_proj(audio_hidden)[:, None, :]

        if audio_mask_value is None:
            audio_mask = jnp.ones((audio.shape[0], 1), dtype=jnp.bool_)
        else:
            audio_mask = jnp.asarray(audio_mask_value, dtype=jnp.bool_)[:, None]
        audio_token = jnp.where(audio_mask[..., None], audio_token, 0.0)
        return audio_token, audio_mask

    def _apply_audio_limoe(
        self,
        tokens: at.Float[at.Array, "b s emb"],
        input_mask: at.Bool[at.Array, "b s"],
        *,
        train: bool,
    ) -> at.Float[at.Array, "b s emb"]:
        batch_size, seq_len, _ = tokens.shape
        pad_seq = 0
        while (batch_size * (seq_len + pad_seq)) % self.audio_limoe_num_experts != 0:
            pad_seq += 1

        limoe_tokens = tokens.astype(jnp.float32)
        padded_mask = input_mask
        limoe_tokens = limoe_tokens * padded_mask[..., None].astype(limoe_tokens.dtype)
        if pad_seq:
            limoe_tokens = jnp.pad(limoe_tokens, ((0, 0), (0, pad_seq), (0, 0)))
            padded_mask = jnp.pad(padded_mask, ((0, 0), (0, pad_seq)), constant_values=False)

        limoe_out, _ = self.audio_limoe(limoe_tokens, deterministic=not train)
        limoe_out = limoe_out * padded_mask[..., None].astype(limoe_out.dtype)
        if pad_seq:
            limoe_out = limoe_out[:, :seq_len]
        return limoe_out.astype(tokens.dtype)

    def _drop_audio_conditioning(self, obs: _model.Observation) -> _model.Observation:
        return _model.Observation(
            images=obs.images,
            image_masks=obs.image_masks,
            state=obs.state,
            audio=None,
            audio_mask=None,
            ambient_audio=None,
            ambient_audio_mask=None,
            instruction_audio=None,
            instruction_audio_mask=None,
            route_label=obs.route_label,
            route_label_mask=obs.route_label_mask,
            tokenized_prompt=obs.tokenized_prompt,
            tokenized_prompt_mask=obs.tokenized_prompt_mask,
            token_ar_mask=obs.token_ar_mask,
            token_loss_mask=obs.token_loss_mask,
        )

    def _embed_instruction_audio_tokens(
        self,
        obs: _model.Observation,
    ) -> tuple[at.Float[at.Array, "b s emb"] | None, at.Bool[at.Array, "b s"] | None]:
        return self._embed_audio_prefix(
            obs.instruction_audio,
            obs.instruction_audio_mask,
            in_proj=self.instruction_audio_in_proj,
            resampler=self.instruction_audio_resampler,
            query_tokens_param=self.instruction_audio_query_tokens,
            query_norm=self.instruction_audio_query_norm,
            kv_norm=self.instruction_audio_kv_norm,
            out_proj=self.instruction_audio_out_proj,
            num_tokens=self.instruction_audio_num_tokens,
        )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        # language
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        # 添加audio通过vad识别的指令槽的embed
        instruction_audio_token, instruction_audio_mask = self._embed_instruction_audio_tokens(obs)
        if instruction_audio_token is not None and instruction_audio_mask is not None:
            target_dtype = tokens[0].dtype if tokens else instruction_audio_token.dtype
            instruction_audio_token = instruction_audio_token.astype(target_dtype)
            tokens.append(instruction_audio_token)
            input_mask.append(instruction_audio_mask)
            ar_mask += [False] * instruction_audio_token.shape[1]

        # img
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        ambient_audio_token, ambient_audio_mask = self._embed_audio_prefix(
            obs.ambient_audio if obs.ambient_audio is not None else obs.audio,
            obs.ambient_audio_mask if obs.ambient_audio_mask is not None else obs.audio_mask,
            in_proj=self.ambient_audio_in_proj,
            resampler=self.ambient_audio_resampler,
            query_tokens_param=self.ambient_audio_query_tokens,
            query_norm=self.ambient_audio_query_norm,
            kv_norm=self.ambient_audio_kv_norm,
            out_proj=self.ambient_audio_out_proj,
            num_tokens=self.ambient_audio_num_tokens,
        )
        if ambient_audio_token is not None and ambient_audio_mask is not None:
            target_dtype = tokens[0].dtype if tokens else ambient_audio_token.dtype
            ambient_audio_token = ambient_audio_token.astype(target_dtype)
            tokens.append(ambient_audio_token)
            input_mask.append(ambient_audio_mask)
            ar_mask += [False] * ambient_audio_token.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        # moe
        if self.audio_limoe:
            tokens = self._apply_audio_limoe(tokens, input_mask, train=train)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _pool_instruction_audio(
        self,
        obs: _model.Observation,
    ) -> at.Float[at.Array, "b emb"]:
        instruction_audio_token, instruction_audio_mask = self._embed_instruction_audio_tokens(obs)
        batch_size = obs.state.shape[0]
        if instruction_audio_token is None or instruction_audio_mask is None:
            return jnp.zeros((batch_size, self.instruction_audio_pool_dim), dtype=jnp.float32)

        token_mask = instruction_audio_mask[..., None].astype(instruction_audio_token.dtype)
        token_count = jnp.maximum(jnp.sum(token_mask, axis=1), 1.0)
        return jnp.sum(instruction_audio_token * token_mask, axis=1) / token_count

    def _audio_adaln_cond_from_preprocessed(
        self,
        obs: _model.Observation,
    ) -> at.Float[at.Array, "b emb"] | None:
        if not self.use_audio_adaln_gate:
            return None
        if self.audio_adaln_mlp_in is None or self.audio_adaln_mlp_out is None:
            raise ValueError("AudioAdaLN gate modules are not initialized.")

        audio_cond = self._pool_instruction_audio(obs)
        audio_cond = self.audio_adaln_mlp_in(audio_cond)
        audio_cond = nnx.swish(audio_cond)
        return self.audio_adaln_mlp_out(audio_cond)

    def _merge_adarms_cond(
        self,
        base_cond: at.Float[at.Array, "b ..."] | None,
        audio_cond: at.Float[at.Array, "b emb"] | None,
    ) -> at.Float[at.Array, "b ..."] | None:
        if audio_cond is None:
            return base_cond
        if base_cond is None:
            return audio_cond
        if base_cond.ndim == audio_cond.ndim + 1:
            audio_cond = audio_cond[:, None, :]
        elif base_cond.ndim != audio_cond.ndim:
            raise ValueError(f"Cannot merge adarms cond shapes {base_cond.shape} and {audio_cond.shape}")
        return base_cond + audio_cond.astype(base_cond.dtype)

    def _predict_route_logits_from_preprocessed(
        self,
        obs: _model.Observation,
    ) -> at.Float[at.Array, "b c"]:
        if not self.use_route_head or self.route_head is None:
            raise ValueError("Route head is disabled for this model.")

        return self.route_head(self._pool_instruction_audio(obs))

    def predict_route_logits(
        self,
        observation: _model.Observation,
    ) -> at.Float[at.Array, "b c"]:
        observation = _model.preprocess_observation(None, observation, train=False)
        return self._predict_route_logits_from_preprocessed(observation)

    def _compute_route_loss(
        self,
        obs: _model.Observation,
        route_logits: at.Float[at.Array, "b c"],
    ) -> at.Float[at.Array, "b"]:  # noqa: F821
        if obs.route_label is None:
            return jnp.zeros((obs.state.shape[0],), dtype=route_logits.dtype)

        labels = jnp.asarray(obs.route_label, dtype=jnp.int32)
        if obs.route_label_mask is None:
            valid = labels >= 0
        else:
            valid = jnp.asarray(obs.route_label_mask, dtype=jnp.bool_)
            valid = jnp.logical_and(valid, labels >= 0)
        if self.route_loss_requires_instruction_audio and obs.instruction_audio_mask is not None:
            valid = jnp.logical_and(valid, jnp.asarray(obs.instruction_audio_mask, dtype=jnp.bool_))

        clipped_labels = jnp.clip(labels, 0, self.route_num_classes - 1)
        log_probs = jax.nn.log_softmax(route_logits, axis=-1)
        ce = -jnp.take_along_axis(log_probs, clipped_labels[:, None], axis=-1)[:, 0]
        valid_f = valid.astype(ce.dtype)
        valid_count = jnp.maximum(jnp.sum(valid_f), 1.0)
        batch_count = jnp.asarray(labels.shape[0], dtype=ce.dtype)
        return ce * valid_f * (batch_count / valid_count)

    def _time_embedding(
        self,
        timestep: at.Float[at.Array, "..."],
        batch_size: int,
    ) -> at.Float[at.Array, "... emb"]:
        timestep = jnp.asarray(timestep, dtype=jnp.float32)
        if timestep.ndim == 0:
            timestep = jnp.broadcast_to(timestep, (batch_size,))
        if timestep.ndim == 1:
            return posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if timestep.ndim != 2:
            raise ValueError(f"timestep must be scalar, [B], or [B, H], got shape {timestep.shape}")
        flat_time = jnp.reshape(timestep, (-1,))
        flat_emb = posemb_sincos(flat_time, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        return jnp.reshape(flat_emb, (*timestep.shape, flat_emb.shape[-1]))

    def _prepare_action_prefix(
        self,
        action_prefix: at.Float[at.Array, "..."] | None,
        batch_size: int,
    ) -> at.Float[at.Array, "b ah ad"]:
        if action_prefix is None:
            return jnp.zeros((batch_size, self.action_horizon, self.action_dim), dtype=jnp.float32)

        prefix = jnp.asarray(action_prefix, dtype=jnp.float32)
        if prefix.ndim == 2:
            prefix = prefix[None, ...]
        if prefix.ndim != 3:
            raise ValueError(f"action_prefix must have shape [H, D] or [B, H, D], got {prefix.shape}")
        if prefix.shape[0] == 1 and batch_size != 1:
            prefix = jnp.broadcast_to(prefix, (batch_size, prefix.shape[1], prefix.shape[2]))
        elif prefix.shape[0] != batch_size:
            raise ValueError(
                f"action_prefix batch size {prefix.shape[0]} does not match observation batch size {batch_size}"
            )

        if prefix.shape[1] < self.action_horizon:
            prefix = jnp.pad(prefix, ((0, 0), (0, self.action_horizon - prefix.shape[1]), (0, 0)))
        elif prefix.shape[1] > self.action_horizon:
            prefix = prefix[:, : self.action_horizon]
        if prefix.shape[2] < self.action_dim:
            prefix = jnp.pad(prefix, ((0, 0), (0, 0), (0, self.action_dim - prefix.shape[2])))
        elif prefix.shape[2] > self.action_dim:
            prefix = prefix[:, :, : self.action_dim]
        return prefix

    def _prepare_delay(
        self,
        delay: int | at.Int[at.Array, "..."] | None,
        batch_size: int,
    ) -> at.Int[at.Array, "b"]:  # noqa: F821
        if delay is None:
            return jnp.zeros((batch_size,), dtype=jnp.int32)
        delay = jnp.asarray(delay, dtype=jnp.int32)
        if delay.ndim == 0:
            delay = jnp.broadcast_to(delay, (batch_size,))
        elif delay.ndim != 1:
            raise ValueError(f"delay must be scalar or [B], got shape {delay.shape}")
        if delay.shape[0] == 1 and batch_size != 1:
            delay = jnp.broadcast_to(delay, (batch_size,))
        elif delay.shape[0] != batch_size:
            raise ValueError(f"delay batch size {delay.shape[0]} does not match observation batch size {batch_size}")
        return jnp.clip(delay, 0, self.action_horizon)

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, "..."],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b ..."] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = self._time_embedding(timestep, obs.state.shape[0])
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            if time_emb.ndim == 2:
                time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            else:
                time_tokens = time_emb
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        audio_adaln_cond = self._audio_adaln_cond_from_preprocessed(obs)
        adarms_cond = self._merge_adarms_cond(adarms_cond, audio_adaln_cond)
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001

        prefix_mask = None
        if self.rtc_train_max_delay > 0:
            delay = jax.random.randint(delay_rng, batch_shape, 0, self.rtc_train_max_delay + 1)
            prefix_mask = jnp.arange(self.action_horizon) < delay[..., None]
            # In this repo t=0 is clean action and t=1 is noise, so RTC prefix tokens use t=0.
            time = jnp.where(prefix_mask, 0.0, time[..., None])

        time_expanded = time[..., None] if time.ndim == len(batch_shape) + 1 else time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask_tokens, prefix_ar_mask = self.embed_prefix(observation, train=train)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask_tokens, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if prefix_mask is not None:
            postfix_mask = jnp.logical_not(prefix_mask)
            postfix_count = jnp.sum(postfix_mask, axis=-1, keepdims=True)
            loss_scale = self.action_horizon / jnp.maximum(postfix_count, 1)
            loss = loss * postfix_mask * loss_scale

        if observation.action_loss_mask is not None:
            action_loss_mask = jnp.asarray(observation.action_loss_mask, dtype=jnp.bool_)
            action_loss_mask_f = action_loss_mask.astype(loss.dtype)
            valid_count = jnp.maximum(jnp.sum(action_loss_mask_f), 1.0)
            batch_count = jnp.asarray(action_loss_mask_f.size, dtype=loss.dtype)
            loss = loss * action_loss_mask_f[..., None] * (batch_count / valid_count)

        if self.action_loss_weight != 1.0:
            loss = loss * self.action_loss_weight

        if self.use_route_head and self.route_loss_weight > 0:
            route_logits = self._predict_route_logits_from_preprocessed(observation)
            route_loss = self._compute_route_loss(observation, route_logits)
            loss = loss + self.route_loss_weight * route_loss[..., None]
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        action_prefix: at.Float[at.Array, "..."] | None = None,
        delay: int | at.Int[at.Array, "..."] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        action_prefix = self._prepare_action_prefix(action_prefix, batch_size)
        delay = self._prepare_delay(delay, batch_size)
        action_prefix_mask = jnp.arange(self.action_horizon)[None, :] < delay[:, None]
        noise = jnp.where(action_prefix_mask[:, :, None], action_prefix, noise)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation, train=False)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        use_audio_guidance = self.use_audio_guidance and self.audio_guidance_lambda != 1.0
        if use_audio_guidance:
            no_audio_observation = self._drop_audio_conditioning(observation)
            no_audio_prefix_tokens, no_audio_prefix_mask, no_audio_prefix_ar_mask = self.embed_prefix(
                no_audio_observation, train=False
            )
            no_audio_prefix_attn_mask = make_attn_mask(no_audio_prefix_mask, no_audio_prefix_ar_mask)
            no_audio_positions = jnp.cumsum(no_audio_prefix_mask, axis=1) - 1
            _, no_audio_kv_cache = self.PaliGemma.llm(
                [no_audio_prefix_tokens, None],
                mask=no_audio_prefix_attn_mask,
                positions=no_audio_positions,
            )
        else:
            no_audio_observation = None
            no_audio_prefix_tokens = None
            no_audio_prefix_mask = None
            no_audio_kv_cache = None

        def predict_velocity(
            obs,
            current_prefix_tokens,
            current_prefix_mask,
            current_kv_cache,
            x_t,
            timestep,
        ):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(obs, x_t, timestep)
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            current_prefix_attn_mask = einops.repeat(current_prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([current_prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                current_prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(current_prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=current_kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        def step(carry):
            x_t, time = carry
            x_t = jnp.where(action_prefix_mask[:, :, None], action_prefix, x_t)
            timestep = jnp.where(action_prefix_mask, 0.0, time)
            v_t = predict_velocity(
                observation,
                prefix_tokens,
                prefix_mask,
                kv_cache,
                x_t,
                timestep,
            )
            if use_audio_guidance:
                v_no_audio = predict_velocity(
                    no_audio_observation,
                    no_audio_prefix_tokens,
                    no_audio_prefix_mask,
                    no_audio_kv_cache,
                    x_t,
                    timestep,
                )
                v_t = v_no_audio + self.audio_guidance_lambda * (v_t - v_no_audio)

            x_next = x_t + dt * v_t
            x_next = jnp.where(action_prefix_mask[:, :, None], action_prefix, x_next)
            return x_next, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
