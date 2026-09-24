import dataclasses
import functools
from pathlib import Path

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import audio_tools as _audio_tools


def make_audiovla_example() -> dict:
    """Creates a random input example for the AudioVLA policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.ones((7,), dtype=np.float32),
        "observation/gripper_position": np.ones((1,), dtype=np.float32),
        "observation/audio": np.random.randn(16_000).astype(np.float32),
        "observation/audio_sample_rate": np.int32(16_000),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if np.min(image) < 0.0:
            image = ((image + 1.0) / 2.0 * 255.0).astype(np.uint8)
        else:
            image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _get_first_present(data: dict, keys: tuple[str, ...], *, required: bool = True):
    for key in keys:
        if key in data:
            return data[key]
    if required:
        raise KeyError(f"Missing required keys {keys}. Available keys: {tuple(data.keys())}")
    return None


def _as_gripper_position(value) -> np.ndarray:
    gripper_position = np.asarray(value, dtype=np.float32)
    if gripper_position.ndim == 0:
        gripper_position = gripper_position[np.newaxis]
    return gripper_position


@functools.lru_cache(maxsize=4096)
def _load_cached_qwen_instruction_feature(
    cache_dir: str,
    episode_index: int,
    expected_num_frames: int,
    expected_dim: int,
) -> np.ndarray:
    feature_path = Path(cache_dir).expanduser() / f"episode_{episode_index:06d}.npy"
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Cached Qwen instruction feature not found: {feature_path}. "
            "Run scripts/precompute_omniaction_qwen_features.py first or remove --cashed."
        )

    feature = np.load(feature_path, allow_pickle=False)
    expected_shape = (expected_num_frames, expected_dim)
    if feature.shape != expected_shape:
        raise ValueError(
            f"Cached Qwen instruction feature shape mismatch in {feature_path}: "
            f"expected {expected_shape}, got {feature.shape}."
        )

    # Keep the training input dtype identical to the online Qwen extraction path.
    feature = np.asarray(feature, dtype=np.float32)
    feature.setflags(write=False)
    return feature


@dataclasses.dataclass(frozen=True)
class AudiovlaInputs(transforms.DataTransformFn):
    """Prepare real-robot observations for pi0 audio checkpoints."""

    action_dim: int
    audio_num_samples: int
    instruction_audio_num_samples: int | None = None
    audio_sample_rate: int = 16_000
    use_whisper: bool = False
    use_qwen: bool = False
    audio_tokenizer_type: str = "sequence_resampler"
    whisper_model_name: str = "openai/whisper-tiny"
    whisper_output_dim: int | None = None
    whisper_device: str = "cpu"
    qwen_model_name: str = "Qwen/Qwen3-ASR-0.6B"
    qwen_output_dim: int | None = None
    qwen_device: str = "auto"
    qwen_instruction_feature_cache_dir: str | None = None
    use_usad2: bool = False
    usad2_model_name: str = "MIT-SLS/USAD2-Large-Plus"
    usad2_output_dim: int | None = None
    usad2_device: str = "auto"
    usad2_target_layer: int | None = None
    dataset_root: str | None = None
    audio_fallback_fps: float = 20.0
    audio_strict: bool = True
    route_label_strict: bool = False
    model_type: _model.ModelType = _model.ModelType.PI0_AUDIO

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type in (
            _model.ModelType.PI0,
            _model.ModelType.PI0_AUDIO,
        )
        instruction_num_samples = (
            self.audio_num_samples if self.instruction_audio_num_samples is None else self.instruction_audio_num_samples
        )

        joint_position = _get_first_present(
            data,
            ("observation/joint_position", "joint_position"),
            required=False,
        )
        gripper_position = _get_first_present(
            data,
            ("observation/gripper_position", "gripper_position"),
            required=False,
        )
        if joint_position is not None and gripper_position is not None:
            state_raw = np.concatenate(
                [
                    np.asarray(joint_position, dtype=np.float32),
                    _as_gripper_position(gripper_position),
                ],
                axis=-1,
            )
        else:
            state_raw = _get_first_present(data, ("observation/state", "state"))
        state = transforms.pad_to_dim(np.asarray(state_raw, dtype=np.float32), self.action_dim)

        base_image = _parse_image(
            _get_first_present(
                data,
                ("observation/exterior_image_1_left", "observation/image", "image", "observation.image"),
            )
        )
        left_wrist_image = _parse_image(
            _get_first_present(
                data,
                ("observation/wrist_image_left", "observation/wrist_image", "wrist_image_left", "wrist_image"),
            )
        )

        right_wrist_value = _get_first_present(
            data,
            ("observation/wrist_image_right", "wrist_image_right"),
            required=False,
        )
        has_right_wrist = right_wrist_value is not None
        right_wrist_image = _parse_image(right_wrist_value) if has_right_wrist else np.zeros_like(base_image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.asarray(has_right_wrist or not mask_padding),
            },
        }

        actions_value = _get_first_present(data, ("actions", "action"), required=False)
        if actions_value is not None:
            inputs["actions"] = transforms.pad_to_dim(np.asarray(actions_value, dtype=np.float32), self.action_dim)
        action_loss_mask = _get_first_present(data, ("action_loss_mask",), required=False)
        if action_loss_mask is not None:
            inputs["action_loss_mask"] = np.asarray(action_loss_mask, dtype=np.bool_)

        prompt = _get_first_present(data, ("prompt",), required=False)
        if prompt is not None:
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        inputs = transforms.copy_present_audio_fields(data, inputs)
        inputs = transforms.copy_present_route_fields(data, inputs)
        inputs = transforms.copy_present_dialogue_fields(data, inputs)
        outputs = transforms.ExtractRouteLabel(
            dataset_root=self.dataset_root,
            strict=self.route_label_strict,
            fallback_fps=self.audio_fallback_fps,
        )(inputs)
        outputs = transforms.ExtractAudioWindow(
            window_num_samples=self.audio_num_samples,
            target_sample_rate=self.audio_sample_rate,
            dataset_root=self.dataset_root,
            fallback_fps=self.audio_fallback_fps,
            strict=self.audio_strict,
        )(outputs)

        # Cached OmniAction features encode the complete dialogue. Dialogue
        # supervision samples instead expose only audio through the current
        # turn, so they must run the online encoder to avoid future-turn
        # leakage.
        is_dialogue_supervision = _get_first_present(data, ("dialogue_target",), required=False) is not None
        use_cached_qwen_instruction = (
            self.qwen_instruction_feature_cache_dir is not None and not is_dialogue_supervision
        )
        if use_cached_qwen_instruction:
            if not self.use_qwen:
                raise ValueError("Cached instruction features currently require use_qwen=True.")
            if self.audio_tokenizer_type != "sequence_resampler":
                raise ValueError(
                    "Cached instruction features currently require audio_tokenizer_type='sequence_resampler'."
                )

            episode_index_value = outputs.get("episode_index")
            if episode_index_value is None:
                raise ValueError("Cached instruction features require an episode_index field.")
            episode_index = int(np.asarray(episode_index_value).reshape(()).item())
            expected_num_frames = _audio_tools.qwen_asr_num_frames_for_duration(
                instruction_num_samples,
                sample_rate=self.audio_sample_rate,
            )
            expected_dim = self.qwen_output_dim or _audio_tools.infer_qwen_asr_hidden_size(self.qwen_model_name)
            outputs = {
                **outputs,
                "instruction_audio": _load_cached_qwen_instruction_feature(
                    self.qwen_instruction_feature_cache_dir,
                    episode_index,
                    expected_num_frames,
                    expected_dim,
                ),
                "instruction_audio_mask": np.True_,
            }
        else:
            outputs = transforms.ExtractInstructionAudioWindow(
                window_num_samples=instruction_num_samples,
                target_sample_rate=self.audio_sample_rate,
                dataset_root=self.dataset_root,
                fallback_fps=self.audio_fallback_fps,
                strict=self.audio_strict,
            )(outputs)
        if sum([self.use_whisper, self.use_qwen, self.use_usad2]) > 1:
            raise ValueError("At most one of use_whisper / use_qwen / use_usad2 may be True.")
        if not self.use_whisper and not self.use_qwen and not self.use_usad2:
            return outputs

        if self.use_usad2:
            usad2_frame_rate = _audio_tools.infer_usad2_frame_rate(self.usad2_model_name)
            if self.audio_tokenizer_type == "pooled_baseline":
                outputs = transforms.ExtractUsad2AudioEmbedding(
                    model_name=self.usad2_model_name,
                    sample_rate=self.audio_sample_rate,
                    device=self.usad2_device,
                    target_layer=self.usad2_target_layer,
                    output_dim=self.usad2_output_dim,
                    audio_key="ambient_audio",
                    audio_mask_key="ambient_audio_mask",
                )(outputs)
                outputs = transforms.ExtractUsad2AudioEmbedding(
                    model_name=self.usad2_model_name,
                    sample_rate=self.audio_sample_rate,
                    device=self.usad2_device,
                    target_layer=self.usad2_target_layer,
                    output_dim=self.usad2_output_dim,
                    audio_key="instruction_audio",
                    audio_mask_key="instruction_audio_mask",
                )(outputs)
                outputs["audio"] = outputs["ambient_audio"]
                outputs["audio_mask"] = outputs["ambient_audio_mask"]
                return outputs

            outputs = transforms.ExtractUsad2AudioSequence(
                model_name=self.usad2_model_name,
                sample_rate=self.audio_sample_rate,
                device=self.usad2_device,
                target_layer=self.usad2_target_layer,
                num_frames=_audio_tools.usad2_num_frames_for_duration(
                    self.audio_num_samples,
                    sample_rate=self.audio_sample_rate,
                    frame_rate=usad2_frame_rate,
                ),
                output_dim=self.usad2_output_dim,
                audio_key="ambient_audio",
                audio_mask_key="ambient_audio_mask",
            )(outputs)
            outputs = transforms.ExtractUsad2AudioSequence(
                model_name=self.usad2_model_name,
                sample_rate=self.audio_sample_rate,
                device=self.usad2_device,
                target_layer=self.usad2_target_layer,
                num_frames=_audio_tools.usad2_num_frames_for_duration(
                    instruction_num_samples,
                    sample_rate=self.audio_sample_rate,
                    frame_rate=usad2_frame_rate,
                ),
                output_dim=self.usad2_output_dim,
                audio_key="instruction_audio",
                audio_mask_key="instruction_audio_mask",
            )(outputs)
            outputs["audio"] = outputs["ambient_audio"]
            outputs["audio_mask"] = outputs["ambient_audio_mask"]
            return outputs

        if self.use_qwen:
            if self.audio_tokenizer_type == "pooled_baseline":
                outputs = transforms.ExtractQwenASRAudioEmbedding(
                    model_name=self.qwen_model_name,
                    sample_rate=self.audio_sample_rate,
                    device=self.qwen_device,
                    output_dim=self.qwen_output_dim,
                    audio_key="ambient_audio",
                    audio_mask_key="ambient_audio_mask",
                )(outputs)
                outputs = transforms.ExtractQwenASRAudioEmbedding(
                    model_name=self.qwen_model_name,
                    sample_rate=self.audio_sample_rate,
                    device=self.qwen_device,
                    output_dim=self.qwen_output_dim,
                    audio_key="instruction_audio",
                    audio_mask_key="instruction_audio_mask",
                )(outputs)
                outputs["audio"] = outputs["ambient_audio"]
                outputs["audio_mask"] = outputs["ambient_audio_mask"]
                return outputs

            outputs = transforms.ExtractQwenASRAudioSequence(
                model_name=self.qwen_model_name,
                sample_rate=self.audio_sample_rate,
                device=self.qwen_device,
                num_frames=_audio_tools.qwen_asr_num_frames_for_duration(
                    self.audio_num_samples,
                    sample_rate=self.audio_sample_rate,
                ),
                output_dim=self.qwen_output_dim,
                audio_key="ambient_audio",
                audio_mask_key="ambient_audio_mask",
            )(outputs)
            if not use_cached_qwen_instruction:
                outputs = transforms.ExtractQwenASRAudioSequence(
                    model_name=self.qwen_model_name,
                    sample_rate=self.audio_sample_rate,
                    device=self.qwen_device,
                    num_frames=_audio_tools.qwen_asr_num_frames_for_duration(
                        instruction_num_samples,
                        sample_rate=self.audio_sample_rate,
                    ),
                    output_dim=self.qwen_output_dim,
                    audio_key="instruction_audio",
                    audio_mask_key="instruction_audio_mask",
                )(outputs)
            outputs["audio"] = outputs["ambient_audio"]
            outputs["audio_mask"] = outputs["ambient_audio_mask"]
            return outputs

        if self.audio_tokenizer_type == "pooled_baseline":
            outputs = transforms.ExtractWhisperAudioEmbedding(
                model_name=self.whisper_model_name,
                sample_rate=self.audio_sample_rate,
                device=self.whisper_device,
                output_dim=self.whisper_output_dim,
                audio_key="ambient_audio",
                audio_mask_key="ambient_audio_mask",
            )(outputs)
            outputs = transforms.ExtractWhisperAudioEmbedding(
                model_name=self.whisper_model_name,
                sample_rate=self.audio_sample_rate,
                device=self.whisper_device,
                output_dim=self.whisper_output_dim,
                audio_key="instruction_audio",
                audio_mask_key="instruction_audio_mask",
            )(outputs)
            outputs["audio"] = outputs["ambient_audio"]
            outputs["audio_mask"] = outputs["ambient_audio_mask"]
            return outputs

        outputs = transforms.ExtractWhisperAudioSequence(
            model_name=self.whisper_model_name,
            sample_rate=self.audio_sample_rate,
            device=self.whisper_device,
            num_frames=_audio_tools.whisper_num_frames_for_duration(
                self.audio_num_samples,
                sample_rate=self.audio_sample_rate,
            ),
            output_dim=self.whisper_output_dim,
            audio_key="ambient_audio",
            audio_mask_key="ambient_audio_mask",
        )(outputs)
        outputs = transforms.ExtractWhisperAudioSequence(
            model_name=self.whisper_model_name,
            sample_rate=self.audio_sample_rate,
            device=self.whisper_device,
            num_frames=_audio_tools.whisper_num_frames_for_duration(
                instruction_num_samples,
                sample_rate=self.audio_sample_rate,
            ),
            output_dim=self.whisper_output_dim,
            audio_key="instruction_audio",
            audio_mask_key="instruction_audio_mask",
        )(outputs)
        outputs["audio"] = outputs["ambient_audio"]
        outputs["audio_mask"] = outputs["ambient_audio_mask"]
        return outputs


@dataclasses.dataclass(frozen=True)
class AudiovlaOutputs(transforms.DataTransformFn):
    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
