import dataclasses

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
                ("observation/exterior_image_1_left", "image", "observation.image"),
            )
        )
        left_wrist_image = _parse_image(
            _get_first_present(
                data,
                ("observation/wrist_image_left", "wrist_image_left", "wrist_image"),
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

        prompt = _get_first_present(data, ("prompt",), required=False)
        if prompt is not None:
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        inputs = transforms.copy_present_audio_fields(data, inputs)
        inputs = transforms.copy_present_route_fields(data, inputs)
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
        outputs = transforms.ExtractInstructionAudioWindow(
            window_num_samples=instruction_num_samples,
            target_sample_rate=self.audio_sample_rate,
            dataset_root=self.dataset_root,
            fallback_fps=self.audio_fallback_fps,
            strict=self.audio_strict,
        )(outputs)
        if self.use_whisper and self.use_qwen:
            raise ValueError("use_whisper and use_qwen cannot both be True.")
        if not self.use_whisper and not self.use_qwen:
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
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
