from collections.abc import Callable, Mapping, Sequence
import dataclasses
from pathlib import Path
import re
from typing import Literal, Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import audio_tools as _audio_tools
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats

ROUTE_LABEL_NAMES = ("straight", "detour", "none")
ROUTE_LABEL_TO_ID = {name: index for index, name in enumerate(ROUTE_LABEL_NAMES)}
ROUTE_ID_TO_LABEL = dict(enumerate(ROUTE_LABEL_NAMES))
INVALID_ROUTE_LABEL = -1


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def _find_first_present(data: DataDict, keys: Sequence[str]):
    for key in keys:
        if key in data:
            return data[key]
    return None


def route_label_to_id(value) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"route label must be scalar, got shape {array.shape}")
    raw = array.reshape(()).item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in ROUTE_LABEL_TO_ID:
            return ROUTE_LABEL_TO_ID[key]
        try:
            raw = int(key)
        except ValueError as exc:
            raise ValueError(f"Unsupported route label {raw!r}; expected one of {ROUTE_LABEL_NAMES}") from exc
    label_id = int(raw)
    if label_id == INVALID_ROUTE_LABEL:
        return INVALID_ROUTE_LABEL
    if 0 <= label_id < len(ROUTE_LABEL_NAMES):
        return label_id
    raise ValueError(f"route label id must be in [0, {len(ROUTE_LABEL_NAMES) - 1}], got {label_id}")


def _invalid_route_label(data: DataDict) -> DataDict:
    return {
        **data,
        "route_label": np.int32(INVALID_ROUTE_LABEL),
        "route_label_mask": np.False_,
    }


def _as_float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        array = np.asarray(value)
        if array.size != 1:
            return None
        raw = array.reshape(()).item()
        if raw is None:
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _as_int_or_none(value) -> int | None:
    if value is None:
        return None
    try:
        array = np.asarray(value)
        if array.size != 1:
            return None
        raw = array.reshape(()).item()
        if raw is None:
            return None
        return int(raw)
    except (TypeError, ValueError):
        return None


def _time_window_from_mapping(value) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    start_sec = _as_float_or_none(value.get("start_sec"))
    end_sec = _as_float_or_none(value.get("end_sec"))
    if start_sec is None or end_sec is None or end_sec < start_sec:
        return None
    return start_sec, end_sec


def _route_label_window_from_sync(sync_data: Mapping[str, object]) -> tuple[float, float] | None:
    window = _time_window_from_mapping(sync_data)
    if window is not None:
        return window

    for key in ("route_label_window", "route_window", "instruction_audio_window"):
        window = _time_window_from_mapping(sync_data.get(key))
        if window is not None:
            return window

    starts: list[float] = []
    ends: list[float] = []
    raw_segments = sync_data.get("vad_segments")
    if isinstance(raw_segments, list):
        for segment in raw_segments:
            if not isinstance(segment, Mapping):
                continue
            if segment.get("kind") not in (None, "instruction"):
                continue
            start_sec = _as_float_or_none(segment.get("start_sec"))
            end_sec = _as_float_or_none(segment.get("end_sec"))
            if start_sec is None or end_sec is None or end_sec < start_sec:
                continue
            starts.append(start_sec)
            ends.append(end_sec)
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def _frame_time_sec_from_sync(
    data: DataDict,
    sync_data: Mapping[str, object],
    *,
    frame_index_keys: Sequence[str],
    time_keys: Sequence[str],
    fallback_fps: float,
) -> float | None:
    raw_time = _find_first_present(data, time_keys)
    explicit_time = _as_float_or_none(raw_time)
    if explicit_time is not None:
        return explicit_time

    raw_frame_index = _find_first_present(data, frame_index_keys)
    frame_index = _as_int_or_none(raw_frame_index)
    if frame_index is None:
        return None

    fps = _as_float_or_none(sync_data.get("control_frequency")) or float(fallback_fps)
    fps = max(fps, 1e-6)
    audio_start_ns = _as_int_or_none(sync_data.get("audio_start_monotonic_ns"))
    frame_records = sync_data.get("frame_records")
    if audio_start_ns is not None and isinstance(frame_records, list):
        record = _audio_tools.get_frame_record(frame_records, frame_index)
        next_record = _audio_tools.get_frame_record(frame_records, frame_index + 1)
        if record is not None:
            end_ns = None
            if next_record is not None:
                end_ns = _as_int_or_none(next_record.get("host_frame_monotonic_ns"))
            if end_ns is None:
                start_ns = _as_int_or_none(record.get("host_frame_monotonic_ns"))
                if start_ns is not None:
                    end_ns = start_ns + round(1e9 / fps)
            if end_ns is not None:
                return max(0.0, (end_ns - audio_start_ns) / 1e9)

    return (frame_index + 1) / fps


def _route_label_mask_from_sync(
    data: DataDict,
    sync_data: Mapping[str, object],
    *,
    label_id: int,
    frame_index_keys: Sequence[str],
    time_keys: Sequence[str],
    fallback_fps: float,
    window_epsilon_sec: float,
) -> bool:
    if label_id == INVALID_ROUTE_LABEL:
        return False
    window = _route_label_window_from_sync(sync_data)
    if window is None:
        return False
    current_time_sec = _frame_time_sec_from_sync(
        data,
        sync_data,
        frame_index_keys=frame_index_keys,
        time_keys=time_keys,
        fallback_fps=fallback_fps,
    )
    if current_time_sec is None:
        return False
    start_sec, end_sec = window
    return start_sec - window_epsilon_sec <= current_time_sec <= end_sec + window_epsilon_sec


_ROUTE_PASSTHROUGH_KEYS: tuple[tuple[str, Sequence[str]], ...] = (
    ("route_label", ("route_label", "observation/route_label", "label", "observation/label")),
    ("route_label_mask", ("route_label_mask", "observation/route_label_mask", "label_mask", "observation/label_mask")),
)


def copy_present_route_fields(source: Mapping[str, object], dest: dict) -> dict:
    for out_key, candidate_keys in _ROUTE_PASSTHROUGH_KEYS:
        value = _find_first_present(source, candidate_keys)
        if value is not None:
            dest[out_key] = value
    return dest


@dataclasses.dataclass(frozen=True)
class ExtractRouteLabel(DataTransformFn):
    """Materialize a route label id from direct fields or the episode sync sidecar."""

    dataset_root: str | None = None
    strict: bool = False
    label_keys: Sequence[str] = ("route_label", "observation/route_label", "label", "observation/label")
    label_mask_keys: Sequence[str] = (
        "route_label_mask",
        "observation/route_label_mask",
        "label_mask",
        "observation/label_mask",
    )
    episode_index_keys: Sequence[str] = ("episode_index", "observation/episode_index")
    frame_index_keys: Sequence[str] = ("frame_index", "observation/frame_index")
    time_keys: Sequence[str] = ("timestamp", "observation/timestamp", "time_sec", "observation/time_sec")
    fallback_fps: float = 20.0
    window_epsilon_sec: float = 1e-6

    def __call__(self, data: DataDict) -> DataDict:
        raw_label = _find_first_present(data, self.label_keys)
        if raw_label is not None:
            try:
                label_id = route_label_to_id(raw_label)
                raw_mask = _find_first_present(data, self.label_mask_keys)
                label_mask = label_id != INVALID_ROUTE_LABEL if raw_mask is None else bool(_as_bool(raw_mask))
            except Exception:
                if self.strict:
                    raise
                return _invalid_route_label(data)
            return {
                **data,
                "route_label": np.int32(label_id),
                "route_label_mask": np.bool_(label_mask),
            }

        episode_index = _find_first_present(data, self.episode_index_keys)
        if episode_index is None or self.dataset_root is None:
            if self.strict:
                raise ValueError("Route label extraction requires a label field or dataset_root + episode_index.")
            return _invalid_route_label(data)

        try:
            _, sync_path = _audio_tools.resolve_episode_audio_paths(
                self.dataset_root,
                _as_int(episode_index, key="episode_index"),
            )
            sync_data = _audio_tools.load_json(sync_path)
            if not sync_data:
                raise FileNotFoundError(f"Route label sync metadata not found: {sync_path}")
            raw_label = sync_data.get("route_label", sync_data.get("label"))
            if raw_label is None:
                raise KeyError(f"No route label in sync metadata: {sync_path}")
            label_id = route_label_to_id(raw_label)
            label_mask = _route_label_mask_from_sync(
                data,
                sync_data,
                label_id=label_id,
                frame_index_keys=self.frame_index_keys,
                time_keys=self.time_keys,
                fallback_fps=self.fallback_fps,
                window_epsilon_sec=self.window_epsilon_sec,
            )
        except Exception:
            if self.strict:
                raise
            return _invalid_route_label(data)

        return {
            **data,
            "route_label": np.int32(label_id),
            "route_label_mask": np.bool_(label_mask),
        }


_AUDIO_PASSTHROUGH_KEYS: tuple[tuple[str, Sequence[str]], ...] = (
    (
        "ambient_audio",
        (
            "ambient_audio",
            "observation/ambient_audio",
            "audio",
            "observation/audio",
            "audio_window",
            "observation/audio_window",
        ),
    ),
    (
        "ambient_audio_mask",
        ("ambient_audio_mask", "observation/ambient_audio_mask", "audio_mask", "observation/audio_mask"),
    ),
    (
        "ambient_audio_sample_rate",
        (
            "ambient_audio_sample_rate",
            "observation/ambient_audio_sample_rate",
            "audio_sample_rate",
            "observation/audio_sample_rate",
        ),
    ),
    (
        "instruction_audio",
        (
            "instruction_audio",
            "observation/instruction_audio",
            "instruction_audio_window",
            "observation/instruction_audio_window",
        ),
    ),
    (
        "instruction_audio_mask",
        ("instruction_audio_mask", "observation/instruction_audio_mask"),
    ),
    (
        "instruction_audio_sample_rate",
        ("instruction_audio_sample_rate", "observation/instruction_audio_sample_rate"),
    ),
    ("audio", ("audio", "observation/audio", "audio_window", "observation/audio_window")),
    ("audio_mask", ("audio_mask", "observation/audio_mask")),
    ("audio_sample_rate", ("audio_sample_rate", "observation/audio_sample_rate")),
    ("episode_index", ("episode_index", "observation/episode_index")),
    ("frame_index", ("frame_index", "observation/frame_index")),
)


def copy_present_audio_fields(source: Mapping[str, object], dest: dict) -> dict:
    """Copy any available audio-related fields into a transformed input dict."""
    for out_key, candidate_keys in _AUDIO_PASSTHROUGH_KEYS:
        value = _find_first_present(source, candidate_keys)
        if value is not None:
            dest[out_key] = value
    return dest


_AUDIO_VLA_BASE_MODES = ("full_avla", "audio_action", "image_action")
_AUDIO_VLA_CONDITIONING_MODES = (*_AUDIO_VLA_BASE_MODES, "mixed_aa_avla")


def _stable_index_from_fields(
    data: DataDict,
    keys: Sequence[str],
    *,
    modulo: int,
) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be > 0")
    accumulator = 0
    found = False
    for key in keys:
        value = _find_first_present(data, (key,))
        scalar = _as_int_or_none(value)
        if scalar is None:
            continue
        found = True
        accumulator = (accumulator * 1_000_003 + scalar) & 0x7FFF_FFFF
    return int(accumulator % modulo) if found else 0


def _set_audio_masks(data: dict, value: bool) -> None:
    for key in ("audio_mask", "ambient_audio_mask", "instruction_audio_mask"):
        if key in data:
            data[key] = np.bool_(value)


@dataclasses.dataclass(frozen=True)
class AudioVLAConditioning(DataTransformFn):
    """Apply staged AVLA conditioning by masking image or audio tokens."""

    mode: Literal["full_avla", "audio_action", "image_action", "mixed_aa_avla"] = "full_avla"
    mixed_modes: Sequence[Literal["full_avla", "audio_action", "image_action"]] = ("audio_action", "full_avla")
    stable_index_keys: Sequence[str] = ("episode_index", "frame_index")

    def __post_init__(self) -> None:
        if self.mode not in _AUDIO_VLA_CONDITIONING_MODES:
            raise ValueError(f"Unsupported AVLA conditioning mode: {self.mode}")
        invalid = [mode for mode in self.mixed_modes if mode not in _AUDIO_VLA_BASE_MODES]
        if invalid:
            raise ValueError(f"mixed_modes can only contain {_AUDIO_VLA_BASE_MODES}, got {invalid}")
        if self.mode == "mixed_aa_avla" and not self.mixed_modes:
            raise ValueError("mixed_aa_avla requires at least one mixed mode")

    def _resolve_mode(self, data: DataDict) -> str:
        if self.mode != "mixed_aa_avla":
            return self.mode
        index = _stable_index_from_fields(data, self.stable_index_keys, modulo=len(self.mixed_modes))
        return self.mixed_modes[index]

    def __call__(self, data: DataDict) -> DataDict:
        mode = self._resolve_mode(data)
        if mode == "full_avla":
            return data

        output = dict(data)
        if mode == "audio_action":
            output["image_mask"] = {key: np.False_ for key in output.get("image_mask", {})}
        elif mode == "image_action":
            _set_audio_masks(output, False)
        else:
            raise ValueError(f"Unsupported resolved AVLA conditioning mode: {mode}")
        return output


def _as_int(value, *, key: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{key} must be a scalar, got shape {array.shape}")
    return int(array.reshape(()).item())


def _as_bool(value) -> np.bool_:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar boolean mask, got shape {array.shape}")
    return np.bool_(array.reshape(()).item())


@dataclasses.dataclass(frozen=True)
class ExtractAudioWindow(DataTransformFn):
    """Materialize the ambient audio window into a fixed-size float32 array."""

    window_num_samples: int
    target_sample_rate: int = 16_000
    dataset_root: str | None = None
    fallback_fps: float = 20.0
    strict: bool = True
    audio_keys: Sequence[str] = (
        "ambient_audio",
        "observation/ambient_audio",
        "audio",
        "observation/audio",
        "audio_window",
        "observation/audio_window",
    )
    audio_mask_keys: Sequence[str] = (
        "ambient_audio_mask",
        "observation/ambient_audio_mask",
        "audio_mask",
        "observation/audio_mask",
    )
    audio_sample_rate_keys: Sequence[str] = (
        "ambient_audio_sample_rate",
        "observation/ambient_audio_sample_rate",
        "audio_sample_rate",
        "observation/audio_sample_rate",
    )
    episode_index_keys: Sequence[str] = ("episode_index", "observation/episode_index")
    frame_index_keys: Sequence[str] = ("frame_index", "observation/frame_index")

    def __post_init__(self) -> None:
        if self.window_num_samples <= 0:
            raise ValueError("window_num_samples must be > 0")
        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be > 0")

    def __call__(self, data: DataDict) -> DataDict:
        raw_audio = _find_first_present(data, self.audio_keys)
        if raw_audio is not None:
            sample_rate = _find_first_present(data, self.audio_sample_rate_keys)
            sample_rate = (
                self.target_sample_rate
                if sample_rate is None
                else _as_int(sample_rate, key="ambient_audio_sample_rate")
            )
            audio = np.asarray(raw_audio, dtype=np.float32).reshape(-1)
            if sample_rate != self.target_sample_rate:
                audio = _audio_tools.resample_audio(audio, sample_rate, self.target_sample_rate)

            raw_mask = _find_first_present(data, self.audio_mask_keys)
            audio_mask = bool(audio.size > 0) if raw_mask is None else bool(_as_bool(raw_mask))
            audio_window = _audio_tools.fit_audio_window(audio, self.window_num_samples)
            return {
                **data,
                "ambient_audio": audio_window,
                "ambient_audio_mask": np.bool_(audio_mask),
                "audio": audio_window,
                "audio_mask": np.bool_(audio_mask),
            }

        episode_index = _find_first_present(data, self.episode_index_keys)
        frame_index = _find_first_present(data, self.frame_index_keys)
        if episode_index is None or frame_index is None:
            if not self.strict:
                zeros = np.zeros((self.window_num_samples,), dtype=np.float32)
                return {
                    **data,
                    "ambient_audio": zeros,
                    "ambient_audio_mask": np.False_,
                    "audio": zeros,
                    "audio_mask": np.False_,
                }
            raise ValueError(
                "Audio window extraction requires raw audio or episode/frame indices. "
                f"Available keys: {tuple(data.keys())}"
            )

        if self.dataset_root is None:
            raise ValueError("dataset_root is required when audio must be loaded from episode sidecars.")

        try:
            audio, has_audio = _audio_tools.extract_episode_audio_window(
                dataset_root=Path(self.dataset_root),
                episode_index=_as_int(episode_index, key="episode_index"),
                frame_index=_as_int(frame_index, key="frame_index"),
                window_num_samples=self.window_num_samples,
                target_sample_rate=self.target_sample_rate,
                fallback_fps=self.fallback_fps,
            )
        except Exception:
            if self.strict:
                raise
            audio = np.zeros((self.window_num_samples,), dtype=np.float32)
            has_audio = False

        return {
            **data,
            "ambient_audio": audio,
            "ambient_audio_mask": np.bool_(has_audio),
            "audio": audio,
            "audio_mask": np.bool_(has_audio),
        }


@dataclasses.dataclass(frozen=True)
class ExtractInstructionAudioWindow(DataTransformFn):
    """Materialize the active instruction audio into a fixed-size float32 array."""

    window_num_samples: int
    target_sample_rate: int = 16_000
    dataset_root: str | None = None
    fallback_fps: float = 20.0
    strict: bool = True
    audio_keys: Sequence[str] = (
        "instruction_audio",
        "observation/instruction_audio",
        "instruction_audio_window",
        "observation/instruction_audio_window",
    )
    audio_mask_keys: Sequence[str] = ("instruction_audio_mask", "observation/instruction_audio_mask")
    audio_sample_rate_keys: Sequence[str] = (
        "instruction_audio_sample_rate",
        "observation/instruction_audio_sample_rate",
    )
    episode_index_keys: Sequence[str] = ("episode_index", "observation/episode_index")
    frame_index_keys: Sequence[str] = ("frame_index", "observation/frame_index")

    def __post_init__(self) -> None:
        if self.window_num_samples <= 0:
            raise ValueError("window_num_samples must be > 0")
        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be > 0")

    def __call__(self, data: DataDict) -> DataDict:
        raw_audio = _find_first_present(data, self.audio_keys)
        if raw_audio is not None:
            sample_rate = _find_first_present(data, self.audio_sample_rate_keys)
            sample_rate = (
                self.target_sample_rate
                if sample_rate is None
                else _as_int(sample_rate, key="instruction_audio_sample_rate")
            )
            audio = np.asarray(raw_audio, dtype=np.float32).reshape(-1)
            if sample_rate != self.target_sample_rate:
                audio = _audio_tools.resample_audio(audio, sample_rate, self.target_sample_rate)

            raw_mask = _find_first_present(data, self.audio_mask_keys)
            audio_mask = bool(audio.size > 0) if raw_mask is None else bool(_as_bool(raw_mask))
            return {
                **data,
                "instruction_audio": _audio_tools.fit_audio_window(audio, self.window_num_samples),
                "instruction_audio_mask": np.bool_(audio_mask),
            }

        episode_index = _find_first_present(data, self.episode_index_keys)
        frame_index = _find_first_present(data, self.frame_index_keys)
        if episode_index is None or frame_index is None:
            if not self.strict:
                return {
                    **data,
                    "instruction_audio": np.zeros((self.window_num_samples,), dtype=np.float32),
                    "instruction_audio_mask": np.False_,
                }
            raise ValueError(
                "Instruction audio extraction requires raw instruction audio or episode/frame indices. "
                f"Available keys: {tuple(data.keys())}"
            )

        if self.dataset_root is None:
            raise ValueError("dataset_root is required when instruction audio must be loaded from episode sidecars.")

        try:
            audio, has_audio = _audio_tools.extract_episode_instruction_audio(
                dataset_root=Path(self.dataset_root),
                episode_index=_as_int(episode_index, key="episode_index"),
                frame_index=_as_int(frame_index, key="frame_index"),
                window_num_samples=self.window_num_samples,
                target_sample_rate=self.target_sample_rate,
                fallback_fps=self.fallback_fps,
            )
        except Exception:
            if self.strict:
                raise
            audio = np.zeros((self.window_num_samples,), dtype=np.float32)
            has_audio = False

        return {
            **data,
            "instruction_audio": audio,
            "instruction_audio_mask": np.bool_(has_audio),
        }


@dataclasses.dataclass(frozen=True)
class ExtractWhisperAudioSequence(DataTransformFn):
    """Encode an audio waveform window into Whisper time-sequence features."""

    model_name: str = "openai/whisper-tiny"
    sample_rate: int = 16_000
    device: str = "cpu"
    num_frames: int | None = None
    output_dim: int | None = None
    audio_key: str = "audio"
    audio_mask_key: str = "audio_mask"

    def __call__(self, data: DataDict) -> DataDict:
        raw_audio = data.get(self.audio_key)
        if raw_audio is None:
            raise ValueError(f"Whisper sequence extraction requires a `{self.audio_key}` field.")

        raw_mask = data.get(self.audio_mask_key)
        has_audio = bool(_as_bool(raw_mask)) if raw_mask is not None else bool(np.asarray(raw_audio).size > 0)
        expected_dim = self.output_dim or _audio_tools.infer_whisper_hidden_size(self.model_name)
        target_num_frames = self.num_frames or _audio_tools.whisper_num_frames_for_duration(
            int(np.asarray(raw_audio).reshape(-1).shape[0]),
            sample_rate=self.sample_rate,
        )

        if not has_audio:
            return {
                **data,
                self.audio_key: np.zeros((target_num_frames, expected_dim), dtype=np.float32),
                self.audio_mask_key: np.False_,
            }

        sequence = _audio_tools.compute_whisper_audio_sequence(
            np.asarray(raw_audio, dtype=np.float32),
            sample_rate=self.sample_rate,
            model_name=self.model_name,
            device=self.device,
        )
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2:
            raise ValueError(f"Whisper sequence must be rank-2, got shape {sequence.shape}")
        if sequence.shape[-1] != expected_dim:
            raise ValueError(
                f"Whisper sequence hidden size mismatch: expected {expected_dim}, got {sequence.shape[-1]}"
            )
        sequence = _audio_tools.fit_audio_feature_sequence(sequence, target_num_frames)

        return {
            **data,
            self.audio_key: sequence,
            self.audio_mask_key: np.True_,
        }


@dataclasses.dataclass(frozen=True)
class ExtractWhisperAudioEmbedding(DataTransformFn):
    """Encode an audio waveform window into a pooled Whisper embedding."""

    model_name: str = "openai/whisper-tiny"
    sample_rate: int = 16_000
    device: str = "cpu"
    pooling: Literal["mean", "first"] = "mean"
    output_dim: int | None = None
    audio_key: str = "audio"
    audio_mask_key: str = "audio_mask"

    def __call__(self, data: DataDict) -> DataDict:
        raw_audio = data.get(self.audio_key)
        if raw_audio is None:
            raise ValueError(f"Whisper embedding extraction requires a `{self.audio_key}` field.")

        raw_mask = data.get(self.audio_mask_key)
        has_audio = bool(_as_bool(raw_mask)) if raw_mask is not None else bool(np.asarray(raw_audio).size > 0)
        expected_dim = self.output_dim or _audio_tools.infer_whisper_hidden_size(self.model_name)

        if not has_audio:
            return {
                **data,
                self.audio_key: np.zeros((expected_dim,), dtype=np.float32),
                self.audio_mask_key: np.False_,
            }

        embedding = _audio_tools.compute_whisper_audio_embedding(
            np.asarray(raw_audio, dtype=np.float32),
            sample_rate=self.sample_rate,
            model_name=self.model_name,
            device=self.device,
            pooling=self.pooling,
        )
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding.shape != (expected_dim,):
            raise ValueError(f"Whisper embedding shape mismatch: expected {(expected_dim,)}, got {embedding.shape}")

        return {
            **data,
            self.audio_key: embedding,
            self.audio_mask_key: np.True_,
        }


@dataclasses.dataclass(frozen=True)
class ExtractQwenASRAudioSequence(DataTransformFn):
    """Encode an audio waveform window into Qwen3-ASR audio-tower features."""

    model_name: str = "Qwen/Qwen3-ASR-0.6B"
    sample_rate: int = 16_000
    device: str = "auto"
    num_frames: int | None = None
    output_dim: int | None = None
    audio_key: str = "audio"
    audio_mask_key: str = "audio_mask"

    def __call__(self, data: DataDict) -> DataDict:
        raw_audio = data.get(self.audio_key)
        if raw_audio is None:
            raise ValueError(f"Qwen ASR sequence extraction requires a `{self.audio_key}` field.")

        raw_mask = data.get(self.audio_mask_key)
        has_audio = bool(_as_bool(raw_mask)) if raw_mask is not None else bool(np.asarray(raw_audio).size > 0)
        expected_dim = self.output_dim or _audio_tools.infer_qwen_asr_hidden_size(self.model_name)
        target_num_frames = self.num_frames or _audio_tools.qwen_asr_num_frames_for_duration(
            int(np.asarray(raw_audio).reshape(-1).shape[0]),
            sample_rate=self.sample_rate,
        )

        if not has_audio:
            return {
                **data,
                self.audio_key: np.zeros((target_num_frames, expected_dim), dtype=np.float32),
                self.audio_mask_key: np.False_,
            }

        sequence = _audio_tools.compute_qwen_asr_audio_sequence(
            np.asarray(raw_audio, dtype=np.float32),
            sample_rate=self.sample_rate,
            model_name=self.model_name,
            device=self.device,
        )
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2:
            raise ValueError(f"Qwen ASR sequence must be rank-2, got shape {sequence.shape}")
        if sequence.shape[-1] != expected_dim:
            raise ValueError(
                f"Qwen ASR sequence hidden size mismatch: expected {expected_dim}, got {sequence.shape[-1]}"
            )
        sequence = _audio_tools.fit_audio_feature_sequence(sequence, target_num_frames)

        return {
            **data,
            self.audio_key: sequence,
            self.audio_mask_key: np.True_,
        }


@dataclasses.dataclass(frozen=True)
class ExtractQwenASRAudioEmbedding(DataTransformFn):
    """Encode an audio waveform window into a pooled Qwen3-ASR embedding."""

    model_name: str = "Qwen/Qwen3-ASR-0.6B"
    sample_rate: int = 16_000
    device: str = "auto"
    pooling: Literal["mean", "first"] = "mean"
    output_dim: int | None = None
    audio_key: str = "audio"
    audio_mask_key: str = "audio_mask"

    def __call__(self, data: DataDict) -> DataDict:
        raw_audio = data.get(self.audio_key)
        if raw_audio is None:
            raise ValueError(f"Qwen ASR embedding extraction requires a `{self.audio_key}` field.")

        raw_mask = data.get(self.audio_mask_key)
        has_audio = bool(_as_bool(raw_mask)) if raw_mask is not None else bool(np.asarray(raw_audio).size > 0)
        expected_dim = self.output_dim or _audio_tools.infer_qwen_asr_hidden_size(self.model_name)

        if not has_audio:
            return {
                **data,
                self.audio_key: np.zeros((expected_dim,), dtype=np.float32),
                self.audio_mask_key: np.False_,
            }

        embedding = _audio_tools.compute_qwen_asr_audio_embedding(
            np.asarray(raw_audio, dtype=np.float32),
            sample_rate=self.sample_rate,
            model_name=self.model_name,
            device=self.device,
            pooling=self.pooling,
        )
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding.shape != (expected_dim,):
            raise ValueError(f"Qwen ASR embedding shape mismatch: expected {(expected_dim,)}, got {embedding.shape}")

        return {
            **data,
            self.audio_key: embedding,
            self.audio_mask_key: np.True_,
        }


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
