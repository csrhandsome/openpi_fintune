import dataclasses
from typing import Literal, override

from openpi.models import model as _model

Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: Variant = "gemma_2b"
    action_expert_variant: Variant = "gemma_300m"
    # PyTorch uses RLinf's PEFT all-linear adapter path. Legacy *_lora
    # variants select that path when is_lora is unspecified.
    is_lora: bool | None = None
    lora_rank: int = 32

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if type(self.lora_rank) is not int or self.lora_rank <= 0:
            raise ValueError("lora_rank must be a positive integer")
        if self.is_lora is not None and type(self.is_lora) is not bool:
            raise ValueError("is_lora must be boolean or None")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0
