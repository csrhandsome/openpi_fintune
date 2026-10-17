"""PEFT adapters with RLinf or OpenPI training scopes and strict weight loading."""

import torch


def uses_lora(config):
    enabled = getattr(config, "is_lora", None)
    if enabled is not None:
        return enabled
    return any("lora" in variant for variant in (config.paligemma_variant, config.action_expert_variant))


def enable_lora(model):
    if not uses_lora(model.config) or getattr(model, "peft_enabled", False):
        return
    from peft import LoraConfig
    from peft import inject_adapter_in_model

    rank = getattr(model.config, "lora_rank", 32)
    mode = getattr(model.config, "lora_mode", "lora_only")
    if mode not in ("lora_only", "openpi"):
        raise ValueError("lora_mode must be lora_only or openpi")
    targets = "all-linear"
    if mode == "openpi":
        targets = [
            name
            for name, module in model.named_modules()
            if name.startswith("llm.") and isinstance(module, torch.nn.Linear)
        ]
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad=False)
    inject_adapter_in_model(
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            target_modules=targets,
            init_lora_weights="gaussian",
        ),
        model,
    )
    if mode == "openpi":
        # Official JAX freeze filters only freeze non-LoRA parameters in llm.
        # Vision and the external action/state/time projections train directly.
        for name, parameter in model.named_parameters():
            if not name.startswith("llm."):
                parameter.requires_grad_(requires_grad=True)
    model.peft_enabled = True


def load_weights(model, state):
    """Load a base checkpoint or a complete PEFT checkpoint, without missing weights."""
    if any(".lora_a" in key or "_lora_a" in key for key in state):
        raise ValueError("Native headwise LoRA checkpoints need conversion; this model uses RLinf PEFT LoRA")
    adapted = any(".lora_A." in key or ".base_layer." in key for key in state)
    if adapted:
        if not uses_lora(model.config):
            raise ValueError("PEFT checkpoint requires model.is_lora=true or a *_lora variant")
        enable_lora(model)
    # strict=True also rejects old HF names rather than saving a random backbone.
    model.load_state_dict(state, strict=True)
    if not adapted:
        enable_lora(model)
