"""RLinf PEFT adapters and complete checkpoint loading for PyTorch Pi0."""


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
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad=False)
    inject_adapter_in_model(
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        ),
        model,
    )
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
