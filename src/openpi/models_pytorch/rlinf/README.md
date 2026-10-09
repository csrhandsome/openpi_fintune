The modules in this directory come from RLinf commit `61ba34e6`,
`rlinf/models/embodiment/openpi/modules/`, under Apache-2.0.

`../pi0_pytorch.py` ports the network, flow-matching loss, and sampler from
RLinf's `rlinf/models/embodiment/openpi/pi0.py` at the same commit. It accepts
OpenPI's observation and training call interface. `gemma.py` adds headwise
attention LoRA so the existing OpenPI LoRA model variants can train; upstream
RLinf's native attention LoRA branch raises `NotImplementedError`.

Training expects RLinf-layout `model.safetensors` weights (`llm.*`, `img.*`,
action projections). Dataset transforms and normalization remain in OpenPI's
training data loader. The user's existing JAX LoRA checkpoints are outside the
scope of this PyTorch port.
