The modules in this directory come from RLinf commit `61ba34e6`,
`rlinf/models/embodiment/openpi/modules/`, under Apache-2.0.

`../pi0_pytorch.py` ports the network, flow-matching loss, and sampler from
RLinf's `rlinf/models/embodiment/openpi/pi0.py` at the same commit. It accepts
OpenPI's observation and training call interface. The vendored Gemma is
unmodified; LoRA uses RLinf's PEFT `all-linear` injection in `../weights.py`,
with frozen base parameters, rank equal to alpha, Gaussian A and zero B.
Native attention LoRA remains unsupported upstream and is not constructed.

All checkpoint loads require complete RLinf-layout weights (`llm.*`, `img.*`,
action projections). Loading, training and conversion use the same selective
BF16/FP32 policy. Dataset transforms and normalization remain in OpenPI's
training data loader.

`examples/convert_jax_model_to_pytorch.py` converts JAX checkpoints directly
into this layout, strictly validates every tensor before writing, and merges
native JAX LoRA into dense weights. New PEFT adapters can then start from
those merged weights; native JAX optimizer states are not migrated.
