# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLinf JAX parameter transforms, generalized to configured model dimensions."""

import numpy as np
import torch


def convert_siglip(params: dict) -> dict:
    """Convert SigLIP ViT parameters from JAX to the OpenPI layout."""
    pt: dict[str, torch.Tensor] = {}
    img = params["PaliGemma"]["img"]

    # Patch embedding (Conv2d): JAX (H, W, C_in, C_out) -> PT (C_out, C_in, H, W)
    pt["img.stem.weight"] = torch.from_numpy(img["embedding"]["kernel"].transpose(3, 2, 0, 1))
    pt["img.stem.bias"] = torch.from_numpy(img["embedding"]["bias"])

    eb = img["Transformer"]["encoderblock"]
    ln0_scale, ln0_bias = eb["LayerNorm_0"]["scale"], eb["LayerNorm_0"]["bias"]
    ln1_scale, ln1_bias = eb["LayerNorm_1"]["scale"], eb["LayerNorm_1"]["bias"]
    dense0_kernel, dense0_bias = (
        eb["MlpBlock_0"]["Dense_0"]["kernel"],
        eb["MlpBlock_0"]["Dense_0"]["bias"],
    )
    dense1_kernel, dense1_bias = (
        eb["MlpBlock_0"]["Dense_1"]["kernel"],
        eb["MlpBlock_0"]["Dense_1"]["bias"],
    )

    mha = eb["MultiHeadDotProductAttention_0"]
    q_kernel, q_bias = mha["query"]["kernel"], mha["query"]["bias"]
    k_kernel, k_bias = mha["key"]["kernel"], mha["key"]["bias"]
    v_kernel, v_bias = mha["value"]["kernel"], mha["value"]["bias"]
    o_kernel, o_bias = mha["out"]["kernel"], mha["out"]["bias"]

    width = img["embedding"]["bias"].shape[-1]
    for i in range(ln0_scale.shape[0]):
        prefix = f"img.encoder.layers.{i}"
        pt[f"{prefix}.norm1.weight"] = torch.from_numpy(ln0_scale[i])
        pt[f"{prefix}.norm1.bias"] = torch.from_numpy(ln0_bias[i])
        pt[f"{prefix}.norm2.weight"] = torch.from_numpy(ln1_scale[i])
        pt[f"{prefix}.norm2.bias"] = torch.from_numpy(ln1_bias[i])

        # JAX Dense uses x @ kernel; PT Linear uses x @ weight.T, so weight = kernel.T.
        q_w = torch.from_numpy(q_kernel[i].reshape(width, width).T)
        k_w = torch.from_numpy(k_kernel[i].reshape(width, width).T)
        v_w = torch.from_numpy(v_kernel[i].reshape(width, width).T)
        q_b = torch.from_numpy(q_bias[i].reshape(width))
        k_b = torch.from_numpy(k_bias[i].reshape(width))
        v_b = torch.from_numpy(v_bias[i].reshape(width))
        pt[f"{prefix}.attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        pt[f"{prefix}.attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        pt[f"{prefix}.attn.out_proj.weight"] = torch.from_numpy(o_kernel[i].reshape(width, width).T)
        pt[f"{prefix}.attn.out_proj.bias"] = torch.from_numpy(o_bias[i])

        pt[f"{prefix}.mlp.fc1.weight"] = torch.from_numpy(dense0_kernel[i].T)
        pt[f"{prefix}.mlp.fc1.bias"] = torch.from_numpy(dense0_bias[i])
        pt[f"{prefix}.mlp.fc2.weight"] = torch.from_numpy(dense1_kernel[i].T)
        pt[f"{prefix}.mlp.fc2.bias"] = torch.from_numpy(dense1_bias[i])

    pt["img.encoder.norm.weight"] = torch.from_numpy(img["Transformer"]["encoder_norm"]["scale"])
    pt["img.encoder.norm.bias"] = torch.from_numpy(img["Transformer"]["encoder_norm"]["bias"])
    pt["img.head.weight"] = torch.from_numpy(img["head"]["kernel"].T)
    pt["img.head.bias"] = torch.from_numpy(img["head"]["bias"])
    pt["img.pos_embedding"] = torch.from_numpy(img["pos_embedding"])  # (1, 256, 1152)
    return pt


def convert_llm(params: dict, *, pi05: bool) -> dict:
    """Convert the dual-expert Gemma LLM (PaliGemma + action expert) from JAX to PyTorch."""
    pt: dict[str, torch.Tensor] = {}
    llm = params["PaliGemma"]["llm"]
    pt["llm.embedder.embedding.weight"] = torch.from_numpy(llm["embedder"]["input_embedding"])

    layers = llm["layers"]
    pg_w = layers["attn"]["q_einsum"]["w"].shape[-2]
    act_w = layers["attn"]["q_einsum_1"]["w"].shape[-2]

    q_einsum = layers["attn"]["q_einsum"]["w"]
    kv_einsum = layers["attn"]["kv_einsum"]["w"]
    o_einsum = layers["attn"]["attn_vec_einsum"]["w"]
    mlp_gating = layers["mlp"]["gating_einsum"]
    mlp_linear = layers["mlp"]["linear"]
    pre_attn_scale = layers["pre_attention_norm"]["scale"]
    pre_ffw_scale = layers["pre_ffw_norm"]["scale"]

    q_einsum_1 = layers["attn"]["q_einsum_1"]["w"]
    kv_einsum_1 = layers["attn"]["kv_einsum_1"]["w"]
    o_einsum_1 = layers["attn"]["attn_vec_einsum_1"]["w"]
    mlp_gating_1 = layers["mlp_1"]["gating_einsum"]
    mlp_linear_1 = layers["mlp_1"]["linear"]

    n_layers = q_einsum.shape[0]
    for i in range(n_layers):
        # Expert 0 (PaliGemma)
        pt[f"llm.layers.{i}.attn.q_proj.0.weight"] = torch.from_numpy(q_einsum[i].transpose(0, 2, 1).reshape(-1, pg_w))
        pt[f"llm.layers.{i}.attn.k_proj.0.weight"] = torch.from_numpy(
            kv_einsum[i, 0].transpose(0, 2, 1).reshape(-1, pg_w)
        )
        pt[f"llm.layers.{i}.attn.v_proj.0.weight"] = torch.from_numpy(
            kv_einsum[i, 1].transpose(0, 2, 1).reshape(-1, pg_w)
        )
        pt[f"llm.layers.{i}.attn.o_proj.0.weight"] = torch.from_numpy(o_einsum[i].reshape(-1, pg_w).T)
        pt[f"llm.layers.{i}.pre_attention_norms.0.scale"] = torch.from_numpy(pre_attn_scale[i])
        pt[f"llm.layers.{i}.pre_ffw_norms.0.scale"] = torch.from_numpy(pre_ffw_scale[i])
        pt[f"llm.layers.{i}.mlps.0.w_gating"] = torch.from_numpy(mlp_gating[i])
        pt[f"llm.layers.{i}.mlps.0.w_linear"] = torch.from_numpy(mlp_linear[i])

        # Expert 1 (Action Expert)
        pt[f"llm.layers.{i}.attn.q_proj.1.weight"] = torch.from_numpy(
            q_einsum_1[i].transpose(0, 2, 1).reshape(-1, act_w)
        )
        pt[f"llm.layers.{i}.attn.k_proj.1.weight"] = torch.from_numpy(
            kv_einsum_1[i, 0].transpose(0, 2, 1).reshape(-1, act_w)
        )
        pt[f"llm.layers.{i}.attn.v_proj.1.weight"] = torch.from_numpy(
            kv_einsum_1[i, 1].transpose(0, 2, 1).reshape(-1, act_w)
        )
        pt[f"llm.layers.{i}.attn.o_proj.1.weight"] = torch.from_numpy(o_einsum_1[i].reshape(-1, act_w).T)
        pt[f"llm.layers.{i}.mlps.1.w_gating"] = torch.from_numpy(mlp_gating_1[i])
        pt[f"llm.layers.{i}.mlps.1.w_linear"] = torch.from_numpy(mlp_linear_1[i])

        if pi05:
            pre_attn_1 = layers["pre_attention_norm_1"]
            pre_ffw_1 = layers["pre_ffw_norm_1"]
            pt[f"llm.layers.{i}.pre_attention_norms.1.ada_modulation.weight"] = torch.from_numpy(
                pre_attn_1["Dense_0"]["kernel"][i].T
            )
            pt[f"llm.layers.{i}.pre_attention_norms.1.ada_modulation.bias"] = torch.from_numpy(
                pre_attn_1["Dense_0"]["bias"][i]
            )
            pt[f"llm.layers.{i}.pre_ffw_norms.1.ada_modulation.weight"] = torch.from_numpy(
                pre_ffw_1["Dense_0"]["kernel"][i].T
            )
            pt[f"llm.layers.{i}.pre_ffw_norms.1.ada_modulation.bias"] = torch.from_numpy(
                pre_ffw_1["Dense_0"]["bias"][i]
            )
        else:
            pt[f"llm.layers.{i}.pre_attention_norms.1.scale"] = torch.from_numpy(
                layers["pre_attention_norm_1"]["scale"][i]
            )
            pt[f"llm.layers.{i}.pre_ffw_norms.1.scale"] = torch.from_numpy(layers["pre_ffw_norm_1"]["scale"][i])

    pt["llm.final_norms.0.scale"] = torch.from_numpy(llm["final_norm"]["scale"])
    if pi05:
        final_norm_1 = llm["final_norm_1"]
        pt["llm.final_norms.1.ada_modulation.weight"] = torch.from_numpy(final_norm_1["Dense_0"]["kernel"].T)
        pt["llm.final_norms.1.ada_modulation.bias"] = torch.from_numpy(final_norm_1["Dense_0"]["bias"])
    else:
        pt["llm.final_norms.1.scale"] = torch.from_numpy(llm["final_norm_1"]["scale"])
    return pt


def convert_projections(params: dict, *, pi05: bool) -> dict:
    """Convert action/time projection parameters from JAX to PyTorch."""
    pt: dict[str, torch.Tensor] = {}
    if pi05:
        proj_keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    else:
        proj_keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
    for key in proj_keys:
        if key not in params:
            continue
        kernel = params[key]["kernel"]
        bias = params[key]["bias"]
        if isinstance(kernel, dict):
            kernel, bias = kernel["value"], bias["value"]
        pt[f"{key}.weight"] = torch.from_numpy(np.array(kernel).T)
        pt[f"{key}.bias"] = torch.from_numpy(np.array(bias))
    return pt
