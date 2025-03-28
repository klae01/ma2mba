# Copyright (c) 2023, Albert Gu, Tri Dao.
# Copyright (c) 2025, Hosu Lee.

import copy
import json
import math
import os
from collections import namedtuple
from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(
                f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2"
            )
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs,
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP,
            hidden_features=d_intermediate,
            out_features=d_model,
            **factory_kwargs,
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.residual_in_fp32 = residual_in_fp32
        self.activation_checkpointing = False
        self.s_vals_constraint = None
        self.l_vals_constraint = None

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=(1 if d_intermediate == 0 else 2),  # 2 if we have MLP
            )
        )

    def activation_checkpointing_enable(
        self, s_vals: Optional[List[int]] = None, l_vals: Optional[List[int]] = None
    ):
        self.activation_checkpointing = True
        self.s_vals_constraint = s_vals
        self.l_vals_constraint = l_vals

    def activation_checkpointing_disable(self):
        self.activation_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        conv_states: Optional[List[Optional[torch.Tensor]]] = None,
        ssm_states: Optional[List[Optional[torch.Tensor]]] = None,
        return_cache: bool = False,
    ):
        if inputs_embeds is None:
            hidden_states = self.embedding(input_ids)
        else:
            hidden_states = inputs_embeds

        hidden_states, residual, *states = (
            self.checkpointing_forward
            if self.activation_checkpointing
            else self.simple_forward
        )(
            hidden_states,
            seq_idx,
            conv_states or [None] * len(self.layers),
            ssm_states or [None] * len(self.layers),
            return_cache,
        )

        if not self.fused_add_norm:
            residual = hidden_states + residual if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )

        if return_cache:
            return hidden_states, *states
        return hidden_states

    def simple_forward(
        self,
        hidden_states: torch.Tensor,
        seq_idx: Optional[torch.Tensor],
        conv_states: List[Optional[torch.Tensor]],
        ssm_states: List[Optional[torch.Tensor]],
        return_cache: bool = False,
    ):
        return_conv_states = []
        return_ssm_states = []
        residual = None

        for layer, conv_state, ssm_state in zip(self.layers, conv_states, ssm_states):
            output = layer(hidden_states, residual, seq_idx, conv_state, ssm_state, return_cache)
            hidden_states, residual, *states = output
            if return_cache:
                layer_conv_states, layer_ssm_states = states
                return_conv_states.append(layer_conv_states)
                return_ssm_states.append(layer_ssm_states)

        if return_cache:
            return hidden_states, residual, return_conv_states, return_ssm_states
        return hidden_states, residual

    def checkpointing_forward(
        self,
        hidden_states: torch.Tensor,
        seq_idx: Optional[torch.Tensor],
        conv_states: List[Optional[torch.Tensor]],
        ssm_states: List[Optional[torch.Tensor]],
        return_cache: bool = False,
    ):
        B, S, F = hidden_states.shape
        L = len(self.layers)
        layer: Mamba2 = self.layers[0].mixer
        dtype_size: int = hidden_states.element_size()

        fp32_multiplier = 4 // dtype_size
        # SSM always uses float32
        ssm_multiplier = fp32_multiplier
        # Residual uses float32 if 'residual_in_fp32' is True
        res_multiplier = fp32_multiplier if self.residual_in_fp32 else 1
        assert layer.d_model == F

        # conv_states: (batch, dim + 2 * ngroups * dstate, width - 1)
        # ssm_states: (batch, nheads, headdim, dstate)
        conv_dim = layer.d_ssm + 2 * layer.ngroups * layer.d_state
        conv_state_size = conv_dim * (layer.d_conv - 1)
        ssm_state_size = layer.d_ssm * layer.d_state
        block_activation = layer.d_model + layer.d_model * res_multiplier

        C_l_ckpt = block_activation
        C_s_ckpt = conv_state_size + ssm_state_size * ssm_multiplier

        # zxbcdt: batch, seqlen, 2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads
        # out_x: batch, seqlen, nheads, headdim
        # hidden_states, residual: batch, seqlen, d_model
        d_in_proj = 2 * layer.d_inner + 2 * layer.ngroups * layer.d_state + layer.nheads
        zxbcdt_size = d_in_proj
        out_x_size = layer.d_inner

        C_grid = zxbcdt_size + out_x_size

        # recompute: conv state + SSM state
        # backward: gradient of input & intermediate
        # out_x is saved previously in C_grid
        mem_eff_path_conv_state = conv_dim
        mem_eff_path_ssm_state = (
            layer.nheads * fp32_multiplier * 2  # dA_cumsum + dt
            + layer.d_ssm * layer.d_state * fp32_multiplier // layer.chunk_size  # states
            + layer.d_ssm * layer.d_state // layer.chunk_size  # states
            + layer.chunk_size * fp32_multiplier  # CB
        )
        restoration_memory = mem_eff_path_conv_state + mem_eff_path_ssm_state

        C_state = restoration_memory

        # # Expected memory usage:
        # L * S // l * C_l_ckpt
        # L * S // s * C_s_ckpt
        # l * s * C_grid
        # s * C_state

        best_mem = float("inf")
        best_s = S
        best_l = L

        for s in self.s_vals_constraint or range(layer.chunk_size, S + 1, layer.chunk_size):
            for l in self.l_vals_constraint or range(1, L + 1):
                M_l_ckpt = L * ((S - 1) // s) * C_s_ckpt
                M_state = s * C_state
                M_s_ckpt = ((L - 1) // l) * S * C_l_ckpt
                M_grid = l * s * C_grid
                total_memory = M_l_ckpt + M_s_ckpt + M_grid + M_state
                if total_memory < best_mem:
                    best_mem = total_memory
                    best_s = s
                    best_l = l

        hidden_state_chunks = list(hidden_states.split(best_s, dim=1))
        residual_chunks = [None] * len(hidden_state_chunks)
        return_conv_states = []
        return_ssm_states = []

        for layer_start in range(0, L, best_l):
            rng = slice(layer_start, layer_start + best_l)
            layer_conv_states = conv_states[rng]
            layer_ssm_states = ssm_states[rng]

            for i in range(len(hidden_state_chunks)):
                return_state = return_cache if i == len(hidden_state_chunks) - 1 else True
                output = torch.utils.checkpoint.checkpoint(
                    self.partial_forward,
                    self.layers[rng],
                    hidden_state_chunks[i],
                    residual_chunks[i],
                    seq_idx,
                    layer_conv_states,
                    layer_ssm_states,
                    return_state,
                    use_reentrant=False,
                )

                hidden_state_chunks[i], residual_chunks[i], *states = output
                if return_state:
                    layer_conv_states, layer_ssm_states = states

            if return_cache:
                return_conv_states.extend(layer_conv_states)
                return_ssm_states.extend(layer_ssm_states)

        if return_cache:
            return (
                torch.cat(hidden_state_chunks, 1),
                torch.cat(residual_chunks, 1),
                return_conv_states,
                return_ssm_states,
            )

        return torch.cat(hidden_state_chunks, 1), torch.cat(residual_chunks, 1)

    @classmethod
    def partial_forward(
        cls,
        layers: List[nn.Module],
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        conv_states: List[Optional[torch.Tensor]],
        ssm_states: List[Optional[torch.Tensor]],
        return_cache: bool,
    ):
        return_conv_states = []
        return_ssm_states = []
        for layer, conv_state, ssm_state in zip(layers, conv_states, ssm_states):
            if return_cache:
                hidden_states, residual, conv_state_out, ssm_state_out = layer(
                    hidden_states, residual, conv_state, ssm_state, True
                )
                return_conv_states.append(conv_state_out)
                return_ssm_states.append(ssm_state_out)
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, conv_state, ssm_state, False
                )

        if return_cache:
            return hidden_states, residual, return_conv_states, return_ssm_states
        return hidden_states, residual

    @torch.no_grad()
    def step(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        conv_states: Optional[List[Optional[torch.Tensor]]] = None,
        ssm_states: Optional[List[Optional[torch.Tensor]]] = None,
    ):
        if inputs_embeds is None:
            hidden_states = self.embedding(input_ids)
        else:
            hidden_states = inputs_embeds

        assert torch.is_tensor(hidden_states)
        assert hidden_states.ndim == 3

        conv_states = conv_states or [None] * len(self.layers)
        ssm_states = ssm_states or [None] * len(self.layers)
        return_conv_states = []
        return_ssm_states = []
        residual = None
        for layer, conv_state, ssm_state in zip(self.layers, conv_states, ssm_states):
            hidden_states, residual, conv_state_out, ssm_state_out = layer(
                hidden_states, residual, conv_state, ssm_state, True
            )
            return_conv_states.append(conv_state_out)
            return_ssm_states.append(ssm_state_out)

        hidden_states = hidden_states[:, -1:, :]
        residual = residual[:, -1:, :]

        if not self.fused_add_norm:
            residual = hidden_states + residual if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )

        return hidden_states, return_conv_states, return_ssm_states


class MambaLMHeadModel(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def forward(
        self,
        input_ids,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        **mixer_kwargs,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        assert inference_params is None
        hidden_states = self.backbone(input_ids, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(
            load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        )
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=4)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        gradient_checkpointing_kwargs = gradient_checkpointing_kwargs or {}
        self.backbone.activation_checkpointing_enable(**gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.backbone.activation_checkpointing_disable()

    def get_input_embeddings(self):
        return self.backbone.embedding
