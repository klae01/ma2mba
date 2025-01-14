# Copyright (c) 2024, Tri Dao, Albert Gu.
# Copyright (c) 2025, Hosu Lee.

from typing import Optional

import torch
from torch import nn, Tensor

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, conv_state=None, ssm_state=None, return_cache: bool = False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states (Tensor): The input sequence to the encoder layer. 
                This is a tensor of shape (batch, sequence_length, hidden_dim) 
                representing the token embeddings or hidden states from the 
                previous layer.

            residual (Optional[Tensor], optional): Residual connection tensor. If provided, 
                the layer applies the residual connection as 
                `hidden_states = Mixer(LN(residual))`. If not provided, the input 
                `hidden_states` will be used directly. Default is None.

            conv_state (Optional[Tensor], optional): Convolutional state tensor used by the mixer 
                layer. This represents additional information needed for 
                processing sequences with convolutional features. 
                Shape is (batch, width - 1, dim + 2 * ngroups * dstate). 
                Default is None.

            ssm_state (Optional[Tensor], optional): State space model (SSM) state tensor 
                used by the mixer layer. This stores intermediate state information 
                for SSM-based sequence processing. Shape is 
                (batch, nheads, headdim, dstate). Default is None.

            return_cache (bool, optional): If True, the layer returns additional 
                cached states required for future computations. This is useful 
                for improving efficiency in tasks like autoregressive decoding. 
                Default is False.
        Returns:
            hidden_states (Tensor): 
                The processed output tensor of shape (batch, sequence_length, hidden_dim), 
                representing the final hidden states after passing through the mixer layer.

            residual (Tensor): 
                The residual connection tensor used in the layer, of the same shape as 
                `hidden_states`. This is either the updated residual or the final hidden states 
                depending on the normalization and residual connection logic.

            conv_state (Tensor, optional): 
                If `return_cache=True`, this is the updated convolutional state tensor, used for 
                maintaining convolutional features in sequential processing. The shape matches 
                the input `conv_state`. If `return_cache=False`, `conv_state` is not returned.

            ssm_state (Tensor, optional): 
                If `return_cache=True`, this is the updated state space model (SSM) state tensor, 
                used for sequential processing in SSM-based layers. The updated `ssm_state` 
                retains the same shape as the input `ssm_state`. If `return_cache=False`, 
                `ssm_state` is not returned.
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )

        kwargs = dict()
        if conv_state is not None:
            kwargs.update(conv_state=conv_state)
        if ssm_state is not None:
            kwargs.update(ssm_state=ssm_state)
        if return_cache:
            kwargs.update(return_cache=return_cache)

        hidden_states = self.mixer(hidden_states, **kwargs)
        if return_cache:
            hidden_states, *states = hidden_states

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm),
                )
            hidden_states = self.mlp(hidden_states)

        if return_cache:
            return hidden_states, residual, *states
        return hidden_states, residual
