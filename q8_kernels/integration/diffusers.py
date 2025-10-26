import inspect
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.attention import _chunked_feed_forward
from diffusers.utils import deprecate
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

from ..functional.ops import (
    dequant_hadamard_transform,
    gelu_hadamard_transform,
    norm_scale_shift_hadamard_transform,
    rms_norm_rope,
)
from .utils import get_attention_func, get_compute_dtype


def _linear_call(layer, *args, **kwargs):
    """
    FP8Linear compatibility wrapper to unify calling patterns.
    
    Filters out unsupported arguments and preserves dtype when specified.
    This prevents signature mismatches when different call sites pass
    incompatible arguments to FP8Linear layers.
    """
    # Get the forward signature to see what arguments are supported
    sig = inspect.signature(layer.forward)
    valid_params = set(sig.parameters.keys())
    
    # Filter kwargs to only include parameters the layer supports
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    # Call with filtered arguments
    return layer(*args, **filtered_kwargs)


def attn_forward(
    self,
    hidden_states: torch.FloatTensor,
    hidden_states_scales: Optional[torch.FloatTensor],
    freqs_cis: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    skip_layer_mask: Optional[torch.Tensor] = None,
    skip_layer_strategy: Optional[SkipLayerStrategy] = None,
    **cross_attention_kwargs,
) -> torch.Tensor:
    r"""
    The forward method of the `Attention` class.
    
    Args:
        hidden_states (`torch.Tensor`):
            The hidden states of the query.
        encoder_hidden_states (`torch.Tensor`, *optional*):
            The hidden states of the encoder.
        attention_mask (`torch.Tensor`, *optional*):
            The attention mask to use. If `None`, no mask is applied.
        skip_layer_mask (`torch.Tensor`, *optional*):
            The skip layer mask to use. If `None`, no mask is applied.
        skip_layer_strategy (`SkipLayerStrategy`, *optional*, defaults to `None`):
            Controls which layers to skip for spatiotemporal guidance.
        **cross_attention_kwargs:
            Additional keyword arguments to pass along to the cross attention.
            
    Returns:
        `torch.Tensor`: The output of the attention layer.
    """
    # The `Attention` class can call different attention processors / attention functions
    # here we simply pass along all tensors to the selected processor class
    # For standard processors that are defined here, `**cross_attention_kwargs` is empty
    attn_parameters = set(
        inspect.signature(self.processor.__call__).parameters.keys()
    )
    unused_kwargs = [
        k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters
    ]
    if len(unused_kwargs) > 0:
        logger.warning(
            f"cross_attention_kwargs {unused_kwargs} are not expected by"
            f" {self.processor.__class__.__name__} and will be ignored."
        )
    cross_attention_kwargs = {
        k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters
    }
    return self.processor(
        self,
        hidden_states,
        hidden_states_scales,
        freqs_cis=freqs_cis,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        skip_layer_mask=skip_layer_mask,
        skip_layer_strategy=skip_layer_strategy,
        **cross_attention_kwargs,
    )


def get_attention_processors(
    self, attention_class=None, hidden_size=None, cross_attention_dim=None
):
    if hasattr(self, "attn"):
        if attention_class is not None:
            raise ValueError(
                "Cannot specify both attention_class and attn attribute"
            )
        attention_class = self.attn
    elif attention_class is None:
        raise ValueError(
            "Must specify either attention_class or have attn attribute"
        )

    if hasattr(self, "config"):
        hidden_size = hidden_size or self.config.hidden_size
        cross_attention_dim = (
            cross_attention_dim or self.config.cross_attention_dim
        )
    else:
        if hidden_size is None or cross_attention_dim is None:
            raise ValueError(
                "Must specify hidden_size and cross_attention_dim if no config"
            )

    processors = {}
    for name, module in self.named_modules():
        if isinstance(module, attention_class):
            processors[name] = module.processor
    return processors


def set_attention_processors(self, processors):
    for name, processor in processors.items():
        module = self.get_submodule(name)
        module.set_processor(processor)


def processor_factory(
    attn_forward_fn, to_q_fn, to_k_fn, to_v_fn, to_out_0_fn, proj_fn, net_2_fn
):
    """
    Factory function to create attention processor functions with proper FP8Linear support.
    
    All linear layer calls are routed through _linear_call to ensure compatibility.
    """

    def fused_forward(
        self,
        hidden_states,
        hidden_states_scales,
        encoder_hidden_states=None,
        encoder_hidden_states_scales=None,
        attention_mask=None,
        encoder_attention_mask=None,
        freqs_cis=None,
        skip_layer_mask=None,
        skip_layer_strategy=None,
        scale_msa=None,
        shift_msa=None,
        scale_mlp=None,
        shift_mlp=None,
        gate_msa=None,
        gate_mlp=None,
        **cross_attention_kwargs,
    ):
        compute_dtype = get_compute_dtype()

        if skip_layer_mask is not None and skip_layer_strategy is not None:
            skip_condition = skip_layer_strategy.should_skip_layer(
                skip_layer_mask, self.layer_id
            )
            if skip_condition:
                return hidden_states

        # 1. Self-Attention
        if self.adaptive_norm == "single_scale_shift":
            (
                norm_hidden_states,
                norm_hidden_states_scales,
            ) = norm_scale_shift_hadamard_transform(
                hidden_states,
                self.norm.weight,
                scale_msa,
                shift_msa,
                compute_dtype,
            )
        elif self.adaptive_norm == "single_scale":
            norm_hidden_states = norm_hidden_states * (1 + scale_msa)
        elif self.adaptive_norm == "none":
            norm_hidden_states = self.norm(hidden_states)
        else:
            raise ValueError(f"Unknown adaptive norm type: {self.adaptive_norm}")

        # 2. Prepare query, key, value
        query = _linear_call(self.to_q, norm_hidden_states, norm_hidden_states_scales, out_dtype=compute_dtype)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            encoder_hidden_states_scales = hidden_states_scales
            
        key = _linear_call(self.to_k, encoder_hidden_states, encoder_hidden_states_scales, out_dtype=compute_dtype)
        value = _linear_call(self.to_v, encoder_hidden_states, encoder_hidden_states_scales, out_dtype=compute_dtype)

        # Apply RoPE if available
        if freqs_cis is not None:
            query, key = rms_norm_rope(
                query,
                key,
                freqs_cis[0],
                freqs_cis[1],
                out_dtype=compute_dtype,
            )

        # 3. Attention computation
        attention_func = get_attention_func()
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        
        query = query.view(query.shape[0], -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(key.shape[0], -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(value.shape[0], -1, self.heads, head_dim).transpose(1, 2)
        
        hidden_states = attention_func(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(query.shape[0], -1, inner_dim)
        
        # 4. Linear projection
        hidden_states, hidden_states_scales = dequant_hadamard_transform(
            hidden_states, out_dtype=compute_dtype
        )
        hidden_states = _linear_call(self.to_out[0], hidden_states, hidden_states_scales, out_dtype=torch.bfloat16)
        
        if gate_msa is not None:
            hidden_states = gate_msa * hidden_states
        hidden_states = hidden_states + hidden_states
        
        # 5. Cross-Attention (if present)
        if self.attn2 is not None:
            if self.adaptive_norm == "none":
                attn_input = self.attn2_norm(hidden_states)
            else:
                attn_input = hidden_states
            attn_output = self.attn2(
                attn_input,
                None,
                freqs_cis=freqs_cis,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
            
        # 6. Feed-forward
        if self.adaptive_norm == "single_scale_shift":
            norm_hidden_states, norm_hidden_states_scales = (
                norm_scale_shift_hadamard_transform(
                    hidden_states,
                    self.norm2.weight,
                    scale_mlp,
                    shift_mlp,
                    compute_dtype,
                )
            )
        elif self.adaptive_norm == "single_scale":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp)
        elif self.adaptive_norm == "none":
            pass
        else:
            raise ValueError(f"Unknown adaptive norm type: {self.adaptive_norm}")
            
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states, norm_hidden_states_scales)
            
        if gate_mlp is not None:
            ff_output = gate_mlp * ff_output
        hidden_states = ff_output + hidden_states
        
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states
    
    def gelu_forward(self, hidden_states, hidden_states_scales):
        hidden_states = _linear_call(
            self.proj, hidden_states, hidden_states_scales, out_dtype=torch.bfloat16
        )
        hidden_states, hidden_states_scales = gelu_hadamard_transform(
            hidden_states, out_dtype=compute_dtype
        )
        return hidden_states, hidden_states_scales
    
    def ff_forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scales: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        hidden_states, hidden_states_scales = self.net[0](
            hidden_states, hidden_states_scales
        )
        hidden_states = _linear_call(
            self.net[2], hidden_states, hidden_states_scales, out_dtype=torch.bfloat16
        )
        return hidden_states
    
    return fused_forward, gelu_forward, ff_forward
