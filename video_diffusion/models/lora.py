from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

# from diffusers.utils
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import FeedForward, CrossAttention, AdaLayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, stride=1):
        super().__init__()

        if rank > min(in_features, out_features):
            Warning(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}, reset to {min(in_features, out_features)//2}")
            rank = min(in_features, out_features)//2
            

        self.down = nn.Conv1d(in_features, rank, bias=False,
                                kernel_size=3,
                                stride = stride,
                                padding=1,)
        self.up = nn.Conv1d(rank, out_features, bias=False,
                            kernel_size=3,
                            padding=1,)

        nn.init.normal_(self.down.weight, std=1 / rank)
        # nn.init.zeros_(self.down.bias.data)
        
        nn.init.zeros_(self.up.weight)
        # nn.init.zeros_(self.up.bias.data)
        if stride > 1:
            self.skip = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        if hasattr(self, 'skip'):
            hidden_states=self.skip(hidden_states)
        return up_hidden_states.to(orig_dtype)+hidden_states


class LoRACrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, rank=4):
        super().__init__()

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank)

    def __call__(
        self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    
    
    
    
class LoRAXFormersCrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim, rank=4):
        super().__init__()

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank)

    def __call__(
        self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query).contiguous()

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)

        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
