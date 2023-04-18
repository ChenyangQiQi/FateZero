# code mostly taken from https://github.com/huggingface/diffusers
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.models.attention import FeedForward, CrossAttention, AdaLayerNorm
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange


@dataclass
class SpatioTemporalTransformerModelOutput(BaseOutput):
    """torch.FloatTensor of shape [batch x channel x frames x height x width]"""

    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class SpatioTemporalTransformerModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        model_config: dict = {},
        **transformer_kwargs,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioTemporalTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    model_config=model_config,
                    **transformer_kwargs,
                )
                for d in range(num_layers)
            ]
        )

        # Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(
        self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True
    ):
        # 1. Input
        clip_length = None
        is_video = hidden_states.ndim == 5
        if is_video:
            clip_length = hidden_states.shape[2]
            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(clip_length, 0)
        else:
            # To adapt to classifier-free guidance where encoder_hidden_states=2
            batch_size = hidden_states.shape[0]//encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(batch_size, 0)
        *_, h, w = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            hidden_states = rearrange(hidden_states, "b c h w -> b (h w) c") # (bf) (hw) c
        else:
            hidden_states = rearrange(hidden_states, "b c h w -> b (h w) c")
            hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states, # [16, 4096, 320]
                encoder_hidden_states=encoder_hidden_states, # ([1, 77, 768]
                timestep=timestep,
                clip_length=clip_length,
            )

        # 3. Output
        if not self.use_linear_projection:
            hidden_states = rearrange(hidden_states, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = rearrange(hidden_states, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        output = hidden_states + residual
        if is_video:
            output = rearrange(output, "(b f) c h w -> b c f h w", f=clip_length)

        if not return_dict:
            return (output,)

        return SpatioTemporalTransformerModelOutput(sample=output)

import copy
class SpatioTemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_sparse_causal_attention: bool = True,
        temporal_attention_position: str = "after_feedforward",
        model_config: dict = {}
    ):
        super().__init__()

        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.use_sparse_causal_attention = use_sparse_causal_attention
        # For safety, freeze the model_config
        self.model_config = copy.deepcopy(model_config)
        if 'least_sc_channel' in model_config:
            if dim< model_config['least_sc_channel']:
                self.model_config['SparseCausalAttention_index'] = []
        
        self.temporal_attention_position = temporal_attention_position
        temporal_attention_positions = ["after_spatial", "after_cross", "after_feedforward"]
        if temporal_attention_position not in temporal_attention_positions:
            raise ValueError(
                f"`temporal_attention_position` must be one of {temporal_attention_positions}"
            )

        # 1. Spatial-Attn
        spatial_attention = SparseCausalAttention if use_sparse_causal_attention else CrossAttention
        self.attn1 = spatial_attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )  # is a self-attention
        self.norm1 = (
            AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
            )
        else:
            self.attn2 = None
            self.norm2 = None

        # 3. Temporal-Attn
        self.attn_temporal = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        nn.init.zeros_(self.attn_temporal.to_out[0].weight.data)  # initialize as an identity function
        self.norm_temporal = (
            AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        )
        # efficient_attention_backward_cutlass is not implemented for large channels
        self.use_xformers = (dim <= 320) or "3090" not in torch.cuda.get_device_name(0)

        # 4. Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                if use_memory_efficient_attention_xformers is True:
                    
                    _ = xformers.ops.memory_efficient_attention(
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                    )
                else:
                    
                    pass
            except Exception as e:
                raise e
            # self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers and self.use_xformers
            self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers and self.use_xformers
            # self.attn_temporal._use_memory_efficient_attention_xformers = (
            #     use_memory_efficient_attention_xformers
            # ),  # FIXME: enabling this raises CUDA ERROR. Gotta dig in.

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        clip_length=None,
    ):
        # 1. Self-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        kwargs = dict(
            hidden_states=norm_hidden_states,
            attention_mask=attention_mask,
        )
        if self.only_cross_attention:
            kwargs.update(encoder_hidden_states=encoder_hidden_states)
        if self.use_sparse_causal_attention:
            kwargs.update(clip_length=clip_length)
        if 'SparseCausalAttention_index' in self.model_config.keys():
            kwargs.update(SparseCausalAttention_index = self.model_config['SparseCausalAttention_index'])
        
        hidden_states = hidden_states + self.attn1(**kwargs)

        if clip_length is not None and self.temporal_attention_position == "after_spatial":
            hidden_states = self.apply_temporal_attention(hidden_states, timestep, clip_length)

        if self.attn2 is not None:
            # 2. Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, # [16, 4096, 320]
                    encoder_hidden_states=encoder_hidden_states, # [1, 77, 768]
                    attention_mask=attention_mask,
                )
                + hidden_states
            )

        if clip_length is not None and self.temporal_attention_position == "after_cross":
            hidden_states = self.apply_temporal_attention(hidden_states, timestep, clip_length)

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        if clip_length is not None and self.temporal_attention_position == "after_feedforward":
            hidden_states = self.apply_temporal_attention(hidden_states, timestep, clip_length)

        return hidden_states

    def apply_temporal_attention(self, hidden_states, timestep, clip_length):
        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=clip_length)
        norm_hidden_states = (
            self.norm_temporal(hidden_states, timestep)
            if self.use_ada_layer_norm
            else self.norm_temporal(hidden_states)
        )
        hidden_states = self.attn_temporal(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
        return hidden_states


class SparseCausalAttention(CrossAttention):
    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        clip_length: int = None,
        SparseCausalAttention_index: list = [-1, 'first']
    ):
        if (
            self.added_kv_proj_dim is not None
            or encoder_hidden_states is not None
            or attention_mask is not None
        ):
            raise NotImplementedError

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        if clip_length is not None:
            key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)
            value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)


            #  *********************** Start of Spatial-temporal attention **********
            frame_index_list = []
            # print(f'SparseCausalAttention_index {str(SparseCausalAttention_index)}')
            if len(SparseCausalAttention_index) > 0:
                for index in SparseCausalAttention_index:
                    if isinstance(index, str):
                        if index == 'first':
                            frame_index = [0] * clip_length
                        if index == 'last':
                            frame_index = [clip_length-1] * clip_length
                        if (index == 'mid') or (index == 'middle'):
                            frame_index = [int(clip_length-1)//2] * clip_length
                    else:
                        assert isinstance(index, int), 'relative index must be int'
                        frame_index = torch.arange(clip_length) + index
                        frame_index = frame_index.clip(0, clip_length-1)
                        
                    frame_index_list.append(frame_index)
                
                key = torch.cat([   key[:, frame_index] for frame_index in frame_index_list
                                    ], dim=2)
                value = torch.cat([ value[:, frame_index] for frame_index in frame_index_list
                                    ], dim=2)


            #  *********************** End of Spatial-temporal attention **********
            key = rearrange(key, "b f d c -> (b f) d c", f=clip_length)
            value = rearrange(value, "b f d c -> (b f) d c", f=clip_length)
        
        
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)
        
        
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(
                    query, key, value, hidden_states.shape[1], dim, attention_mask
                )

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states