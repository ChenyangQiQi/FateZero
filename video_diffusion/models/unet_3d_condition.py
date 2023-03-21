# code mostly taken from https://github.com/huggingface/diffusers
import os
import glob
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import copy

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .unet_3d_blocks import (
    CrossAttnDownBlockPseudo3D,
    CrossAttnUpBlockPseudo3D,
    DownBlockPseudo3D,
    UNetMidBlockPseudo3DCrossAttn,
    UpBlockPseudo3D,
    get_down_block,
    get_up_block,
)
from .resnet import PseudoConv3d


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNetPseudo3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNetPseudo3DConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockPseudo3D",
            "CrossAttnDownBlockPseudo3D",
            "CrossAttnDownBlockPseudo3D",
            "DownBlockPseudo3D",
        ),
        mid_block_type: str = "UNetMidBlockPseudo3DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlockPseudo3D",
            "CrossAttnUpBlockPseudo3D",
            "CrossAttnUpBlockPseudo3D",
            "CrossAttnUpBlockPseudo3D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        **kwargs
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4
        if 'temporal_downsample' in kwargs and  kwargs['temporal_downsample'] is True:
            kwargs['temporal_downsample_time'] = 3
        self.temporal_downsample_time = kwargs.get('temporal_downsample_time', 0)
        
        # input
        self.conv_in = PseudoConv3d(in_channels, block_out_channels[0], 
                                    kernel_size=3, padding=(1, 1), model_config=kwargs)

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            kwargs_copy=copy.deepcopy(kwargs)
            temporal_downsample_i = ((i >= (len(down_block_types)-self.temporal_downsample_time))
                                    and (not is_final_block))
            kwargs_copy.update({'temporal_downsample': temporal_downsample_i} )
            
            # kwargs_copy.update({'SparseCausalAttention_index': temporal_downsample_i} )
            if temporal_downsample_i:
                print(f'Initialize model temporal downsample at layer {i}')
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                model_config=kwargs_copy
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlockPseudo3DCrossAttn":
            self.mid_block = UNetMidBlockPseudo3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                model_config=kwargs
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False
            
            kwargs_copy=copy.deepcopy(kwargs)
            kwargs_copy.update({'temporal_downsample': 
                i < (self.temporal_downsample_time-1)})
            if i < (self.temporal_downsample_time-1):
                print(f'Initialize model temporal updample at layer {i}')

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                model_config=kwargs_copy
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = PseudoConv3d(block_out_channels[0], out_channels, 
                                     kernel_size=3, padding=1, model_config=kwargs)

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = (
            num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size
        )

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(
            module,
            (CrossAttnDownBlockPseudo3D, DownBlockPseudo3D, CrossAttnUpBlockPseudo3D, UpBlockPseudo3D),
        ):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None, # None
        attention_mask: Optional[torch.Tensor] = None, # None
        return_dict: bool = True,
    ) -> Union[UNetPseudo3DConditionOutput, Tuple]:
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None: # None
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample: # False
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )
        # for i in down_block_res_samples: print(i.shape) 
        # torch.Size([1, 320, 16, 64, 64])
        # torch.Size([1, 320, 16, 64, 64])
        # torch.Size([1, 320, 16, 64, 64])
        # torch.Size([1, 320, 8, 32, 32])
        # torch.Size([1, 640, 8, 32, 32])
        # torch.Size([1, 640, 8, 32, 32])
        # torch.Size([1, 640, 4, 16, 16])
        # torch.Size([1, 1280, 4, 16, 16])
        # torch.Size([1, 1280, 4, 16, 16])
        # torch.Size([1, 1280, 2, 8, 8])
        # torch.Size([1, 1280, 2, 8, 8])
        # torch.Size([1, 1280, 2, 8, 8])

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNetPseudo3DConditionOutput(sample=sample)

    @classmethod
    def from_2d_model(cls, model_path, model_config):
        config_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(config_path):
            raise RuntimeError(f"{config_path} does not exist")
        with open(config_path, "r") as f:
            config = json.load(f)

        config.pop("_class_name")
        config.pop("_diffusers_version")

        block_replacer = {
            "CrossAttnDownBlock2D": "CrossAttnDownBlockPseudo3D",
            "DownBlock2D": "DownBlockPseudo3D",
            "UpBlock2D": "UpBlockPseudo3D",
            "CrossAttnUpBlock2D": "CrossAttnUpBlockPseudo3D",
        }

        def convert_2d_to_3d_block(block):
            return block_replacer[block] if block in block_replacer else block

        config["down_block_types"] = [
            convert_2d_to_3d_block(block) for block in config["down_block_types"]
        ]
        config["up_block_types"] = [convert_2d_to_3d_block(block) for block in config["up_block_types"]]
        if model_config is not None:
            config.update(model_config)

        model = cls(**config)

        state_dict_path_condidates = glob.glob(os.path.join(model_path, "*.bin"))
        if state_dict_path_condidates:
            state_dict = torch.load(state_dict_path_condidates[0], map_location="cpu")
            model.load_2d_state_dict(state_dict=state_dict)

        return model

    def load_2d_state_dict(self, state_dict, **kwargs):
        state_dict_3d = self.state_dict()

        for k, v in state_dict.items():
            if k not in state_dict_3d:
                raise KeyError(f"2d state_dict key {k} does not exist in 3d model")
            elif v.shape != state_dict_3d[k].shape:
                raise ValueError(f"state_dict shape mismatch, 2d {v.shape}, 3d {state_dict_3d[k].shape}")

        for k, v in state_dict_3d.items():
            if "_temporal" in k:
                continue
            if k not in state_dict:
                raise KeyError(f"3d state_dict key {k} does not exist in 2d model")

        state_dict_3d.update(state_dict)
        self.load_state_dict(state_dict_3d, **kwargs)
