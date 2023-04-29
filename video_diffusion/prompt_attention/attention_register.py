"""
register the attention controller into the UNet of stable diffusion
Build a customized attention function `_attention'
Replace the original attention function with `forward' and `spatial_temporal_forward' in attention_controlled_forward function
Most of spatial_temporal_forward is directly copy from `video_diffusion/models/attention.py'
TODO FIXME: merge redundant code with attention.py
"""

from einops import rearrange
import torch
import torch.nn.functional as F


def register_attention_control(model, controller):
    "Connect a model with a controller"
    def attention_controlled_forward(self, place_in_unet, attention_type='cross'):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        def _attention( query, key, value, is_cross, attention_mask=None):
            if self.upcast_attention:
                query = query.float()
                key = key.float()

            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query,
                key.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            if self.upcast_softmax:
                attention_scores = attention_scores.float()

            attention_probs = attention_scores.softmax(dim=-1)

            # cast back to the original dtype
            attention_probs = attention_probs.to(value.dtype)

            # START OF CORE FUNCTION
            # Record during inversion and edit the attention probs during editing
            attention_probs = controller(reshape_batch_dim_to_temporal_heads(attention_probs), 
                                         is_cross, place_in_unet)
            attention_probs = reshape_temporal_heads_to_batch_dim(attention_probs)
            # END OF CORE FUNCTION
            
            # compute attention output
            hidden_states = torch.bmm(attention_probs, value)

            # reshape hidden_states
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

        def reshape_temporal_heads_to_batch_dim( tensor):
            head_size = self.heads
            tensor = rearrange(tensor, " b h s t -> (b h) s t ", h = head_size)
            return tensor

        def reshape_batch_dim_to_temporal_heads(tensor):
            head_size = self.heads
            tensor = rearrange(tensor, "(b h) s t -> b h s t", h = head_size)
            return tensor
        
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            # hidden_states: torch.Size([16, 4096, 320])
            # encoder_hidden_states: torch.Size([16, 77, 768])
            is_cross = encoder_hidden_states is not None
            
            encoder_hidden_states = encoder_hidden_states

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)
            query = self.reshape_heads_to_batch_dim(query)

            if self.added_kv_proj_dim is not None:
                key = self.to_k(hidden_states)
                value = self.to_v(hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                key = self.reshape_heads_to_batch_dim(key)
                value = self.reshape_heads_to_batch_dim(value)
                encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
                encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

                key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
                value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
            else:
                encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                key = self.reshape_heads_to_batch_dim(key)
                value = self.reshape_heads_to_batch_dim(value)

            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                    attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)


            if self._use_memory_efficient_attention_xformers and query.shape[-2] > 32 ** 2:
                # for large attention map of 64X64, use xformers to save memory
                hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
            
                hidden_states = _attention(query, key, value, is_cross=is_cross, attention_mask=attention_mask)
                # else:
                #     hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

            # linear proj
            hidden_states = to_out(hidden_states)

            # dropout
            # hidden_states = self.to_out[1](hidden_states)
            return hidden_states


        def spatial_temporal_forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            clip_length: int = None,
            SparseCausalAttention_index: list = [-1, 'first']
        ):
            """
            Most of spatial_temporal_forward is directly copy from `video_diffusion.models.attention.SparseCausalAttention'
            We add two modification
            1. use self defined attention function that is controlled by AttentionControlEdit module
            2. remove the dropout to reduce randomness
            FIXME: merge redundant code with attention.py

            """
            if (
                self.added_kv_proj_dim is not None
                or encoder_hidden_states is not None
                or attention_mask is not None
            ):
                raise NotImplementedError

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)
            query = self.reshape_heads_to_batch_dim(query)

            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            if clip_length is not None:
                key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)
                value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)


                #  *********************** Start of Spatial-temporal attention **********
                frame_index_list = []
                
                if len(SparseCausalAttention_index) > 0:
                    for index in SparseCausalAttention_index:
                        if isinstance(index, str):
                            if index == 'first':
                                frame_index = [0] * clip_length
                            if index == 'last':
                                frame_index = [clip_length-1] * clip_length
                            if (index == 'mid') or (index == 'middle'):
                                frame_index = [int((clip_length-1)//2)] * clip_length
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

            if self._use_memory_efficient_attention_xformers and query.shape[-2] > 32 ** 2:
                # FIXME there should be only one variable to control whether use xformers
                # if self._use_memory_efficient_attention_xformers:
                # for large attention map of 64X64, use xformers to save memory
                hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
            # if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = _attention(query, key, value, attention_mask=attention_mask, is_cross=False)
            # else:
            #     hidden_states = self._sliced_attention(
            #         query, key, value, hidden_states.shape[1], dim, attention_mask
            #     )

            # linear proj
            hidden_states = to_out(hidden_states)

            # dropout
            # hidden_states = self.to_out[1](hidden_states)
            return hidden_states
        if attention_type == 'CrossAttention':
            return forward
        elif attention_type == "SparseCausalAttention":
            return spatial_temporal_forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()
    
    def register_recr(net_, count, place_in_unet):
        if net_[1].__class__.__name__ == 'CrossAttention' \
            or net_[1].__class__.__name__ == 'SparseCausalAttention':
            net_[1].forward = attention_controlled_forward(net_[1], place_in_unet, attention_type = net_[1].__class__.__name__)
            return count + 1
        elif hasattr(net_[1], 'children'):
            for net in net_[1].named_children():
                if net[0] !='attn_temporal':

                    count = register_recr(net, count, place_in_unet)

        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net, 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net, 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net, 0, "mid")
    print(f"Number of attention layer registered {cross_att_count}")
    controller.num_att_layers = cross_att_count
