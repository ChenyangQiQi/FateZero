"""
Collect all function in prompt_attention folder.
Provide a API `make_controller' to return an initialized AttentionControlEdit class object in the main validation loop.
"""

from typing import Optional, Union, Tuple, List, Dict
import abc
import numpy as np
import copy
from einops import rearrange

import torch
import torch.nn.functional as F

import video_diffusion.prompt_attention.ptp_utils as ptp_utils
import video_diffusion.prompt_attention.seq_aligner as seq_aligner
from video_diffusion.prompt_attention.spatial_blend import SpatialBlender
from video_diffusion.prompt_attention.visualization import show_cross_attention, show_self_attention_comp
from video_diffusion.prompt_attention.attention_store import AttentionStore, AttentionControl
from video_diffusion.prompt_attention.attention_register import register_attention_control
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


        
        
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControlEdit(AttentionStore, abc.ABC):
    """Decide self or cross-attention. Call the reweighting cross attention module

    Args:
        AttentionStore (_type_): ([1, 4, 8, 64, 64])
        abc (_type_): [8, 8, 1024, 77]
    """
    
    def step_callback(self, x_t):
        x_t = super().step_callback(x_t)
        x_t_device = x_t.device
        x_t_dtype = x_t.dtype
        if self.latent_blend is not None:
            if self.use_inversion_attention:
                step_in_store = len(self.additional_attention_store.latents_store) - self.cur_step
            else:
                step_in_store = self.cur_step
            
            inverted_latents = self.additional_attention_store.latents_store[step_in_store]
            inverted_latents = inverted_latents.to(device =x_t_device, dtype=x_t_dtype)
            # [prompt, channel, clip, res, res] = [1, 4, 2, 64, 64]
            
            blend_dict = self.get_empty_cross_store()
            # each element in blend_dict have (prompt head) clip_length (res res) words, 
            # to better align with  (b c f h w)
            
            step_in_store_atten_dict = self.additional_attention_store.attention_store_all_step[step_in_store]
            if isinstance(step_in_store_atten_dict, str): 
                step_in_store_atten_dict = torch.load(step_in_store_atten_dict)
            
            for key in blend_dict.keys():
                place_in_unet_cross_atten_list = step_in_store_atten_dict[key]
                for i, attention in enumerate(place_in_unet_cross_atten_list):

                    concate_attention = torch.cat([attention[None, ...], self.attention_store[key][i][None, ...]], dim=0)
                    blend_dict[key].append(copy.deepcopy(concate_attention))
            x_t = self.latent_blend(x_t = copy.deepcopy(torch.cat([inverted_latents, x_t], dim=0)), attention_store = copy.deepcopy(blend_dict))
            return x_t[1:, ...]
        else:
            return x_t
        
    def replace_self_attention(self, attn_base, att_replace, reshaped_mask=None):
        if att_replace.shape[-2] <= 32 ** 2:
            target_device = att_replace.device
            target_dtype  = att_replace.dtype
            attn_base = attn_base.to(target_device, dtype=target_dtype)
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            if reshaped_mask is not None:
                return_attention = reshaped_mask*att_replace + (1-reshaped_mask)*attn_base
                return return_attention
            else:
                return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def update_attention_position_dict(self, current_attention_key):
        self.attention_position_counter_dict[current_attention_key] +=1


    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if attn.shape[-2] <= 32 ** 2:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            current_pos = self.attention_position_counter_dict[key]

            if self.use_inversion_attention:
                step_in_store = len(self.additional_attention_store.attention_store_all_step) - self.cur_step -1
            else:
                step_in_store = self.cur_step
                
            step_in_store_atten_dict = self.additional_attention_store.attention_store_all_step[step_in_store]
            
            if isinstance(step_in_store_atten_dict, str): 
                step_in_store_atten_dict = torch.load(step_in_store_atten_dict)
            
            # Note that attn is append to step_store, 
            # if attn is get through clean -> noisy, we should inverse it
            attn_base = step_in_store_atten_dict[key][current_pos]          
            
            self.update_attention_position_dict(key)
            # save in format of [temporal, head, resolution, text_embedding]
            if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
                clip_length = attn.shape[0] // (self.batch_size)
                attn = attn.reshape(self.batch_size, clip_length, *attn.shape[1:])
                # Replace att_replace with attn_base
                attn_base, attn_repalce = attn_base, attn[0:]
                if is_cross:
                    alpha_words = self.cross_replace_alpha[self.cur_step]
                    attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                    attn[0:] = attn_repalce_new # b t h p n = [1, 1, 8, 1024, 77]
                else:
                    
                    # start of masked self-attention
                    if self.attention_blend is not None and attn_repalce.shape[-2] <= 32 ** 2:
                        # ca_this_step = step_in_store_atten_dict
                        # query 1024, key 2048
                        h = int(np.sqrt(attn_repalce.shape[-2]))
                        w = h
                        mask = self.attention_blend(target_h = h, target_w =w, attention_store= step_in_store_atten_dict, step_in_store=step_in_store)
                        # reshape from ([ 1, 2, 32, 32]) -> [2, 1, 1024, 1]
                        reshaped_mask = rearrange(mask, "d c h w -> c d (h w)")[..., None]
                        
                        # input has shape  (h) c res words
                        # one meens using target self-attention, zero is using source
                        # Previous implementation us all zeros
                        # mask should be repeat.
                    else: 
                        reshaped_mask = None
                    attn[0:] = self.replace_self_attention(attn_base, attn_repalce, reshaped_mask)

                
                
                attn = attn.reshape(self.batch_size * clip_length, *attn.shape[2:])
                # save in format of [temporal, head, resolution, text_embedding]
                
        return attn
    def between_steps(self):

        super().between_steps()
        self.step_store = self.get_empty_store()
        
        self.attention_position_counter_dict = {
            'down_cross': 0,
            'mid_cross': 0,
            'up_cross': 0,
            'down_self': 0,
            'mid_self': 0,
            'up_self': 0,
        }        
        return 
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 latent_blend: Optional[SpatialBlender], tokenizer=None, 
                 additional_attention_store: AttentionStore =None,
                 use_inversion_attention: bool=False,
                 attention_blend: SpatialBlender= None,
                 save_self_attention: bool=True,
                 disk_store=False
                 ):
        super(AttentionControlEdit, self).__init__(
            save_self_attention=save_self_attention,
            disk_store=disk_store)
        self.additional_attention_store = additional_attention_store
        self.batch_size = len(prompts)
        self.attention_blend = attention_blend
        if self.additional_attention_store is not None:
            # the attention_store is provided outside, only pass in one promp
            self.batch_size = len(prompts) //2
            assert self.batch_size==1, 'Only support single video editing with additional attention_store'

        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.latent_blend = latent_blend
        # We need to know the current position in attention
        self.prev_attention_key_name = 0
        self.use_inversion_attention = use_inversion_attention
        self.attention_position_counter_dict = {
            'down_cross': 0,
            'mid_cross': 0,
            'up_cross': 0,
            'down_self': 0,
            'mid_self': 0,
            'up_self': 0,
        }

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        # torch.Size([8, 4096, 77]), torch.Size([1, 77, 77]) -> [1, 8, 4096, 77]
        # Can be extend to temporal, use temporal as batch size
        target_device = att_replace.device
        target_dtype  = att_replace.dtype
        attn_base = attn_base.to(target_device, dtype=target_dtype)
        
        if attn_base.dim()==3:
            return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
        elif attn_base.dim()==4:
            return torch.einsum('thpw,bwn->bthpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 latent_blend: Optional[SpatialBlender] = None, tokenizer=None,
                 additional_attention_store=None,
                 use_inversion_attention = False,
                 attention_blend: SpatialBlender=None,
                 save_self_attention: bool = True,
                 disk_store=False):
        super(AttentionReplace, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, latent_blend, tokenizer=tokenizer,
            additional_attention_store=additional_attention_store, use_inversion_attention = use_inversion_attention,
            attention_blend=attention_blend,
            save_self_attention = save_self_attention,
            disk_store=disk_store
            )
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        
        target_device = att_replace.device
        target_dtype  = att_replace.dtype
        attn_base = attn_base.to(target_device, dtype=target_dtype)
        if attn_base.dim()==3:
            attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        elif attn_base.dim()==4:
            attn_base_replace = attn_base[:, :, :, self.mapper].permute(3, 0, 1, 2, 4)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 latent_blend: Optional[SpatialBlender] = None, tokenizer=None,
                 additional_attention_store=None,
                 use_inversion_attention = False,
                 attention_blend: SpatialBlender=None,
                 save_self_attention : bool=True,
                 disk_store = False
                 ):
        super(AttentionRefine, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, latent_blend, tokenizer=tokenizer,
            additional_attention_store=additional_attention_store, use_inversion_attention = use_inversion_attention,
            attention_blend=attention_blend,
            save_self_attention = save_self_attention,
            disk_store = disk_store
            )
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):
    """First replace the weight, than increase the attention at a area

    Args:
        AttentionControlEdit (_type_): _description_
    """

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                latent_blend: Optional[SpatialBlender] = None, controller: Optional[AttentionControlEdit] = None, tokenizer=None,
                additional_attention_store=None,
                use_inversion_attention = False,
                attention_blend: SpatialBlender=None,
                save_self_attention:bool = True,
                disk_store = False
                ):
        super(AttentionReweight, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, latent_blend, tokenizer=tokenizer,
            additional_attention_store=additional_attention_store,
            use_inversion_attention = use_inversion_attention,
            attention_blend=attention_blend,
            save_self_attention=save_self_attention,
            disk_store = disk_store
            )
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller

def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]], tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer



def make_controller(tokenizer, prompts: List[str], is_replace_controller: bool,
                    cross_replace_steps: Dict[str, float], self_replace_steps: float=0.0, 
                    blend_words=None, equilizer_params=None, 
                    additional_attention_store=None, use_inversion_attention = False, blend_th: float=(0.3, 0.3),
                    NUM_DDIM_STEPS=None,
                    blend_latents = False,
                    blend_self_attention=False,
                    save_path = None,
                    save_self_attention = True,
                    disk_store = False
                    ) -> AttentionControlEdit:
    if (blend_words is None) or (blend_words == 'None'):
        latent_blend = None
        attention_blend =None
    else:
        if blend_latents:
            latent_blend = SpatialBlender( prompts, blend_words, 
                                       start_blend = 0.2, end_blend=0.8,
                                       tokenizer=tokenizer, th=blend_th, NUM_DDIM_STEPS=NUM_DDIM_STEPS,
                            save_path=save_path+f'/latent_blend_mask',
                            prompt_choose='both')
            print(f'Blend latent mask with threshold {blend_th}')
        else:
            latent_blend = None
        if blend_self_attention:
            attention_blend = SpatialBlender( prompts, blend_words, 
                                                    start_blend = 0.0, end_blend=2,
                                                  tokenizer=tokenizer, th=blend_th, NUM_DDIM_STEPS=NUM_DDIM_STEPS,
                           save_path=save_path+f'/attention_blend_mask',
                           prompt_choose='source')
            print(f'Blend self attention mask with threshold {blend_th}')
        else:
            attention_blend = None
    if is_replace_controller:
        print('use replace controller')
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, 
                                      cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                      latent_blend=latent_blend, tokenizer=tokenizer,
                                      additional_attention_store=additional_attention_store,
                                      use_inversion_attention = use_inversion_attention,
                                      attention_blend=attention_blend,
                                      save_self_attention = save_self_attention,
                                      disk_store=disk_store
                                      )
    else:
        print('use refine controller')
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS,
                                     cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps,
                                     latent_blend=latent_blend, tokenizer=tokenizer,
                                     additional_attention_store=additional_attention_store,
                                     use_inversion_attention = use_inversion_attention,
                                     attention_blend=attention_blend,
                                     save_self_attention = save_self_attention,
                                     disk_store=disk_store
                                     )
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"], tokenizer=tokenizer)
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, 
                                       cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                       equalizer=eq, latent_blend=latent_blend, controller=controller, 
                                        tokenizer=tokenizer,
                                        additional_attention_store=additional_attention_store,
                                        use_inversion_attention = use_inversion_attention,
                                        attention_blend=attention_blend,
                                        save_self_attention = save_self_attention,
                                        disk_store=disk_store
                                       )
    return controller

