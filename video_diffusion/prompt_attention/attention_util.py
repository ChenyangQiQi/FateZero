"""
Code for prompt2prompt local editing and attention visualization

"""

from typing import Optional, Union, Tuple, List, Dict
import abc
import os
import datetime
import numpy as np
from PIL import Image
import copy
import torchvision.utils as tvu
from einops import rearrange

import torch
import torch.nn.functional as F

from video_diffusion.common.util import get_time_string
import video_diffusion.prompt_attention.ptp_utils as ptp_utils
import video_diffusion.prompt_attention.seq_aligner as seq_aligner
from video_diffusion.common.image_util import save_gif_mp4_folder_type
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class LocalBlend:
    """Called in make_controller
    self.alpha_layers.shape = torch.Size([2, 1, 1, 1, 1, 77]), 1 denotes the world to be replaced
    """
    def get_mask(self, maps, alpha, use_pool, x_t, step_in_store: int=None, prompt_choose='source'):
        k = 1
        # ([2, 40, 4, 16, 16, 77]) * ([2, 1, 1, 1, 1, 77]) -> [2, 1, 16, 16]
        if maps.dim() == 5: alpha = alpha[:, None, ...]
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = F.interpolate(maps, size=(x_t.shape[-2:]))
        mask = mask / mask.max(-2, keepdims=True)[0].max(-1, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        if self.save_path is not None:
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

            save_path = f'{self.save_path}/{prompt_choose}/'
            if step_in_store is not None:
                save_path += f'step_in_store_{step_in_store:04d}'
                # f'{self.save_path}/step_in_store_{step_in_store:04d}/mask_{now}_{self.count:02d}.png'    
            save_path +=f'/mask_{now}_{self.count:02d}.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            tvu.save_image(rearrange(mask[1:].float(), "c p h w -> p c h w"), save_path,normalize=True)
            self.count +=1
        return mask
    
    def __call__(self, x_t, attention_store):
        """_summary_

        Args:
            x_t (_type_): [1,4,8,64,64] # (prompt, channel, clip_length, res, res)
            attention_store (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.counter += 1
        if (self.counter > self.start_blend) and (self.counter < self.end_blend):
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            if maps[0].dim() == 4:
                (ph, c, r, w)= maps[0].shape
                assert r == 16*16
                # a list of len(5), elements has shape [16, 256, 77]
                maps = [rearrange(item, "(p h) c (res_h res_w) w -> p h c res_h res_w w ", 
                                  p=self.alpha_layers.shape[0], res_h=16, res_w=16) for item in maps]
                maps = torch.cat(maps, dim=1)
                mask = self.get_mask(maps, self.alpha_layers, True, x_t)
                if self.substruct_layers is not None:
                    maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                    mask = mask * maps_sub
                mask = mask.float()
                # only for debug
                # mask = torch.zeros_like(mask)
                # "mask is one: use geenerated information"
                # "mask is zero: use geenerated information"
                self.mask_list.append(mask[0][:, None, :, :].float().cpu().detach())
                if x_t.dim()==5: 
                    mask = mask[:, None, ...]
                    # x_t [2,4,2,64,64]
                x_t = x_t[:1] + mask * (x_t - x_t[:1])
            else:
                (ph, r, w)= maps[0].shape
                # a list of len(5), elements has shape [16, 256, 77]

                maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, self.MAX_NUM_WORDS) for item in maps]
                maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, self.MAX_NUM_WORDS) for item in maps]
                maps = torch.cat(maps, dim=1)
                mask = self.get_mask(maps, self.alpha_layers, True, x_t)
                if self.substruct_layers is not None:
                    maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                    mask = mask * maps_sub
                mask = mask.float()
                x_t = x_t[:1] + mask * (x_t - x_t[:1])
                
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, 
                 start_blend=0.2, end_blend=0.8,
                 th=(0.9, 0.9), tokenizer=None, NUM_DDIM_STEPS =None,
                 save_path =None):
        self.count = 0
        self.MAX_NUM_WORDS = 77
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        if save_path is not None:
            self.save_path = save_path+'/latents_mask'
            os.makedirs(self.save_path, exist_ok='True')
        else:
            self.save_path = None
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, self.MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                # debug me
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, self.MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * self.NUM_DDIM_STEPS)
        self.end_blend = int(end_blend * self.NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th
        self.mask_list = []



class MaskBlend:
    """
    First, we consider only source prompt
    Called in make_controller
    self.alpha_layers.shape = torch.Size([2, 1, 1, 1, 1, 77]), 1 denotes the world to be replaced
    """
    def get_mask(self, maps, alpha, use_pool, h=None, w=None, step_in_store: int=None, prompt_choose='source'):
        """
        # ([1, 40, 2, 16, 16, 77]) * ([1, 1, 1, 1, 1, 77]) -> [2, 1, 16, 16]
        mask have dimention of [clip_length, dim, res, res]
        """
        k = 1
        
        if maps.dim() == 5: alpha = alpha[:, None, ...]
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = F.interpolate(maps, size=(h, w))
        mask = mask / mask.max(-2, keepdims=True)[0].max(-1, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        if self.save_path is not None:
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

            save_path = f'{self.save_path}/{prompt_choose}/'
            if step_in_store is not None:
                save_path += f'step_in_store_{step_in_store:04d}'
            save_path +=f'/mask_{now}_{self.count:02d}.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            tvu.save_image(rearrange(mask.float(), "c p h w -> p c h w"), save_path,normalize=True)
            self.count +=1
        return mask

    def __call__(self, target_h, target_w, attention_store, step_in_store: int=None):
        """
            input has shape  (heads) clip res words
            one meens using target self-attention, zero is using source
            Previous implementation us all zeros
            mask should be repeat.

        Args:
            x_t (_type_): [1,4,8,64,64] # (prompt, channel, clip_length, res, res)
            attention_store (_type_): _description_

        Returns:
            _type_: _description_
        """

        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        
        # maps = attention_store # [2,8,1024, 77] = [frames, head, (res, res), word_embedding]
        assert maps[0].dim() == 4, "only support temporal data"
        ( c, heads, r, w)= maps[0].shape
        res_h = int(np.sqrt(r))
        assert r == res_h* res_h
        # a list of len(5), elements has shape [16, 256, 77]
        target_device = self.alpha_layers.device
        target_dtype  = self.alpha_layers.dtype
        maps = [rearrange(item, " c h (res_h res_w) w -> h c res_h res_w w ", 
                            h=heads, res_h=res_h, res_w=res_h)[None, ...].to(target_device, dtype=target_dtype)
                for item in maps]
        
        
        maps = torch.cat(maps, dim=1)
        # We only support self-attention blending using source prompt 
        masked_alpah_layers = self.alpha_layers[0:1]
        mask = self.get_mask(maps, masked_alpah_layers, True, target_h, target_w, step_in_store=step_in_store, prompt_choose='source')
    
        if self.substruct_layers is not None:
            maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
            mask = mask * maps_sub
        mask = mask.float()

        # "mask is one: use geenerated information"
        # "mask is zero: use geenerated information"
        self.mask_list.append(mask[0][:, None, :, :].float().cpu().detach())

        return mask
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, 
                 start_blend=0.2, end_blend=0.8,
                 th=(0.9, 0.9), tokenizer=None, NUM_DDIM_STEPS =None,
                 save_path = None):
        self.count = 0
        # self.config_dict = copy.deepcopy(config_dict)
        self.MAX_NUM_WORDS = 77
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        if save_path is not None:
            self.save_path = save_path+'/blend_mask'
            os.makedirs(self.save_path, exist_ok='True')
        else:
            self.save_path = None
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, self.MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                # debug me
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, self.MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        
        self.alpha_layers = alpha_layers.to(device)
        print('the index mask of edited word in the prompt')
        print(self.alpha_layers[0][..., 0:(len(prompts[0].split(" "))+2)])
        print(self.alpha_layers[1][..., 0:(len(prompts[1].split(" "))+2)])
        
        self.start_blend = int(start_blend * self.NUM_DDIM_STEPS)
        self.end_blend = int(end_blend * self.NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th
        self.mask_list = []


        
        
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        self.cur_att_layer = 0
        self.cur_step += 1
        self.between_steps()
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        """I guess the diffusion of google has some unconditional attention layer
        No unconditional attention layer in Stable diffusion

        Returns:
            _type_: _description_
        """
        # return self.num_att_layers if config_dict['LOW_RESOURCE'] else 0
        return 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.LOW_RESOURCE:
                # For inversion without null text file 
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                # For classifier-free guidance scale!=1
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1

        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, 
                 ):
        self.LOW_RESOURCE = False # assume the edit have cfg
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float, NUM_DDIM_STEPS=None):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
    

class AttentionStore(AttentionControl):
    def step_callback(self, x_t):


        x_t = super().step_callback(x_t)
        self.latents_store.append(x_t.cpu().detach())
        return x_t
    
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    @staticmethod
    def get_empty_cross_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[-2] <= 32 ** 2:  # avoid memory overhead
            # print(f"Store attention map {key} of shape {attn.shape}")
            if is_cross or self.save_self_attention:
                if attn.shape[-2] == 32**2:
                    append_tensor = attn.cpu().detach()
                else:
                    append_tensor = attn
                self.step_store[key].append(copy.deepcopy(append_tensor))
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        
        if self.disk_store:
            path = self.store_dir + f'/{self.cur_step:03d}.pt'
            torch.save(copy.deepcopy(self.step_store), path)
            self.attention_store_all_step.append(path)
        else:
            self.attention_store_all_step.append(copy.deepcopy(self.step_store))
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        "divide the attention map value in attention store by denoising steps"
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store_all_step = []
        self.attention_store = {}

    def __init__(self, save_self_attention:bool=True, disk_store=False):
        super(AttentionStore, self).__init__()
        self.disk_store = disk_store
        if self.disk_store:
            time_string = get_time_string()
            path = f'./trash/attention_cache_{time_string}'
            os.makedirs(path, exist_ok=True)
            self.store_dir = path
        else:
            self.store_dir =None
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.save_self_attention = save_self_attention
        self.latents_store = []
        self.attention_store_all_step = []


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
        if self.local_blend is not None:
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
            
            attention_store_step = self.additional_attention_store.attention_store_all_step[step_in_store]
            if isinstance(place_in_unet_cross_atten_list, str): attention_store_step = torch.load(attention_store_step)
            
            for key in blend_dict.keys():
                place_in_unet_cross_atten_list = attention_store_step[key]
                for i, attention in enumerate(place_in_unet_cross_atten_list):

                    concate_attention = torch.cat([attention[None, ...], self.attention_store[key][i][None, ...]], dim=0)
                    blend_dict[key].append(copy.deepcopy(rearrange(concate_attention, ' p c h res words -> (p h) c res words')))
            x_t = self.local_blend(copy.deepcopy(torch.cat([inverted_latents, x_t], dim=0)), copy.deepcopy(blend_dict))
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
                
            place_in_unet_cross_atten_list = self.additional_attention_store.attention_store_all_step[step_in_store]
            if isinstance(place_in_unet_cross_atten_list, str): place_in_unet_cross_atten_list = torch.load(place_in_unet_cross_atten_list)
            # breakpoint()
            # Note that attn is append to step_store, 
            # if attn is get through clean -> noisy, we should inverse it
            attn_base = place_in_unet_cross_atten_list[key][current_pos]          
            
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
                    if self.MB is not None and attn_repalce.shape[-2] <= 32 ** 2:
                        # ca_this_step = place_in_unet_cross_atten_list
                        # query 1024, key 2048
                        h = int(np.sqrt(attn_repalce.shape[-2]))
                        w = h
                        mask = self.MB(target_h = h, target_w =w, attention_store= place_in_unet_cross_atten_list, step_in_store=step_in_store)
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
                 local_blend: Optional[LocalBlend], tokenizer=None, 
                 additional_attention_store: AttentionStore =None,
                 use_inversion_attention: bool=False,
                 MB: MaskBlend= None,
                 save_self_attention: bool=True,
                 disk_store=False
                 ):
        super(AttentionControlEdit, self).__init__(
            save_self_attention=save_self_attention,
            disk_store=disk_store)
        self.additional_attention_store = additional_attention_store
        self.batch_size = len(prompts)
        self.MB = MB
        if self.additional_attention_store is not None:
            # the attention_store is provided outside, only pass in one promp
            self.batch_size = len(prompts) //2
            assert self.batch_size==1, 'Only support single video editing with additional attention_store'

        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
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
                 local_blend: Optional[LocalBlend] = None, tokenizer=None,
                 additional_attention_store=None,
                 use_inversion_attention = False,
                 MB: MaskBlend=None,
                 save_self_attention: bool = True,
                 disk_store=False):
        super(AttentionReplace, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer=tokenizer,
            additional_attention_store=additional_attention_store, use_inversion_attention = use_inversion_attention,
            MB=MB,
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
                 local_blend: Optional[LocalBlend] = None, tokenizer=None,
                 additional_attention_store=None,
                 use_inversion_attention = False,
                 MB: MaskBlend=None,
                 save_self_attention : bool=True,
                 disk_store = False
                 ):
        super(AttentionRefine, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer=tokenizer,
            additional_attention_store=additional_attention_store, use_inversion_attention = use_inversion_attention,
            MB=MB,
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
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None, tokenizer=None,
                additional_attention_store=None,
                use_inversion_attention = False,
                MB: MaskBlend=None,
                save_self_attention:bool = True,
                disk_store = False
                ):
        super(AttentionReweight, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer=tokenizer,
            additional_attention_store=additional_attention_store,
            use_inversion_attention = use_inversion_attention,
            MB=MB,
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

def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.dim() == 3:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
            elif item.dim() == 4:
                t, h, res_sq, token = item.shape
                if item.shape[2] == num_pixels:
                    cross_maps = item.reshape(len(prompts), t, -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
                    
    out = torch.cat(out, dim=-4)
    out = out.sum(-4) / out.shape[-4]
    return out.cpu()


def make_controller(tokenizer, prompts: List[str], is_replace_controller: bool,
                    cross_replace_steps: Dict[str, float], self_replace_steps: float=0.0, 
                    blend_words=None, equilizer_params=None, 
                    additional_attention_store=None, use_inversion_attention = False, bend_th: float=(0.3, 0.3),
                    NUM_DDIM_STEPS=None,
                    masked_latents = False,
                    masked_self_attention=False,
                    save_path = None,
                    save_self_attention = True,
                    disk_store = False
                    ) -> AttentionControlEdit:
    if (blend_words is None) or (blend_words == 'None'):
        lb = None
        MB =None
    else:
        if masked_latents:
            lb = LocalBlend( prompts, blend_words, tokenizer=tokenizer, th=bend_th, NUM_DDIM_STEPS=NUM_DDIM_STEPS,
                            save_path=save_path)
        else:
            lb = None
        if masked_self_attention:
            MB = MaskBlend( prompts, blend_words, tokenizer=tokenizer, th=bend_th, NUM_DDIM_STEPS=NUM_DDIM_STEPS,
                           save_path=save_path)
            print(f'Control self attention mask with threshold {bend_th}')   
        else:
            MB = None
    if is_replace_controller:
        print('use replace controller')
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, 
                                      cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                      local_blend=lb, tokenizer=tokenizer,
                                      additional_attention_store=additional_attention_store,
                                      use_inversion_attention = use_inversion_attention,
                                      MB=MB,
                                      save_self_attention = save_self_attention,
                                      disk_store=disk_store
                                      )
    else:
        print('use refine controller')
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS,
                                     cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps,
                                     local_blend=lb, tokenizer=tokenizer,
                                     additional_attention_store=additional_attention_store,
                                     use_inversion_attention = use_inversion_attention,
                                     MB=MB,
                                     save_self_attention = save_self_attention,
                                     disk_store=disk_store
                                     )
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"], tokenizer=tokenizer)
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, 
                                       cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                       equalizer=eq, local_blend=lb, controller=controller, 
                                        tokenizer=tokenizer,
                                        additional_attention_store=additional_attention_store,
                                        use_inversion_attention = use_inversion_attention,
                                        MB=MB,
                                        save_self_attention = save_self_attention,
                                        disk_store=disk_store
                                       )
    return controller


def show_cross_attention(tokenizer, prompts, attention_store: AttentionStore, 
                         res: int, from_where: List[str], select: int = 0, save_path = None):
    """_summary_

        tokenizer (_type_): _description_
        prompts (_type_): _description_
        attention_store (AttentionStore): _description_
            ["down", "mid", "up"] X ["self", "cross"]
            4,         1,    6
            head*res*text_token_len = 8*res*77
            res=1024 -> 64 -> 1024
        res (int): res
        from_where (List[str]): "up", "down'
        select (int, optional): _description_. Defaults to 0.
    """
    if isinstance(prompts, str):
        prompts = [prompts,]
    tokens = tokenizer.encode(prompts[select]) # list of length 9, [0-49 K]
    decoder = tokenizer.decode
    # 16, 16, 7, 7
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    os.makedirs('trash', exist_ok=True)
    attention_list = []
    if attention_maps.dim()==3: attention_maps=attention_maps[None, ...]
    for j in range(attention_maps.shape[0]):
        images = []
        for i in range(len(tokens)):
            image = attention_maps[j, :, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)
        ptp_utils.view_images(np.stack(images, axis=0), save_path=save_path)
        atten_j = np.concatenate(images, axis=1)
        attention_list.append(atten_j)
    if save_path is not None:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        video_save_path = f'{save_path}/{now}.gif'
        save_gif_mp4_folder_type(attention_list, video_save_path)
    return attention_list
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))


def register_attention_control(model, controller):
    "Connect a model with a controller"
    def ca_forward(self, place_in_unet, attention_type='cross'):
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

            # KEY FUNCTION:
            # Record and edit the attention probs
            attention_probs_th = reshape_batch_dim_to_temporal_heads(attention_probs)
            attention_probs = controller(reshape_batch_dim_to_temporal_heads(attention_probs), 
                                         is_cross, place_in_unet)
            attention_probs = reshape_temporal_heads_to_batch_dim(attention_probs_th)
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


        def scforward(
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
            query = self.reshape_heads_to_batch_dim(query)

            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            if clip_length is not None:
                key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)
                value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)


                #  ***********************Start of SparseCausalAttention_index**********
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


                #  ***********************End of SparseCausalAttention_index**********
                key = rearrange(key, "b f d c -> (b f) d c", f=clip_length)
                value = rearrange(value, "b f d c -> (b f) d c", f=clip_length)
            
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            if self._use_memory_efficient_attention_xformers and query.shape[-2] > 32 ** 2:
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
            return scforward

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
            net_[1].forward = ca_forward(net_[1], place_in_unet, attention_type = net_[1].__class__.__name__)
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
