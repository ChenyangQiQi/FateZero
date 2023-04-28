"""
Code of spatial blending module for latents and self-attention
"""

from typing import List
import os
import datetime
import numpy as np
import torchvision.utils as tvu
from einops import rearrange

import torch
import torch.nn.functional as F

import video_diffusion.prompt_attention.ptp_utils as ptp_utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SpatialBlender:
    """
    Return a blending mask using the cross attention produced by both source during the inversion and target prompt during editing.
    Called in make_controller
    """
    def get_mask(self, maps, alpha, use_pool, h=None, w=None, x_t=None, step_in_store: int=None):
        """
        ([1, 40, 2, 16, 16, 77]) * ([1, 1, 1, 1, 1, 77]) -> [2, 1, 16, 16]
        mask have dimension of [clip_length, dim, res, res]
        """
        if h is None and w is None and x_t is not None:
            h, w = x_t.shape[-2:] # record the shape if only latent is provided
        k = 1
        
        if maps.dim() == 5: alpha = alpha[:, None, ...]
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = F.interpolate(maps, size=(h, w))
        mask = mask / mask.max(-2, keepdims=True)[0].max(-1, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        if self.prompt_choose == 'both':
            assert mask.shape[0] == 2, "If using both source and target prompt"
            mask = mask[:1] + mask
        if self.save_path is not None:
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            save_path = f'{self.save_path}/{self.prompt_choose}/'
            if step_in_store is not None:
                save_path += f'step_in_store_{step_in_store:04d}'
            save_path +=f'/mask_{now}_{self.count:02d}.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if mask.shape[0] == 2:
                save_mask = mask[1:]
            else:
                save_mask = mask
            tvu.save_image(rearrange(save_mask.float(), "c p h w -> p c h w"), save_path,normalize=True)
            self.count +=1
        return mask

    def __call__(self, attention_store, step_in_store: int=None, target_h=None, target_w=None, x_t=None):
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
        if target_h is None and target_w is None and x_t is not None:
            target_h, target_w = x_t.shape[-2:] # record the shape if only latent is provided
        
        self.counter += 1
        # if (self.counter > self.start_blend) and (self.counter < self.end_blend):
            
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        # breakpoint()
        assert len(attention_store["down_cross"][0].shape) in [5, 4], \
            (f"the maps in attention_store must have shape [p c h (res_h res_w) w], or [c h (res_h res_w) w] \
            not {attention_store['down_cross'][0].shape} ")
        # maps = attention_store # [2,8,1024, 77] = [frames, head, (res, res), word_embedding]
        # a list of len(5), elements has shape [16, 256, 77]
        target_device = self.alpha_layers.device
        target_dtype  = self.alpha_layers.dtype
        # maps = [rearrange(  item, "p c h (res_h res_w) w -> p h c res_h res_w w ", 
                            # h=heads, res_h=res_h, res_w=res_h).to(target_device, dtype=target_dtype)
                            # for item in maps]
        rearranged_maps = []
        for item in maps:
            if len(item.shape) == 4: item = item[None, ...]
            ( p, c, heads, r, w)= item.shape
            res_h = int(np.sqrt(r))
            assert r == res_h* res_h, "the shape of attention map must be a squire"
            rearranged_item = rearrange(  item, "p c h (res_h res_w) w -> p h c res_h res_w w ", 
                            h=heads, res_h=res_h, res_w=res_h)
            rearranged_maps.append(rearranged_item.to(target_device, dtype=target_dtype))
        maps = torch.cat(rearranged_maps, dim=1)

        if self.prompt_choose == 'source':
            # We support self-attention blending using only source prompt
            masked_alpah_layers = self.alpha_layers[0:1]
        else:
            masked_alpah_layers = self.alpha_layers
        mask = self.get_mask(maps, masked_alpah_layers, True, target_h, target_w, step_in_store=step_in_store)
    
        if self.substruct_layers is not None:
            maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
            mask = mask * maps_sub
        mask = mask.float()

        # mask is one: use generated information
        # mask is zero: use inverted information
        self.mask_list.append(mask[0][:, None, :, :].float().cpu().detach())
        if x_t is not None:
            if x_t.dim()==5: 
                mask = mask[:, None, ...]
            # x_t [2,4,2,64,64]
            if (self.counter > self.start_blend) and (self.counter < self.end_blend):
                x_t = x_t[:1] + mask * (x_t - x_t[:1])
            return x_t
        else:
            return mask
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, 
                 start_blend=0.2, end_blend=0.8,
                 th=(0.9, 0.9), tokenizer=None, NUM_DDIM_STEPS =None,
                 save_path = None, prompt_choose='source'):
        """
        Args:
            start_blend (float, optional): For latent blending, defaults to 0.2, for attention fusion better to be 0.0
            end_blend (float, optional): For latent blending, defaults to 0.8, for attention fusion better to be 1.0
        """
        self.count = 0
        self.MAX_NUM_WORDS = 77
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        if save_path is not None:
            self.save_path = save_path
            os.makedirs(self.save_path, exist_ok=True)
        else:
            self.save_path = None
        assert prompt_choose in ['source', 'both'], "choose to generate the mask by only source prompt or both the source and target"
        self.prompt_choose = prompt_choose
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, self.MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                # debug me
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        # self.alpha_layers.shape = torch.Size([2, 1, 1, 1, 1, 77]), 1 denotes the world to be replaced
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

