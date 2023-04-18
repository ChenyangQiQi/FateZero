"""
Code of spatial blending module for latents and self-attention
TODO FIXME: Merge the LatentBlend and AttentionBlend class in this file
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



class LatentBlend:
    """
    We consider the cross attention produced by both source during the inversion and target prompt during editing.
    Called in make_controller
    """
    def get_mask(self, maps, alpha, use_pool, x_t, step_in_store: int=None, prompt_choose='source'):
        """
        ([2, 40, 4, 16, 16, 77]) * ([2, 1, 1, 1, 1, 77]) -> [2, 1, 16, 16]
        mask have dimension of [clip_length, dim, res, res]
        """
        k = 1
        
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
                                  p=self.alpha_layers.shape[0], res_h=16, res_w=16) 
                        for item in maps]
                maps = torch.cat(maps, dim=1)
                mask = self.get_mask(maps, self.alpha_layers, True, x_t)
                if self.substruct_layers is not None:
                    maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                    mask = mask * maps_sub
                mask = mask.float()

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
        self.start_blend = int(start_blend * self.NUM_DDIM_STEPS)
        self.end_blend = int(end_blend * self.NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th
        self.mask_list = []



class AttentionBlend:
    """
    Currently, we only consider the cross attention produced by source prompt during the inversion.
    Called in make_controller
    
    """
    def get_mask(self, maps, alpha, use_pool, h=None, w=None, step_in_store: int=None, prompt_choose='source'):
        """
        ([1, 40, 2, 16, 16, 77]) * ([1, 1, 1, 1, 1, 77]) -> [2, 1, 16, 16]
        mask have dimension of [clip_length, dim, res, res]
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
            self.save_path = save_path+'/attention_blend_mask'
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

