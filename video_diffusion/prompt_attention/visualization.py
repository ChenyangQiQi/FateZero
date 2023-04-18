from typing import List
import os
import datetime
import numpy as np
from PIL import Image

import torch

import video_diffusion.prompt_attention.ptp_utils as ptp_utils
from video_diffusion.common.image_util import save_gif_mp4_folder_type
from video_diffusion.prompt_attention.attention_store import AttentionStore


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


def show_cross_attention(tokenizer, prompts, attention_store: AttentionStore, 
                         res: int, from_where: List[str], select: int = 0, save_path = None):
    """
        attention_store (AttentionStore): 
            ["down", "mid", "up"] X ["self", "cross"]
            4,         1,    6
            head*res*text_token_len = 8*res*77
            res=1024 -> 64 -> 1024
        res (int): res
        from_where (List[str]): "up", "down'
    """
    if isinstance(prompts, str):
        prompts = [prompts,]
    tokens = tokenizer.encode(prompts[select]) 
    decoder = tokenizer.decode
    
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
