import os

import numpy as np
from PIL import Image
from einops import rearrange
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .transform import short_size_scale, random_crop, center_crop, offset_crop
from ..common.image_util import IMAGE_EXTENSION


class ImageSequenceDataset(Dataset):
    def __init__(
        self,
        path: str,
        prompt_ids: torch.Tensor,
        prompt: str,
        start_sample_frame: int=0,
        n_sample_frame: int = 8,
        sampling_rate: int = 1,
        stride: int = -1, # only used during tuning to sample a long video
        image_mode: str = "RGB",
        image_size: int = 512,
        crop: str = "center",
                
        class_data_root: str = None,
        class_prompt_ids: torch.Tensor = None,
        
        offset: dict = {
            "left": 0,
            "right": 0,
            "top": 0,
            "bottom": 0
        },
        **args
        
    ):
        self.path = path
        self.images = self.get_image_list(path)
        self.n_images = len(self.images)
        self.offset = offset
        self.start_sample_frame = start_sample_frame
        if n_sample_frame < 0:
            n_sample_frame = len(self.images)        
        self.n_sample_frame = n_sample_frame
        # local sampling rate from the video
        self.sampling_rate = sampling_rate

        self.sequence_length = (n_sample_frame - 1) * sampling_rate + 1
        if self.n_images < self.sequence_length:
            raise ValueError(f"self.n_images  {self.n_images } < self.sequence_length {self.sequence_length}: Required number of frames {self.sequence_length} larger than total frames in the dataset {self.n_images }")
        
        # During tuning if video is too long, we sample the long video every self.stride globally
        self.stride = stride if stride > 0 else (self.n_images+1)
        self.video_len = (self.n_images - self.sequence_length) // self.stride + 1

        self.image_mode = image_mode
        self.image_size = image_size
        crop_methods = {
            "center": center_crop,
            "random": random_crop,
        }
        if crop not in crop_methods:
            raise ValueError
        self.crop = crop_methods[crop]

        self.prompt = prompt
        self.prompt_ids = prompt_ids
        # Negative prompt for regularization to avoid overfitting during one-shot tuning
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_images_path = sorted(list(self.class_data_root.iterdir()))
            self.num_class_images = len(self.class_images_path)
            self.class_prompt_ids = class_prompt_ids
        
        
    def __len__(self):
        max_len = (self.n_images - self.sequence_length) // self.stride + 1
        
        if hasattr(self, 'num_class_images'):
            max_len = max(max_len, self.num_class_images)
        
        return max_len

    def __getitem__(self, index):
        return_batch = {}
        frame_indices = self.get_frame_indices(index%self.video_len)
        frames = [self.load_frame(i) for i in frame_indices]
        frames = self.transform(frames)

        return_batch.update(
            {
            "images": frames,
            "prompt_ids": self.prompt_ids,
            }
        )

        if hasattr(self, 'class_data_root'):
            class_index = index % (self.num_class_images - self.n_sample_frame)
            class_indices = self.get_class_indices(class_index)           
            frames = [self.load_class_frame(i) for i in class_indices]
            return_batch["class_images"] = self.tensorize_frames(frames)
            return_batch["class_prompt_ids"] = self.class_prompt_ids
        return return_batch
    
    def transform(self, frames):
        frames = self.tensorize_frames(frames)
        frames = offset_crop(frames, **self.offset)
        frames = short_size_scale(frames, size=self.image_size)
        frames = self.crop(frames, height=self.image_size, width=self.image_size)
        return frames

    @staticmethod
    def tensorize_frames(frames):
        frames = rearrange(np.stack(frames), "f h w c -> c f h w")
        return torch.from_numpy(frames).div(255) * 2 - 1

    def load_frame(self, index):
        image_path = os.path.join(self.path, self.images[index])
        return Image.open(image_path).convert(self.image_mode)

    def load_class_frame(self, index):
        image_path = self.class_images_path[index]
        return Image.open(image_path).convert(self.image_mode)

    def get_frame_indices(self, index):
        if self.start_sample_frame is not None:
            frame_start = self.start_sample_frame + self.stride * index
        else:
            frame_start = self.stride * index
        return (frame_start + i * self.sampling_rate for i in range(self.n_sample_frame))

    def get_class_indices(self, index):
        frame_start = index
        return (frame_start + i  for i in range(self.n_sample_frame))

    @staticmethod
    def get_image_list(path):
        images = []
        for file in sorted(os.listdir(path)):
            if file.endswith(IMAGE_EXTENSION):
                images.append(file)
        return images
