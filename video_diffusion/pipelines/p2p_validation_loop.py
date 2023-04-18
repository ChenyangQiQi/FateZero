import os
import numpy as  np
from typing import List, Union
import PIL
import copy
from einops import rearrange

import torch
import torch.utils.data
import torch.utils.checkpoint

from diffusers.pipeline_utils import DiffusionPipeline
from tqdm.auto import tqdm
from video_diffusion.common.image_util import make_grid, annotate_image
from video_diffusion.common.image_util import save_gif_mp4_folder_type


class P2pSampleLogger:
    def __init__(
        self,
        editing_prompts: List[str],
        clip_length: int,
        logdir: str,
        subdir: str = "sample",
        num_samples_per_prompt: int = 1,
        sample_seeds: List[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7,
        strength: float = None,
        annotate: bool = False,
        annotate_size: int = 15,
        use_make_grid: bool = True,
        grid_column_size: int = 2,
        prompt2prompt_edit: bool=False,
        p2p_config: dict = None,
        use_inversion_attention: bool = True,
        source_prompt: str = None,
        traverse_p2p_config: bool = False,
        **args
    ) -> None:
        self.editing_prompts = editing_prompts
        self.clip_length = clip_length
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength

        if sample_seeds is None:
            max_num_samples_per_prompt = int(1e5)
            if num_samples_per_prompt > max_num_samples_per_prompt:
                raise ValueError
            sample_seeds = torch.randint(0, max_num_samples_per_prompt, (num_samples_per_prompt,))
            sample_seeds = sorted(sample_seeds.numpy().tolist())
        self.sample_seeds = sample_seeds

        self.logdir = os.path.join(logdir, subdir)
        os.makedirs(self.logdir)

        self.annotate = annotate
        self.annotate_size = annotate_size
        self.make_grid = use_make_grid
        self.grid_column_size = grid_column_size
        self.prompt2prompt_edit = prompt2prompt_edit
        self.p2p_config = p2p_config
        self.use_inversion_attention = use_inversion_attention
        self.source_prompt = source_prompt
        self.traverse_p2p_config =traverse_p2p_config

    def log_sample_images(
        self, pipeline: DiffusionPipeline,
        device: torch.device, step: int,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        latents: torch.FloatTensor = None,
        uncond_embeddings_list: List[torch.FloatTensor] = None,
        save_dir = None,
    ):
        torch.cuda.empty_cache()
        samples_all = []
        attention_all = []
        # handle input image
        if image is not None:
            input_pil_images = pipeline.numpy_to_pil(tensor_to_numpy(image))[0]
            if self.annotate :
                samples_all.append([
                            annotate_image(image, "input sequence", font_size=self.annotate_size) for image in input_pil_images
                        ])
            else:
                samples_all.append(input_pil_images)
        for idx, prompt in enumerate(tqdm(self.editing_prompts, desc="Generating sample images")):
            if self.prompt2prompt_edit:
                if self.traverse_p2p_config:
                    p2p_config_now = copy.deepcopy(self.p2p_config[idx])
                else:
                    p2p_config_now = copy.deepcopy(self.p2p_config[idx])

                if idx == 0 and not self.use_inversion_attention:
                    edit_type = 'save'
                    p2p_config_now.update({'save_self_attention': True})
                    print('Reflash the attention map in pipeline')

                else:
                    edit_type = 'swap'
                    p2p_config_now.update({'save_self_attention': False})

                p2p_config_now.update({'use_inversion_attention': self.use_inversion_attention})
            else:
                edit_type = None

            input_prompt = prompt
            for seed in self.sample_seeds:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                sequence_return = pipeline(
                    prompt=input_prompt,
                    source_prompt = self.editing_prompts[0] if self.source_prompt is None else self.source_prompt,
                    edit_type = edit_type,
                    image=image, # torch.Size([8, 3, 512, 512])
                    strength=self.strength,
                    generator=generator,
                    num_inference_steps=self.num_inference_steps,
                    clip_length=self.clip_length,
                    guidance_scale=self.guidance_scale,
                    num_images_per_prompt=1,
                    # used in null inversion
                    latents = latents,
                    uncond_embeddings_list = uncond_embeddings_list,
                    save_path = save_dir,
                    **p2p_config_now,
                )
                if self.prompt2prompt_edit:
                    sequence = sequence_return['sdimage_output'].images[0]
                    attention_output = sequence_return['attention_output']
                    
                else:
                    sequence = sequence_return.images[0]
                torch.cuda.empty_cache()

                if self.annotate:
                    images = [
                        annotate_image(image, prompt, font_size=self.annotate_size) for image in sequence
                    ]
                else:
                    images = sequence
                if self.make_grid:
                    samples_all.append(images)
                    if self.prompt2prompt_edit:
                        if attention_output is not None:
                            attention_all.append(attention_output)

                save_path = os.path.join(self.logdir, f"step_{step}_{idx}_{seed}.gif")
                save_gif_mp4_folder_type(images, save_path)

                if self.prompt2prompt_edit:

                    if attention_output is not None:
                        save_gif_mp4_folder_type(attention_output, save_path.replace('.gif', 'atten.gif'))

        if self.make_grid:
            samples_all = [make_grid(images, cols=int(np.ceil(np.sqrt(len(samples_all))))) for images in zip(*samples_all)]
            save_path = os.path.join(self.logdir, f"step_{step}.gif")
            save_gif_mp4_folder_type(samples_all, save_path)
            if self.prompt2prompt_edit:
                if len(attention_all) > 0 :
                    attention_all = [make_grid(images, cols=1) for images in zip(*attention_all)]
                if len(attention_all) > 0:
                    save_gif_mp4_folder_type(attention_all, save_path.replace('.gif', 'atten.gif'))
        return samples_all




def tensor_to_numpy(image, b=1):
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16

    image = image.cpu().float().numpy()
    image = rearrange(image, "(b f) c h w -> b f h w c", b=b)
    return image
