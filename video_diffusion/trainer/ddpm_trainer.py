from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ..models.unet_3d_condition import UNetPseudo3DConditionModel
from video_diffusion.pipelines.stable_diffusion import SpatioTemporalStableDiffusionPipeline

class DDPMTrainer(SpatioTemporalStableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNetPseudo3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        **kwargs
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
        )
        for name, module in kwargs.items():
            setattr(self, name, module)

    def step(self, 
             batch: dict = dict()):
        if 'class_images' in batch:
            self.step2d(batch["class_images"], batch["class_prompt_ids"])
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.train()
        if self.prior_preservation is not None:
            print('Use prior_preservation loss')
            self.unet2d.eval()

        # Convert images to latent space
        images = batch["images"].to(dtype=self.weight_dtype)
        b = images.shape[0]
        images = rearrange(images, "b c f h w -> (b f) c h w")
        latents = self.vae.encode(images).latent_dist.sample() # shape=torch.Size([8, 3, 512, 512]), min=-1.00, max=0.98, var=0.21, -0.96875
        latents = rearrange(latents, "(b f) c h w -> b c f h w", b=b)
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["prompt_ids"])[0]

        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if self.prior_preservation is not None:
            model_pred_2d = self.unet2d(noisy_latents[:, :, 0], timesteps, encoder_hidden_states).sample
            loss = (
                loss
                + F.mse_loss(model_pred[:, :, 0].float(), model_pred_2d.float(), reduction="mean")
                * self.prior_preservation
            )

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.unet.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        return loss
    
    def step2d(self, class_images, prompt_ids
             ):
        
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.train()
        if self.prior_preservation is not None:
            self.unet2d.eval()


        # Convert images to latent space
        images = class_images.to(dtype=self.weight_dtype)
        b = images.shape[0]
        images = rearrange(images, "b c f h w -> (b f) c h w")
        latents = self.vae.encode(images).latent_dist.sample() # shape=torch.Size([8, 3, 512, 512]), min=-1.00, max=0.98, var=0.21, -0.96875
        
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(prompt_ids)[0]

        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if self.prior_preservation is not None:
            model_pred_2d = self.unet2d(noisy_latents[:, :, 0], timesteps, encoder_hidden_states).sample
            loss = (
                loss
                + F.mse_loss(model_pred[:, :, 0].float(), model_pred_2d.float(), reduction="mean")
                * self.prior_preservation
            )

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.unet.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        return loss