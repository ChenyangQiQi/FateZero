# EditingGuidance

## Prompt Engineering
For the results in the paper and webpage, we get the source prompt using the BLIP model embedded in the [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui/).

Move the middle frame into the UI and click the "interrogate CLIP". Then, we will get a source prompt automatically. Finally, we remove the last few useless words.

<img src="../docs/blip.png" height="220px"/> 

During stylization, you may use a very simple source prompt "A photo" as a baseline if your input video is too complicated to describe by one sentence.

### Validate the prompt

- Put the source prompt into the stable diffusion. If the generated image is close to our input video, it can be a good source prompt.
- A good prompt describes each frame and most objects in video. Especially, it has the object or attribute that we want to edit or preserve.
- Put the target prompt into the stable diffusion. We can check the upper bound of our editing effect. A reasonable composition of video may achieve better results(e.g., "sunflower" video with "Van Gogh" prompt is better than "sunflower" with "Monet")






## FateZero hyperparameters
We give a simple analysis of the involved hyperparaters as follows:
``` yaml
# For edited words (e.g., posche car) , whether to directly copy the cross attention from source according to the word index, although the original word is different (e.g., silver jeed)
# True: directly copy, better for object and shape editing
# False: keep source attention, better for style editing (e.g., water color style)
is_replace_controller: False

# Semantic layout preserving. Value is in [0, 1]. Higher steps, replace more cross attention to preserve semantic layout as source image
cross_replace_steps: 
    default_: 0.8

# Source background structure preserving. Value is in [0, 1]. 
# e.g., =0.8 Replace the first 80% steps self-attention
self_replace_steps: 0.8


# Equalize and amplify the target-words cross attention.
eq_params: 
    words: ["watercolor"]
    values: [10]

# Blend the self-attention and latents for better local editing
# Blending is usefull in local shape editing.
# Without following three lines, self-attention maps at all HXW spatial pixels will be replaced
blend_words: [['jeep',], ["car",]] 
blend_self_attention:  True # Attention map of spatial-temporal self attention
blend_latents: True   # Latents at each time step. False for style editing. Can be True for local shape or attribute editing.
blend_th: [0.3, 0.3] # Threshold of blending mask, where the cross attention has beed normalized to [0, 1]. 0.3 can be a good choice
# e.g., if blend_th: [2, 2], we replace full-resolution spatial-temporal self-attention maps with the source maps. Thus, the geometry of generated image can be very similar to the imput.
# if blend_th -> [0.0, 0.0], mask -> 1. We use full-resolution spatial-temporal self-attention maps obtained by denoising editing. None of them is blended with those from inversion.
```

## DDIM hyperparameters

We profile the cost of editing 8 frames on an Nvidia 3090, fp16 of accelerator, xformers.

| Configs | Attention location | DDIM Inver. Step | CPU memory         | GPU memory        | Inversion time | Editing time time | Quality
|------------------|------------------  |------------------|------------------|------------------|------------------|----| ---- |
| [basic](../config/teaser/jeep_watercolor.yaml)  | RAM | 50  | 100G    | 12G  | 60s | 40s | Full support
| [low cost](../config/low_resource_teaser/jeep_watercolor_ddim_10_steps.yaml) | RAM | 10  | 15G    | 12G  | 10s | 10s | OK for Style, not work for shape
| [lower cost](../config/low_resource_teaser/jeep_watercolor_ddim_10_steps_disk_store.yaml) | DISK | 10  | 6G    | 12G  | 33 s | 100s | OK for Style, not work for shape
