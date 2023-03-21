## FateZero: Fusing Attentions for Zero-shot Text-based Video Editing

[Chenyang Qi](https://chenyangqiqi.github.io/), [Xiaodong Cun](http://vinthony.github.io/), [Yong Zhang](https://yzhang2016.github.io), [Chenyang Lei](https://chenyanglei.github.io/), [Xintao Wang](https://xinntao.github.io/), [Ying Shan](https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ), and [Qifeng Chen](https://cqf.io)

[Paper](https://arxiv.org/abs/2303.09535) | [Project Page](https://fate-zero-edit.github.io/) | [Code](https://github.com/ChenyangQiQi/FateZero)

<!-- ![fatezero_demo](./docs/teaser.png) -->

<table class="center">
  <td><img src="docs/gif_results/17_car_posche_01_concat_result.gif"></td>
  <td><img src="docs/gif_results/3_sunflower_vangogh_conat_result.gif"></td>
  <tr>
  <td width=25% style="text-align:center;">"Cat ➜ Posche Car*"</td>
  <td width=25% style="text-align:center;">"+ Van Gogh Style"</td>
  <!-- <td width=25% style="text-align:center;">"Wonder Woman, wearing a cowboy hat, is skiing"</td>
  <td width=25% style="text-align:center;">"A man, wearing pink clothes, is skiing at sunset"</td> -->
</tr>
</table >

## Abstract
TL;DR: Using FateZero, Edits your video via pretrained Diffusion models without training.

<details><summary>CLICK for full abstract</summary>


> The diffusion-based generative models have achieved
remarkable success in text-based image generation. However,
since it contains enormous randomness in generation
progress, it is still challenging to apply such models for
real-world visual content editing, especially in videos. In
this paper, we propose FateZero, a zero-shot text-based editing method on real-world videos without per-prompt
training or use-specific mask. To edit videos consistently,
we propose several techniques based on the pre-trained
models. Firstly, in contrast to the straightforward DDIM
inversion technique, our approach captures intermediate
attention maps during inversion, which effectively retain
both structural and motion information. These maps are
directly fused in the editing process rather than generated
during denoising. To further minimize semantic leakage of
the source video, we then fuse self-attentions with a blending
mask obtained by cross-attention features from the source
prompt. Furthermore, we have implemented a reform of the
self-attention mechanism in denoising UNet by introducing
spatial-temporal attention to ensure frame consistency. Yet
succinct, our method is the first one to show the ability of
zero-shot text-driven video style and local attribute editing
from the trained text-to-image model. We also have a better
zero-shot shape-aware editing ability based on the text-tovideo
model. Extensive experiments demonstrate our
superior temporal consistency and editing capability than
previous works.
</details>

## Changelog

- 2023.03.21 `Update the codebase and configuration`. Now, it can be run on the lower resources computers(16G GPU and 16G CPU RAM) with new configuration in `config/low_resource_teaser`. We also add an option to store all the attentions in hard disk, which require less ram than the original configuration.
- 2023.03.17 Release Code and Paper!

## Todo

- [x] Release the edit config for teaser
- [x] Memory and runtime profiling
- [x] Hands-on guidance of hyperparameters tuning
- [ ] Colab and hugging-face
- [ ] Tune-a-video optimization
- [ ] Release configs for other result and in-the-wild dataset
- [ ] Release more application

## Setup Environment
Our method is tested using cuda11, fp16 of accelerator and xformers on a single A100 or 3090.

```bash
conda create -n fatezero38 python=3.8
conda activate fatezero38

pip install -r requirements.txt
```

`xformers` is recommended for A100 GPU to save memory and running time. 

<details><summary>Click for xformers installation </summary>

We find its installation not stable. You may try the following wheel:
```bash
wget https://github.com/ShivamShrirao/xformers-wheels/releases/download/4c06c79/xformers-0.0.15.dev0+4c06c79.d20221201-cp38-cp38-linux_x86_64.whl
pip install xformers-0.0.15.dev0+4c06c79.d20221201-cp38-cp38-linux_x86_64.whl
```

</details>

Validate the installation by 
```
python test_install.py
```

Our environment is similar to Tune-A-video ([official](https://github.com/showlab/Tune-A-Video), [unofficial](https://github.com/bryandlee/Tune-A-Video))  and [prompt-to-prompt](https://github.com/google/prompt-to-prompt/). You may check them for more details.


## FateZero Editing

Download the [stable diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) (or other interesting image diffusion model) and put it to `./ckpt/stable-diffusion-v1-4`. 

<details><summary>Click for bash command: </summary>
 
```
mkdir ./ckpt
# download from huggingface face, takes 20G space
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
cd ./ckpt
ln -s ../stable-diffusion-v1-4 .
```
</details>

We also provide a `Tune-A-Video` [checkpoint](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cqiaa_connect_ust_hk/EviSTWoAOs1EmHtqZruq50kBZu1E8gxDknCPigSvsS96uQ?e=492khj). You may download the it and move it to `./ckpt/jeep_tuned_200/`.
<!-- We provide the [Tune-a-Video](https://drive.google.com/file/d/166eNbabM6TeJVy7hxol2gL1kUGKHi3Do/view?usp=share_link), you could download the data, unzip and put it to `data`. : -->

<details><summary>Click for directory structure </summary>

The directory structure should like this:

```
ckpt
├── stable-diffusion-v1-4
├── jeep_tuned_200
...
data
├── car-turn
│   ├── 00000000.png
│   ├── 00000001.png
│   ├── ...
video_diffusion
```

</details>

You could reproduce style and shape editing result in our teaser by running:

```bash
accelerate launch test_fatezero.py --config config/teaser/jeep_watercolor.yaml
accelerate launch test_fatezero.py --config config/teaser/jeep_posche.yaml
```
Editing 8 frames on an Nvidia 3090, use 100G CPU memory, 12G GPU memory, 60 seconds inversion/input video + 40 seconds editing/prompt.
<!-- <details><summary>Click for fast and low-resource edting </summary> -->

For fast style and other easy editings, you could using 10 DDIM steps

```bash
accelerate launch test_fatezero.py --config config/low_resource_teaser/jeep_watercolor_ddim_10_steps.yaml
```
On an Nvidia 3090 GPU, the above setting only takes 10G GPU memory, 15G CPU memory, 10 seconds inversion per input video + 10 seconds editing per target prompt.

If your CPU memory is less than 16G, you may try to save the attention on the disk by the following commands

```bash
accelerate launch test_fatezero.py --config config/low_resource_teaser/jeep_watercolor_ddim_10_steps_disk_store.yaml
```
The running time depends on the machine. Our 3090 server use 33 seconds invertion + 100 seconds editing.



<!-- </details> -->

<details><summary>Click for result structure </summary>

The result is saved as follows:
```

result
├── teaser
│   ├── jeep_posche
│   ├── jeep_watercolor
│           ├── cross-attention
│           ├── sample
│           ├── train_samples

```
where `cross-attention` is the visualization of cross-attention during inversion;
sample is the result videos obtained from target prompt;
train_sample is the input video;

</details>

## Tuning guidance to edit your video
We provided a tuning guidance to edit in-the-wild video at [here](config/TuningGuidance.md). The work is still in progress. Welcome to give your feedback in issues.

## Style Editing Results with Stable Diffusion
We show the difference of source prompt and target prompt in the box below each video.

Note mp4 and gif files in this github page are compressed. 
Please check our [Project Page](https://fate-zero-edit.github.io/) for mp4 files of original video editing results.
<table class="center">

<tr>
  <td><img src="docs/gif_results/style/1_surf_ukiyo_01_concat_result.gif"></td>
  <td><img src="docs/gif_results/style/2_car_watercolor_01_concat_result.gif"></td>
    <td><img src="docs/gif_results/style/6_lily_monet_01_concat_result.gif"></td>
  <!-- <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-skiing/wonder-woman.gif"></td>              
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-skiing/pink-sunset.gif"></td> -->
</tr>
<tr>
  <td width=25% style="text-align:center;">"+ Ukiyo-e Style"</td>
  <td width=25% style="text-align:center;">"+ Watercolor Painting"</td>
  <td width=25% style="text-align:center;">"+ Monet Style"</td>
</tr>

<tr>
  <td><img src="docs/gif_results/style/4_rabit_pokemon_01_concat_result.gif"></td>
  <td><img src="docs/gif_results/style/5_train_shikai_01_concat_result.gif"></td>
  <td><img src="docs/gif_results/style/7_swan_carton_01_concat_result.gif"></td>

</tr>
<tr>

</tr>
<tr>
  <td width=25% style="text-align:center;">"+ Pokémon Cartoon Style"</td>
  <td width=25% style="text-align:center;">"+ Makoto Shinkai Style"</td>
  <td width=25% style="text-align:center;">"+ Cartoon Style"</td>
</tr>
</table>

## Attribute Editing Results with Stable Diffusion
<table class="center">

<tr>

  <td><img src="docs/gif_results/attri/16_sq_eat_04_concat_result.gif"></td>
  <td><img src="docs/gif_results/attri/16_sq_eat_02_concat_result.gif"></td>
  <td><img src="docs/gif_results/attri/16_sq_eat_03_concat_result.gif"></td>

</tr>
<tr>
  <td width=25% style="text-align:center;">"Squirrel ➜ robot squirrel"</td>
  <td width=25% style="text-align:center;">"Squirrel, Carrot ➜ Rabbit, Eggplant"</td>
  <td width=25% style="text-align:center;">"Squirrel, Carrot ➜ Robot mouse, Screwdriver"</td>

</tr>

<tr>

  <td><img src="docs/gif_results/attri/13_bear_tiger_leopard_lion_01_concat_result.gif"></td>
  <td><img src="docs/gif_results/attri/13_bear_tiger_leopard_lion_02_concat_result.gif"></td>
  <td><img src="docs/gif_results/attri/13_bear_tiger_leopard_lion_03_concat_result.gif"></td>

</tr>
<tr>
  <td width=25% style="text-align:center;">"Bear ➜ A Red Tiger"</td>
  <td width=25% style="text-align:center;">"Bear ➜ A yellow leopard"</td>
  <td width=25% style="text-align:center;">"Bear ➜ A yellow lion"</td>

</tr>
<tr>

  <td><img src="docs/gif_results/attri/14_cat_grass_tiger_corgin_02_concat_result.gif"></td>
  <td><img src="docs/gif_results/attri/14_cat_grass_tiger_corgin_03_concat_result.gif"></td>
  <td><img src="docs/gif_results/attri/14_cat_grass_tiger_corgin_04_concat_result.gif"></td>

</tr>
<tr>
  <td width=25% style="text-align:center;">"Cat ➜ Black Cat, Grass..."</td>
  <td width=25% style="text-align:center;">"Cat ➜ Red Tiger"</td>
  <td width=25% style="text-align:center;">"Cat ➜ Shiba-Inu"</td>

</tr>


</table>

## Shape and large motion editing with Tune-A-Video
<table class="center">

<tr>
  <td><img src="docs/gif_results/shape/17_car_posche_01_concat_result.gif"></td>
  <td><img src="docs/gif_results/shape/18_swan_01_concat_result.gif"></td>
    <td><img src="docs/gif_results/shape/18_swan_02_concat_result.gif"></td>
  <!-- <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-skiing/wonder-woman.gif"></td>              
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-skiing/pink-sunset.gif"></td> -->
</tr>
<tr>
  <td width=25% style="text-align:center;">"Cat ➜ Posche Car"</td>
  <td width=25% style="text-align:center;">"Swan ➜ White Duck"</td>
  <td width=25% style="text-align:center;">"Swan ➜ Pink flamingo"</td>
</tr>

<tr>
  <td><img src="docs/gif_results/shape/19_man_wonder_01_concat_result.gif"></td>
  <td><img src="docs/gif_results/shape/19_man_wonder_02_concat_result.gif"></td>
  <td><img src="docs/gif_results/shape/19_man_wonder_03_concat_result.gif"></td>

</tr>
<tr>

</tr>
<tr>
  <td width=25% style="text-align:center;">"A man ➜ A Batman"</td>
  <td width=25% style="text-align:center;">"A man ➜ A Wonder Woman, With cowboy hat"</td>
  <td width=25% style="text-align:center;">"A man ➜ A Spider-Man"</td>
</tr>
</table>


## Demo Video

https://user-images.githubusercontent.com/45789244/225698509-79c14793-3153-4bba-9d6e-ede7d811d7f8.mp4

The video here is compressed due to the size limit of github.
The original full resolution video is [here](https://hkustconnect-my.sharepoint.com/:v:/g/personal/cqiaa_connect_ust_hk/EXKDI_nahEhKtiYPvvyU9SkBDTG2W4G1AZ_vkC7ekh3ENw?e=Xhgtmk).

## Citation 

```
@misc{qi2023fatezero,
      title={FateZero: Fusing Attentions for Zero-shot Text-based Video Editing}, 
      author={Chenyang Qi and Xiaodong Cun and Yong Zhang and Chenyang Lei and Xintao Wang and Ying Shan and Qifeng Chen},
      year={2023},
      eprint={2303.09535},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` 


## Acknowledgements

This repository borrows heavily from [Tune-A-Video](https://github.com/showlab/Tune-A-Video) and [prompt-to-prompt](https://github.com/google/prompt-to-prompt/). thanks the authors for sharing their code and models.

## Maintenance

This is the codebase for our research work. We are still working hard to update this repo and more details are coming in days. If you have any questions or ideas to discuss, feel free to contact [Chenyang Qi](cqiaa@connect.ust.hk) or [Xiaodong Cun](vinthony@gmail.com).

