## FateZero: Fusing Attentions for Zero-shot Text-based Video Editing

[Chenyang Qi](https://chenyangqiqi.github.io/), [Xiaodong Cun](http://vinthony.github.io/), [Yong Zhang](https://yzhang2016.github.io), [Chenyang Lei](https://chenyanglei.github.io/), [Xintao Wang](https://xinntao.github.io/), [Ying Shan](https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ), and [Qifeng Chen](https://cqf.io)

[Paper]() | [Project Page](https://fate-zero-edit.github.io/) | [Code](https://github.com/ChenyangQiQi/FateZero)

![Teaser](./docs/teaser.png)


## Abstract

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

## Changelog

- 2023.03.17 Release Code and Paper!

## Todo

- [x] Release the edit config for teaser
- [ ] Memory and runtime profiling and tune-a-video optimization
- [ ] More detailed description of our environment and More Hands-on guidance
- [ ] Release configs for other result and in-the-wild dataset
- [ ] Release more application

## Setup Environment
Our method is tested using cuda11, fp16 and xformers on a single A100 or 3090.

```bash
conda create -n fatezero38 python=3.8
conda activate fatezero38

pip install -r requirements.txt
```
`xformers` is recommended for A100 GPU to save memory and running time. We find its installation not stable. You may try the following wheel:
```bash
wget https://github.com/ShivamShrirao/xformers-wheels/releases/download/4c06c79/xformers-0.0.15.dev0+4c06c79.d20221201-cp38-cp38-linux_x86_64.whl
pip install xformers-0.0.15.dev0+4c06c79.d20221201-cp38-cp38-linux_x86_64.whl
```
Validate the installation by 
```
python test_install.py
```

Our environment is similar to Tune-A-video ([official](https://github.com/showlab/Tune-A-Video), [unofficial](https://github.com/bryandlee/Tune-A-Video))  and [prompt-to-prompt](https://github.com/google/prompt-to-prompt/). You may check them for more details.

## FateZero Editing

Download the [stable diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) (or other interesting image diffusion model) and put it to `./ckpt/stable-diffusion-v1-4`. You may refer to the following bash command:
```
mkdir ./ckpt
# download from huggingface face, takes 20G space
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
cd ./ckpt
ln -s ../stable-diffusion-v1-4 .
```
<!-- We provide the [Tune-a-Video](https://drive.google.com/file/d/166eNbabM6TeJVy7hxol2gL1kUGKHi3Do/view?usp=share_link), you could download the data, unzip and put it to `data`. : -->
The directory structure should like this:

```
ckpt
├── stable-diffusion-v1-4
├── stable-diffusion-v1-5
...
data
├── car-turn
│   ├── 00000000.png
│   ├── 00000001.png
│   ├── ...
video_diffusion
```

You could generate style editing result in our teaser by running:
```bash
CUDA_VISIBLE_DEVICES=0 python test_fatezero.py --config config/teaser/jeep_watercolor.yaml
```
The result is saved as follows:
```

result
├── teaser
│   ├── jeep_watercolor
│           ├── cross-attention
│           ├── sample
│           ├── train_samples

```
where `cross-attention` is the visualization of cross-attention during inversion;
sample is the result videos obtained from target prompt;
train_sample is the input video;

We also provide a `Tune-A-Video` [checkpoint](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cqiaa_connect_ust_hk/EviSTWoAOs1EmHtqZruq50kBZu1E8gxDknCPigSvsS96uQ?e=492khj). You may download the check point is this link and move it to `./ckpt/jeep_tuned_200/`.
Run following command to get the result:
```bash
CUDA_VISIBLE_DEVICES=0 python test_fatezero.py --config config/teaser/jeep_posche.yaml
```

<!-- ## Citing MetaPortrait

```
@misc{zhang2022metaportrait,
      title={MetaPortrait: Identity-Preserving Talking Head Generation with Fast Personalized Adaptation}, 
      author={Bowen Zhang and Chenyang Qi and Pan Zhang and Bo Zhang and HsiangTao Wu and Dong Chen and Qifeng Chen and Yong Wang and Fang Wen},
      year={2022},
      eprint={2212.08062},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` -->

## Acknowledgements

This repository borrows heavily from [Tune-A-Video](https://github.com/showlab/Tune-A-Video) and [prompt-to-prompt](https://github.com/google/prompt-to-prompt/). thanks the authors for sharing their code and models.

## Maintenance

This is the codebase for our research work. We are still working hard to update this repo and more details are coming in days. If you have any questions or ideas to discuss, feel free to contact [Chenyang Qi](cqiaa@connect.ust.hk) or [Xiaodong Cun](vinthony@gmail.com).
