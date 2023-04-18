import os
import math
import textwrap

import imageio
import numpy as np
from typing import Sequence
import requests
import cv2
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision import transforms
from einops import rearrange


IMAGE_EXTENSION = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".JPEG")

FONT_URL = "https://raw.github.com/googlefonts/opensans/main/fonts/ttf/OpenSans-Regular.ttf"
FONT_PATH = "./docs/OpenSans-Regular.ttf"


def pad(image: Image.Image, top=0, right=0, bottom=0, left=0, color=(255, 255, 255)) -> Image.Image:
    new_image = Image.new(image.mode, (image.width + right + left, image.height + top + bottom), color)
    new_image.paste(image, (left, top))
    return new_image


def download_font_opensans(path=FONT_PATH):
    font_url = FONT_URL
    response = requests.get(font_url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(response.content)


def annotate_image_with_font(image: Image.Image, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    image_w = image.width
    _, _, text_w, text_h = font.getbbox(text)
    line_size = math.floor(len(text) * image_w / text_w)

    lines = textwrap.wrap(text, width=line_size)
    padding = text_h * len(lines)
    image = pad(image, top=padding + 3)

    ImageDraw.Draw(image).text((0, 0), "\n".join(lines), fill=(0, 0, 0), font=font)
    return image


def annotate_image(image: Image.Image, text: str, font_size: int = 15):
    if not os.path.isfile(FONT_PATH):
        download_font_opensans()
    font = ImageFont.truetype(FONT_PATH, size=font_size)
    return annotate_image_with_font(image=image, text=text, font=font)


def make_grid(images: Sequence[Image.Image], rows=None, cols=None) -> Image.Image:
    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(i) for i in images]

    if rows is None:
        assert cols is not None
        rows = math.ceil(len(images) / cols)
    else:
        cols = math.ceil(len(images) / rows)

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        if image.size != (w, h):
            image = image.resize((w, h))
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def save_images_as_gif(
    images: Sequence[Image.Image],
    save_path: str,
    loop=0,
    duration=100,
    optimize=False,
) -> None:

    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        loop=loop,
        duration=duration,
    )

def save_images_as_mp4(
    images: Sequence[Image.Image],
    save_path: str,
) -> None:

    writer_edit = imageio.get_writer(
        save_path,
        fps=10)
    for i in images:
        init_image = i.convert("RGB")
        writer_edit.append_data(np.array(init_image))
    writer_edit.close()



def save_images_as_folder(
    images: Sequence[Image.Image],
    save_path: str,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    for index, image in enumerate(images):
        init_image = image
        if len(np.array(init_image).shape) == 3:
            cv2.imwrite(os.path.join(save_path, f"{index:05d}.png"), np.array(init_image)[:, :, ::-1])
        else:
            cv2.imwrite(os.path.join(save_path, f"{index:05d}.png"), np.array(init_image))

def log_train_samples(
    train_dataloader,
    save_path,
    num_batch: int = 4,
):
    train_samples = []
    for idx, batch in enumerate(train_dataloader):
        if idx >= num_batch:
            break
        train_samples.append(batch["images"])

    train_samples = torch.cat(train_samples).numpy()
    train_samples = rearrange(train_samples, "b c f h w -> b f h w c")
    train_samples = (train_samples * 0.5 + 0.5).clip(0, 1)
    train_samples = numpy_batch_seq_to_pil(train_samples)
    train_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(train_samples))))) for images in zip(*train_samples)]
    # save_images_as_gif(train_samples, save_path)
    save_gif_mp4_folder_type(train_samples, save_path)

def log_train_reg_samples(
    train_dataloader,
    save_path,
    num_batch: int = 4,
):
    train_samples = []
    for idx, batch in enumerate(train_dataloader):
        if idx >= num_batch:
            break
        train_samples.append(batch["class_images"])

    train_samples = torch.cat(train_samples).numpy()
    train_samples = rearrange(train_samples, "b c f h w -> b f h w c")
    train_samples = (train_samples * 0.5 + 0.5).clip(0, 1)
    train_samples = numpy_batch_seq_to_pil(train_samples)
    train_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(train_samples))))) for images in zip(*train_samples)]
    # save_images_as_gif(train_samples, save_path)
    save_gif_mp4_folder_type(train_samples, save_path)


def save_gif_mp4_folder_type(images, save_path, save_gif=True):

    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(i) for i in images]
    elif isinstance(images[0], torch.Tensor):
        images = [transforms.ToPILImage()(i.cpu().clone()[0]) for i in images]
    save_path_mp4 = save_path.replace('gif', 'mp4')
    save_path_folder = save_path.replace('.gif', '')
    if save_gif: save_images_as_gif(images, save_path)
    save_images_as_mp4(images, save_path_mp4)
    save_images_as_folder(images, save_path_folder)

# copy from video_diffusion/pipelines/stable_diffusion.py
def numpy_seq_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

# copy from diffusers-0.11.1/src/diffusers/pipeline_utils.py
def numpy_batch_seq_to_pil(images):
    pil_images = []
    for sequence in images:
        pil_images.append(numpy_seq_to_pil(sequence))
    return pil_images
