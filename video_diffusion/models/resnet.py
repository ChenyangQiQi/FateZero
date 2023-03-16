# code mostly taken from https://github.com/huggingface/diffusers
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from einops import rearrange
from .lora import LoRALinearLayer, LoRACrossAttnProcessor, LoRAXFormersCrossAttnProcessor

class PseudoConv3d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, temporal_kernel_size=None, model_config: dict={}, temporal_downsample=False, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )
        if temporal_kernel_size is None:
            temporal_kernel_size = kernel_size
        
        if temporal_downsample is True:
            temporal_stride = 2
        else:
            temporal_stride = 1
            
            
        if 'lora' in model_config.keys() :
            self.conv_temporal = (
                LoRALinearLayer(
                    out_channels,
                    out_channels,
                    rank=model_config['lora'],
                    stride=temporal_stride
                    
                )
                if kernel_size > 1
                else None
            )
        else:
            self.conv_temporal = (
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=temporal_kernel_size,
                    padding=temporal_kernel_size // 2,
                )
                if kernel_size > 1
                else None
            )

            if self.conv_temporal is not None:
                nn.init.dirac_(self.conv_temporal.weight.data)  # initialized to be identity
                nn.init.zeros_(self.conv_temporal.bias.data)

    def forward(self, x):
        b = x.shape[0]

        is_video = x.ndim == 5
        if is_video:
            x = rearrange(x, "b c f h w -> (b f) c h w")

        x = super().forward(x)

        if is_video:
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b)

        if self.conv_temporal is None or not is_video:
            return x

        *_, h, w = x.shape

        x = rearrange(x, "b c f h w -> (b h w) c f")

        x = self.conv_temporal(x)

        x = rearrange(x, "(b h w) c f -> b c f h w", h=h, w=w)

        return x


class UpsamplePseudo3D(nn.Module):
    """
    An upsampling layer with an optional convolution.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        use_conv_transpose:
        out_channels:
    """

    def __init__(
        self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv", model_config: dict={}, **kwargs
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.model_config = copy.deepcopy(model_config)
        
        conv = None
        if use_conv_transpose:
            raise NotImplementedError
            conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            # Do NOT downsample in upsample block
            td =  False
            
            conv = PseudoConv3d(self.channels, self.out_channels, 3, padding=1, 
                                model_config=model_config,  temporal_downsample=td)
            # conv = PseudoConv3d(self.channels, self.out_channels, 3, kwargs['lora'], padding=1)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        b = hidden_states.shape[0]
        is_video = hidden_states.ndim == 5
        if is_video:
            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
        
        if is_video:
            td =  ('temporal_downsample' in self.model_config and self.model_config['temporal_downsample'] is True)

                
            if td:
                hidden_states = rearrange(hidden_states, " (b f) c h w -> b c h w f ", b=b)
                t_b, t_c, t_h, t_w, t_f = hidden_states.shape
                hidden_states = rearrange(hidden_states, " b c h w f -> (b c) (h w) f ", b=b)
                
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="linear")
                hidden_states = rearrange(hidden_states, " (b c) (h w) f  ->  (b f) c h w ", b=t_b, h=t_h)
            
        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if is_video:
            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", b=b)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class DownsamplePseudo3D(nn.Module):
    """
    A downsampling layer with an optional convolution.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        out_channels:
        padding:
    """

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, model_config: dict={}, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        self.model_config = copy.deepcopy(model_config)
        # self.model_config = copy.deepcopy(model_config)
        
        if use_conv:
            td =  ('temporal_downsample' in model_config and model_config['temporal_downsample'] is True)

            conv = PseudoConv3d(self.channels, self.out_channels, 3, stride=stride, padding=padding, 
                                model_config=model_config, temporal_downsample=td)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels
        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        else:
            b = hidden_states.shape[0]
            is_video = hidden_states.ndim == 5
            if is_video:
                hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
            hidden_states = self.conv(hidden_states)
            if is_video:
                hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", b=b)

        return hidden_states


class ResnetBlockPseudo3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        model_config: dict={},
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True
        )

        self.conv1 = PseudoConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, model_config=model_config)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(
            num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = PseudoConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, model_config=model_config)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = UpsamplePseudo3D(in_channels, use_conv=False, model_config=model_config)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = DownsamplePseudo3D(in_channels, use_conv=False, padding=1, name="op", model_config=model_config)

        self.use_in_shortcut = (
            self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = PseudoConv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, model_config=model_config
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            is_video = hidden_states.ndim == 5
            if is_video:
                b, c, f, h, w = hidden_states.shape
                hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                temb = temb.repeat_interleave(f, 0)

            hidden_states = hidden_states + temb

            if is_video:
                hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", b=b)

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            is_video = hidden_states.ndim == 5
            if is_video:
                b, c, f, h, w = hidden_states.shape
                hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                temb = temb.repeat_interleave(f, 0)

            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

            if is_video:
                hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", b=b)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))


def upsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    r"""Upsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and upsamples each image with the given
    filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its shape is
    a: multiple of the upsampling factor.

    Args:
        hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
        factor: Integer upsampling factor (default: 2).
        gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output: Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)

    kernel = kernel * (gain * (factor**2))
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states,
        kernel.to(device=hidden_states.device),
        up=factor,
        pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
    )
    return output


def downsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    r"""Downsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and downsamples each image with the
    given filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the
    specified `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the downsampling factor.

    Args:
        hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to average pooling.
        factor: Integer downsampling factor (default: 2).
        gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output: Tensor of the shape `[N, C, H // factor, W // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)

    kernel = kernel * gain
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states,
        kernel.to(device=hidden_states.device),
        down=factor,
        pad=((pad_value + 1) // 2, pad_value // 2),
    )
    return output


def upfirdn2d_native(tensor, kernel, up=1, down=1, pad=(0, 0)):
    up_x = up_y = up
    down_x = down_y = down
    pad_x0 = pad_y0 = pad[0]
    pad_x1 = pad_y1 = pad[1]

    _, channel, in_h, in_w = tensor.shape
    tensor = tensor.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = tensor.shape
    kernel_h, kernel_w = kernel.shape

    out = tensor.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out.to(tensor.device)  # Move back to mps if necessary
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)
