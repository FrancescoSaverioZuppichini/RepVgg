import torch
from torch import nn, Tensor
from torchvision.ops import Conv2dNormActivation
from typing import Dict, List

torch.manual_seed(0)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.weight = nn.Sequential(
            Conv2dNormActivation(
                in_channels, out_channels, kernel_size=3, stride=stride
            ),
            Conv2dNormActivation(
                out_channels, out_channels, kernel_size=3, activation_layer=None
            ),
        )
        self.shortcut = (
            Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                activation_layer=None,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.shortcut(x)  # <- 2x memory
        x = self.weight(x)
        x += res
        x = self.act(x)  # <- 1x memory
        return x


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.weight = nn.Sequential(
            Conv2dNormActivation(
                in_channels, out_channels, kernel_size=3, stride=stride
            ),
            Conv2dNormActivation(
                out_channels, out_channels, kernel_size=3, activation_layer=None
            ),
        )
        self.shortcut = (
            Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                activation_layer=None,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.weight(x)
        x += res
        x = self.act(x)
        return x


# To make most of the mem-
# bers shallower or simpler, we use ResNet-like identity (only
# if the dimensions match) and 1 × 1 branches so that the
# training-time information flow of a building block is y =
# x + g(x) + f (x).


def get_fused_bn_to_conv_state_dict(
    conv: nn.Conv2d, bn: nn.BatchNorm2d
) -> Dict[str, Tensor]:
    # in the paper, weights is gamma and bias is beta
    bn_mean, bn_var, bn_gamma, bn_beta = (
        bn.running_mean,
        bn.running_var,
        bn.weight,
        bn.bias,
    )
    # we need the std!
    bn_std = (bn_var + bn.eps).sqrt()
    # eq (3)
    conv_weight = nn.Parameter((bn_gamma / bn_std).reshape(-1, 1, 1, 1) * conv.weight)
    # still eq (3)
    conv_bias = nn.Parameter(bn_beta - bn_mean * bn_gamma / bn_std)
    return {"weight": conv_weight, "bias": conv_bias}


def get_fused_conv_state_dict_from_block(block: RepVGGBlock) -> Dict[str, Tensor]:
    fused_block_conv_state_dict = get_fused_bn_to_conv_state_dict(
        block.block[0], block.block[1]
    )

    if block.shortcut:
        # fuse the 1x1 shortcut
        conv_1x1_state_dict = get_fused_bn_to_conv_state_dict(
            block.shortcut[0], block.shortcut[1]
        )
        # we pad the 1x1 to a 3x3
        conv_1x1_state_dict["weight"] = torch.nn.functional.pad(
            conv_1x1_state_dict["weight"], [1, 1, 1, 1]
        )
        fused_block_conv_state_dict["weight"] += conv_1x1_state_dict["weight"]
        fused_block_conv_state_dict["bias"] += conv_1x1_state_dict["bias"]
    if block.identity:
        # create our identity 3x3 conv kernel
        identify_conv = nn.Conv2d(
            block.block[0].in_channels,
            block.block[0].in_channels,
            kernel_size=3,
            bias=True,
            padding=1,
        ).to(block.block[0].weight.device)
        # set them to zero!
        identify_conv.weight.zero_()
        # set the middle element to zero for the right channel
        in_channels = identify_conv.in_channels
        for i in range(identify_conv.in_channels):
            identify_conv.weight[i, i % in_channels, 1, 1] = 1
        # fuse the 3x3 identity
        identity_state_dict = get_fused_bn_to_conv_state_dict(
            identify_conv, block.identity
        )
        fused_block_conv_state_dict["weight"] += identity_state_dict["weight"]
        fused_block_conv_state_dict["bias"] += identity_state_dict["bias"]

    fused_conv_state_dict = {
        k: nn.Parameter(v) for k, v in fused_block_conv_state_dict.items()
    }

    return fused_conv_state_dict


class RepVGGFastBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.relu = nn.ReLU(inplace=True)


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            stride=stride,
            activation_layer=None,
            # the original model may also have groups > 1
        )

        self.shortcut = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            activation_layer=None,
        )

        self.identity = (
            nn.BatchNorm2d(out_channels) if in_channels == out_channels else None
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x  # <- 2x memory
        x = self.block(x)
        x += self.shortcut(res)
        if self.identity:
            x += self.identity(res)
        x = self.relu(x)  # <- 1x memory
        return x

    def to_fast(self) -> RepVGGFastBlock:
        fused_conv_state_dict = get_fused_conv_state_dict_from_block(self)
        fast_block = RepVGGFastBlock(
            self.block[0].in_channels,
            self.block[0].out_channels,
            stride=self.block[0].stride,
        )

        fast_block.conv.load_state_dict(fused_conv_state_dict)

        return fast_block


class RepVGGStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
    ):
        super().__init__(
            RepVGGBlock(in_channels, out_channels, stride=2),
            *[RepVGGBlock(out_channels, out_channels) for _ in range(depth - 1)],
        )


class RepVGG(nn.Sequential):
    def __init__(self, widths: List[int], depths: List[int], in_channels: int = 3):
        super().__init__()
        in_out_channels = zip(widths, widths[1:])

        self.stages = nn.Sequential(
            RepVGGStage(in_channels, widths[0], depth=1),
            *[
                RepVGGStage(in_channels, out_channels, depth)
                for (in_channels, out_channels), depth in zip(in_out_channels, depths)
            ],
        )

        # omit classification head for simplicity

    def switch_to_fast(self):
        for stage in self.stages:
            for i, block in enumerate(stage):
                stage[i] = block.to_fast()
        return self
