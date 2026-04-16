"""
Normalized PyTorch Model Definitions.

This module defines a simple convolutional network architecture including
ConvBlock, InceptionBlock, and SimpleNet.
"""

from typing import Literal
import torch
import torch.nn as nn

__all__ = ["ConvBlock", "InceptionBlock", "SimpleNet"]

class ConvBlock(nn.Module):
    """
    A convolutional block consisting of Conv2d, InstanceNorm2d, and LeakyReLU.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int, tuple): Size of the convolving kernel. Default: 1.
        stride (int, tuple): Stride of the convolution. Default: 1.
        padding (int, tuple): Zero-padding added to both sides of the input. Default: 0.
        padding_mode (str): 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode=padding_mode,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class InceptionBlock(nn.Module):
    """
    An Inception-like block with three parallel branches and a fusion layer.

    Architecture:
        - Branch 1: 1x1 Conv
        - Branch 2: 1x1 Conv -> 3x3 Conv
        - Branch 3: 1x1 Conv -> 3x3 Conv -> 3x3 Conv
        - Fusion: Concatenate branches -> 1x1 Conv to reduce channels

    Args:
        in_channels (int): Number of input channels.
        out_per_branch (int): Number of output channels for each branch before fusion.
        padding_mode (str): Padding mode for convolution layers.
    """
    def __init__(
        self,
        in_channels: int,
        out_per_branch: int,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    ):
        super().__init__()
        self.branch1 = ConvBlock(
            in_channels, out_per_branch, kernel_size=1, padding=0, padding_mode=padding_mode
        )

        self.branch2 = nn.Sequential(
            ConvBlock(
                in_channels, out_per_branch, kernel_size=1, padding=0, padding_mode=padding_mode
            ),
            ConvBlock(
                out_per_branch, out_per_branch, kernel_size=3, padding=1, padding_mode=padding_mode,
            ),
        )

        self.branch3 = nn.Sequential(
            ConvBlock(
                in_channels, out_per_branch, kernel_size=1, padding=0, padding_mode=padding_mode
            ),
            ConvBlock(
                out_per_branch, out_per_branch, kernel_size=3, padding=1, padding_mode=padding_mode,
            ),
            ConvBlock(
                out_per_branch, out_per_branch, kernel_size=3, padding=1, padding_mode=padding_mode,
            ),
        )

        # Fuse 3 * out_per_branch channels back to out_per_branch
        self.fuse = ConvBlock(
            3 * out_per_branch, out_per_branch, kernel_size=1, padding_mode=padding_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # Concatenate along channel dimension
        x_cat = torch.cat([x1, x2, x3], dim=1)
        return self.fuse(x_cat)

class SimpleNet(nn.Module):
    """
    A simple network wrapping Input Projection, Inception Block, and Output Head.

    Args:
        input_channels (int): Number of channels in the input image. Default: 3.
        hidden_channels (int): Number of hidden channels. Default: 64.
        output_channels (int): Number of output channels. Default: 28.
        padding_mode (str): Padding mode for convolution layers.
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 64,
        output_channels: int = 28,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    ):
        super().__init__()
        self.input_proj = ConvBlock(
            input_channels, hidden_channels, kernel_size=1, padding_mode=padding_mode
        )
        self.inception = InceptionBlock(
            hidden_channels, hidden_channels, padding_mode=padding_mode
        )
        # Output head typically does not need activation or normalization if it's the final layer
        self.output_head = nn.Conv2d(
            hidden_channels, output_channels, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.inception(x)
        x = self.output_head(x)
        return x