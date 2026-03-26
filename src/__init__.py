# MS-HVED for Super-Resolution
# PyTorch implementation of Hetero-Orientation Variational Encoder-Decoder
# Adapted from: https://github.com/ReubenDo/U-HVED
# Paper: https://arxiv.org/abs/1907.11150

from .mshved import MSHVED, create_mshved
from .encoder import SegResEncoder
from .decoder import SegResDecoder
from .fusion import ProductOfGaussians
from .blocks import RegressionResBlock, UpsampleBlock

from .losses import (
    SSIM3DLoss,
    MSHVEDLoss,
    create_loss
)

__all__ = [
    'ProductOfGaussians',
    'SSIM3DLoss',
    'MSHVEDLoss',
    'create_loss',
    'create_mshved',
    'SegResEncoder',
    'SegResDecoder',
    'MSHVED',
    'RegressionResBlock',
    'UpsampleBlock',
]
