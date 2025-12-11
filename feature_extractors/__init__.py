"""
Feature Extractors Package
Contiene diferentes extractores de características para imágenes.
"""

from .base_extractor import BaseExtractor
from .resnet_extractor import ResNetExtractor
from .vgg_extractor import VGGExtractor
from .color_texture_extractor import ColorTextureExtractor
from .hog_extractor import HOGExtractor
from .color_shape_extractor import ColorShapeExtractor

__all__ = [
    'BaseExtractor',
    'ResNetExtractor',
    'VGGExtractor',
    'ColorTextureExtractor',
    'HOGExtractor',
    'ColorShapeExtractor'
]
