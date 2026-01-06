from espnet2.enh_se.spatial_encoder.abs_spatial_encoder import AbsSpatialEncoder

try:
    from espnet2.enh_se.spatial_encoder.resnet2d_spatial_encoder import (
        ResNet2DSpatialEncoder,
    )

    __all__ = ["AbsSpatialEncoder", "ResNet2DSpatialEncoder"]
except ImportError:
    __all__ = ["AbsSpatialEncoder"]

