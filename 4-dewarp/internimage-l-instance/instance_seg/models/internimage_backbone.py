"""Detectron2 backbone adapter for the audited semantic InternImage-L."""

from collections import OrderedDict
from typing import Dict, Iterable, Mapping

import torch
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone

from .internimage_impl import InternImage


@BACKBONE_REGISTRY.register()
class InternImageBackbone(Backbone):
    """Expose InternImage-L C2-C5 features through Detectron2's contract."""

    _FEATURE_NAMES = ("res2", "res3", "res4", "res5")
    _CHANNELS = (160, 320, 640, 1280)
    _STRIDES = (4, 8, 16, 32)

    def __init__(
        self,
        *,
        out_features: Iterable[str] = _FEATURE_NAMES,
        with_cp: bool = True,
        drop_path_rate: float = 0.4,
    ) -> None:
        super().__init__()
        self._out_features = tuple(out_features)
        unknown = set(self._out_features) - set(self._FEATURE_NAMES)
        if unknown:
            raise ValueError(f"Unknown InternImage output features: {sorted(unknown)}")
        self.bottom_up = InternImage(
            core_op="DCNv3",
            channels=160,
            depths=[5, 5, 22, 5],
            groups=[10, 20, 40, 80],
            mlp_ratio=4.0,
            drop_path_rate=drop_path_rate,
            norm_layer="LN",
            layer_scale=1.0,
            offset_scale=2.0,
            post_norm=True,
            with_cp=with_cp,
            out_indices=(0, 1, 2, 3),
            init_cfg=None,
        )
        self._out_feature_channels = dict(zip(self._FEATURE_NAMES, self._CHANNELS))
        self._out_feature_strides = dict(zip(self._FEATURE_NAMES, self._STRIDES))

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected NCHW RGB input, received {tuple(images.shape)}")
        tensors = self.bottom_up(images)
        features = OrderedDict(zip(self._FEATURE_NAMES, tensors))
        return OrderedDict((name, features[name]) for name in self._out_features)

    def output_shape(self) -> Mapping[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self) -> int:
        return 32


@BACKBONE_REGISTRY.register()
def build_internimage_l_backbone(cfg=None, input_shape=None) -> InternImageBackbone:
    """Detectron2 builder kept explicit for YAML/LazyConfig integration."""
    del input_shape
    out_features = InternImageBackbone._FEATURE_NAMES
    with_cp = True
    drop_path_rate = 0.4
    if cfg is not None:
        node = cfg.MODEL.BACKBONE
        out_features = tuple(getattr(node, "OUT_FEATURES", out_features))
        with_cp = bool(getattr(node, "WITH_CP", with_cp))
        drop_path_rate = float(getattr(node, "DROP_PATH_RATE", drop_path_rate))
    return InternImageBackbone(
        out_features=out_features,
        with_cp=with_cp,
        drop_path_rate=drop_path_rate,
    )
