# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .instance_postprocessing import (
    restore_letterbox_boxes,
    restore_letterbox_masks,
    select_topk_query_classes,
)
from .utils import box_ops


@META_ARCH_REGISTRY.register()
class MaskDINO(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            transform_eval: transform sigmoid score into softmax score to make score sharper
            semantic_ce_loss: whether use cross-entroy loss in classification
        """
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        criterion = nn.Module()
        criterion.register_buffer("empty_weight", torch.ones(sem_seg_head.num_classes + 1))

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD,
            "metadata": None,
            "size_divisibility": cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
                or cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MaskDINO.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "data_loader": cfg.INPUT.DATASET_MAPPER_NAME,
            "focus_on_box": cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX,
            "transform_eval": cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL,
            "pano_temp": cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE,
            "semantic_ce_loss": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        if self.training:
            raise RuntimeError("This deployment model requires model.eval().")
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)
        outputs, _ = self.sem_seg_head(features)
        padded_size = tuple(images.tensor.shape[-2:])
        return [
            {"instances": self.memory_safe_instance_inference(
                mask_cls_result=classes, mask_pred_result=masks,
                mask_box_result=boxes, input_per_image=record,
                image_size=size, padded_size=padded_size,
            )}
            for classes, masks, boxes, record, size in zip(
                outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"],
                batched_inputs, images.image_sizes,
            )
        ]






    def memory_safe_instance_inference(
        self,
        *,
        mask_cls_result,
        mask_pred_result,
        mask_box_result,
        input_per_image,
        image_size,
        padded_size,
    ):
        """Instance-only postprocessing without all-query full-size tensors."""
        scores, labels, query_indices = select_topk_query_classes(
            mask_cls_result, self.test_topk_per_image)
        selected_masks = mask_pred_result[query_indices]
        selected_boxes = mask_box_result[query_indices].to(selected_masks)

        letterbox_meta = input_per_image.get("letterbox_meta")
        if letterbox_meta is not None:
            pred_masks, mask_scores = restore_letterbox_masks(
                selected_masks, tuple(map(int, image_size)), letterbox_meta)
            pred_boxes = restore_letterbox_boxes(
                selected_boxes, tuple(map(int, image_size)), letterbox_meta)
            output_size = (
                int(letterbox_meta["original_height"]),
                int(letterbox_meta["original_width"]),
            )
        else:
            # Compatibility path for non-letterbox MaskDINO dataset mappers.
            height = int(input_per_image.get("height", image_size[0]))
            width = int(input_per_image.get("width", image_size[1]))
            selected_masks = F.interpolate(
                selected_masks[None], size=padded_size,
                mode="bilinear", align_corners=False)[0]
            selected_masks = sem_seg_postprocess(
                selected_masks, image_size, height, width)
            pred_masks = selected_masks > 0
            mask_scores = (
                selected_masks.sigmoid().flatten(1) * pred_masks.flatten(1)
            ).sum(1) / (pred_masks.flatten(1).sum(1) + 1e-6)
            pred_boxes = self.box_postprocess(selected_boxes, height, width)
            output_size = (height, width)

        result = Instances(output_size)
        result.pred_masks = pred_masks
        result.pred_boxes = Boxes(pred_boxes)
        if self.focus_on_box:
            mask_scores = torch.ones_like(scores)
        result.scores = scores.to(mask_scores) * mask_scores
        result.pred_classes = labels.to(mask_scores.device)
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes

