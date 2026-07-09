# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class DocSegDataset(CustomDataset):
    """Binary document foreground segmentation dataset.

    The on-disk masks may use 0/255 for readability. The training pipeline
    converts every non-zero value to label 1 before formatting.
    """

    CLASSES = ('background', 'document')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(DocSegDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
