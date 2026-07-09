# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class BinaryMaskFormat(object):
    """Convert 0/255 or colored foreground masks to class ids 0/1."""

    def __call__(self, results):
        for key in results.get('seg_fields', []):
            mask = results[key]
            if mask.ndim == 3:
                mask = np.any(mask > 0, axis=2)
            else:
                mask = mask > 0
            results[key] = mask.astype(np.uint8)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'
