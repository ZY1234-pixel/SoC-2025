import unittest

import torch

from nets.deeplabv3_plus import DeepLab


class MobileNetV3OutputStrideTest(unittest.TestCase):
    def test_configured_stride_reaches_backbone_and_preserves_output_size(self):
        # Include odd image dimensions to exercise residual branches and padding.
        for stride in (8, 16):
            with self.subTest(stride=stride), torch.inference_mode():
                model = DeepLab(2, backbone="mobilenetv3", downsample_factor=stride).eval()
                inputs = torch.rand(1, 3, 129, 161)
                features = []
                handle = model.backbone.register_forward_hook(
                    lambda _module, _args, result: features.append(result)
                )
                try:
                    logits = model(inputs)
                finally:
                    handle.remove()
                low, high = features[0]
                self.assertEqual(tuple(low.shape), (1, 24, 33, 41))
                self.assertEqual(
                    tuple(high.shape),
                    (1, 960, (129 + stride - 1) // stride, (161 + stride - 1) // stride),
                )
                self.assertEqual(tuple(logits.shape), (1, 2, 129, 161))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_rejects_unsupported_stride(self):
        with self.assertRaisesRegex(ValueError, "8 or 16"):
            DeepLab(2, backbone="mobilenetv3", downsample_factor=32)


if __name__ == "__main__":
    unittest.main()
