from .network import WatermarkMaskNet
from .paired_network import PairedWatermarkMaskNet
from .residual_guided_network import ResidualGuidedWatermarkMaskNet
from .aligned_difference_network import AlignedDifferenceMaskNet
from .difference_prior_network import DifferencePriorMaskNet
from .aligned_difference_v2_network import AlignedDifferenceV2MaskNet
from .difference_first_network import DifferenceFirstMaskNet
from .difference_gate_network import DifferenceGateMaskNet

__all__ = [
    "PairedWatermarkMaskNet",
    "ResidualGuidedWatermarkMaskNet",
    "AlignedDifferenceMaskNet",
    "DifferencePriorMaskNet",
    "AlignedDifferenceV2MaskNet",
    "DifferenceFirstMaskNet",
    "DifferenceGateMaskNet",
    "WatermarkMaskNet",
    "paired_model_from_checkpoint",
]


def paired_model_from_checkpoint(checkpoint: dict):
    """Instantiate the paired architecture recorded in a training checkpoint."""

    arguments = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    architecture = arguments.get("architecture", "paired")
    if architecture == "aligned_difference_v2":
        model = AlignedDifferenceV2MaskNet(pretrained=False)
    elif architecture == "difference_first":
        model = DifferenceFirstMaskNet(pretrained=False)
    elif architecture == "difference_gate":
        model = DifferenceGateMaskNet(pretrained=False)
    elif architecture == "difference_prior":
        model = DifferencePriorMaskNet(pretrained=False)
    elif architecture == "aligned_difference":
        model = AlignedDifferenceMaskNet(pretrained=False)
    elif architecture == "residual_guided":
        model = ResidualGuidedWatermarkMaskNet(pretrained=False)
    elif architecture == "paired":
        model = PairedWatermarkMaskNet(pretrained=False)
    else:
        raise ValueError(f"Unknown paired architecture in checkpoint: {architecture}")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return model, architecture
