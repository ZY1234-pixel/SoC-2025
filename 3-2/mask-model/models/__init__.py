"""Detector architectures.

Live architectures:
  ``AppearanceGateMaskNet``   current paired detector (``appearance_gate``):
                              appearance is the main criterion; the candidate
                              difference only supplies a multiplicative gate
  ``DifferenceGateMaskNet``   previous paired detector (``difference_gate``,
                              and ``difference_gate_scale`` when large_region=True)

Retained for older scripts that still import them:
  ``WatermarkMaskNet``        original RGB-only single-image model (train.py)
  ``PairedWatermarkMaskNet``  first paired model (evaluate_paired_full_resolution.py)

Historical experimental variants were moved to ``mask-model/_archived/models/``:
residual_guided, aligned_difference, aligned_difference_v2, difference_prior,
difference_first.
"""

from .network import WatermarkMaskNet
from .paired_network import PairedWatermarkMaskNet
from .difference_gate_network import DifferenceGateMaskNet
from .appearance_gate_network import AppearanceGateMaskNet

__all__ = [
    "WatermarkMaskNet",
    "PairedWatermarkMaskNet",
    "DifferenceGateMaskNet",
    "AppearanceGateMaskNet",
    "paired_model_from_checkpoint",
]


def paired_model_from_checkpoint(checkpoint: dict):
    """Instantiate the paired architecture recorded in a training checkpoint.

    Supported architectures are ``appearance_gate``, ``difference_gate`` and its
    large-region variant ``difference_gate_scale``. Historical variants live in
    ``mask-model/_archived/models/`` and are no longer loadable from here.
    """

    arguments = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    architecture = arguments.get("architecture") or _architecture_from_state(state)
    if architecture == "appearance_gate":
        model = AppearanceGateMaskNet(pretrained=False)
    elif architecture == "difference_gate_scale":
        model = DifferenceGateMaskNet(pretrained=False, large_region=True)
    elif architecture == "difference_gate":
        model = DifferenceGateMaskNet(pretrained=False)
    elif architecture == "paired":
        # Retained only so legacy checkpoints still load.
        model = PairedWatermarkMaskNet(pretrained=False)
    elif architecture is None:
        raise ValueError(
            "Checkpoint records no architecture; it is either the old single-image "
            "model or an unsupported historical variant. Use a difference_gate / "
            "difference_gate_scale checkpoint."
        )
    else:
        raise ValueError(
            f"Unknown paired architecture in checkpoint: {architecture}. "
            "Historical variants live in mask-model/_archived/models/."
        )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        # Checkpoints written before the appearance branch gained a real decoder
        # lack the source_* keys and still carry the old 1x1 source_head.weight /
        # .bias.  Load what matches and leave the new appearance decoder at its
        # initialisation rather than refusing the checkpoint.
        missing = [key for key in model.state_dict() if key not in state]
        unexpected = [key for key in state if key not in model.state_dict()]
        superseded = ("source_head.weight", "source_head.bias")
        if (missing and all(key.startswith("source_") for key in missing)
                and all(key in superseded for key in unexpected)):
            model.load_state_dict(state, strict=False)
            return model, architecture
        raise RuntimeError(
            f"Checkpoint does not match architecture '{architecture}'. It is probably "
            "the old single-image weight (mask-model/weights/watermark_mask.pt) or an "
            "unsupported historical variant. Use a difference_gate / "
            "difference_gate_scale checkpoint from mask-model/runs/."
        ) from error
    return model, architecture


def _architecture_from_state(state: dict) -> str | None:
    """Recover the architecture for checkpoints saved without an args entry."""

    keys = set(state)
    if any(key.startswith("side_heads.") for key in keys):
        return "appearance_gate"
    if any(key.startswith("large_region_head.") for key in keys):
        return "difference_gate_scale"
    if any(key.startswith("gates.") for key in keys):
        return "difference_gate"
    if any(key.startswith("aux_heads.") for key in keys):
        return "paired"
    return None
