"""水印 Mask 检测网络（架构名 ``difference_gate_scale``）。"""

from .difference_gate_network import DifferenceGateMaskNet

__all__ = ["DifferenceGateMaskNet"]


def model_from_checkpoint(checkpoint):
    """按 checkpoint 里记录的架构建模型并载入权重。"""
    arguments = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    architecture = arguments.get("architecture") or "difference_gate_scale"
    large_region = architecture == "difference_gate_scale"
    model = DifferenceGateMaskNet(pretrained=False, large_region=large_region)
    # strict=False：旧权重没有 source_* 这些新增的外观分支参数
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(
            "checkpoint 与 DifferenceGateMaskNet 不匹配，多余参数："
            f"{sorted(unexpected)[:5]}"
        )
    return model, architecture
