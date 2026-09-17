"""Model loading, detection filtering and CUDA timing."""

from pathlib import Path
from detectron2.modeling import build_model
import torch


def filter_and_sort_instances(instances, score_threshold: float):
    keep = instances.scores >= float(score_threshold)
    instances = instances[keep]
    order = torch.argsort(instances.scores, descending=True)
    return instances[order]


def synchronize_cuda() -> None:
    """Synchronize only when CUDA is active so timing includes GPU work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(cfg, checkpoint_path: Path):
    model = build_model(cfg)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model.eval()
