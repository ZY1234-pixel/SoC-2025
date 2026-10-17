"""Local image encoder."""
import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from .dinov2.models.vision_transformer import vit_small


def load_encoder(path, device):
    model = vit_small(
        img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp",
        block_chunks=0, num_register_tokens=0, interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return model.eval().requires_grad_(False).to(device)


def extract_features(model, image, layer_ids, size):
    image = F.interpolate(image.add(1).mul(0.5), size=(size, size),
                          mode="bicubic", align_corners=False, antialias=True).clamp_(0, 1)
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    tokens = model.prepare_tokens_with_masks((image - mean) / std, None)
    features = {}
    for index, block in enumerate(model.blocks):
        tokens = block(tokens)
        features[index] = tokens[:, 1:]
    features[max(features)] = model.norm(tokens)[:, 1:]
    return [features[index] for index in layer_ids]
