from typing import Callable, Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F

class SwiGLUFFN(nn.Module):

    def __init__(self, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, act_layer: Callable[..., nn.Module]=None, drop: float=0.0, bias: bool=True) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        if x12.dtype == torch.float16:
            # Gated activations can exceed the FP16 range.
            with torch.autocast(device_type=x.device.type, enabled=False):
                hidden = F.silu(x1.float()) * x2.float()
                return self.w3(hidden)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
