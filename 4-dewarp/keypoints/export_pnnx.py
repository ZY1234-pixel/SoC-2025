import torch
from nets.deeplabv3_plus import DeepLabKpt

# 加载模型
model = DeepLabKpt(num_keypoints=4, backbone="mobilenetv3", downsample_factor=8)
model.load_state_dict(torch.load("kpt_model_256x256.pth", map_location='cpu'))
model.eval()

# 生成 TorchScript
example = torch.randn(1, 3, 256, 256)
traced_model = torch.jit.trace(model, example)
traced_model.save("kpt_model_256x256.pt")
print("TorchScript 模型已保存")