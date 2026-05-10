import torch
from nets.deeplabv3_plus import DeepLabKpt

# 配置
num_keypoints = 4
backbone = "mobilenetv3"
old_weights = "logs/2026_05_10_03_49_03/best_epoch_weights.pth"
new_weights = "all_kpt_model_256x256.pth"

# 新建只有关键点头的模型
model = DeepLabKpt(num_keypoints=num_keypoints, backbone=backbone, downsample_factor=8)

# 加载旧权重
old_dict = torch.load(old_weights, map_location='cpu')
# 如果旧权重是 DataParallel 格式，去掉 'module.' 前缀
if any(k.startswith('module.') for k in old_dict.keys()):
    old_dict = {k[7:]: v for k, v in old_dict.items()}

# 只保留新模型中存在的 key
new_dict = model.state_dict()
matched_dict = {k: v for k, v in old_dict.items() if k in new_dict and new_dict[k].shape == v.shape}
new_dict.update(matched_dict)
model.load_state_dict(new_dict, strict=False)

# 保存纯关键点模型权重
torch.save(model.state_dict(), new_weights)
print(f"转换完成，纯关键点模型已保存至 {new_weights}")