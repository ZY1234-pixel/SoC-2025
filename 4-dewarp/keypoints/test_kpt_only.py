import torch
import numpy as np
from PIL import Image
from nets.deeplabv3_plus import DeepLabKpt
from utils.utils import cvtColor, preprocess_input, resize_image

# 1. 加载纯关键点模型
model = DeepLabKpt(num_keypoints=4, backbone="mobilenetv3", downsample_factor=8)
model.load_state_dict(torch.load("all_kpt_model_256x256.pth", map_location='cpu'))
model.eval()

# 2. 读取图片并预处理（与训练/完整推理完全一致）
image_path = "VOCdevkit/VOC2007/JPEGImages/1.jpg"
image = Image.open(image_path)
image = cvtColor(image)

orininal_h = np.array(image).shape[0]
orininal_w = np.array(image).shape[1]

input_shape = (256, 256)
image_data, nw, nh = resize_image(image, input_shape)   # letterbox
image_data = np.transpose(preprocess_input(np.array(image_data, np.float32)), (2,0,1))
image_tensor = torch.from_numpy(image_data).unsqueeze(0)
image_tensor[0].numpy().astype(np.float32).tofile("python_preprocess.bin")
print("python_preprocess.bin saved")

# 3. 推理
with torch.no_grad():
    heatmaps = model(image_tensor)[0].cpu().numpy()  # (4, 256, 256)

# 4. 解码角点
from utils.keypoint_utils import decode_keypoints
raw_coords = decode_keypoints(heatmaps, (input_shape[1], input_shape[0]))  # 输入尺寸上的坐标
# print("256x256 峰值坐标:")
# for i, pt in enumerate(raw_coords):
#     print(f"  角点{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
scale_x = orininal_w / nw
scale_y = orininal_h / nh
offset_x = (input_shape[1] - nw) // 2
offset_y = (input_shape[0] - nh) // 2
keypoints = raw_coords.copy()
keypoints[:, 0] = (keypoints[:, 0] - offset_x) * scale_x
keypoints[:, 1] = (keypoints[:, 1] - offset_y) * scale_y
# print(f"offset_x: {offset_x}, offset_y: {offset_y}, nw: {nw}, nh: {nh}")
# print(f"scale_x: {scale_x}, scale_y: {scale_y}")

# 5. 打印结果
print("纯关键点模型预测的角点坐标：")
for i, pt in enumerate(keypoints):
    print(f"  角点{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")

# 6. 对比之前完整模型的结果
# 之前图 1.jpg 的 256 训练的输出为：
#   角点1: (737.4, 602.2)
#   角点2: (4407.8, 599.3)
#   角点3: (4400.9, 3572.9)
#   角点4: (742.4, 3563.1)