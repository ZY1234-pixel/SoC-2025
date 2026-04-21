import sys

sys.path.append('../')
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import os

from model.DocDiff import DocDiff
from schedule.diffusionSample import GaussianDiffusion
from schedule.schedule import Schedule


# -------------------------- 核心工具函数（尺寸严格对齐） --------------------------
def min_max(array):
    """安全归一化到0-1范围"""
    return (array - array.min()) / (array.max() - array.min() + 1e-8)


def crop_concat_preserve_size(img, crop_size=128):
    """
    分块但严格保留原始尺寸信息
    img: [1, C, H, W] 原始输入
    return: 分块张量 + 原始尺寸(H,W)
    """
    B, C, H, W = img.shape

    # 计算需要填充的尺寸（仅填充到crop_size的整数倍）
    pad_h = (crop_size - H % crop_size) % crop_size
    pad_w = (crop_size - W % crop_size) % crop_size

    # 填充图片（仅在需要时）
    img_padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, H_pad, W_pad = img_padded.shape

    # 分块
    patches = []
    for i in range(H_pad // crop_size):
        for j in range(W_pad // crop_size):
            patch = img_padded[:, :,
            i * crop_size:(i + 1) * crop_size,
            j * crop_size:(j + 1) * crop_size]
            patches.append(patch)

    return torch.cat(patches, dim=0), (H, W)  # 只返回原始尺寸，不返回分块数


def crop_concat_back_preserve_size(patches, original_size, crop_size=128):
    """
    拼接回原始尺寸（精确裁剪，无尺寸误差）
    patches: 分块推理结果
    original_size: (H, W) 原始输入尺寸
    return: [1, C, H, W] 与输入尺寸完全一致的输出
    """
    H_ori, W_ori = original_size
    B = 1  # 固定batch_size=1
    C = patches.shape[1]

    # 计算分块数
    n_patches = patches.shape[0] // B
    n_w = int((W_ori + crop_size - 1) // crop_size)  # 向上取整
    n_h = n_patches // n_w

    # 按行拼接
    rows = []
    for i in range(n_h):
        row_patches = patches[i * n_w * B: (i + 1) * n_w * B]
        row = torch.cat([row_patches[j * B:(j + 1) * B] for j in range(n_w)], dim=3)
        rows.append(row)

    # 按列拼接
    full_img = torch.cat(rows, dim=2)

    # 精确裁剪回原始尺寸（关键！）
    full_img = full_img[:, :, :H_ori, :W_ori]

    return full_img


# -------------------------- 主函数（无强制裁剪，保留原始尺寸） --------------------------
def main():
    # ====================== 只需改这里的路径 ======================
    init_path = "/watermark/DocDiff-main/checksave/model_init_200000.pth"
    denoiser_path = "/watermark/DocDiff-main/checksave/model_denoiser_200000.pth"
    blur_img_path = "/home/fauyn/Aproject/watermark/DocDiff-main/demo/d238fbad8ae61fcaeb22ebc8a4afd6bd_720.jpg"
    out_final = "./final_result.png"  # 输出图（尺寸和输入完全一致）
    crop_size = 128  # 分块尺寸（可调整，建议128/256）
    # ========================================================

    # 检查文件
    for f in [blur_img_path, init_path, denoiser_path]:
        if not os.path.exists(f):
            print(f"文件不存在: {f}")
            sys.exit(1)

    # 加载图片（不做任何裁剪！保留原始尺寸）
    img = Image.open(blur_img_path).convert('RGB')
    original_width, original_height = img.size  # 记录原始尺寸
    print(f"输入图原始尺寸: {original_width} x {original_height}")

    # 仅转换为张量，不做任何裁剪/缩放
    img_tensor = ToTensor()(img).unsqueeze(0)  # [1, 3, H, W] 完全匹配原始尺寸
    B, C, H, W = img_tensor.shape
    print(f"输入张量尺寸: {H} x {W} (H x W)")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    network = DocDiff(
        input_channels=6,
        output_channels=3,
        n_channels=32,
        ch_mults=[1, 2, 3, 4],
        n_blocks=1,
    ).to(device)

    # 加载权重
    network.init_predictor.load_state_dict(
        torch.load(init_path, map_location=device),
        strict=False
    )
    network.denoiser.load_state_dict(
        torch.load(denoiser_path, map_location=device),
        strict=False
    )
    network.eval()

    # 采样器
    schedule = Schedule('linear', 100)
    sampler = GaussianDiffusion(network.denoiser, 100, schedule).to(device)

    # ====================== 分块推理（严格保留原始尺寸） ======================
    print("生成最终修复图（尺寸与输入一致）...")
    with torch.no_grad():
        # 分块（保留原始尺寸信息）
        img_cropped, original_size = crop_concat_preserve_size(img_tensor, crop_size)
        img_cropped_device = img_cropped.to(device)

        # 模型推理
        noisyImage = torch.randn_like(img_cropped_device).to(device)
        init_predict = network.init_predictor(img_cropped_device, 0)
        sampledImgs = sampler(noisyImage, init_predict, 'True')
        finalImgs = sampledImgs + init_predict

        # 拼接回原始尺寸（精确裁剪，无误差）
        finalImgs = crop_concat_back_preserve_size(
            finalImgs.cpu(),
            original_size,
            crop_size
        )

        # 归一化（保证图片正常显示）
        finalImgs = min_max(finalImgs)

    # 保存最终图（尺寸和输入100%一致）
    save_image(finalImgs, out_final, normalize=False)

    # 验证尺寸
    output_img = Image.open(out_final)
    output_width, output_height = output_img.size
    print(f"输出图最终尺寸: {output_width} x {output_height}")
    print(f"尺寸是否匹配: {output_width == original_width and output_height == original_height}")
    print(f"✅ 最终修复图已保存：{out_final}")


if __name__ == "__main__":
    main()
