import argparse
import json
import math
import random
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models import WatermarkMaskNet


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def resize_for_crop(images: list[Image.Image], size: int, resamples: list[int]) -> list[Image.Image]:
    """按同一比例放大一组对齐图像，确保任一边都不小于裁剪尺寸。"""
    width, height = images[0].size
    scale = max(size / width, size / height, 1.0)
    if scale == 1.0:
        return images
    shape = (math.ceil(width * scale), math.ceil(height * scale))
    return [image.resize(shape, resample) for image, resample in zip(images, resamples)]


def random_crop_box(width: int, height: int, size: int, rng=random) -> tuple[int, int, int, int]:
    left = rng.randint(0, width - size)
    top = rng.randint(0, height - size)
    return left, top, left + size, top + size


def crop_aligned(images: list[Image.Image], size: int, training: bool, seed: int = 0) -> list[Image.Image]:
    """使用同一个裁剪框处理 RGB、Mask 等配对数据，避免像素错位。"""
    width, height = images[0].size
    # 验证阶段为每个 index 使用固定随机种子，使多次评估看到完全相同的裁块。
    rng = random if training else random.Random(seed)
    box = random_crop_box(width, height, size, rng)
    return [image.crop(box) for image in images]


def training_crop_box(
    mask: Image.Image, size: int, positive_probability: float, negative_probability: float
) -> tuple[int, int, int, int]:
    """按水印感知策略生成训练裁剪框。

    默认配置下：70% 的裁块包含随机选中的水印像素，10% 尝试裁出纯背景，
    剩余 20% 均匀随机裁剪。这样既避免小水印被大量空裁块淹没，也保留
    足够的负样本来抑制误检。
    """
    width, height = mask.size
    choice = random.random()
    bounds = mask.getbbox()
    if choice < positive_probability and bounds:
        # 在所有前景像素中随机选一个锚点，再随机改变它在裁块中的相对位置。
        # 这比始终把水印放在中心更接近真实推理分布。
        positive = np.argwhere(np.asarray(mask.crop(bounds)) > 127)
        if len(positive):
            row, column = positive[random.randrange(len(positive))]
            x, y = bounds[0] + int(column), bounds[1] + int(row)
            left = min(max(0, x - random.randint(size // 4, 3 * size // 4)), width - size)
            top = min(max(0, y - random.randint(size // 4, 3 * size // 4)), height - size)
            return left, top, left + size, top + size
    if choice < positive_probability + negative_probability:
        # 最多尝试 12 次寻找完全不含水印的区域；找不到时回退到普通随机裁剪。
        for _ in range(12):
            box = random_crop_box(width, height, size)
            if not mask.crop(box).getbbox():
                return box
    return random_crop_box(width, height, size)


def rgb_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def gray_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255.0).unsqueeze(0)


def degrade_image(image: Image.Image) -> Image.Image:
    """模拟线上常见的轻微模糊和 JPEG 压缩，不改变监督 Mask。"""
    if random.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 0.8)))
    if random.random() < 0.35:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(55, 95))
        image = Image.open(buffer).convert("RGB")
    return image


class CLWDDataset(Dataset):
    """读取 CLWD 的真实有水印图与二值 Mask。

    目录约定：
        root/Watermarked_image/<name>.jpg
        root/Mask/<name>.png

    训练阶段执行水印感知裁剪和图像增强；验证阶段执行确定性裁剪。
    返回值为归一化 RGB Tensor [3,H,W] 和二值 Mask Tensor [1,H,W]。
    """

    def __init__(
        self,
        root: Path,
        size: int,
        training: bool,
        positive_crop_probability: float = 0.0,
        negative_crop_probability: float = 0.0,
    ):
        self.root = root
        self.size = size
        self.training = training
        self.positive_crop_probability = positive_crop_probability
        self.negative_crop_probability = negative_crop_probability
        self.images = image_paths(root / "Watermarked_image")
        if not self.images:
            raise FileNotFoundError(f"No CLWD images found in {root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 图像和 Mask 通过相同文件 stem 一一对应。
        image_path = self.images[index]
        mask_path = self.root / "Mask" / f"{image_path.stem}.png"
        images = [
            Image.open(image_path).convert("RGB"),
            Image.open(mask_path).convert("L"),
        ]
        images = resize_for_crop(
            images,
            self.size,
            # RGB 使用双线性插值；Mask 必须使用最近邻，避免产生不存在的灰度标签。
            [Image.Resampling.BILINEAR, Image.Resampling.NEAREST],
        )
        if self.training:
            # 所有配对图像共用一个裁剪框，输出固定为 size x size。
            box = training_crop_box(
                images[1], self.size, self.positive_crop_probability, self.negative_crop_probability
            )
            images = [image.crop(box) for image in images]
        else:
            images = crop_aligned(images, self.size, False, seed=index)
        if self.training and random.random() < 0.5:
            # 翻转同样同步作用于 RGB 和 Mask。
            images = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in images]
        if self.training:
            # 光度与退化增强仅作用于输入图像，不能改变 GT Mask。
            brightness, contrast = random.uniform(0.85, 1.15), random.uniform(0.85, 1.15)
            images[0] = ImageEnhance.Brightness(images[0]).enhance(brightness)
            images[0] = ImageEnhance.Contrast(images[0]).enhance(contrast)
            images[0] = degrade_image(images[0])
        # 即使源 Mask 存在压缩灰度，最终监督也严格二值化为 0/1。
        mask = (gray_tensor(images[1]) > 0.5).float()
        return rgb_tensor(images[0]), mask


def paste_clipped(canvas: Image.Image, mark: Image.Image, left: int, top: int) -> None:
    """将水印贴到画布上，并正确处理超出边界的残缺水印。"""
    source_left, source_top = max(0, -left), max(0, -top)
    destination_left, destination_top = max(0, left), max(0, top)
    width = min(mark.width - source_left, canvas.width - destination_left)
    height = min(mark.height - source_top, canvas.height - destination_top)
    if width > 0 and height > 0:
        piece = mark.crop((source_left, source_top, source_left + width, source_top + height))
        canvas.alpha_composite(piece, (destination_left, destination_top))


class SyntheticWatermarkDataset(Dataset):
    """在线生成“有水印 RGB + 二值 Mask”训练对。

    每次 __getitem__ 都重新随机选择背景、水印、尺度、角度、颜色、透明度和位置，
    因此同一个 index 在不同 epoch 中也会产生不同样本。length 只控制每个 epoch
    生成多少个合成样本，并不对应磁盘上的固定图片数量。
    """

    def __init__(
        self,
        clean_root: Path,
        watermark_root: Path,
        size: int,
        length: int,
    ):
        self.clean = image_paths(clean_root)
        self.watermarks = image_paths(watermark_root)
        self.size = size
        self.length = length
        if not self.clean or not self.watermarks:
            raise FileNotFoundError("Synthetic data requires clean images and watermark assets")

    def __len__(self) -> int:
        return self.length

    def _clean_crop(self, index: int) -> Image.Image:
        # 使用 index 轮询背景图库，再做随机裁剪，避免完全依赖随机抽样造成覆盖不均。
        image = Image.open(self.clean[index % len(self.clean)]).convert("RGB")
        image = resize_for_crop([image], self.size, [Image.Resampling.BICUBIC])[0]
        return crop_aligned([image], self.size, True)[0]

    def _mark(self, scale: float) -> Image.Image:
        """随机生成经过缩放、旋转、变色和透明度调整的 RGBA 水印。"""
        mark = Image.open(random.choice(self.watermarks)).convert("RGBA")
        longest = max(mark.size)
        factor = self.size * scale / longest
        shape = (max(2, round(mark.width * factor)), max(2, round(mark.height * factor)))
        mark = mark.resize(shape, Image.Resampling.LANCZOS)
        mark = mark.rotate(random.uniform(-35, 35), Image.Resampling.BICUBIC, expand=True)
        array = np.asarray(mark).copy()
        if random.random() < 0.45:
            # 随机变色扩大水印颜色分布，同时保留素材原始透明通道。
            color = np.random.randint(0, 256, 3, dtype=np.uint8)
            array[..., :3] = color
        # 45% 样本重点覆盖肉眼较难识别的低透明水印，其余覆盖常规透明度。
        opacity = random.uniform(0.02, 0.30) if random.random() < 0.45 else random.uniform(0.30, 0.90)
        array[..., 3] = (array[..., 3].astype(np.float32) * opacity).astype(np.uint8)
        return Image.fromarray(array, "RGBA")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean = self._clean_crop(index)
        # overlay 保存本次所有水印的 RGB 和透明通道，初始为完全透明。
        overlay = Image.new("RGBA", clean.size)

        # 10% 概率保持空 overlay，显式生成无水印负样本。
        if random.random() >= 0.10:
            # 25% 的正样本使用小水印平铺，其余放置 1~3 个不同大小的水印。
            tiled = random.random() < 0.25
            if tiled:
                mark = self._mark(random.uniform(0.08, 0.22))
                gap_x = mark.width + random.randint(20, max(21, self.size // 5))
                gap_y = mark.height + random.randint(20, max(21, self.size // 5))
                offset_x = random.randint(-gap_x, 0)
                offset_y = random.randint(-gap_y, 0)
                for top in range(offset_y, self.size, gap_y):
                    for left in range(offset_x, self.size, gap_x):
                        paste_clipped(canvas=overlay, mark=mark, left=left, top=top)
            else:
                for _ in range(random.randint(1, 3)):
                    mark = self._mark(random.uniform(0.10, 0.75))
                    # 允许水印部分超出图像边界，模拟线上截断和残缺形态。
                    left = random.randint(-mark.width // 3, self.size - 2 * mark.width // 3)
                    top = random.randint(-mark.height // 3, self.size - 2 * mark.height // 3)
                    paste_clipped(canvas=overlay, mark=mark, left=left, top=top)

        clean_array = np.asarray(clean, dtype=np.float32) / 255.0
        overlay_array = np.asarray(overlay, dtype=np.float32) / 255.0
        # 这里的 alpha 只用于物理混合生成有水印图，不会返回给模型作为监督目标。
        alpha = overlay_array[..., 3:4]
        image = clean_array * (1.0 - alpha) + overlay_array[..., :3] * alpha
        # 只要透明通道非零就属于水印区域，最终输出严格的单通道二值 Mask。
        mask = (alpha > 1.0 / 255.0).astype(np.float32)
        # 混合完成后再退化 RGB，模拟压缩传播；Mask 保持原始几何边界。
        degraded = degrade_image(Image.fromarray((image.clip(0, 1) * 255).astype(np.uint8)))
        return rgb_tensor(degraded), torch.from_numpy(mask.copy()).permute(2, 0, 1)


def mask_losses(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    correct_probability = probability * target + (1.0 - probability) * (1.0 - target)
    focal = (((1.0 - correct_probability) ** 2) * bce).mean()
    intersection = (probability * target).sum((1, 2, 3))
    dice = 1.0 - (
        (2.0 * intersection + 1.0)
        / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1.0)
    ).mean()
    return focal, dice


def loss_function(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_weight: float = 0.0,
    auxiliary: tuple[torch.Tensor, ...] = (),
    auxiliary_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask_target = target[:, :1]
    mask_logits = logits[:, :1]
    probability = mask_logits.sigmoid()
    focal, dice = mask_losses(mask_logits, mask_target)

    dilated = F.max_pool2d(mask_target, 5, stride=1, padding=2)
    eroded = -F.max_pool2d(-mask_target, 5, stride=1, padding=2)
    boundary_region = (dilated - eroded).clamp(0, 1)
    boundary = ((probability - mask_target).abs() * boundary_region).sum()
    boundary = boundary / boundary_region.sum().clamp_min(1.0)

    auxiliary_loss = logits.new_zeros(())
    for auxiliary_logits in auxiliary:
        auxiliary_focal, auxiliary_dice = mask_losses(auxiliary_logits, mask_target)
        auxiliary_loss = auxiliary_loss + auxiliary_focal + auxiliary_dice
    total = focal + dice + boundary_weight * boundary + auxiliary_weight * auxiliary_loss
    return total, {
        "focal": focal.item(),
        "dice": dice.item(),
        "boundary": boundary.item(),
        "aux": auxiliary_loss.item(),
    }


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int,
) -> dict[str, float]:
    model.eval()
    true_positive = false_positive = false_negative = 0.0
    for step, (image, target) in enumerate(loader, 1):
        image, target = image.to(device), target.to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.float16):
            prediction = model(image)[:, :1].sigmoid()
        binary = prediction > 0.5
        truth = target[:, :1] > 0.5
        true_positive += (binary & truth).sum().item()
        false_positive += (binary & ~truth).sum().item()
        false_negative += (~binary & truth).sum().item()
        if max_steps and step >= max_steps:
            break
    iou = true_positive / max(1.0, true_positive + false_positive + false_negative)
    f1 = 2.0 * true_positive / max(1.0, 2.0 * true_positive + false_positive + false_negative)
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train a lightweight standalone watermark mask model")
    parser.add_argument("--clwd-root", type=Path, default=project / "0-数据/data（CLWD 格式）/clwd_crop")
    parser.add_argument("--clean-root", type=Path, default=project / "0-数据/干净图片")
    parser.add_argument("--watermark-root", type=Path, default=project / "0-数据/水印素材/watermark_logo_27kpng/train_images_wm")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs/mask_v1")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--synthetic-length", type=int, default=20000)
    parser.add_argument("--positive-crop-prob", type=float, default=0.7)
    parser.add_argument("--negative-crop-prob", type=float, default=0.1)
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--weights", type=Path, help="Load model weights but start a fresh optimizer/schedule")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument("--check-crops", action="store_true")
    return parser.parse_args()


def check_crop_policy() -> None:
    mask = Image.new("L", (1024, 1024))
    mask.paste(255, (32, 32, 96, 96))
    random.seed(7)
    positive = training_crop_box(mask, 512, 1.0, 0.0)
    negative = training_crop_box(mask, 512, 0.0, 1.0)
    assert mask.crop(positive).getbbox()
    assert not mask.crop(negative).getbbox()
    print(f"positive_box={positive} negative_box={negative}")


def main() -> None:
    args = parse_args()
    if args.check_crops:
        check_crop_policy()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training configuration")
    if args.resume and args.weights:
        raise ValueError("Use either --resume or --weights, not both")
    if not 0.0 <= args.positive_crop_prob <= 1.0 or not 0.0 <= args.negative_crop_prob <= 1.0:
        raise ValueError("Crop probabilities must be between 0 and 1")
    if args.positive_crop_prob + args.negative_crop_prob > 1.0:
        raise ValueError("Positive and negative crop probabilities cannot sum above 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2))

    train_data = ConcatDataset(
        [
            CLWDDataset(
                args.clwd_root / "train",
                args.image_size,
                training=True,
                positive_crop_probability=args.positive_crop_prob,
                negative_crop_probability=args.negative_crop_prob,
            ),
            SyntheticWatermarkDataset(
                args.clean_root,
                args.watermark_root,
                args.image_size,
                args.synthetic_length,
            ),
        ]
    )
    validation_data = CLWDDataset(
        args.clwd_root / "test",
        args.image_size,
        training=False,
    )
    loader_options = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_options)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, **loader_options)

    device = torch.device("cuda")
    model = WatermarkMaskNet().to(device)
    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        weights = checkpoint.get("model", checkpoint)
        current = model.state_dict()
        compatible = {
            name: value
            for name, value in weights.items()
            if name in current and current[name].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
        print(f"transferred {len(compatible)}/{len(current)} tensors from {args.weights}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, best_iou, global_step = 1, -1.0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_iou = checkpoint.get("best_iou", -1.0)
        global_step = checkpoint.get("global_step", 0)

    writer = SummaryWriter(args.output / "tensorboard")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={torch.cuda.get_device_name()} params={parameters:,} train={len(train_data):,} val={len(validation_data):,}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        for step, (image, target) in enumerate(train_loader, 1):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, auxiliary = model(image, return_aux=True)
                loss, parts = loss_function(
                    logits,
                    target,
                    args.boundary_weight,
                    auxiliary,
                    args.aux_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            global_step += 1
            if step % 50 == 0 or step == 1:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running / step:.4f} parts={parts}", flush=True)
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            if args.max_train_steps and step >= args.max_train_steps:
                break

        metrics = validate(model, validation_loader, device, args.max_val_steps)
        scheduler.step()
        for name, value in metrics.items():
            writer.add_scalar(f"validation/{name}", value, epoch)
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_iou": max(best_iou, metrics["iou"]),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        }
        torch.save(state, args.output / "last.pt")
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            torch.save(state, args.output / "best.pt")
        print(
            f"epoch={epoch} metrics={metrics} seconds={time.time() - started:.1f} "
            f"best_iou={best_iou:.4f}",
            flush=True,
        )
    writer.close()


if __name__ == "__main__":
    main()
