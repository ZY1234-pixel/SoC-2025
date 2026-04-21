"""
平衡训练脚本：从检查点继续训练，增加灰度图像在训练中的权重
用于改善模型对灰度图像的处理效果
"""
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from dataLoader import DirtyDocumentsDataset
from model2 import DecontaminationModel
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim import Adam
from torchvision.utils import make_grid
import random
import torch.multiprocessing as mp
import cv2
import glob

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='图像去污模型平衡训练（增强灰度图像）')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小（GPU内存不足时可减小）')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, 
                       help='梯度累积步数（实际batch_size = batch_size * gradient_accumulation_steps）')
    parser.add_argument('--lr', type=float, default=5e-4, help='初始学习率（建议比初始训练小）')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载器工作进程数')
    parser.add_argument('--logdir', type=str, default='./runs', help='TensorBoard日志目录')
    parser.add_argument('--grayscale_weight', type=float, default=20.0, 
                       help='灰度图像采样权重（相对于彩色图像，默认3.0表示灰度图像出现频率是彩色的3倍）')
    parser.add_argument('--checkpoint_path', type=str, 
                       default='./checkpoints/model2.pth',
                       help='检查点路径')
    parser.add_argument('--best_model_path', type=str,
                       default='./checkpoints/best_model2.pth',
                       help='最佳模型路径（用于加载初始权重）')
    return parser.parse_args()

# YUV转RGB辅助函数（从train.py复制）
def yuv_to_rgb(yuv_tensor):
    yuv = yuv_tensor.permute(1, 2, 0).numpy()
    uv_norm = np.sqrt((yuv[:, :, 1] - 0.5)**2 + (yuv[:, :, 2] - 0.5)**2).mean()
    if uv_norm < 0.01:
        Y = yuv[:, :, 0]
        rgb = np.stack([Y, Y, Y], axis=-1)
    else:
        yuv_uint8 = (yuv * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(yuv_uint8, cv2.COLOR_YUV2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype("float32") / 255.0
    rgb_tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    return rgb_tensor

# 检测图像是否为灰度图像
def is_grayscale_image(img_path):
    """
    检测图像是否为灰度图像（3通道伪彩色图）
    通过检查YUV颜色空间中的UV通道来判断
    
    灰度图像特征：UV通道的所有像素值都应该等于128/255（即0.5）
    使用高效的向量化操作验证：
    1. 检查U和V通道的最小值和最大值是否都接近0.5（最快方法）
    2. 同时检查U和V通道是否相同
    
    效率说明：
    - Python循环：O(n)，每个像素执行Python解释器指令，非常慢（~1000ms for 1MP图像）
    - NumPy向量化：O(n)，C实现，比Python循环快100-1000倍（~1-10ms for 1MP图像）
    - min/max方法：只需2次遍历（找最小最大值），比全量比较更快（~0.5-2ms for 1MP图像）
    """
    img = cv2.imread(img_path)
    if img is None:
        return False
    
    # 转换为YUV颜色空间
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv = img_yuv.astype("float32") / 255.0
        
        # 提取U和V通道
        u_channel = img_yuv[:, :, 1]  # U通道
        v_channel = img_yuv[:, :, 2]  # V通道
        
        # 使用min/max检查（最高效）
        # 如果所有像素值都等于0.5，那么min和max都应该等于0.5
        # 允许小的浮点误差（约1/255的容差）
        tolerance = 1.0 / 255.0  # 约0.004，允许1个像素值的误差
        
        u_min, u_max = u_channel.min(), u_channel.max()
        v_min, v_max = v_channel.min(), v_channel.max()
        
        # 检查U通道是否所有值都接近0.5
        u_is_constant = (np.abs(u_min - 0.5) < tolerance and 
                        np.abs(u_max - 0.5) < tolerance)
        
        # 检查V通道是否所有值都接近0.5
        v_is_constant = (np.abs(v_min - 0.5) < tolerance and 
                        np.abs(v_max - 0.5) < tolerance)
        
        # 检查U和V通道是否相同（灰度图像的U和V应该相同）
        uv_diff_max = np.abs(u_channel - v_channel).max()
        uv_are_same = uv_diff_max < tolerance
        
        is_grayscale = u_is_constant and v_is_constant and uv_are_same
        return is_grayscale
    else:
        # 单通道图像直接认为是灰度图
        return True

# 计算数据集权重和任务ID
def calculate_dataset_weights(dataset, grayscale_weight=3.0):
    """
    计算数据集中每个样本的权重和任务ID
    参数:
        dataset: DirtyDocumentsDataset实例
        grayscale_weight: 灰度图像的权重（相对于彩色图像）
    
    返回:
        weights: 权重列表
        task_ids: 任务ID列表（0=彩色，1=灰度）
        grayscale_count: 灰度图像数量
        color_count: 彩色图像数量
    """
    # 获取所有图像文件路径
    clean_files = dataset.clean_files
    
    # 为每个样本分配权重和任务ID
    weights = []
    task_ids = []
    grayscale_count = 0
    color_count = 0
    
    print("正在分析数据集，计算采样权重和任务ID...")
    for i, clean_path in enumerate(tqdm(clean_files, desc="分析图像")):
        is_gray = is_grayscale_image(clean_path)
        if is_gray:
            weights.append(grayscale_weight)
            task_ids.append(1)  # 灰度图像
            grayscale_count += 1
        else:
            weights.append(1.0)
            task_ids.append(0)  # 彩色图像
            color_count += 1
    
    # 如果有数据增强，需要为增强样本也分配权重和任务ID
    if dataset.apply_augmentation:
        num_augmentations = len(dataset.augmentations)
        original_size = len(clean_files)
        # 为每个原始图像的所有增强版本分配相同的权重和任务ID
        expanded_weights = []
        expanded_task_ids = []
        for i in range(original_size):
            base_weight = weights[i]
            base_task_id = task_ids[i]
            # 原始图像 + 所有增强版本
            expanded_weights.extend([base_weight] * (1 + num_augmentations))
            expanded_task_ids.extend([base_task_id] * (1 + num_augmentations))
        weights = expanded_weights
        task_ids = expanded_task_ids
    
    print(f"数据集统计: 彩色图像 {color_count} 张, 灰度图像 {grayscale_count} 张")
    print(f"采样权重: 彩色图像=1.0, 灰度图像={grayscale_weight}")
    if grayscale_count + color_count > 0:
        total_weight = color_count * 1.0 + grayscale_count * grayscale_weight
        print(f"预期采样比例: 彩色图像={color_count*1.0/total_weight*100:.1f}%, "
              f"灰度图像={grayscale_count*grayscale_weight/total_weight*100:.1f}%")
    return weights, task_ids, grayscale_count, color_count

# 包装数据集类，添加任务ID
class TaskAwareDataset(Dataset):
    """
    包装数据集，在返回数据时包含任务ID
    """
    def __init__(self, base_dataset, task_ids):
        """
        参数:
            base_dataset: 基础数据集（DirtyDocumentsDataset或Subset）
            task_ids: 任务ID列表，与数据集索引对应
        """
        self.base_dataset = base_dataset
        self.task_ids = task_ids
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        dirty_img, clean_img = self.base_dataset[idx]
        task_id = self.task_ids[idx]
        return dirty_img, clean_img, task_id

# SSIM损失函数（从train.py复制）
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = None

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if self.window is not None and channel == self.channel and self.window.is_cuda == img1.is_cuda:
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.device)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

# 训练函数（支持梯度累积和内存优化）
def train_one_epoch(model, train_loader, optimizer, device, gradient_accumulation_steps=1):
    model.train()
    epoch_loss = 0
    batch_count = 0
    optimizer.zero_grad()  # 在epoch开始时清零梯度
    
    with tqdm(train_loader, desc="Training") as pbar:
        for step, batch in enumerate(pbar):
            if len(batch) == 3:
                dirty_imgs, clean_imgs, task_ids = batch
            elif len(batch) == 2:
                dirty_imgs, clean_imgs = batch
                task_ids = None  # 如果没有提供task_id，使用自动检测
            else:
                print("警告: 训练集返回格式不符合预期")
                continue
                
            dirty_imgs = dirty_imgs.to(device, non_blocking=True)
            clean_imgs = clean_imgs.to(device, non_blocking=True)
            
            # 将task_ids传递到设备（DataLoader已自动堆叠为张量）
            if task_ids is not None:
                task_ids = task_ids.to(device, non_blocking=True)
            
            # 显式传递task_id给模型
            outputs = model(dirty_imgs, task_id=task_ids)
            
            mse = mse_loss(outputs, clean_imgs)
            ssim = ssim_loss(outputs, clean_imgs)
            loss = 0.7 * mse + 0.3 * ssim
            
            # 梯度累积：将loss除以累积步数
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # 每gradient_accumulation_steps步或最后一步，执行优化
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            batch_count += 1
            epoch_loss += loss.item() * gradient_accumulation_steps  # 恢复原始loss值用于显示
            pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})
            
            # 定期清理GPU缓存
            if (step + 1) % 10 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return epoch_loss / batch_count if batch_count > 0 else 0

# PSNR计算函数
def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    return psnr.item()

# 验证函数（内存优化版本）
def validate(model, val_loader, device):
    model.eval()
    val_loss = 0
    total_psnr = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if len(batch) == 3:
                dirty_imgs, clean_imgs, task_ids = batch
            elif len(batch) == 2:
                dirty_imgs, clean_imgs = batch
                task_ids = None  # 如果没有提供task_id，使用自动检测
            else:
                print("警告: 验证集返回格式不符合预期")
                continue
                
            dirty_imgs = dirty_imgs.to(device, non_blocking=True)
            clean_imgs = clean_imgs.to(device, non_blocking=True)
            
            # 将task_ids传递到设备（DataLoader已自动堆叠为张量）
            if task_ids is not None:
                task_ids = task_ids.to(device, non_blocking=True)
            
            # 显式传递task_id给模型
            outputs = model(dirty_imgs, task_id=task_ids)
            
            mse = mse_loss(outputs, clean_imgs)
            ssim = ssim_loss(outputs, clean_imgs)
            loss = 0.7 * mse + 0.3 * ssim
            
            psnr = calculate_psnr(outputs, clean_imgs)
            
            batch_count += 1
            val_loss += loss.item()
            total_psnr += psnr
            
            # 定期清理GPU缓存
            if (batch_idx + 1) % 5 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
    
    avg_val_loss = val_loss / batch_count if batch_count > 0 else 0
    avg_psnr = total_psnr / batch_count if batch_count > 0 else 0
    
    # 验证结束后清理缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return avg_val_loss, avg_psnr

# 主训练逻辑
def main():
    set_seed()
    args = parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"\n=== 平衡训练配置 ===")
    print(f"灰度图像采样权重: {args.grayscale_weight}x")
    print(f"学习率: {args.lr}")
    print(f"批次大小: {args.batch_size}")
    print(f"梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"有效批次大小: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"训练轮数: {args.epochs}")
    
    # 设置路径
    dirty_dirs = '../data/dataset/input'
    clean_dirs = '../data/dataset/target'
    
    os.makedirs('./model2/checkpoints/', exist_ok=True)
    checkpoint_path = args.checkpoint_path
    best_model_path = os.path.join('./model2/checkpoints/best_decontamination_model_balanced.pth')
    
    # 加载数据集
    print("\n正在加载数据集...")
    dataset = DirtyDocumentsDataset(clean_dirs, dirty_dirs, apply_augmentation=True)
    
    # 计算数据集权重和任务ID（在划分数据集之前）
    print("\n计算数据集权重和任务ID...")
    full_weights, full_task_ids, grayscale_count, color_count = calculate_dataset_weights(
        dataset, args.grayscale_weight
    )
    # 划分训练集和验证集
    dataset_size = len(dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(dataset_size), [train_size, val_size]
    )
    train_indices = train_indices.indices
    val_indices = val_indices.indices
    
    # 创建子集数据集
    train_base_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_base_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # 为训练集和验证集提取对应的任务ID
    train_task_ids = [full_task_ids[i] for i in train_indices]
    val_task_ids = [full_task_ids[i] for i in val_indices]
    
    # 创建包装数据集（包含任务ID）
    train_dataset = TaskAwareDataset(train_base_dataset, train_task_ids)
    val_dataset = TaskAwareDataset(val_base_dataset, val_task_ids)
    
    # 为训练集创建加权采样器（只采样训练集的索引）
    train_weights = [full_weights[i] for i in train_indices]
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_indices),
        replacement=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,  # 使用加权采样器，不能同时使用shuffle
        num_workers=args.num_workers
    )
    # 验证集使用更小的batch_size以节省内存
    val_batch_size = max(1, args.batch_size // 2)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")    
    # 初始化模型
    model = DecontaminationModel(in_channels=3, out_channels=3).to(device)
    
    # 定义损失函数
    global mse_loss, ssim_loss
    mse_loss = nn.MSELoss()
    ssim_loss = SSIMLoss()
    
    # 优化器和学习率调度器（使用较小的学习率）
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练状态初始化
    start_epoch = 0
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # 加载检查点或最佳模型
    if args.resume and os.path.exists(checkpoint_path):
        print(f"\n正在从检查点加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"已从第 {start_epoch-1} 轮恢复训练")
    elif os.path.exists(args.best_model_path):
        print(f"\n正在加载最佳模型权重: {args.best_model_path}")
        checkpoint = torch.load(args.best_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("已加载最佳模型权重，开始平衡训练")
    else:
        print("\n警告: 未找到检查点或最佳模型，将从随机初始化开始训练")
    
    # 训练循环
    print(f"\n开始训练，共 {args.epochs} 轮...")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, 
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        # 每个epoch后清理GPU缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        val_loss, val_psnr = validate(model, val_loader, device)
        scheduler.step(val_loss)
        
        print(f"轮次 {epoch+1}/{args.epochs} - 训练损失: {train_loss:.4f}, "
              f"验证损失: {val_loss:.4f}, PSNR: {val_psnr:.2f} dB, "
              f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存当前模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            early_stop_counter = 0
            print(f"✓ 新的最佳模型已保存！验证损失: {val_loss:.4f}")
        else:
            early_stop_counter += 1
        
        # 早停检查
        if early_stop_counter >= args.patience:
            print(f"\n验证损失 {args.patience} 轮未改善，提前停止训练")
            break
    
    print("\n训练完成！")
    print(f"最佳模型已保存至: {best_model_path}")

if __name__ == '__main__':
    mp.freeze_support()
    main()

