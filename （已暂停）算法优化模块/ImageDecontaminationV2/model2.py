"""
模型定义：图像去污模型2（在model1基础上增加支持条件任务编码）
包含 Encoder、Decoder 和 DecontaminationModel 类
"""
from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
                
    def forward(self, x):
        # 编码器第一层
        enc1 = self.enc1(x)
        p1 = self.pool1(enc1)
        
        # 编码器第二层
        enc2 = self.enc2(p1)
        p2 = self.pool2(enc2)
        
        # 编码器第三层
        enc3 = self.enc3(p2)
        p3 = self.pool3(enc3)
        
        # 瓶颈层
        bottleneck = self.bottleneck(p3)
        
        return bottleneck, enc3, enc2, enc1


class Decoder(nn.Module):
    def __init__(self, out_channels=1, use_conditional=False):
        super(Decoder, self).__init__()
        self.use_conditional = use_conditional
        
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 256 = 128 + 128 (skip connection)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 128 = 64 + 64 (skip connection)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 64 = 32 + 32 (skip connection)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(32, out_channels, 1),
            nn.Sigmoid()  # 输出YUV通道，使用Sigmoid确保值在[0,1]范围内
        )
        
        # FiLM调制层：为每个解码层的通道数预定义gamma和beta生成器
        if use_conditional:
            # dec1输出通道数: 128
            self.film_gamma_1 = nn.Linear(128, 128)
            self.film_beta_1 = nn.Linear(128, 128)
            # dec2输出通道数: 64
            self.film_gamma_2 = nn.Linear(128, 64)
            self.film_beta_2 = nn.Linear(128, 64)
            # dec3输出通道数: 32
            self.film_gamma_3 = nn.Linear(128, 32)
            self.film_beta_3 = nn.Linear(128, 32)

    def forward(self, bottleneck, skip3, skip2, skip1, cond_params=None):
        """
        前向传播
        
        参数:
            bottleneck: 瓶颈层特征 [B, 256, H/8, W/8]
            skip3: 跳跃连接3 [B, 128, H/4, W/4]
            skip2: 跳跃连接2 [B, 64, H/2, W/2]
            skip1: 跳跃连接1 [B, 32, H, W]
            cond_params: 条件参数列表，每个元素为 [B, 128] 的向量
        """
        # 第一次上采样 + 跳跃连接
        up1 = self.upconv1(bottleneck)
        merge1 = torch.cat([up1, skip3], dim=1)
        dec1 = self.dec1(merge1)
        
        # 如果使用条件参数，应用FiLM调制
        if self.use_conditional and cond_params is not None and len(cond_params) > 0:
            dec1 = self._apply_film(dec1, cond_params[0], layer_idx=1)
        
        # 第二次上采样 + 跳跃连接
        up2 = self.upconv2(dec1)
        merge2 = torch.cat([up2, skip2], dim=1)
        dec2 = self.dec2(merge2)
        
        # 应用条件参数
        if self.use_conditional and cond_params is not None and len(cond_params) > 1:
            dec2 = self._apply_film(dec2, cond_params[1], layer_idx=2)
        
        # 第三次上采样 + 跳跃连接
        up3 = self.upconv3(dec2)
        merge3 = torch.cat([up3, skip1], dim=1)
        dec3 = self.dec3(merge3)
        
        # 应用条件参数
        if self.use_conditional and cond_params is not None and len(cond_params) > 2:
            dec3 = self._apply_film(dec3, cond_params[2], layer_idx=3)
        
        # 输出层
        output = self.output(dec3)
        
        return output
    
    def _apply_film(self, feature, cond_param, layer_idx):
        """
        应用FiLM (Feature-wise Linear Modulation) 条件注入
        
        参数:
            feature: 特征图 [B, C, H, W]
            cond_param: 条件参数 [B, 128]
            layer_idx: 解码层索引 (1, 2, 或 3)
        
        返回:
            调制后的特征图 [B, C, H, W]
        """
        # 根据层索引选择对应的gamma和beta生成器
        if layer_idx == 1:
            gamma = self.film_gamma_1(cond_param)  # [B, 128]
            beta = self.film_beta_1(cond_param)    # [B, 128]
        elif layer_idx == 2:
            gamma = self.film_gamma_2(cond_param)  # [B, 64]
            beta = self.film_beta_2(cond_param)     # [B, 64]
        elif layer_idx == 3:
            gamma = self.film_gamma_3(cond_param)  # [B, 32]
            beta = self.film_beta_3(cond_param)    # [B, 32]
        else:
            raise ValueError(f"Invalid layer_idx: {layer_idx}")
        
        # 扩展维度以匹配特征图 [B, C, 1, 1]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        # 应用FiLM: gamma * feature + beta
        modulated = gamma * feature + beta
        
        return modulated


class DecontaminationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels, use_conditional=True)
        
        # 任务嵌入：0=彩色, 1=灰度
        self.task_embedding = nn.Embedding(2, 64)
        
        # 条件融合层（在每个解码块前注入任务信息）
        # 为每个解码层生成条件参数，输出维度128用于FiLM调制
        self.task_fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU()
            ) for _ in range(3)  # 对应3个解码层
        ])
    
    def forward(self, x, task_id=None):
        """
        前向传播
        
        参数:
            x: 输入图像 [B, C, H, W]，C=3 (YUV格式)
            task_id: 任务ID，0=彩色图像，1=灰度图像。如果为None，则自动检测
        
        返回:
            输出图像 [B, C, H, W]
        """
        # 自动检测任务类型（根据UV通道是否接近0或0.5）
        if task_id is None:
            # 计算每个样本的UV通道的平均绝对值 [B]
            if x.shape[1] >= 2:
                uv_mean = x[:, 1:, :, :].abs().mean(dim=(1, 2, 3))  # [B] - 每个样本的平均值
                # 检测UV通道是否接近0（真实灰度）或接近0.5（伪彩色灰度）
                # 0.5对应YUV中的中性色（128/255），也是灰度图像的特征
                uv_deviation_from_neutral = (uv_mean - 0.5).abs()
                is_grayscale = (uv_mean < 0.05) | (uv_deviation_from_neutral < 0.05)
                task_id = torch.where(is_grayscale, 
                                     torch.tensor(1, device=x.device, dtype=torch.long),
                                     torch.tensor(0, device=x.device, dtype=torch.long))
            else:
                # 如果输入通道数小于2，默认为灰度图像
                task_id = torch.tensor(1, device=x.device, dtype=torch.long)
        
        # 确保task_id是标量或与batch size匹配
        if task_id.dim() == 0:
            task_id = task_id.unsqueeze(0).expand(x.shape[0])
        elif task_id.shape[0] != x.shape[0]:
            task_id = task_id[0].expand(x.shape[0])
            
        # 获取任务嵌入
        task_emb = self.task_embedding(task_id)  # [B, 64]
        
        # 编码器部分
        bottleneck, skip3, skip2, skip1 = self.encoder(x)
        
        # 在解码时注入任务信息
        output = self._conditional_decode(
            bottleneck, skip3, skip2, skip1, task_emb
        )
        return output
    
    def _conditional_decode(self, bottleneck, skip3, skip2, skip1, task_emb):
        """
        条件解码：在解码过程中注入任务信息
        
        参数:
            bottleneck: 瓶颈层特征 [B, 256, H/8, W/8]
            skip3: 跳跃连接3 [B, 128, H/4, W/4]
            skip2: 跳跃连接2 [B, 64, H/2, W/2]
            skip1: 跳跃连接1 [B, 32, H, W]
            task_emb: 任务嵌入向量 [B, 64]
        
        返回:
            输出图像 [B, C, H, W]
        """
        # 为每个解码层生成条件参数
        cond_params = []
        for fusion_layer in self.task_fusion_layers:
            cond = fusion_layer(task_emb)  # [B, 128]
            cond_params.append(cond)
        
        # 调用支持条件参数的解码器
        output = self.decoder(bottleneck, skip3, skip2, skip1, cond_params)
        
        return output

