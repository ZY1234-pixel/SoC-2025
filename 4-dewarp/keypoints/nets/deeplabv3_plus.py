import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from nets.starnet import StarNetBackbone
from nets.mobilenetv3 import MobileNetV3_Small
from nets.mobilenetv3 import MobileNetV3_Large
class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x 


#-----------------------------------------#
class MobileNetV3(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=False):
        super(MobileNetV3, self).__init__()

        model = MobileNetV3_Large(num_classes=2)

        # -------------------------#
        # 拆 backbone
        # -------------------------#
        self.conv1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.hs1
        )

        self.bneck = model.bneck
        self.conv2 = nn.Sequential(
            model.conv2,
            model.bn2,
            model.hs2
        )

        # -------------------------#
        # 记录stage分界（经验值）
        # -------------------------#
        self.low_level_idx = 2   # 前2个Block输出作为浅层特征
        self.total_idx = len(self.bneck)
        self.down_idx = [1, 3, 6, 12]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.bneck[i].apply(lambda m: self._nostride_dilate(m, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.bneck[i].apply(lambda m: self._nostride_dilate(m, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.bneck[i].apply(lambda m: self._nostride_dilate(m, dilate=2))
        else:
            raise ValueError("MobileNetV3 supports downsample_factor 8 or 16.")

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') == -1:
            return

        kernel_size = getattr(m, 'kernel_size', None)
        stride = getattr(m, 'stride', None)
        if not isinstance(kernel_size, tuple) or kernel_size == (1, 1):
            return

        if stride == (2, 2):
            new_dilate = max(1, dilate // 2)
            m.stride = (1, 1)
            m.dilation = (new_dilate, new_dilate)
            m.padding = ((kernel_size[0] // 2) * new_dilate, (kernel_size[1] // 2) * new_dilate)
        else:
            m.dilation = (dilate, dilate)
            m.padding = ((kernel_size[0] // 2) * dilate, (kernel_size[1] // 2) * dilate)

    def forward(self, x):
        x = self.conv1(x)

        low_level_features = None
        for i, block in enumerate(self.bneck):
            x = block(x)
            if i == self.low_level_idx:
                low_level_features = x

        x = self.conv2(x)

        return low_level_features, x
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#
class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenetv3", pretrained=True, downsample_factor=16, num_keypoints=0):
        super(DeepLab, self).__init__()
        self.num_keypoints = num_keypoints
        if backbone=="xception":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24

        elif backbone == "starnet":
            self.backbone = StarNetBackbone(
                variant="s2",
                pretrained=pretrained,
                downsample_factor=downsample_factor
            )

            in_channels = 256
            low_level_channels = 32
        elif backbone == "mobilenetv3":
            self.backbone = MobileNetV3(
                downsample_factor=downsample_factor,
                pretrained=pretrained
            )
            in_channels = 960  # MobileNetV3-Large conv2 输出通道
            low_level_channels = 24  # 第2个block输出通道
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )		

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
        # 关键点检测头
        if num_keypoints > 0:
            self.kpt_head = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_keypoints, 1)
            )
        else:
            self.kpt_head = None

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        features = self.cat_conv(torch.cat((x, low_level_features), dim=1))

        seg = self.cls_conv(features)
        seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=True)

        if self.num_keypoints > 0 and self.kpt_head is not None:
            kpt = self.kpt_head(features)
            kpt = F.interpolate(kpt, size=(H, W), mode='bilinear', align_corners=True)
            return seg, kpt
        else:
            return seg

class DeepLabKpt(nn.Module):
    def __init__(self, num_keypoints=4, backbone="mobilenetv3", pretrained=False, downsample_factor=8):
        super().__init__()
        self.num_keypoints = num_keypoints
        if backbone == "mobilenetv3":
            self.backbone = MobileNetV3(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 960
            low_level_channels = 24
        else:
            raise ValueError("仅支持 mobilenetv3")

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.kpt_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints, 1)
        )

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)),
                          mode='bilinear', align_corners=True)
        features = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        kpt = self.kpt_head(features)
        kpt = F.interpolate(kpt, size=(H, W), mode='bilinear', align_corners=True)
        kpt = torch.sigmoid(kpt)   # 重要：输出 0~1
        return kpt
