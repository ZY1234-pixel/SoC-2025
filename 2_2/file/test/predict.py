# 代码示例
# python predict.py [src_image_dir] [results]

import os
import sys
import glob
import json
import cv2
import numpy as np
import paddle
from paddle import nn

# paddle.summary(UNet(), (1, 3, 600, 600))
import paddle.nn.functional as F
import numpy as np
from paddle.autograd import PyLayer
import numbers

class Identity(nn.Layer):
    def __init_(self):
        super().__init__()

    def forward(self, x):
        return x

def to_3d(x):
    b, c, h, w = x.shape
    x = paddle.reshape(x, [b, c, h * w])
    x = paddle.transpose(x, [0, 2, 1])
    return x
    # return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    b, hw, c = x.shape
    x = paddle.reshape(x, [b, h, w, c])
    x = paddle.transpose(x, [0, 3, 1, 2])
    return x
    # return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Layer):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        # normalized_shape = [normalized_shape]

        assert len(normalized_shape) == 1

        # self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.weight = paddle.create_parameter(shape=normalized_shape,dtype='float32',
                                              default_initializer=nn.initializer.Constant(1.0))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / paddle.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Layer):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        # normalized_shape = normalized_shape.shape

        assert len(normalized_shape) == 1

        # self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.weight = paddle.create_parameter(shape=normalized_shape,dtype='float32',
                                              default_initializer=nn.initializer.Constant(1.0))
        # self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.bias = paddle.create_parameter(shape=normalized_shape,dtype='float32',
                                              default_initializer=nn.initializer.Constant(0.0))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / paddle.sqrt(sigma+1e-6) * self.weight + self.bias


class LayerNorm(nn.Layer):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SimpleGate(nn.Layer):
    def forward(self, x):
        x1, x2 = x.chunk(2, axis=1)
        return x1 * x2


class NAFBlock(nn.Layer):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2D(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias_attr=True)
        self.conv2 = nn.Conv2D(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias_attr=True)
        self.conv3 = nn.Conv2D(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias_attr=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias_attr=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2D(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias_attr=True)
        self.conv5 = nn.Conv2D(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias_attr=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else Identity()

        self.beta = paddle.create_parameter(shape=[1, c, 1, 1],
                        dtype='float32',
                        default_initializer=paddle.nn.initializer.Assign(paddle.zeros([1, c, 1, 1])))
        self.gamma = paddle.create_parameter(shape=[1, c, 1, 1],
                        dtype='float32',
                        default_initializer=paddle.nn.initializer.Assign(paddle.zeros([1, c, 1, 1])))

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Layer):

    def __init__(self, img_channel=3, width=12, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2D(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias_attr=True)
        self.ending = nn.Conv2D(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias_attr=True)

        self.encoders = nn.LayerList()
        self.decoders = nn.LayerList()
        self.middle_blks = nn.LayerList()
        self.ups = nn.LayerList()
        self.downs = nn.LayerList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2D(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2D(chan, chan * 2, 1, bias_attr=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = paddle.rand(train_size)
        with paddle.no_grad():
            self.forward(imgs)


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 2.5), int(W * 2.5))

        self.eval()
        with paddle.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class AvgPool2d(nn.Layer):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.shape[-2] and self.kernel_size[1] >= x.shape[-1]:
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(axis=-1).cumsum(axis=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = paddle.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(axis=-1).cumsum(axis=-2)
            s = paddle.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = paddle.nn.functional.pad(out, pad2d, mode='replicate')

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound Layer, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2D):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m._output_size == 1
            setattr(model, n, pool)


#################

def clip_img(img):
    h, w, c = img.shape
    img_top = img[0:int(h/2), ...]
    img_bot = img[int(h/2):h, ...]

    img_ltop = img_top[:,0:int(w/2), ...]
    img_rtop = img_top[:,int(w/2):w, ...]
    img_lbot = img_bot[:,0:int(w/2), ...]
    img_rbot = img_bot[:,int(w/2):w, ...]

    return [img_ltop, img_rtop, img_lbot, img_rbot]

def concat_img(patch_list):
    img_ltop, img_rtop, img_lbot, img_rbot = patch_list
    re_img_top = np.concatenate((img_ltop, img_rtop), 1)
    re_img_bot = np.concatenate((img_lbot, img_rbot), 1)
    re_img = np.concatenate((re_img_top, re_img_bot), 0)

    return re_img

def process(src_image_dir, save_dir):
    model = NAFNetLocal(img_channel=3, width=24, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 10], dec_blk_nums=[1, 1, 1, 1])
    param_dict = paddle.load('file/test/model.pdparams')
    model.load_dict(param_dict)
    # model = FBCNN()
    image_paths = glob.glob(os.path.join(src_image_dir, "*.png"))
    with paddle.no_grad():
        for idx, image_path in enumerate(image_paths):
            print('The {} img'.format(idx))
            print(image_path)
            # do something
            img = cv2.imread(image_path)
            h, w, c = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_patches = clip_img(img)
            pre_patches = []
            for img in img_patches:
                img = img.transpose((2,0,1))
                img = img/255
                img = paddle.to_tensor(img).astype('float32')
                img = img.reshape([1]+img.shape)
                # print(img.shape)
                pre = model(img)[0].numpy()
                pre = pre.squeeze()
                # pre[pre>0.9]=1
                # pre[pre<0.1]=0
                pre = pre*255.
                pre = pre.transpose((1,2,0))
                pre_patches.append(pre)
            out_image = concat_img(pre_patches)
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)

            # 保存结果图片
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, out_image)
        

if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    process(src_image_dir, save_dir)


