"""
Official Code Implementation of:
"D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping"
"""

import argparse
import sys
import time

import cv2
import glob
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
from d2dewarp.networks.d2dewarp_model import *

from PIL import Image
from vis_utils import *
from d2dewarp.loader.dataset_doc3d_grid_HV import gradient
from d2dewarp.predict_utils.draw_grid import *
from d2dewarp.predict_utils.edge_util import *
from d2dewarp.predict_utils.dewarp_core import dewarp_document
#------------------------------------------------------------------------------#
#   text_seg：文本 h / v 特征分割模型（UNet）
#   text_seg/predict.py 依赖其自身目录下的 nets / utils 包，因此这里先把 text_seg
#   加入 sys.path，之后即可直接复用它的 Unet 封装与前后处理工具函数。
#------------------------------------------------------------------------------#
TEXT_SEG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'text_seg')
if TEXT_SEG_DIR not in sys.path:
    sys.path.insert(0, TEXT_SEG_DIR)

from text_seg.unet import Unet as TextSegUnet                              # noqa: E402
from text_seg.utils.utils import cvtColor, resize_image, preprocess_input  # noqa: E402

#------------------------------------------------------------------------------#
#   internimage-l-instance：文档区域实例分割模型（InternImage-L + MaskDINO）
#   用来在线预测文档区域"实心掩码"，替代原来从磁盘读取的固定边缘掩码。
#   该模型输出的是实心区域（文档内部像素为 255），下游 predict 再把它处理成
#   文档边缘掩码（边界一圈 255）后接入原有的形变矫正流程。
#------------------------------------------------------------------------------#
INSTANCE_SEG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'internimage-l-instance')
if INSTANCE_SEG_DIR not in sys.path:
    sys.path.insert(0, INSTANCE_SEG_DIR)

from internimage_l_instance.instance_seg.config import load_config                              # noqa: E402
from internimage_l_instance.instance_seg.runtime import load_model, filter_and_sort_instances     # noqa: E402
from internimage_l_instance.instance_seg.preprocessing import DocLetterboxDatasetMapper          # noqa: E402
from internimage_l_instance.instance_seg import settings as seg_settings                         # noqa: E402


class TextSegHVPredictor(TextSegUnet):
    """在 text_seg 的 Unet 之上增加“直接返回类别图”的接口。

    前向流程与 text_seg/predict.py -> Unet.detect_image 完全一致：
    转 RGB -> 不失真（letterbox）resize -> /255 -> softmax -> 裁掉灰条
    -> resize 回原图尺寸 -> argmax。
    区别仅在于这里不做可视化 / 落盘，而是把每个像素的类别编号返回
    （0=background, 1=horizon_line, 2=vertical_line），
    供 D2DewarpModel_my 直接当作 h / v 特征图使用。
    """

    def predict_pr(self, image):
        """返回与原图同尺寸的类别图 (H, W)，取值 0 / 1 / 2。"""
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   resize 回原图尺寸后取出每一个像素点的种类
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
        return pr

    def predict_hv_mask(self, image):
        """返回 (h_mask, v_mask)：uint8，(H, W)，文本行 / 竖线像素为 255，其余为 0。"""
        pr      = self.predict_pr(image)
        h_mask  = (pr == 1).astype(np.uint8) * 255
        v_mask  = (pr == 2).astype(np.uint8) * 255
        return h_mask, v_mask


class DocRegionPredictor:
    """用 internimage-l-instance 的 InternImage-L + MaskDINO 在线预测文档区域。

    前向流程与 instance_seg/pipeline.py 的 Stage-1 完全一致：
    转 RGB -> 不失真（letterbox）resize 到 1024 -> 送入模型 -> 取回原图分辨率的
    逐实例掩码 -> 按置信度排序、合并为整幅文档的实心区域掩码。

    返回的是「实心区域掩码」（文档内部像素为 255，背景为 0），它的内部是填满的；
    predict 里再调用 solid_region_to_edge 把它处理成文档边缘掩码（边界一圈 255），
    即可无缝接入原有的 homography 第一次矫正 / 掩码生成 / prune / snap 流程。
    """

    def __init__(self, cuda=True):
        self.cuda = cuda
        self.score_threshold = seg_settings.DEFAULT_SCORE_THRESHOLD
        self.max_detections = seg_settings.DEFAULT_MAX_DETECTIONS
        self.cfg = load_config(
            seg_settings.DEFAULT_CONFIG,
            [
                "MODEL.WEIGHTS", str(seg_settings.DEFAULT_CHECKPOINT),
                "TEST.DETECTIONS_PER_IMAGE", str(self.max_detections),
            ],
        )
        self.model = load_model(self.cfg, seg_settings.DEFAULT_CHECKPOINT)
        self.mapper = DocLetterboxDatasetMapper(
            is_train=False,
            image_size=self.cfg.INPUT.IMAGE_SIZE,
            image_format=self.cfg.INPUT.FORMAT,
            random_flip=False,
        )
        print(f'doc region model loaded from {seg_settings.DEFAULT_CHECKPOINT}')

    def predict(self, image_rgb):
        """返回与原图同尺寸的文档区域实心掩码 (H, W)，uint8，内部 255 / 背景 0。

        Args:
            image_rgb: uint8 (H, W, 3)，RGB 顺序（与 PIL Image.open(...).convert("RGB") 一致）。
        """
        h, w = image_rgb.shape[:2]
        record = self.mapper(
            {
                "file_name": "<mem>",
                "height": h,
                "width": w,
                "image_id": 0,
                "_preloaded_image": image_rgb,
            }
        )
        self.model.eval()
        with torch.inference_mode(), torch.cuda.amp.autocast(
            enabled=self.cuda, cache_enabled=False
        ):
            instances = self.model([record])[0]["instances"].to("cpu")
        instances = filter_and_sort_instances(instances, self.score_threshold)

        solid = np.zeros((h, w), dtype=np.uint8)
        if len(instances) == 0:
            return solid
        for mask_tensor in instances.pred_masks:
            # MaskDINO 后处理已把掩码还原到原图分辨率；偶尔因整数取整差 1px，安全对齐。
            m = mask_tensor.numpy().astype(np.uint8) * 255
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            solid = np.maximum(solid, m)
        return solid


def solid_region_to_edge(solid_mask, thickness=2):
    """由文档区域实心掩码（内部 255）得到文档边缘掩码（边界一圈 255，其余 0）。

    做法：先保留最大连通域（剔除碎裂的噪声实例），再取该实心区域的最大外轮廓，
    重画成一条干净的闭合文档边界线。这条轮廓既能让下游 create_document_mask
    完美回填成实心文档掩码，也能让 fit_document_quad 稳健拟合文档四边形——
    比「实心区域减腐蚀」更稳，不会因模型边界存在缝隙、或文档几乎铺满画幅贴到
    图像边缘而退化。

    该边缘掩码与原先磁盘读取的固定边缘掩码语义一致，可直接用于 homography 第一次
    矫正、create_document_mask、prune / snap 等下游流程。

    Args:
        solid_mask: uint8 (H, W)，文档区域内部为 255，背景为 0（或任意 0/255 掩码）。
        thickness: 重画的文档边界线宽度（像素），默认 2。
    """
    solid = (np.asarray(solid_mask) > 127).astype(np.uint8)
    if solid.sum() == 0:
        return np.zeros_like(solid_mask, dtype=np.uint8)
    # 仅保留最大连通域，剔除碎裂的噪声实例
    num, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        solid = (labels == largest).astype(np.uint8)
    # 取最大外轮廓，重画成一条干净的闭合文档边界线
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(solid_mask, dtype=np.uint8)
    edge = np.zeros_like(solid_mask, dtype=np.uint8)
    cv2.drawContours(edge, contours, -1, 255, thickness=max(1, int(thickness)))
    return edge


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', default=448, help='image size')
    parser.add_argument('--model_path',
                        default='output/d2dewarp_add_edge_mask_448_uvfinal/200.pt',
                        help='model path')
    parser.add_argument('--img_path', default='test_common_test_dataset/img/perspective',
                        help='image path or path to folder containing images (set multi as true)')
    parser.add_argument('--save_path', default='test_common_test_dataset/dewarp', help='save path')
    parser.add_argument('--prune_bm', type=str2bool, default=False,
                        help='删除 bm0/bm1 中落到文档边缘之外的采样点，并把剩余有效区域重参数化'
                             '（拉伸）到整幅输出图，使矫正后的文档边缘与图像边缘完全贴合')
    parser.add_argument('--prune_smooth', type=int, default=9,
                        help='重参数化时行/列有效边界的平滑窗口（奇数），越大越平滑，默认 9')
    parser.add_argument('--fit_bm', type=str2bool, default=True,
                        help='把 bm 的采样区域（bm 的像）拉伸到整幅输入图，使形变网格铺满输入图、'
                             '不丢文档最外圈内容（纯几何后处理，在 --prune_bm 之前执行）')
    parser.add_argument('--fit_mode', default='homography',
                        help="铺满方式：homography=先把采样区四角单应顶到图像四角再逐轴拉伸（默认，"
                             "可处理旋转/透视）；bbox=只做逐轴线性拉伸；off=关闭")
    parser.add_argument('--fit_quantile', type=float, default=0.5,
                        help='统计采样区支撑区间时两端各忽略的比例（%%），用于抑制纸边弯曲造成的'
                             '离群凸出点；0 表示严格取 min/max（默认 0.5）')
    parser.add_argument('--fit_clamp', type=str2bool, default=True,
                        help='铺满后把采样坐标夹到图像范围内，避免 grid_sample 采到界外产生黑边')
    parser.add_argument('--prune_inward', type=int, default=1,
                        help='判定采样点是否越界时把文档掩码向内腐蚀的像素数（默认 1）：'
                             '让输出图最外圈采样点落在文档边缘内侧，既贴合又不掺入被置零的背景')
    parser.add_argument('--snap_lbl', type=str2bool, default=False,
                        help='把 lbl 对齐到文档四边形：先用单应变换把"映射框"贴到文档边缘，'
                             '再把仍越界的点重定向到最近文档边缘像素。与 --prune_bm 是两种可互相'
                             '替代的边界贴合后处理，默认已由 --prune_bm 完成，故此处默认关闭')
    parser.add_argument('--inward_offset', type=float, default=5.0,
                        help='兜底重定向后再往文档内部偏移的像素数，避免双线性插值掺入边缘外侧被置零的背景（默认 2.0）')
    parser.add_argument('--grid_size', type=int, default=20,
    help='叠加到原图上的形变场网格密度（每边的网格数）')
    parser.add_argument('--homo_rectify', type=str2bool, default=True,
                        help='第一次矫正：用文档边缘掩码拟合文档四边形，通过单应性变换把'
                             '倾斜/透视的文档拉正（文档四边贴合图像四边），再把矫正结果'
                             '（含同步变换后的边缘掩码）送入后续流程')
    parser.add_argument('--homo_preserve_ar', type=str2bool, default=False,
                        help='第一次矫正的输出按文档四边形的长宽比取尺寸（面积与原图相当），'
                             '默认 False：沿用原图尺寸')
    parser.add_argument('--homo_canvas_margin', type=int, default=200,
                        help='第一次矫正后把拉正的图贴到更大的画布上，四周各留的边距像素数'
                             '（默认 200，即上/下/左/右各留 200px 空白；设为 0 表示不贴画布）')
    parser.add_argument('--homo_grid', type=int, default=10,
                        help='网格化第一级矫正的网格密度（每边的网格数）。把文档边界掩码'
                             '按该密度细分、逐格拟合成矩形来矫正弯曲文档，避免单应性变换'
                             '用四个角点近似而丢失弯曲区域（默认 10）')
    parser.add_argument('--k', type=int, default=2,
                        help='迭代预测次数：每轮跑一遍完整预测流程（文档区域预测 -> 网格化'
                             '第一级矫正 -> 形变场网络 -> 后处理），上一轮输出的矫正图裁掉'
                             '黑边后作为下一轮输入，迭代修正以提升最终矫正效果（默认 3）')

    """ 文本分割模型（在线预测 h / v 特征图） """
    parser.add_argument('--seg_model_path',
                        default=os.path.join(TEXT_SEG_DIR, 'logs', 'best_epoch_weights.pth'),
                        help='text_seg 分割模型权重路径')
    parser.add_argument('--seg_backbone', default='starnet',
                        help='text_seg 分割模型主干：vgg / resnet50 / starnet')
    parser.add_argument('--seg_input_size', type=int, default=640,
                        help='text_seg 分割模型输入尺寸')
    parser.add_argument('--seg_num_classes', type=int, default=3,
                        help='text_seg 分割类别数（background + horizon_line + vertical_line）')

    """ Model """
    parser.add_argument('--d_model', type=int, default=448, help='last layer dim in UNet and all layers in attention')
    parser.add_argument('--in_chans', type=int, default=3, help='input channels (binary mask)')
    return parser.parse_args()


parser = get_args()

recti_model = D2DewarpModel_my(img_size=parser.input_size, in_chans=parser.in_chans,
                            d_model=parser.d_model).cuda()

# recti_model = torch.nn.DataParallel(recti_model).cuda()
state_dict = torch.load(parser.model_path, map_location="cuda")
recti_model.load_state_dict(state_dict['state_dict'])
print(f'model loaded')

# ========== 文本分割模型：用于在线预测 h / v 特征图 ==========
seg_model = TextSegHVPredictor(
    model_path  = parser.seg_model_path,
    num_classes = parser.seg_num_classes,
    backbone    = parser.seg_backbone,
    input_shape = [parser.seg_input_size, parser.seg_input_size],
    cuda        = True,
)
seg_model.net.eval()
print(f'text seg model loaded from {parser.seg_model_path}')

# ========== 文档区域实例分割模型：用于在线预测文档区域实心掩码 ==========
doc_region_predictor = DocRegionPredictor(cuda=True)

# ========== Hook 容器 ==========
features = {}

def make_hook(name):
    def hook(module, inp, out):
        features[name] = inp
    return hook

# ========== 注册 Hook ==========
# CoordAtt 输出的 fused h / v map
recti_model.fusion_block.register_forward_hook(make_hook('fusion_out'))


def _predict_core(image_rgb, save_path, base_name, it, is_last):
    """单轮预测流程（被 predict 的迭代循环调用，详见 predict）。"""
    dewarp_path = os.path.join(save_path, 'perspective/')
    bm_grid_path = os.path.join(save_path, 'bm_grid/perspective')
    debug_path = os.path.join(save_path, 'debug_masked_input/')
    feat_path = os.path.join(save_path, 'features/')
    for p in (dewarp_path, bm_grid_path, debug_path, feat_path):
        os.makedirs(p, exist_ok=True)

    img_size = parser.input_size
    # ---------- 当前轮彩色图像，并用实例分割模型在线预测文档区域掩码 ----------
    first_pass_img = np.asarray(image_rgb)
    img_h, img_w = first_pass_img.shape[:2]

    # 文档区域实心掩码（内部 255，背景 0）-> 文档边缘掩码（边界一圈 255），
    doc_region_mask = doc_region_predictor.predict(first_pass_img)
    edge_raw = solid_region_to_edge(doc_region_mask)
    print('[iter %d] doc region coverage: %.1f%%' % (it, 100.0 * (doc_region_mask > 127).mean()))

    # ========== 第一级矫正：用 dewarp_document 把弯曲/倾斜的文档展平为矩形 ==========
    if parser.homo_rectify:
        # 记录基准（去 margin 后）尺寸
        _margin = int(parser.homo_canvas_margin)
        if _margin > 0:
            img_w0 = first_pass_img.shape[1] - 2 * _margin
            img_h0 = first_pass_img.shape[0] - 2 * _margin
        else:
            img_w0, img_h0 = first_pass_img.shape[1], first_pass_img.shape[0]
        try:
            result = dewarp_document(
                first_pass_img, doc_region_mask,
                grid_columns=int(parser.homo_grid) * 8,
                grid_rows=int(parser.homo_grid) * 6,
            )
        except Exception as e:
            print('[iter %d] grid first pass skipped: %s' % (it, e))
            result = None
        if result is not None:
            # 展平后的文档裁剪到了自身包围盒，resize 回原图尺寸以对齐下游分辨率
            rect_img = cv2.resize(result.image, (img_w0, img_h0),
                                  interpolation=cv2.INTER_LINEAR)
            rh, rw = rect_img.shape[:2]
            rect_edge = solid_region_to_edge(np.full((rh, rw), 255, dtype=np.uint8))
            # ---------- 把展平后的图贴到更大的画布上，四周各留 margin px 空白 ----------
            margin = int(parser.homo_canvas_margin)
            if margin > 0:
                ch, cw = rh + 2 * margin, rw + 2 * margin
                canvas_img = np.full((ch, cw, 3), 0, dtype=np.uint8)
                canvas_img[margin:margin + rh, margin:margin + rw] = rect_img
                canvas_edge = np.zeros((ch, cw), dtype=np.uint8)
                canvas_edge[margin:margin + rh, margin:margin + rw] = rect_edge
                first_pass_img = canvas_img
                edge_raw = canvas_edge
                img_h, img_w = ch, cw
                print('[iter %d] dewarp_document + canvas(%d): %s -> %s (angle=%.1f)' % (
                    it, margin, (rw, rh), (cw, ch), result.rotation_degrees))
            else:
                first_pass_img = rect_img
                edge_raw = rect_edge
                img_h, img_w = first_pass_img.shape[:2]   # 同步最终输入图尺寸
                print('[iter %d] dewarp_document: %s (angle=%.1f)' % (
                    it, first_pass_img.shape[1::-1], result.rotation_degrees))
        else:
            print('[iter %d] grid first pass skipped: no valid document' % it)

    # # ---------- 由边缘图像生成文档掩码 ----------
    fill_mask_, fill_mask = create_document_mask(edge_raw, img_size)

    # ---------- 获取 h_img 和 v_img（原始分辨率）----------
    # 由文本行分割模型对【第一次矫正后的图像】在线预测得到（类别 1 -> h，类别 2 -> v）
    original_pil = Image.fromarray(first_pass_img)
    h_img, v_img = seg_model.predict_hv_mask(original_pil)


    # # ---------- 新增：用 fill_mask 过滤 h_img 和 v_img ----------
    # h_img_filtered = filter_by_document_mask(h_img, fill_mask_)
    # v_img_filtered = filter_by_document_mask(v_img, fill_mask_)
    #
    # h_combined_mask = or_with_edge(h_img_filtered, edge_raw)
    # v_combined_mask = or_with_edge(v_img_filtered, edge_raw)

    # ---------- 归一化到 [0, 1]，resize 到模型输入尺寸 ----------
    h_input = cv2.resize(h_img, (img_size, img_size)).astype(np.float32) / 255.0
    v_input = cv2.resize(v_img, (img_size, img_size)).astype(np.float32) / 255.0
    edge = cv2.resize(edge_raw, (img_size, img_size)).astype(np.float32) / 255.0


    # 第一级矫正后的图像即为本流程的输入 image
    original_img = first_pass_img
    # original_img = (original_img.astype(np.float32) * fill_mask_[..., np.newaxis]).astype(np.uint8)  # 乘以掩码，外部背景变为 0（黑色）
    input = cv2.resize(original_img, (img_size, img_size)) / 255.

    # edge = gradient(original_img)
    # edge = edge[:, :, np.newaxis]
    # edge = cv2.resize(edge.astype(np.uint8), (img_size, img_size)) / 255.

    # ========== 新增：保存被掩码限制后的 input 以供检查 ==========
    debug_path = os.path.join(save_path, 'debug_masked_input/')
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    # 迭代标记：最终轮不带后缀，中间轮带 _it{it} 便于对照
    itag = '' if is_last else ('_it%d' % it)
    cv2.imwrite(os.path.join(debug_path, base_name + ".png"), original_img)
    # 分割模型直接预测出的 h / v 特征图
    cv2.imwrite(os.path.join(debug_path, base_name + "_h_pred.png"), h_img)
    cv2.imwrite(os.path.join(debug_path, base_name + "_v_pred.png"), v_img)
    # 第一级矫正直出的图像与同步变换后的边缘掩码，用于检查是否正确摆正
    cv2.imwrite(os.path.join(debug_path, base_name + "_homo.png"),
                first_pass_img[:, :, ::-1])
    cv2.imwrite(os.path.join(debug_path, base_name + "_homo_edge.png"), edge_raw)

    recti_model.eval()

    with torch.no_grad():
        h_input_ = torch.from_numpy(h_input).unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, H, W)
        v_input_ = torch.from_numpy(v_input).unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, H, W)
        input_ = torch.from_numpy(input).permute(2, 0, 1).cuda()
        input_ = input_.unsqueeze(0)
        edge = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).cuda()

        start = time.time()
        pred_bm = recti_model(h_input_.float(),v_input_.float(),input_.float(),edge.float())
        # 处理输出形变场
        bm = (2. * (pred_bm / (parser.input_size)) - 1) * 1.004
        ps_time = time.time() - start


    bm = bm.cpu()
    print("bm:",bm.shape)
    bm0 = cv2.resize(bm[0, 0].numpy(), (img_w, img_h))  # x flow
    bm1 = cv2.resize(bm[0, 1].numpy(), (img_w, img_h))  # y flow

    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))

    # ========== 删除落在文档边缘之外的采样点，并把有效区域拉伸到整幅输出图 ==========
    # 使矫正后的文档边缘与图像边缘完全贴合（此时文档边缘即整幅图像的边界）
    if parser.prune_bm:
        bm0, bm1, valid_mask, prune_outside = prune_samples_outside_document(
            bm0, bm1, edge_raw, doc_mask=fill_mask_, smooth=parser.prune_smooth,
            inward_offset=parser.prune_inward, return_debug=True)
        print('prune: deleted %.3f%% | still outside: %.3f%%'
              % (100.0 * (~valid_mask).mean(), 100.0 * prune_outside.mean()))
        cv2.imwrite(os.path.join(debug_path, base_name + "_prune_valid.png"),
                    (valid_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(debug_path, base_name + "_prune_outside.png"),
                    (prune_outside.astype(np.uint8) * 255))

    # ========== 把 bm 的采样区域拉伸到整幅输入图，使形变网格铺满输入图 ==========
    if parser.fit_bm:
        bm0, bm1, fit_info = fit_bm_to_image(
            bm0, bm1, mode=parser.fit_mode, quantile=parser.fit_quantile,
            clamp=parser.fit_clamp, return_debug=True)
        print('fit_bm: coverage %.3f/%.3f -> %.3f/%.3f (homo=%s)'
              % (fit_info['cov_before'][0], fit_info['cov_before'][1],
                 fit_info['cov_after'][0], fit_info['cov_after'][1],
                 fit_info['used_homo']))



    # default
    lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)  # h * w * 2
    print("lbl:",lbl.shape)
    # 形变场以网格形式叠加到原图上，用于检查形变场是否正确
    img_grid = draw_deformation_grid(original_img, bm0, bm1, grid_size=parser.grid_size)
    cv2.imwrite(os.path.join(bm_grid_path, base_name + itag + ".png"), img_grid[:, :, ::-1])  # save

    # ========== 用文档边缘掩码约束采样点：落到文档边缘之外的点重定向到最近边缘 ==========
    # 使矫正后的文档边缘与图像边缘贴合（理想情况下文档边缘即整幅图像的边界）
    if parser.snap_lbl:
        lbl, outside_mask, redirect_mask = snap_lbl_to_document_edge(
            lbl, edge_raw, doc_mask=fill_mask_, inward_offset=parser.inward_offset)
        # outside_mask 为重定向之后仍在文档外的采样点，正常应全为 False（存图全黑）
        print('redirect: %.3f%% | still outside: %.3f%%'
              % (100.0 * redirect_mask.mean(), 100.0 * outside_mask.mean()))
        cv2.imwrite(os.path.join(debug_path, base_name + "_snapmask.png"),
                    (outside_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(debug_path, base_name + "_redirect.png"),
                    (redirect_mask.astype(np.uint8) * 255))

    out = F.grid_sample(torch.from_numpy(original_img / 255.).permute(2, 0, 1).unsqueeze(0).float().cuda(), lbl.cuda(),
                        align_corners=True)
    img_geo = ((out[0] * 255).permute(1, 2, 0).cpu().numpy()).astype(np.uint8)

    # ========== 可视化 ==========
    base = base_name + itag
    feat_path = os.path.join(save_path, 'features/')
    os.makedirs(feat_path,exist_ok=True)
    # Fusion 输出
    if 'fusion_out' in features:
        save_feature_map(
            features['fusion_out'][0], feat_path, f'{base}_fusion_out_vmap'
        )
        save_spatial_map(
            features['fusion_out'][0], feat_path, f'{base}_fusion_out_vmap'
        )
        save_feature_map(
            features['fusion_out'][1], feat_path, f'{base}_fusion_out_hmap'
        )
        save_spatial_map(
            features['fusion_out'][1], feat_path, f'{base}_fusion_out_hmap'
        )

    return img_geo, fill_mask, img_h, img_w, ps_time

def predict(img_path, save_path, filename):
    """迭代预测入口：加载首轮原图，循环运行 k 轮完整预测流程。

    每一轮调用 predict_one 得到矫正图；除最后一轮外，把矫正图按画布边距裁掉黑边，
    作为下一轮的输入（避免逐轮把画布边距滚雪球式放大）。最后一轮的矫正图即最终输出。
    """
    assert os.path.exists(img_path), 'Incorrect Image Path'
    # assert os.path.exists(save_path), 'Incorrect Save Path'
    os.makedirs(save_path,exist_ok=True)

    dewarp_path = os.path.join(save_path, 'perspective/')
    os.makedirs(dewarp_path, exist_ok=True)

    init_img = np.array(Image.open(img_path).convert("RGB"))
    # 输入四周补 margin 像素空白（黑边），使文档区域检测 / 展平在每轮都有余量；
    # 该 margin 由这里统一提供，最终在 predict 末尾按 margin 裁掉，故不放大输出尺寸。
    _margin = int(parser.homo_canvas_margin)
    if _margin > 0:
        ih, iw = init_img.shape[:2]
        init_img = np.pad(init_img,
                          ((_margin, _margin), (_margin, _margin), (0, 0)),
                          mode='constant', constant_values=0)
    k = max(1, int(parser.k))
    base_name = filename.rsplit('/', 1)[-1].split('.')[0]

    cur_img = init_img
    total_ps = 0.0
    for it in range(k):
        is_last = (it == k - 1)
        img_geo, fill_mask, H, W, ps_time = _predict_core(cur_img, save_path, base_name, it, is_last)
        total_ps += ps_time

        # 去掉黑边：第一级矫正后画布在四周各加了 homo_canvas_margin 像素空白，
        # 直接按该 margin 值裁掉上下左右对应距离的像素，使保存结果尺寸与输入一致，
        margin = int(parser.homo_canvas_margin)
        ch, cw = img_geo.shape[:2]
        if margin > 0 and 2 * margin < min(ch, cw):
            img_geo_crop = img_geo[margin:ch - margin, margin:cw - margin].copy()
        else:
            img_geo_crop = img_geo.copy()

        # 保存本轮回正结果：最终轮输出主文件名，中间轮带 _it{it} 便于对照
        out_name = (base_name if is_last else '%s_it%d' % (base_name, it)) + '.png'
        cv2.imwrite(os.path.join(dewarp_path, out_name), img_geo_crop[:, :, ::-1])

        if not is_last:
            # 上一轮去黑边后的矫正图作为下一轮输入；四周补 margin 空白，
            _margin = int(parser.homo_canvas_margin)
            if _margin > 0:
                ch, cw = img_geo_crop.shape[:2]
                cur_img = np.pad(img_geo_crop,
                                ((_margin, _margin), (_margin, _margin), (0, 0)),
                                mode='constant', constant_values=0)
            else:
                cur_img = img_geo_crop.copy()

    return total_ps



if __name__ == '__main__':
    img_path = parser.img_path
    save_path = parser.save_path
    total_time = 0.0

    start = time.time()
    img_num = 0.0

    for file in glob.glob(img_path + "/*"):
        print("file: ", file)
        filename = (save_path + "/" + file[file.rindex("/") + 1:file.rindex(".")] + ".png")

        total_time += predict(file, save_path, filename)
        print("total_time: ", total_time)
        img_num += 1
    print('FPS: %.1f' % (1.0 / (total_time / img_num)))

