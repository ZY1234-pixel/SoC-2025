import argparse
import os
import sys

import cv2
import hdf5storage as h5
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import IMG_SIZE, bilinear_unwarping, load_model


# -----------------------------------------------------------------------------#
#   internimage-l-instance：文档区域实例分割模型（InternImage-L + MaskDINO）
#   用来在线预测文档区域"实心掩码"（文档内部像素为 255），替代原来从磁盘读取的
#   固定边缘掩码。下游预测再基于该掩码做背景去除 + 单应性摆正（去除透视/旋转）。
#   加载与使用方式与本仓库参考实现（d2dewarp 集成脚本）保持一致：
#   sys.path 加入项目根目录后即可直接 import，模型权重由 settings 默认给出。
# -----------------------------------------------------------------------------#
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from internimage_l_instance.instance_seg.config import load_config                              # noqa: E402
from internimage_l_instance.instance_seg.runtime import load_model as load_seg_model, filter_and_sort_instances  # noqa: E402
from internimage_l_instance.instance_seg.preprocessing import DocLetterboxDatasetMapper          # noqa: E402
from internimage_l_instance.instance_seg import settings as seg_settings                         # noqa: E402


class DocRegionPredictor:
    """用 internimage-l-instance 的 InternImage-L + MaskDINO 在线预测文档区域。

    前向流程与 instance_seg/pipeline.py 的 Stage-1 完全一致：
    转 RGB -> 不失真（letterbox）resize 到 1024 -> 送入模型 -> 取回原图分辨率的
    逐实例掩码 -> 按置信度排序、合并为整幅文档的实心区域掩码。

    返回的是「实心区域掩码」（文档内部像素为 255，背景为 0），下游据此做背景去除
    与文档四边形拟合（单应性摆正）。
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
        self.model = load_seg_model(self.cfg, seg_settings.DEFAULT_CHECKPOINT)
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


# -----------------------------------------------------------------------------#
#   文档四边形拟合 + 单应性摆正：用文档区域掩码去除背景后，把倾斜/透视的文档
#   拉正为正面视角（文档四边贴合输出矩形四边），只做透视/旋转一级矫正；
#   残留的弯曲（曲率）交给下游 UVDoc 模型修正，不做迭代。
# -----------------------------------------------------------------------------#
def order_points(pts):
    """把 4 个角点排序为 [左上, 右上, 右下, 左下]，与 cv2.getPerspectiveTransform
    的目标矩形角点顺序一致。"""
    pts = pts.copy()
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上：x+y 最小
    rect[2] = pts[np.argmax(s)]   # 右下：x+y 最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上：y-x 最小
    rect[3] = pts[np.argmax(diff)]  # 左下：y-x 最大
    return rect


def detect_document_quad(solid_mask):
    """由文档区域实心掩码（内部 255）得到用于摆正的 4 个源角点 (4, 2) float32，
    顺序为 [左上, 右上, 右下, 左下]；检测不到时返回 None。

    采用**最大外接矩形**（而非最小外接矩形，也非内缩的多边形）：
    1) 保留最大连通域、取最大外轮廓的凸包；
    2) 用最小面积矩形只估计文档的**主方向（旋转角）**，把凸包旋正到该方向；
    3) 在旋正后的坐标系下取凸包的**完整包围盒**（即能完整包含文档区域全部像素
       的最大外接矩形），再反旋回原坐标得到 4 个角点。
    该矩形完整包住文档区域，因此摆正时**不会裁减任何文档内容**；把它单应映射到
    轴对齐矩形即可同时去除旋转与（近似的）透视。
    """
    bin_mask = (solid_mask > 127).astype(np.uint8)
    if bin_mask.sum() == 0:
        return None
    # 仅保留最大连通域，剔除碎裂的噪声实例
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bin_mask = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)

    # 用最小面积矩形估计文档主方向（仅取旋转角，不用它作为外接矩形本身）
    (cx, cy), (w, h), angle = cv2.minAreaRect(hull)
    # 旋转凸包到文档主方向，使其边与坐标轴平行
    R = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    pts = hull.reshape(-1, 2).astype(np.float32)
    pts_rot = cv2.transform(pts.reshape(-1, 1, 2), R).reshape(-1, 2)
    # 取完整包围盒（最大外接矩形）：完整包含文档区域所有像素，保证不裁减内容
    x0, y0 = float(pts_rot[:, 0].min()), float(pts_rot[:, 1].min())
    x1, y1 = float(pts_rot[:, 0].max()), float(pts_rot[:, 1].max())
    box_rot = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    # 反旋回原坐标，得到 4 个源角点
    Rinv = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    box = cv2.transform(box_rot.reshape(-1, 1, 2), Rinv).reshape(-1, 2)
    return order_points(box)


def rectify_to_frontal(image_rgb, quad):
    """把文档四边形 quad 单应变换到轴对齐矩形，去除透视 + 旋转，得到正面视角。"""
    tl, tr, br, bl = quad
    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    h_left = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))
    max_w = max(1, int(round(max(w_top, w_bot))))
    max_h = max(1, int(round(max(h_left, h_right))))
    dst = np.array([[0, 0], [max_w, 0], [max_w, max_h], [0, max_h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(
        image_rgb, H, (max_w, max_h), flags=cv2.INTER_LINEAR, borderValue=0
    )


def preprocess_document(img_bgr, doc_region_predictor, debug_path=None, base_name="img", mask_erode=0):
    """对单张输入图执行：文档区域分割 -> 背景去除 -> 单应性摆正（去除透视/旋转）。

    Args:
        img_bgr: BGR 原图 (H, W, 3)
        doc_region_predictor: DocRegionPredictor 实例
        debug_path: 可选，保存中间结果（实心掩码 / 去背景图 / 摆正图）
        base_name: 调试文件名前缀
        mask_erode: 背景去除时把文档掩码向内收缩的像素距离（>=0）。仅在背景去除
            这一步生效（用于剔除文档边缘附近的背景残留/锯齿），不影响用于摆正的四边形检测。
    Returns:
        rectified_bgr: 已去除背景且摆正为正面视角的图 (H', W', 3)
    """
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1) 文档区域分割
    solid_mask = doc_region_predictor.predict(image_rgb)
    coverage = float((solid_mask > 127).mean())
    print('[doc region] coverage: %.1f%%' % (100.0 * coverage))

    # 2) 背景去除：文档区域外置零（黑色背景）
    #    可选：先把掩码向内腐蚀 mask_erode 像素，收缩文档边界，避免边缘处的背景残留/锯齿
    if mask_erode > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * mask_erode + 1, 2 * mask_erode + 1)
        )
        bg_mask = cv2.erode(solid_mask, kernel, iterations=1)
    else:
        bg_mask = solid_mask
    mask3 = (bg_mask > 127)[..., np.newaxis].astype(np.float32)
    masked_rgb = (image_rgb.astype(np.float32) * mask3).astype(np.uint8)

    # 3) 单应性摆正：去除透视变换 + 旋转，使输入视角为正
    quad = detect_document_quad(solid_mask)
    if quad is not None:
        rectified_rgb = rectify_to_frontal(masked_rgb, quad)
        print('[rectify] frontal view size: %s' % (rectified_rgb.shape[1::-1],))
    else:
        rectified_rgb = masked_rgb
        print('[rectify] no document quad detected, skip frontal rectification')

    rectified_bgr = cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2BGR)

    if debug_path is not None:
        os.makedirs(debug_path, exist_ok=True)
        cv2.imwrite(os.path.join(debug_path, base_name + "_solid_mask.png"), solid_mask)
        cv2.imwrite(os.path.join(debug_path, base_name + "_bg_removed.png"),
                    cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(debug_path, base_name + "_frontal.png"), rectified_bgr)

    return rectified_bgr


def infer_uvdoc(model, doc_region_predictor, img_list, device, save_path, k=1, margin=0, mask_erode=0, debug_path=None):
    """
    对每张输入图：先经文档区域分割/去背景/摆正，再送入 UVDoc 模型做形变场预测与反变形。

    迭代修正（迭代次数由 k 控制）：第 i 次迭代以“第 i-1 次迭代矫正后的结果”作为模型输入，
    再次预测形变场并做反变形，因此每次迭代都在前一次的矫正结果上继续修正（即把上一次
    矫正后的图送回模型）。

    margin（命令行参数，单位像素）：除最后一次外，每次迭代的矫正结果会在上下左右各加
    margin 宽度的黑边后再作为下一次迭代的输入（给模型留出周边上下文）。无论 margin 多大，
    最后一次保存的迭代结果都会 resize 回“原始输入尺寸”（用 resize 而非裁剪，避免裁掉因
    加 margin 而外扩/位移的文档内容），以保证最终输出与原图同尺寸。

    每次迭代的矫正图与形变场都会分别保存到 uwp_img_iter{i} / bm_iter{i} 目录；最终
    （第 k 次，已裁剪回原尺寸）的结果同时保存到原约定目录 uwp_img / bm。

    Args:
        k: 迭代次数。k=1 表示仅一次前向预测，等价于不迭代（与原行为一致）。
        margin: 迭代间加入的黑边宽度（像素）。margin=0 时等价于不加边。
    """
    model.eval()
    k = max(1, int(k))
    margin = max(0, int(margin))

    # 为每一次迭代创建独立的保存目录，并保留原约定目录用于存放最终结果
    iter_img_dirs, iter_bm_dirs = [], []
    for i in range(1, k + 1):
        d_img = os.path.join(save_path, "uwp_img_iter%d" % i)
        d_bm = os.path.join(save_path, "bm_iter%d" % i)
        os.makedirs(d_img, exist_ok=True)
        os.makedirs(d_bm, exist_ok=True)
        iter_img_dirs.append(d_img)
        iter_bm_dirs.append(d_bm)
    final_img_dir = os.path.join(save_path, "uwp_img")
    final_bm_dir = os.path.join(save_path, "bm")
    os.makedirs(final_img_dir, exist_ok=True)
    os.makedirs(final_bm_dir, exist_ok=True)

    for name in tqdm(img_list):
        img_bgr = cv2.imread(name)
        if img_bgr is None:
            print(f"[skip] cannot read {name}")
            continue
        base = os.path.splitext(os.path.basename(name))[0]

        # ---- 文档区域分割 -> 背景去除 -> 单应性摆正（正面视角）----
        rectified_bgr = preprocess_document(
            img_bgr, doc_region_predictor, debug_path=debug_path, base_name=base, mask_erode=mask_erode
        )
        rectified_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)

        # 原始输入尺寸（用于最后一次裁剪回原尺寸）
        H, W = rectified_bgr.shape[0], rectified_bgr.shape[1]

        # 当前送入模型的图：迭代起点为“摆正后的原图”，后续迭代替换为上一次矫正后的结果
        # （除最后一次外会先加 margin 黑边）。
        cur_img = torch.unsqueeze(
            torch.from_numpy(rectified_rgb.transpose(2, 0, 1) / 255.0).float(), dim=0
        ).to(device)  # [1,3,H,W]，值域 [0,1]

        last_unwarped = None
        last_G = None
        for i in range(1, k + 1):
            # 当前图的实际尺寸 (宽, 高)，随 margin 累积可能变大
            cur_size = (cur_img.shape[3], cur_img.shape[2])
            # 模型输入需 resize 到 IMG_SIZE=(W,H)；F.interpolate 的 size 为 (H,W)
            model_input = F.interpolate(
                cur_img, size=(IMG_SIZE[1], IMG_SIZE[0]), mode="bilinear", align_corners=True
            )
            with torch.no_grad():
                point_positions2D, _ = model(model_input)
            G = torch.unsqueeze(point_positions2D[0], dim=0)  # [1,2,GH,GW]

            # 对当前图做双线性反变形，得到本次迭代的矫正结果（即“矫正后的结果”）
            unwarped = bilinear_unwarping(cur_img, G, tuple(cur_size))
            last_unwarped = unwarped
            last_G = G

            if i < k:
                # 中间迭代：保存本次自然尺寸的矫正结果与形变场
                unwarped_np = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                unwarped_bgr = cv2.cvtColor(unwarped_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(iter_img_dirs[i - 1], base + ".png"), unwarped_bgr)
                h5.savemat(
                    os.path.join(iter_bm_dirs[i - 1], base + ".mat"),
                    {"bm": G[0].detach().cpu().numpy().transpose(1, 2, 0)},
                )
                # 在上下左右各加 margin 宽度的黑边，作为下一次迭代的输入
                cur_img = F.pad(
                    unwarped, (margin, margin, margin, margin), mode="constant", value=0.0
                )
            # i == k 时不在循环内保存，留待循环外统一裁剪回原尺寸后保存

        # 最后一次（第 k 次）迭代结果：直接 resize 回原始输入尺寸后再保存。
        # 用 resize 而非裁剪，避免把因加 margin 而在周边外扩/位移的文档内容裁掉；
        # 同时写入该次迭代目录与原约定目录，保证最终输出与原图同尺寸。
        unwarped_resized = F.interpolate(last_unwarped, size=(H, W), mode="bilinear", align_corners=True)
        unwarped_np = (unwarped_resized[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        unwarped_bgr = cv2.cvtColor(unwarped_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(iter_img_dirs[k - 1], base + ".png"), unwarped_bgr)
        h5.savemat(
            os.path.join(iter_bm_dirs[k - 1], base + ".mat"),
            {"bm": last_G[0].detach().cpu().numpy().transpose(1, 2, 0)},
        )
        cv2.imwrite(os.path.join(final_img_dir, base + ".png"), unwarped_bgr)
        h5.savemat(
            os.path.join(final_bm_dir, base + ".mat"),
            {"bm": last_G[0].detach().cpu().numpy().transpose(1, 2, 0)},
        )


def gather_images(img_path):
    if os.path.isdir(img_path):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        return sorted(
            os.path.join(img_path, f)
            for f in os.listdir(img_path)
            if f.lower().endswith(exts)
        )
    return [img_path]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-path", type=str, default="./model/best_model.pkl", help="Path to the UVDoc model weights (.pkl)."
    )
    parser.add_argument(
        "--img_path", type=str, default="data/test_common_test_dataset/img/curved",
        help="输入图像路径，或包含图像的文件夹（会被逐个处理）"
    )
    parser.add_argument(
        "--save_path", type=str, default="data/test_common_test_dataset/dewarp/curved", help="矫正结果保存目录"
    )
    parser.add_argument(
        "--debug", default="data/test_common_test_dataset/dewarp/curved/debug", help="同时保存中间结果（实心掩码/去背景图/摆正图）"
    )
    parser.add_argument(
        "--k", type=int, default=3, help="UVDoc 形变场迭代修正次数（k=1 即不迭代，与原行为一致）"
    )
    parser.add_argument(
        "--margin", type=int, default=100, help="迭代间在上下左右各加的黑边宽度（像素）；最后一次结果会 resize 回原尺寸"
    )
    parser.add_argument(
        "--mask_erode", type=int, default=10, help="背景去除时把文档掩码向内收缩的像素距离（0 表示不收缩）"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 UVDoc 模型
    model = load_model(args.ckpt_path)
    model.to(device)
    print(f"UVDoc model loaded from {args.ckpt_path}")

    # 加载文档区域分割模型
    doc_region_predictor = DocRegionPredictor(cuda=torch.cuda.is_available())

    img_list = gather_images(args.img_path)
    assert len(img_list) > 0, f"No input images found at {args.img_path}"
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Results will be saved at {args.save_path}")

    infer_uvdoc(model, doc_region_predictor, img_list, device, args.save_path, k=args.k, margin=args.margin, mask_erode=args.mask_erode, debug_path=args.debug)
