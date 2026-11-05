import argparse
import os
import sys

import cv2
import hdf5storage as h5
import numpy as np
import torch
from tqdm import tqdm

from utils import IMG_SIZE, bilinear_unwarping, load_model

# ---------------------------------------------------------------------------- #
#   internimage-l-instance：文档区域实例分割模型（InternImage-L + MaskDINO）
#   用来在线预测文档区域"实心掩码"，作为 dewarp_document 第一级矫正的输入。
# ---------------------------------------------------------------------------- #
INSTANCE_SEG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'internimage_l_instance')
if INSTANCE_SEG_DIR not in sys.path:
    sys.path.insert(0, INSTANCE_SEG_DIR)

from internimage_l_instance.instance_seg.config import load_config                              # noqa: E402
from internimage_l_instance.instance_seg.runtime import load_model as seg_load_model            # noqa: E402
from internimage_l_instance.instance_seg.runtime import filter_and_sort_instances               # noqa: E402
from internimage_l_instance.instance_seg.preprocessing import DocLetterboxDatasetMapper         # noqa: E402
from internimage_l_instance.instance_seg import settings as seg_settings                         # noqa: E402

from predict_utils.dewarp_core import dewarp_document, DewarpError                              # noqa: E402


class DocRegionPredictor:
    """用 internimage-l-instance 的 InternImage-L + MaskDINO 在线预测文档区域。

    前向流程与 instance_seg/pipeline.py 的 Stage-1 完全一致：
    转 RGB -> 不失真（letterbox）resize 到 1024 -> 送入模型 -> 取回原图分辨率的
    逐实例掩码 -> 按置信度排序、合并为整幅文档的实心区域掩码。

    返回的「实心区域掩码」（文档内部像素为 255，背景为 0）即可直接作为
    dewarp_document 的输入，完成一次基于文档区域的网格化第一级矫正。
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
        self.model = seg_load_model(self.cfg, seg_settings.DEFAULT_CHECKPOINT)
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
            m = mask_tensor.numpy().astype(np.uint8) * 255
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            solid = np.maximum(solid, m)
        return solid


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class UVDocBenchmarkLoader(torch.utils.data.Dataset):
    """
    Torch dataset class for the UVDoc benchmark dataset.
    """

    def __init__(
        self,
        data_path,
        img_size=(488, 712),
    ):
        self.dataroot = data_path
        self.im_list = os.listdir(os.path.join(self.dataroot))
        self.img_size = img_size

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        im_name = self.im_list[index]
        img_path = os.path.join(self.dataroot,im_name)
        img_RGB = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_RGB = torch.from_numpy(cv2.resize(img_RGB, self.img_size).transpose(2, 0, 1))
        return img_RGB, im_name


def first_pass_rectify(image_rgb, doc_region_predictor, homo_grid):
    """先对输入图像做一次矫正：用文档区域实例分割得到实心掩码，
    再用 dewarp_document 的网格化 Coons/Harmonic 方法把弯曲/倾斜文档展平为矩形。

    Args:
        image_rgb: uint8 (H, W, 3) RGB 原图。
        doc_region_predictor: DocRegionPredictor 实例。
        homo_grid: 第一级矫正的网格密度基数（homo_grid*8 列 / homo_grid*6 行）。

    Returns:
        一次矫正后的 RGB 图（uint8, (H, W, 3)）；若矫正失败则返回原图。
    """
    doc_region_mask = doc_region_predictor.predict(image_rgb)
    print('[first-pass] doc region coverage: %.1f%%'
          % (100.0 * (doc_region_mask > 127).mean()))
    try:
        result = dewarp_document(
            image_rgb, doc_region_mask,
            grid_columns=int(homo_grid) * 8,
            grid_rows=int(homo_grid) * 6,
        )
    except (DewarpError, Exception) as e:
        print('[first-pass] rectify skipped: %s' % e)
        return image_rgb
    print('[first-pass] dewarp_document done (angle=%.1f, size=%s)'
          % (result.rotation_degrees, result.output_size))
    return result.image


def fit_to_canvas_with_margin(img, orig_h, orig_w, margin):
    """把一次矫正结果放到与原图同尺寸的画布中央，四周各留 margin 像素空白。

    工作图尺寸恒为 orig_h x orig_w：给文档四周留白以容纳边缘内容，但整幅图尺寸
    不放大，避免后续裁切导致内容丢失（与参考 d2dewarp 代码一致）。
    """
    if margin <= 0:
        return cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    tw, th = orig_w - 2 * margin, orig_h - 2 * margin
    if tw <= 0 or th <= 0:
        return cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    inner = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((orig_h, orig_w, 3), 0, dtype=np.uint8)
    canvas[margin:margin + th, margin:margin + tw] = inner
    return canvas


def uvdoc_iterate(model, device, image_rgb, k, margin):
    """将当前 RGB 图（已为原图尺寸，四周含 margin 留白）送入 UVDoc model 迭代 k 次修正。

    与参考 d2dewarp 代码一致：每轮反变形得到原图尺寸的结果后，向其四周补 margin 留白，
    再整体 resize 回原图尺寸作为下一次迭代输入。这样工作图尺寸始终保持原图大小，
    最终保存结果自然与原图尺寸一致，且不会因裁切而丢失内容。

    Args:
        image_rgb: uint8 (H, W, 3) RGB 图，尺寸恒为原图尺寸（四周可含 margin 留白）。
        k: 迭代次数（>=1）。
        margin: 四周补的空白边距像素数（0 表示不补）。

    Returns:
        result_rgb: 最后一轮矫正后的 RGB 图（原图尺寸，uint8）。
        bm: 最后一轮的 backward map 张量（B, 2, Gh, Gw）。
    """
    model.eval()
    cur = image_rgb
    bm = None
    n = max(1, int(k))
    for it in range(n):
        h, w = cur.shape[:2]
        # 模型输入：resize 到 IMG_SIZE 并归一化到 [0, 1]
        img_RGB = cv2.resize(cur, IMG_SIZE).astype(np.float32) / 255.0
        img_RGB = torch.from_numpy(img_RGB.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            point_positions2D, _ = model(img_RGB)
        bm = point_positions2D

        # 对整幅当前图做双线性反变形（backward map 上采样到当前图尺寸）
        warped = torch.from_numpy(
            cur.transpose(2, 0, 1) / 255.0
        ).float().unsqueeze(0).to(device)
        unwarped = bilinear_unwarping(warped, point_positions2D, (w, h))
        cur = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        # 向四周补 margin 留白，再整体 resize 回原图尺寸，作为下一次迭代输入
        # （与参考代码一致：保留留白给边缘内容，但工作图尺寸始终保持不变，
        #   因此最终输出即原图尺寸，无需裁切，内容不会丢失）
        if margin > 0:
            ch, cw = h + 2 * margin, w + 2 * margin
            canvas = np.full((ch, cw, 3), 0, dtype=np.uint8)
            canvas[margin:margin + h, margin:margin + w] = cur
            cur = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)
        print('[uvdoc-iter %d] done' % it)
    return cur, bm


def infer_uvdoc(model, dataloader, device, save_path, doc_region_predictor, args):
    """
    Unwarp all images in the UVDoc benchmark and save them, along with the mappings.
    """
    model.eval()

    os.makedirs(os.path.join(save_path, "uwp_img"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "bm"), exist_ok=True)

    for _, im_names in tqdm(dataloader):
        im_name = im_names[0]

        # 读取整幅原图（第一级矫正与迭代修正都需要全分辨率）
        img_path = os.path.join(dataloader.dataset.dataroot,im_name)
        full_bgr = cv2.imread(img_path)
        full_rgb = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = full_rgb.shape[:2]

        # ---------- 先对输入图像做一次矫正 ----------
        cur_rgb = full_rgb
        if args.first_rectify:
            cur_rgb = first_pass_rectify(full_rgb, doc_region_predictor, args.homo_grid)

        # ---------- 一次矫正结果放到与原图同尺寸的画布中央（四周留 margin），作为迭代输入 ----------
        margin = int(args.margin)
        cur_rgb = fit_to_canvas_with_margin(cur_rgb, orig_h, orig_w, margin)

        # ---------- 送入 UVDoc model 迭代 k 次修正（工作图尺寸始终为原图尺寸）----------
        result_rgb, bm = uvdoc_iterate(model, device, cur_rgb, args.k, margin)

        # result_rgb 已是原图尺寸（工作图全程保持原图大小），无需裁切，直接保存
        unwarped_BGR = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(save_path, "uwp_img", im_name.split(" ")[0].split(".")[0] + ".png"),
            unwarped_BGR,
        )
        # Save Backward Map（最后一次迭代的形变场）
        assert bm is not None
        h5.savemat(
            os.path.join(save_path, "bm", im_name.split(" ")[0].split(".")[0] + ".mat"),
            {"bm": bm[0].detach().cpu().numpy().transpose(1, 2, 0)},
        )


def create_uvdoc_results(ckpt_path, uvdoc_path, img_size, args):
    """
    Create results for the UVDoc benchmark.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model, create dataset and save directory
    model = load_model(ckpt_path)
    model.to(device)

    # 文档区域实例分割模型（仅在第一级矫正开启时需要）
    doc_region_predictor = None
    if args.first_rectify:
        doc_region_predictor = DocRegionPredictor(cuda=device.type == "cuda")

    dataset = UVDocBenchmarkLoader(data_path=uvdoc_path, img_size=img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    save_path = os.path.join("/".join(ckpt_path.split("/")[:-1]), "output_uvdoc/rotate")
    os.makedirs(save_path, exist_ok=True)
    print(f"    Results will be saved at {save_path}", flush=True)

    # Infer results
    infer_uvdoc(model, dataloader, "cuda:0", save_path, doc_region_predictor, args)
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-path", type=str, default="./model/best_model.pkl", help="Path to the model weights as pkl."
    )
    parser.add_argument(
        "--uvdoc-path", type=str, default="./data/test_common_test_dataset/img/rotate/", help="Path to the UVDocBenchmark dataset."
    )
    parser.add_argument(
        "--first-rectify", type=str2bool, default=True,
        help="是否先对输入图像做一次矫正（用文档区域实例分割 + dewarp_document 网格化"
             "把弯曲/倾斜文档展平为矩形），再把矫正结果送入 UVDoc model 迭代修正"
    )
    parser.add_argument(
        "--k", type=int, default=1,
        help="UVDoc model 迭代修正次数：每轮把上一步矫正结果送入模型重新预测形变场并"
             "反变形，k 次迭代以进一步提升最终矫正效果（默认 1）"
    )
    parser.add_argument(
        "--homo-grid", type=int, default=10,
        help="第一级矫正的网格密度基数（homo_grid*8 列 / homo_grid*6 行），用于把文档"
             "边界网格化逐格拟合成矩形来矫正弯曲文档（默认 10）"
    )
    parser.add_argument(
        "--margin", type=int, default=200,
        help="在图像四周补的空白边距像素数：一次矫正后给结果加 margin 作为输入，"
             "且每次迭代也对输出加 margin 作为下一次迭代输入（先裁掉再补回以保持恒定）；"
             "最终保存结果会裁掉 margin 并 resize 回原始输入尺寸（默认 200）"
    )
    args = parser.parse_args()

    create_uvdoc_results(args.ckpt_path, os.path.abspath(args.uvdoc_path), IMG_SIZE, args)
