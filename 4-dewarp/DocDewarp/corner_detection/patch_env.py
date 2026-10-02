# -*- coding: utf-8 -*-
"""一键为 ultralytics 打齐 5 处补丁（幂等 + 自检）

这 5 处补丁是角点检测的前置条件，缺任何一处输出都会变：
  1. data/utils.py        verify_image_label: 放宽关键点坐标校验（允许负值/越界到 ±999）
  2. data/augment.py      RandomPerspective.apply_keypoints: 不再把画外角点置为不可见
  3. utils/instance.py    Instances.clip: 新增 clip_keypoints 开关
  4. utils/ops.py + models/yolo/pose/predict.py  scale_coords 的 clip 开关
     **推理必须**：不打这一处，落在画面外的预测点会被夹回画面
  5. utils/loss.py        KeypointLoss 改为顺序无关（4! 排列取最小）——仅影响训练

用法:
  python -X utf8 patch_env.py --check                 # 只检查（默认）
  python -X utf8 patch_env.py --apply                 # 缺哪处补哪处（先备份 .bak_release）
  python -X utf8 patch_env.py --check --venv D:\some\venv   # 指定 venv 根目录

默认按运行本脚本的解释器推导 site-packages（即在目标 venv 里跑即可）。
"""
import argparse
import os
import py_compile
import re
import shutil
import sys

def default_site_packages(venv_root=None):
    """优先按 --venv 推导，其次按当前解释器推导"""
    if venv_root:
        cands = [os.path.join(venv_root, "Lib", "site-packages"),
                 os.path.join(venv_root, "lib", "site-packages")]
        for c in cands:
            if os.path.isdir(c):
                return c
        return cands[0]
    exe = os.path.dirname(sys.executable)                    # .../venv/Scripts
    root = os.path.dirname(exe)                              # .../venv
    for c in (os.path.join(root, "Lib", "site-packages"),
              os.path.join(root, "lib", "site-packages")):
        if os.path.isdir(c):
            return c
    return os.path.join(root, "Lib", "site-packages")


SP = os.path.join(default_site_packages(), "ultralytics")
TAG = "_release"


def read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def backup(p):
    bak = p + f".bak{TAG}"
    if not os.path.exists(bak):
        shutil.copy2(p, bak)
    return bak


def sub_line_regex(text, pattern, repl, count=1, skip_comment=True):
    """按行正则替换，保留缩进；默认跳过注释行（避免误改已注释的原文）"""
    out, n = [], 0
    for line in text.split("\n"):
        if n < count and re.search(pattern, line) and not (skip_comment and line.lstrip().startswith("#")):
            out.append(re.sub(pattern, repl, line, count=1))
            n += 1
        else:
            out.append(line)
    return "\n".join(out), n


# ---- 各补丁: 文件 / 判据(已打的标志) / 修改动作 ----
def p1_utils():
    p = os.path.join(SP, "data", "utils.py")
    t = read(p)
    # 只放宽两个坐标校验断言；points 取值那一行保持原样
    t, n2 = sub_line_regex(t, r"(?<=assert )points\.max\(\) <= 1\.01", "points.max() <= 999.0")
    t, n3 = sub_line_regex(t, r"non-normalized or out of bounds coordinates \{points\[points > 1\.01\]\}",
                           "non-normalized or out of bounds coordinates {points[points > 999.0]}")
    t, n4 = sub_line_regex(t, r"(?<=assert )lb\.min\(\) >= -0\.01", "lb[:, :5].min() >= -0.01")
    t, n5 = sub_line_regex(t, r"negative class labels or coordinate \{lb\[lb < -0\.01\]\}",
                           "negative class labels or bbox {lb[:, :5][lb[:, :5] < -0.01]}")
    write(p, t)
    return n2 + n3 + n4 + n5


P1_CHECK = "points.max() <= 999.0"


def p2_augment():
    p = os.path.join(SP, "data", "augment.py")
    t = read(p)
    old = ("        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | "
           "(xy[:, 0] > self.size[0]) | (xy[:, 1] > self.size[1])\n"
           "        visible[out_mask] = 0\n")
    new = "        # [qbj patch] 保留画外关键点：坐标可越界、可见性不被清零\n"
    if old in t:
        t = t.replace(old, new, 1)
        t = t.replace("        new_instances.clip(*self.size)\n",
                      "        new_instances.clip(*self.size, clip_keypoints=False)\n", 1)
        write(p, t)
        return 1
    return 0


P2_CHECK = "clip_keypoints=False"


def p3_instance():
    p = os.path.join(SP, "utils", "instance.py")
    t = read(p)
    if "def clip(self, w: int, h: int, clip_keypoints: bool = True)" in t:
        return 0
    t = t.replace("    def clip(self, w: int, h: int) -> None:",
                  "    def clip(self, w: int, h: int, clip_keypoints: bool = True) -> None:", 1)
    # 注意: instance.py 里有 9 处同名的 `if self.keypoints is not None:`，
    # 必须只改 clip() 函数体内那一处，否则会误改 scale/denormalize 等方法。
    lines = t.split("\n")
    start = next((i for i, l in enumerate(lines)
                  if "def clip(self, w: int, h: int, clip_keypoints" in l), None)
    if start is None:
        return -1
    for j in range(start, len(lines)):
        if lines[j].strip() == "if self.keypoints is not None:":
            lines[j] = lines[j].replace("if self.keypoints is not None:",
                                        "if self.keypoints is not None and clip_keypoints:")
            break
    else:
        return -1
    write(p, "\n".join(lines))
    return 1


P3_CHECK = "clip_keypoints: bool = True"


def p4_ops():
    p = os.path.join(SP, "utils", "ops.py")
    t = read(p)
    if "if clip:  # [qbj patch]" in t or "padding: bool = True, clip: bool = True" in t:
        return 0
    t, n1 = sub_line_regex(
        t, r"def scale_coords\(img1_shape, coords, img0_shape, ratio_pad=None, "
           r"normalize: bool = False, padding: bool = True\):",
        "def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, "
        "normalize: bool = False, padding: bool = True, clip: bool = True):")
    t, n2 = sub_line_regex(t, r"^(\s*)coords = clip_coords\(coords, img0_shape\)$",
                           r"\1if clip:  # [qbj patch] 画外关键点保留负/越界坐标\n"
                           r"\1    coords = clip_coords(coords, img0_shape)")
    write(p, t)
    return n1 + n2


P4A_CHECK = "padding: bool = True, clip: bool = True"


def p4b_predict():
    p = os.path.join(SP, "models", "yolo", "pose", "predict.py")
    t = read(p)
    if "clip=False" in t:
        return 0
    t = t.replace("ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)",
                  "ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape, clip=False)", 1)
    write(p, t)
    return 1


P4B_CHECK = "clip=False"


# 补丁 5 的原文/替换文本（与 ultralytics 8.4.48 一致；换版本若匹配不到会提示手动改）
P5_OLD = '''class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()'''
P5_NEW = '''class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """顺序无关 keypoint loss: 对 4 个角点取 4! 排列里误差最小者

        避免逐通道监督把"点集正确、角色错位"判成超大误差, 从而逼模型输出折中几何。
        K>6 (如 COCO 17 点) 时回退原逐通道实现。
        """
        import itertools

        K = pred_kpts.shape[1]
        if K > 6:
            d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
            kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
            e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
            return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

        perms = list(itertools.permutations(range(K)))
        P = len(perms)
        dev = pred_kpts.device
        perm_idx = torch.as_tensor(perms, dtype=torch.long, device=dev)  # (P,K)
        N = pred_kpts.shape[0]
        # (N,P,K,2): 每个排列下预测点坐标
        pred = pred_kpts[:, None, :, :2].expand(N, P, K, 2)
        idx = perm_idx[None, :, :, None].expand(N, P, K, 2)
        pred_p = pred.gather(2, idx)
        gt = gt_kpts[..., :2].unsqueeze(1)  # (N,1,K,2)
        d = (pred_p - gt).pow(2).sum(-1)  # (N,P,K)
        e = d / ((2 * self.sigmas).pow(2).view(1, 1, K) * (area[:, None] + 1e-9) * 2)
        term = (1 - torch.exp(-e)) * kpt_mask.unsqueeze(1)  # (N,P,K)
        best = term.sum(-1).min(-1).values  # (N,)
        vis = torch.sum(kpt_mask != 0, dim=1)
        return (best / (vis + 1e-9)).mean()'''


def p5_loss():
    p = os.path.join(SP, "utils", "loss.py")
    t = read(p)
    if "顺序无关" in t:
        return 0
    if P5_OLD is None or P5_NEW is None:
        print("  [跳过] loss.py：本脚本未内置该补丁文本（仅影响训练，不影响推理）")
        return -1
    n = t.count(P5_OLD)
    if n != 1:
        print(f"  [跳过] loss.py：待替换文本匹配到 {n} 处（期望 1 处），"
              f"可能 ultralytics 版本不同，需手动按 README 3.3 表格修改")
        return -1
    write(p, t.replace(P5_OLD, P5_NEW, 1))
    return 1


P5_CHECK = "顺序无关"


def build_checks():
    """每次按当前 SP 重新构造检查表（--venv 可能在 main 里改过 SP）"""
    return [
        ("1 坐标校验放宽      data/utils.py",
         os.path.join(SP, "data", "utils.py"), P1_CHECK),
        ("2 增强保留画外点    data/augment.py",
         os.path.join(SP, "data", "augment.py"), P2_CHECK),
        ("3 clip_keypoints    utils/instance.py",
         os.path.join(SP, "utils", "instance.py"), P3_CHECK),
        ("4a scale_coords     utils/ops.py",
         os.path.join(SP, "utils", "ops.py"), P4A_CHECK),
        ("4b 推理端 clip=False models/yolo/pose/predict.py",
         os.path.join(SP, "models", "yolo", "pose", "predict.py"), P4B_CHECK),
        ("5 顺序无关 kpt loss utils/loss.py（仅训练需要）",
         os.path.join(SP, "utils", "loss.py"), P5_CHECK),
    ]


def check():
    ok = True
    print("补丁自检:")
    for name, path, marker in build_checks():
        if not os.path.exists(path):
            print(f"  [缺文件] {name}"); ok = False; continue
        has = marker in read(path)
        print(f"  [{'OK ' if has else '未打'}] {name}")
        ok &= has
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="补齐缺失的补丁")
    ap.add_argument("--check", action="store_true", help="只检查（默认行为）")
    ap.add_argument("--venv", default=None, help="venv 根目录（默认按当前解释器推导）")
    args = ap.parse_args()
    global SP
    SP = os.path.join(default_site_packages(args.venv), "ultralytics")
    print(f"ultralytics 路径: {SP}")
    if not os.path.isdir(SP):
        print(f"找不到 ultralytics 包: {SP}\n（用 --venv 指定 venv 根目录，或在该 venv 里运行本脚本）")
        sys.exit(2)
    if not args.apply:
        sys.exit(0 if check() else 1)
    for name, path, marker in build_checks():
        if marker in read(path):
            print(f"  [跳过] {name}（已打）")
            continue
        backup(path)
    done = {"1": p1_utils(), "2": p2_augment(), "3": p3_instance(),
            "4a": p4_ops(), "4b": p4b_predict()}
    p5_loss()
    for key, n in done.items():
        print(f"  [应用] 补丁 {key}: {n} 处改动")
    print()
    ok = check()
    for _n, path, _m in build_checks():
        try:
            py_compile.compile(path, doraise=True)
        except Exception as e:
            print(f"  [语法错误] {path}: {e}"); ok = False
    print("\n结果:", "全部就绪" if ok else "仍有缺失, 见上")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
