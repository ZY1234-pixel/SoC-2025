"""
debug_save.py

在 opt() 方法运行时保存关键中间变量，供后续分析。
保存格式：numpy .npz 压缩文件（可用 np.load 读取）

使用方法：
    from optim.debug_save import save_opt_debug_vars
    save_opt_debug_vars(u, v, uv, coord_S, top, right, bottom, left,
                        textline, textline1, grid1=None)

读取方法：
    import numpy as np
    data = np.load('optim/debug_vars.npz', allow_pickle=True)
    uv = data['uv']
    coord_S = data['coord_S']
    # 等等...
"""

import os
import numpy as np


# 默认保存路径（与 opt.py 同目录）
_DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_vars.npz")


def save_opt_debug_vars(u, v, uv, coord_S,
                        top, right, bottom, left,
                        textline, textline1,
                        grid1=None,
                        save_path=None):
    """
    保存 opt() 中的关键变量到 .npz 文件。

    参数
    ----
    u         : (n, n) ndarray  — grid(top, bottom, ...) 的求解结果（水平参数坐标场）
    v         : (n, n) ndarray  — grid(left, right, ...) 的求解结果（垂直参数坐标场）
    uv        : (n*n, 2) ndarray — np.stack([u.T, v], -1).reshape(n*n, 2)
    coord_S   : (n*n, 2) ndarray — 均匀参数网格坐标，值域 [0,1]×[0,1]
    top       : (n, 2) ndarray  — 上边界点（网格坐标系）
    right     : (n, 2) ndarray  — 右边界点（网格坐标系）
    bottom    : (n, 2) ndarray  — 下边界点（网格坐标系）
    left      : (n, 2) ndarray  — 左边界点（网格坐标系）
    textline  : list of ndarray — 水平文本行点集（已缩放到 [0, n-1]）
    textline1 : list of ndarray — 垂直文本行点集（已缩放到 [0, n-1]）
    grid1     : (n*n, 2) ndarray or None — 插值结果（[-1,1] 范围），可选
    save_path : str or None     — 保存路径，默认为 optim/debug_vars.npz
    """
    if save_path is None:
        save_path = _DEFAULT_SAVE_PATH

    # ── 统计信息（打印到控制台）──────────────────────────────────
    print("\n" + "="*60)
    print("  [debug_save] opt() 关键变量统计")
    print("="*60)

    def _stat(name, arr):
        if arr is None:
            print(f"  {name:20s}: None")
            return
        arr = np.asarray(arr)
        print(f"  {name:20s}: shape={arr.shape}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  "
              f"mean={arr.mean():.4f}  std={arr.std():.4f}")

    _stat("u", u)
    _stat("v", v)
    _stat("uv", uv)
    _stat("coord_S", coord_S)
    _stat("top", top)
    _stat("right", right)
    _stat("bottom", bottom)
    _stat("left", left)
    print(f"  {'textline':20s}: {len(textline)} 条水平文本行")
    # for k, tl in enumerate(textline):
        # _stat(f"    textline[{k}]", tl)
    print(f"  {'textline1':20s}: {len(textline1)} 条垂直文本行")
    # for k, tl in enumerate(textline1):
        # _stat(f"    textline1[{k}]", tl)
    if grid1 is not None:
        _stat("grid1", grid1)
    print("="*60 + "\n")

    # ── 额外诊断：检查 uv 是否超出 [0,1] ────────────────────────
    uv_arr = np.asarray(uv)
    out_of_range = np.sum((uv_arr < 0) | (uv_arr > 1))
    total = uv_arr.size
    print(f"  [诊断] uv 超出 [0,1] 的元素数: {out_of_range} / {total} "
          f"({100*out_of_range/total:.1f}%)")

    # ── 保存到 .npz ──────────────────────────────────────────────
    save_dict = dict(
        u=np.asarray(u),
        v=np.asarray(v),
        uv=np.asarray(uv),
        coord_S=np.asarray(coord_S),
        top=np.asarray(top),
        right=np.asarray(right),
        bottom=np.asarray(bottom),
        left=np.asarray(left),
        textline=np.array(textline, dtype=object),
        textline1=np.array(textline1, dtype=object),
    )
    if grid1 is not None:
        save_dict['grid1'] = np.asarray(grid1)

    np.savez_compressed(save_path, **save_dict)
    print(f"  [debug_save] 变量已保存到: {save_path}\n")


def load_and_report(save_path=None):
    """
    读取并打印已保存的 debug 变量（供分析时调用）。
    """
    if save_path is None:
        save_path = _DEFAULT_SAVE_PATH

    data = np.load(save_path, allow_pickle=True)
    print(f"\n已加载: {save_path}")
    print(f"包含的键: {list(data.keys())}\n")

    for key in data.files:
        arr = data[key]
        if arr.dtype == object:
            print(f"  {key:20s}: object array, len={len(arr)}")
            for k, item in enumerate(arr):
                item = np.asarray(item)
                print(f"    [{k}] shape={item.shape}  "
                      f"min={item.min():.4f}  max={item.max():.4f}")
        else:
            print(f"  {key:20s}: shape={arr.shape}  "
                  f"min={arr.min():.4f}  max={arr.max():.4f}  "
                  f"mean={arr.mean():.4f}")
    return data


if __name__ == "__main__":
    # 直接运行此脚本时，读取并报告已保存的变量
    load_and_report()
