import os
import shutil
import random

# ========== 配置区 ==========
src_dir = r"YoloV8_ClassFor6/dataset"          # 源文件夹（按类别存放图片）
dst_dir = r"YoloV8_ClassFor6/cls_dataset"      # 输出文件夹（YOLO 分类格式）
train_ratio = 0.85            # 训练集比例
seed = 42                     # 随机种子，保证每次划分一致

# 中文类名 -> 英文类名
class_map = {
    "报纸或海报": "newspaper_poster",
    "单侧书本页面": "single_page",
    "非刚体票据": "receipt",
    "双页面左右结构展开书本": "double_page_book",
    "显示器或投影屏": "screen",
    "unclassified": "unclassified"
}
# ============================

random.seed(seed)

for cn_name, en_name in class_map.items():
    src_path = os.path.join(src_dir, cn_name)
    if not os.path.isdir(src_path):
        print(f"⚠️ 跳过不存在的文件夹: {cn_name}")
        continue

    # 获取所有图片文件
    imgs = [f for f in os.listdir(src_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    random.shuffle(imgs)

    # 计算划分点
    n_total = len(imgs)
    n_train = int(n_total * train_ratio)

    # 分别复制到 train/val 对应的英文目录下
    for subset, subset_imgs in [("train", imgs[:n_train]), ("val", imgs[n_train:])]:
        target_dir = os.path.join(dst_dir, subset, en_name)
        os.makedirs(target_dir, exist_ok=True)
        for fname in subset_imgs:
            shutil.copy2(os.path.join(src_path, fname),
                         os.path.join(target_dir, fname))

    print(f"✅ {cn_name}: {n_total} 张 → train {n_train} / val {n_total - n_train}")

print(f"\n数据集划分完成，保存至: {dst_dir}")
print("结构如下:")
print(f"  {dst_dir}/")
print(f"    ├── train/")
for en_name in class_map.values():
    print(f"    │   ├── {en_name}/")
print(f"    └── val/")
for en_name in class_map.values():
    print(f"        ├── {en_name}/")