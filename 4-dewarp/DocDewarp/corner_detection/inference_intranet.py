import os
import cv2
import numpy as np
import shutil
import tempfile
import time
import torch
from ultralytics import YOLO

# 基础目录
BASE_DIR = r"D:\奔图\deeplabv3p_zzh\YoloV8_Pose"
RUNS_DIR = os.path.join(BASE_DIR, "runs")


class Document4CornerDetector:
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        self.input_size = 192

        # 类别名称
        self.class_names = [
            "double_page_book",
            "newspaper_poster",
            "receipt",
            "screen",
            "single_page",
            "unclassified"
        ]

        self._safe_model_path = None
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型文件: {model_path}")

        print(f"正在准备加载模型: {model_path}")
        # 如果是 TorchScript 模型，使用纯英文临时路径进行映射
        if model_path.endswith('.torchscript'):
            temp_dir = os.path.join(tempfile.gettempdir(), "yolo_pose_infer")
            os.makedirs(temp_dir, exist_ok=True)
            self._safe_model_path = os.path.join(temp_dir, "best.torchscript")
            shutil.copy2(model_path, self._safe_model_path)
            # 明确 task='pose'，确保 TorchScript 加载正确的后处理头
            self.model = YOLO(self._safe_model_path, task='pose')
        else:
            self.model = YOLO(model_path)

    def __del__(self):
        # 析构时清理临时模型文件
        if self._safe_model_path and os.path.exists(self._safe_model_path):
            try:
                os.remove(self._safe_model_path)
            except Exception:
                pass

    def preprocess(self, image):
        """4K图像预处理：区域插值下采样+letterbox填充"""
        h, w = image.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        # 区域插值下采样（比双线性插值更适合角点特征）
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 添加黑边到192x192
        dw = (self.input_size - new_w) / 2
        dh = (self.input_size - new_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        letterboxed = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return letterboxed, scale, dw, dh

    def postprocess(self, results, original_image, scale, dw, dh):
        """后处理：坐标映射+亚像素精修+凸四边形校验"""
        if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return None, None, None

        # 获取最高置信度的检测结果
        result = results[0]
        if len(result.boxes) == 0:
            return None, None, None

        best_idx = result.boxes.conf.argmax().item()
        cls_id = int(result.boxes.cls[best_idx].item())
        conf = result.boxes.conf[best_idx].item()
        cls_name = self.class_names[cls_id]

        # 获取4个角点坐标（192x192坐标系）
        kpts = result.keypoints.xy[best_idx].cpu().numpy()

        # 映射回原图坐标系
        kpts[:, 0] = (kpts[:, 0] - dw) / scale
        kpts[:, 1] = (kpts[:, 1] - dh) / scale

        # 亚像素角点精修
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
        refined_kpts = cv2.cornerSubPix(
            gray, kpts.astype(np.float32), (9, 9), (-1, -1), criteria
        )

        # 严格的凸四边形校验
        if not self._validate_convex_quadrilateral(refined_kpts):
            return None, None, None

        # 按顺时针顺序排序角点
        sorted_kpts = self._sort_corners(refined_kpts)

        return cls_name, conf, sorted_kpts

    def _validate_convex_quadrilateral(self, corners):
        """校验四个点是否构成凸四边形"""
        if len(corners) != 4:
            return False

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        signs = []
        for i in range(4):
            o = corners[i]
            a = corners[(i + 1) % 4]
            b = corners[(i + 2) % 4]
            c = cross(o, a, b)
            if abs(c) < 1e-6:
                return False
            signs.append(np.sign(c))

        return all(s == signs[0] for s in signs)

    def _sort_corners(self, corners):
        """将四个角点按顺时针顺序排序：左上→右上→右下→左下"""
        sum_coords = corners[:, 0] + corners[:, 1]
        tl_idx = np.argmin(sum_coords)
        br_idx = np.argmax(sum_coords)

        diff_coords = corners[:, 0] - corners[:, 1]
        tr_idx = np.argmax(diff_coords)
        bl_idx = np.argmin(diff_coords)

        return np.array([
            corners[tl_idx],
            corners[tr_idx],
            corners[br_idx],
            corners[bl_idx]
        ])

    def detect(self, image, conf_threshold=0.5):
        """完整检测流程"""
        input_image, scale, dw, dh = self.preprocess(image)

        # 执行推理，屏蔽控制台冗余输出
        results = self.model(
            input_image,
            device=self.device,
            conf=conf_threshold,
            iou=0.5,
            verbose=False
        )

        return self.postprocess(results, image, scale, dw, dh)

    def draw_result(self, image, cls_name, conf, corners):
        """绘制检测结果 (多边形 + 顺时针角点)"""
        if corners is None:
            return image

        # 复制原图以避免修改原始图像内存
        out_img = image.copy()

        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out_img, [pts], True, (0, 255, 0), 3)

        corner_names = ["TL", "TR", "BR", "BL"]
        for i, (x, y) in enumerate(corners):
            cv2.circle(out_img, (int(x), int(y)), 8, (0, 0, 255), -1)
            cv2.putText(
                out_img, corner_names[i], (int(x) + 15, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3
            )

        text = f"{cls_name} {conf:.2f}"
        cv2.putText(
            out_img, text, (int(corners[0][0]), int(corners[0][1]) - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
        )

        return out_img


# ==========================================
# 测试代码执行
# ==========================================
if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 替换为你导出的 TorchScript 模型路径
    ts_model_path = os.path.join(RUNS_DIR, "pose_train_final", "weights", "best.torchscript")

    # 初始化你的检测器
    detector = Document4CornerDetector(model_path=ts_model_path, device=device)

    # 修改为你的真实测试图片路径
    test_image_path = os.path.join(BASE_DIR, "data", "val", "images", "5.jpg")
    image = cv2.imread(test_image_path)

    if image is None:
        print(f" 无法读取图片: {test_image_path}")
        exit()

    # 预热推理（CUDA初次加载会有延迟）
    print("模型预热中...")
    for _ in range(3):
        _ = detector.detect(image)

    # 包含前后处理的全流程计时
    start_time = time.time()
    cls_name, conf, corners = detector.detect(image)
    end_time = time.time()

    print(f"\n--- 推理完成 (含前后处理总耗时: {(end_time - start_time) * 1000:.2f} ms) ---")

    if corners is not None:
        print(f"检测结果类别: {cls_name}, 置信度: {conf:.4f}")
        print(f"四个角点坐标 :")
        for i, name in enumerate(["左上(TL)", "右上(TR)", "右下(BR)", "左下(BL)"]):
            print(f"  {name}: ({corners[i][0]:.2f}, {corners[i][1]:.2f})")

        result_image = detector.draw_result(image, cls_name, conf, corners)
        output_path = os.path.join(BASE_DIR, "result_4corners_custom.jpg")
        cv2.imwrite(output_path, result_image)
        print(f"\n 结果已使用多边形框渲染，并保存为: {output_path}")
    else:
        print("未检测到有效文档 (可能未通过置信度或凸四边形校验)")