import numpy as np
import torch
import warnings
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
import os
import glob
import argparse
import sys
import time
from typing import Optional

# -------------------------- 导入模型相关模块 --------------------------
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from Hi_SAM.hi_sam.modeling.build import model_registry
from Hi_SAM.hi_sam.modeling.predictor import SamPredictor
from ultralytics import YOLO
warnings.filterwarnings("ignore")


# -------------------------- COCO格式工具函数 --------------------------
def init_coco_format():
    """初始化COCO格式数据结构"""
    return {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "text", "supercategory": "object"},
            {"id": 2, "name": "edge", "supercategory": "object"},
            {"id": 3, "name": "image", "supercategory": "object"},
            {"id": 4, "name": "person", "supercategory": "object"}
        ],
        "images": [],
        "annotations": []
    }


def mask_to_coco_rle(mask):
    """将二值掩码转换为COCO RLE格式"""
    mask = mask.astype(np.uint8)
    rle = {"counts": [], "size": list(mask.shape)}
    counts = []
    prev = 0

    for pixel in mask.flatten(order='F'):
        if pixel != prev:
            counts.append(1)
            prev = pixel
        else:
            if counts:
                counts[-1] += 1
            else:
                counts.append(1)

    if not counts:
        counts = [mask.size]

    rle["counts"] = counts
    return rle


def add_coco_annotation(coco_data, img_id, mask, category_id):
    """向COCO数据中添加标注"""
    if np.sum(mask) == 0:
        print(f"跳过空掩码标注（类别ID: {category_id}，图像ID: {img_id}）")
        return

    area = int(np.sum(mask))
    where = np.argwhere(mask)
    if len(where) == 0:
        print(f"掩码无有效像素（类别ID: {category_id}，图像ID: {img_id}）")
        return

    y1, x1 = where.min(axis=0)
    y2, x2 = where.max(axis=0)
    bbox = [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]
    rle = mask_to_coco_rle(mask)

    annotation = {
        "id": len(coco_data["annotations"]) + 1,
        "image_id": img_id,
        "category_id": category_id,
        "segmentation": rle,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
        "attributes": {}
    }

    coco_data["annotations"].append(annotation)


# -------------------------- 全局配置与工具函数 --------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('Fast-SAM + Hi-SAM + YOLOv8-Seg', add_help=False)
    # 通用配置
    parser.add_argument("--input", type=str, default="./input", help="输入图像文件夹路径")
    parser.add_argument("--output", type=str, default="./final_results", help="结果保存根目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="运行设备")

    # Fast-SAM配置
    parser.add_argument("--fastsam_checkpoint", type=str, default="FastSAM/weights/FastSAM-x.pt", help="Fast-SAM权重路径")
    parser.add_argument("--fastsam_conf", type=float, default=0.4, help="Fast-SAM置信度阈值")
    parser.add_argument("--fastsam_iou", type=float, default=0.9, help="Fast-SAM IoU阈值")
    parser.add_argument("--fastsam_imgsz", type=int, default=640, help="Fast-SAM输入图像尺寸")

    # yolo配置
    parser.add_argument("--yolo_checkpoint", type=str, default="yolo_weights/yolov8m-seg.pt", help="yolo权重路径")

    # Hi-SAM配置
    parser.add_argument("--hisam_model_type", type=str, default="vit_s",
                        help="Hi-SAM模型类型 ['vit_h', 'vit_l', 'vit_b','vit_s']")
    parser.add_argument("--hisam_checkpoint", type=str, default="Hi_SAM/pretrained_checkpoint/efficient_hi_sam_s.pth", help="Hi-SAM权重路径")
    parser.add_argument("--hisam_hier_det", action='store_true', help="Hi-SAM是否启用层级检测")
    parser.add_argument("--hisam_patch_mode", action='store_true', help="Hi-SAM是否启用patch模式")
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument('--attn_layers', default=1, type=int, help='cross attention layers数')
    parser.add_argument('--prompt_len', default=12, type=int, help='prompt token数')

    # 后处理配置
    parser.add_argument("--text_dilate_pixel", type=int, default=10, help="文本掩码膨胀像素数")
    parser.add_argument("--edge_white_value", type=int, default=255, help="边缘掩码白色值")
    parser.add_argument("--fill_black_value", type=int, default=0, help="重叠区域填充黑色值")

    return parser.parse_args()


# -------------------------- Hi-SAM工具函数 --------------------------
def patchify_sliding(image: np.array, patch_size: int = 512, stride: int = 256):
    h, w = image.shape[:2]
    patch_list = []
    h_slice_list = []
    w_slice_list = []
    for j in range(0, h, stride):
        start_h, end_h = j, j + patch_size
        if end_h > h:
            start_h = max(h - patch_size, 0)
            end_h = h
        for i in range(0, w, stride):
            start_w, end_w = i, i + patch_size
            if end_w > w:
                start_w = max(w - patch_size, 0)
                end_w = w
            h_slice = slice(start_h, end_h)
            h_slice_list.append(h_slice)
            w_slice = slice(start_w, end_w)
            w_slice_list.append(w_slice)
            patch_list.append(image[h_slice, w_slice])
    return patch_list, h_slice_list, w_slice_list


def unpatchify_sliding(patch_list, h_slice_list, w_slice_list, ori_size):
    assert len(ori_size) == 2
    whole_logits = np.zeros(ori_size)
    assert len(patch_list) == len(h_slice_list) == len(w_slice_list)
    for idx in range(len(patch_list)):
        h_slice = h_slice_list[idx]
        w_slice = w_slice_list[idx]
        whole_logits[h_slice, w_slice] += patch_list[idx]
    return whole_logits


# -------------------------- 掩码优化函数 --------------------------
def refine_edge_mask(
        edge_mask: np.ndarray,
        text_mask: Optional[np.ndarray] = None,
        edge_white_value: int = 255,
        fill_black_value: int = 0,
        text_dilate_pixel: int = 20
) -> np.ndarray:
    """优化SAM边缘掩码"""
    if len(edge_mask.shape) == 3:
        edge_mask_gray = cv2.cvtColor(edge_mask, cv2.COLOR_BGR2GRAY)
    else:
        edge_mask_gray = edge_mask.copy()
    _, edge_mask_bin = cv2.threshold(
        edge_mask_gray,
        edge_white_value - 1,
        edge_white_value,
        cv2.THRESH_BINARY
    )

    refined_edge_mask = edge_mask_bin.copy()

    if text_mask is not None:
        if len(text_mask.shape) == 3:
            text_mask_gray = cv2.cvtColor(text_mask, cv2.COLOR_BGR2GRAY)
        else:
            text_mask_gray = text_mask.copy()
        _, text_mask_bin = cv2.threshold(text_mask_gray, 1, 255, cv2.THRESH_BINARY)

        dilate_kernel = np.ones((text_dilate_pixel * 2 + 1, text_dilate_pixel * 2 + 1), np.uint8)
        text_mask_dilated = cv2.dilate(text_mask_bin, dilate_kernel, iterations=1)

        text_edge_overlap = np.logical_and(edge_mask_bin == edge_white_value, text_mask_dilated == 255)
        refined_edge_mask[text_edge_overlap] = fill_black_value

    return refined_edge_mask




# -------------------------- 模型推理函数 --------------------------
def run_fastsam_inference(img_path, fastsam_model, device, imgsz=1024, conf=0.4, iou=0.9):
    """Fast-SAM推理：返回边缘掩码数组、原始物体掩码列表，推理耗时"""
    try:
        start_time = time.time()
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        everything_results = fastsam_model(
            img_path,
            device=device,
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou
        )

        prompt_process = FastSAMPrompt(img_path, everything_results, device=device)
        ann = prompt_process.everything_prompt()


        edge_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        object_masks = []

        for mask in ann:
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            object_masks.append(mask_np)
            edges = cv2.Canny(mask_np, threshold1=50, threshold2=150)
            edge_mask = cv2.bitwise_or(edge_mask, edges)

        fastsam_infer_time = round((time.time() - start_time) * 1000, 1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "img_name": Path(img_path).stem,
            "img_size": (img_h, img_w),
            "sam_edge_mask": edge_mask,
            "object_masks": object_masks,
            "sam_infer_time": fastsam_infer_time
        }
    except Exception as e:
        return {
            "status": "failed",
            "img_path": img_path,
            "error": str(e),
            "sam_infer_time": 0.0
        }

def run_yolov8_inference(img_path, yolo_model, device, imgsz=640, conf=0.25, iou=0.7):
    """yolov8推理"""
    try:
        start_time = time.time()
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        results=yolo_model(image)

        # Create an empty mask for segmentation
        person_mask = np.zeros((img_h,img_w), dtype=np.uint8)
        obj_mask = np.zeros((img_h,img_w), dtype=np.uint8)
        # obj_masks = []

        # Iterate over the results
        for i, r in enumerate(results):
            # Iterate through the detected masks
            if r.masks is not None:
                for j, mask in enumerate(r.masks.xy):
                    # Convert the class tensor to an integer
                    class_id = int(r.boxes.cls[j].item())  # Extract the class ID as an integer

                    # Check if the detected class corresponds to 'person' (class ID 0)
                    if class_id == 0:
                        # Convert mask coordinates to an integer format for drawing
                        mask = np.array(mask, dtype=np.int32)

                        # Fill the segmentation mask with color (e.g., white for people)
                        cv2.fillPoly(person_mask, [mask], 255)
                    elif class_id>0:
                        mask = np.array(mask, dtype=np.int32)
                        # obj_mask=np.zeros((img_h,img_w), dtype=np.uint8)
                        cv2.fillPoly(obj_mask, [mask], 255)
                        # Fill the segmentation mask with color (e.g., white for people)
                        # obj_masks.append(obj_mask)

        yolo_infer_time = round((time.time() - start_time) * 1000, 1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "img_name": Path(img_path).stem,
            "img_size": (img_h, img_w),
            "person_mask": person_mask,
            "obj_mask": obj_mask,
            "yolo_infer_time": yolo_infer_time
        }
    except Exception as e:
        return {
            "status": "failed",
            "img_path": img_path,
            "error": str(e),
            "yolo_infer_time": 0.0
        }

def run_hisam_inference(img_path, hisam_model, hier_det=False, patch_mode=False):
    """Hi-SAM推理：返回文本掩码数组 + 推理耗时"""
    try:
        start_time = time.time()
        predictor = SamPredictor(hisam_model)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_name = Path(img_path).stem

        if patch_mode:
            ori_size = image.shape[:2]
            patch_list, h_slice_list, w_slice_list = patchify_sliding(image_rgb, 512, 384)
            mask_512 = []
            for patch in patch_list:
                predictor.set_image(patch)
                m, hr_m, score, hr_score = predictor.predict(multimask_output=False, return_logits=True)
                mask_512.append(hr_m[0])
            mask_512 = unpatchify_sliding(mask_512, h_slice_list, w_slice_list, ori_size)
            text_mask = (mask_512 > predictor.model.mask_threshold).astype(np.uint8) * 255
        else:
            predictor.set_image(image_rgb)
            if hier_det:
                input_point = np.array([[125, 275]])
                input_label = np.ones(input_point.shape[0])
                mask, hr_mask, score, hr_score, hi_mask, hi_iou, word_mask = predictor.predict(
                    multimask_output=False,
                    hier_det=True,
                    point_coords=input_point,
                    point_labels=input_label,
                )
                text_mask = hr_mask[0].astype(np.uint8) * 255
            else:
                mask, hr_mask, score, hr_score = predictor.predict(multimask_output=False)
                text_mask = hr_mask[0].astype(np.uint8) * 255

        hisam_infer_time = round((time.time() - start_time) * 1000, 1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "img_name": img_name,
            "hisam_text_mask": text_mask,
            "hisam_infer_time": hisam_infer_time
        }
    except Exception as e:
        return {
            "status": "failed",
            "img_path": img_path,
            "error": str(e),
            "hisam_infer_time": 0.0
        }


# -------------------------- 主函数 --------------------------
def main():
    args = get_args_parser()

    # 创建结果目录
    os.makedirs(args.output, exist_ok=True)
    print(f"结果保存目录：{args.output}")

    # 加载模型
    print("\n加载模型...")
    fastsam = FastSAM(args.fastsam_checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    hisam = model_registry[args.hisam_model_type](args)
    hisam.eval()
    hisam.to(device)

    yolo_seg=YOLO(args.yolo_checkpoint)
    print(f"模型加载完成，使用设备：{device}")

    # 获取输入图像列表
    input_images = []
    if os.path.isdir(args.input):
        for fname in os.listdir(args.input):
            img_path = os.path.join(args.input, fname)
            if cv2.haveImageReader(img_path):
                input_images.append(img_path)
    else:
        input_images = glob.glob(os.path.expanduser(args.input))

    assert len(input_images) > 0, "未找到有效输入图像"
    print(f"\n待处理图像数量：{len(input_images)}")

    # 初始化时间统计变量
    total_sam_time = 0.0
    total_hisam_time = 0.0
    total_yolo_time = 0.0
    success_sam_count = 0
    success_hisam_count = 0
    success_yolo_count = 0
    time_stats = []

    # 串行运行推理
    print("\n开始串行推理...")
    inference_results = {}
    success_count = 0

    for img_idx, img_path in enumerate(tqdm(input_images, desc="推理")):
        img_name = Path(img_path).stem
        inference_results[img_name] = {}

        # 初始化COCO格式数据
        coco_data = init_coco_format()

        # 添加图像信息到COCO数据
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        coco_data["images"].append({
            "id": img_idx + 1,
            "width": img_w,
            "height": img_h,
            "file_name": os.path.basename(img_path)
        })

        # 执行Fast-SAM推理
        sam_result = run_fastsam_inference(
            img_path=img_path,
            fastsam_model=fastsam,
            device=device,
            imgsz=args.fastsam_imgsz,
            conf=args.fastsam_conf,
            iou=args.fastsam_iou
        )
        inference_results[img_name]["sam"] = sam_result

        if sam_result["status"] == "success":
            total_sam_time += sam_result["sam_infer_time"]
            success_sam_count += 1

        # 执行Hi-SAM推理
        hisam_result = run_hisam_inference(
            img_path=img_path,
            hisam_model=hisam,
            hier_det=args.hisam_hier_det,
            patch_mode=args.hisam_patch_mode
        )
        inference_results[img_name]["hisam"] = hisam_result

        if hisam_result["status"] == "success":
            total_hisam_time += hisam_result["hisam_infer_time"]
            success_hisam_count += 1

        yolo_result=run_yolov8_inference(img_path=img_path,yolo_model=yolo_seg,device=device)
        inference_results[img_name]["yolo"] = yolo_result

        if yolo_result["status"] == "success":
            total_yolo_time += yolo_result["yolo_infer_time"]
            success_yolo_count += 1

        # 记录单张图片耗时
        time_stats.append({
            "img_name": img_name,
            "sam_time": sam_result["sam_infer_time"],
            "hisam_time": hisam_result["hisam_infer_time"],
            "yolo_time": yolo_result["yolo_infer_time"],
            "sam_status": sam_result["status"],
            "hisam_status": hisam_result["status"],
            "yolo_status": yolo_result["status"]
        })

        # 掩码优化 + 保存 + 生成COCO标注
        if (sam_result["status"] == "success" and
                hisam_result["status"] == "success" and yolo_result["status"]=="success"):
            # 获取各类掩码
            sam_edge_mask = sam_result["sam_edge_mask"]
            hisam_text_mask = hisam_result["hisam_text_mask"]
            # object_masks = sam_result["object_masks"]
            object_masks = yolo_result["obj_mask"]
            person_mask = yolo_result["person_mask"]  # 人体掩码
            img_h, img_w = sam_result["img_size"]

            # 1. 保存文本掩码并添加到COCO
            text_mask_path = os.path.join(args.output, f"{img_name}_hisam_text_mask.png")
            cv2.imwrite(text_mask_path, hisam_text_mask)
            text_mask_bin = (hisam_text_mask > 127).astype(np.uint8)
            add_coco_annotation(coco_data, img_idx + 1, text_mask_bin, 1)

            # 2. 优化边缘掩码并添加到COCO
            refined_edge_mask = refine_edge_mask(
                edge_mask=sam_edge_mask,
                text_mask=hisam_text_mask,
                edge_white_value=args.edge_white_value,
                fill_black_value=args.fill_black_value,
                text_dilate_pixel=args.text_dilate_pixel
            )
            refined_mask_path = os.path.join(args.output, f"{img_name}_edge_mask.png")
            cv2.imwrite(refined_mask_path, refined_edge_mask)
            edge_mask_bin = (refined_edge_mask > 127).astype(np.uint8)
            add_coco_annotation(coco_data, img_idx + 1, edge_mask_bin, 2)

            # 3. 处理物体掩码并添加到COCO
            combined_object_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            mask_bin = (object_masks > 127).astype(np.uint8)
            combined_object_mask = np.logical_or(combined_object_mask, mask_bin).astype(np.uint8)
            # for mask in object_masks:
            #     mask_bin = (mask > 127).astype(np.uint8)
            #     combined_object_mask = np.logical_or(combined_object_mask, mask_bin).astype(np.uint8)

            text_mask_dilated = cv2.dilate(text_mask_bin, np.ones((5, 5), np.uint8), iterations=1)
            # exclude_mask = np.logical_or(text_mask_dilated, edge_mask_bin).astype(np.uint8)
            combined_object_mask = np.logical_and(combined_object_mask, 1 - text_mask_dilated).astype(np.uint8)

            object_mask_path = os.path.join(args.output, f"{img_name}_object_mask.png")
            cv2.imwrite(object_mask_path, combined_object_mask * 255)
            add_coco_annotation(coco_data, img_idx + 1, combined_object_mask, 3)

            # 4. 处理人体掩码并添加到COCO
            person_mask_bin = (person_mask > 127).astype(np.uint8)
            # 应用置信度阈值过滤
            # if np.sum(person_mask_bin) > 0:
            person_mask_path = os.path.join(args.output, f"{img_name}_person_mask.png")
            cv2.imwrite(person_mask_path, person_mask)
            add_coco_annotation(coco_data, img_idx + 1, person_mask_bin, 4)  # 类别4: person
            inference_results[img_name]["person_mask_path"] = person_mask_path

            # 保存COCO格式JSON文件
            coco_json_path = os.path.join(args.output, f"{img_name}_coco_annotations.json")
            with open(coco_json_path, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, ensure_ascii=False, indent=2)

            inference_results[img_name]["refined_edge_mask_path"] = refined_mask_path
            inference_results[img_name]["hisam_text_mask_path"] = text_mask_path
            inference_results[img_name]["object_mask_path"] = object_mask_path
            inference_results[img_name]["coco_json_path"] = coco_json_path

            success_count += 1
        else:
            print(f"\n跳过{img_name}：推理失败")
            if sam_result["status"] == "failed":
                print(f"   - Fast-SAM失败原因：{sam_result['error']}")
            if hisam_result["status"] == "failed":
                print(f"   - Hi-SAM失败原因：{hisam_result['error']}")
            if yolo_result["status"] == "failed":
                print(f"   - YOLO失败原因：{yolo_result['error']}")


        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # 时间统计结果输出
    print("\n" + "-" * 60)
    print("推理时间统计（单位：毫秒 ms）")
    print("-" * 60)
    print(f"总处理图片数：{len(input_images)}")
    print(
        f"Fast-SAM成功推理数：{success_sam_count} | Hi-SAM成功推理数：{success_hisam_count}| YOLO成功推理数：{success_yolo_count}")
    print(f"Fast-SAM总耗时：{total_sam_time:.1f} ms | 平均每张：{total_sam_time / max(success_sam_count, 1):.1f} ms")
    print(f"Hi-SAM总耗时：{total_hisam_time:.1f} ms | 平均每张：{total_hisam_time / max(success_hisam_count, 1):.1f} ms")
    print(f"YOLO总耗时：{total_yolo_time:.1f} ms | 平均每张：{total_yolo_time / max(success_yolo_count, 1):.1f} ms")

    # 单张图片明细
    print("\n单张图片耗时明细：")
    for stat in time_stats:
        status = f"Fast-SAM: {stat['sam_status']} | Hi-SAM: {stat['hisam_status']}"
        print(
            f"  {stat['img_name']} | Fast-SAM: {stat['sam_time']:.1f}ms | Hi-SAM: {stat['hisam_time']:.1f}ms| YOLO: {stat['yolo_time']:.1f}ms")

    # 最终结果输出
    print("\n任务完成！成功处理 " + f"{success_count}/{len(input_images)} 张图像")
    print(f"结果保存目录：{args.output}")
    print("保存文件包括：")
    print("   - {img_name}_hisam_text_mask.png: 文本掩码")
    print("   - {img_name}_refined_edge_mask.png: 边缘掩码")
    print("   - {img_name}_object_mask.png: 物体掩码")
    print("   - {img_name}_person_mask.png: 人体掩码")  # 新增
    print("   - {img_name}_coco_annotations.json: COCO格式标注文件")


if __name__ == '__main__':
    main()