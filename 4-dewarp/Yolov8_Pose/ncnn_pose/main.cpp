#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "net.h"

// 定义检测结果结构体
struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<cv::Point2f> kpts;
};

// ==========================================
// 辅助函数：计算 NMS 的 IoU
// ==========================================
static inline float intersection_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    const int n = faceobjects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

// ==========================================
// 几何校验：凸四边形检查
// ==========================================
bool validate_convex_quadrilateral(const std::vector<cv::Point2f>& corners) {
    if (corners.size() != 4) return false;

    auto cross = [](cv::Point2f o, cv::Point2f a, cv::Point2f b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    };

    int sign = 0;
    for (int i = 0; i < 4; i++) {
        cv::Point2f o = corners[i];
        cv::Point2f a = corners[(i + 1) % 4];
        cv::Point2f b = corners[(i + 2) % 4];
        float c = cross(o, a, b);
        
        if (std::abs(c) < 1e-6) return false;
        
        int current_sign = c > 0 ? 1 : -1;
        if (sign == 0) {
            sign = current_sign;
        } else if (sign != current_sign) {
            return false;
        }
    }
    return true;
}

// ==========================================
// 几何处理：顺时针排序 (TL, TR, BR, BL)
// ==========================================
std::vector<cv::Point2f> sort_corners(const std::vector<cv::Point2f>& corners) {
    std::vector<cv::Point2f> sorted(4);
    int tl_idx = 0, br_idx = 0, tr_idx = 0, bl_idx = 0;
    float min_sum = 1e9, max_sum = -1e9;
    float min_diff = 1e9, max_diff = -1e9;

    for (int i = 0; i < 4; ++i) {
        float sum = corners[i].x + corners[i].y;
        float diff = corners[i].x - corners[i].y;

        if (sum < min_sum) { min_sum = sum; tl_idx = i; }
        if (sum > max_sum) { max_sum = sum; br_idx = i; }
        if (diff > max_diff) { max_diff = diff; tr_idx = i; }
        if (diff < min_diff) { min_diff = diff; bl_idx = i; }
    }

    sorted[0] = corners[tl_idx];
    sorted[1] = corners[tr_idx];
    sorted[2] = corners[br_idx];
    sorted[3] = corners[bl_idx];
    return sorted;
}

// ==========================================
// 主类：文档四角点检测器 (NCNN版)
// ==========================================
class DocumentDetectorNCNN {
public:
    DocumentDetectorNCNN(const char* param_path, const char* bin_path) {
        yolo.opt.use_vulkan_compute = false; // 视端侧环境决定是否开启 Vulkan
        yolo.load_param(param_path);
        yolo.load_model(bin_path);
        input_size = 192;
        class_names = {"double_page_book", "newspaper_poster", "single_page", "receipt", "screen", "unclassified"};
    }

    bool detect(const cv::Mat& bgr_img, Object& best_result, float conf_thresh = 0.5f) {
        int w = bgr_img.cols;
        int h = bgr_img.rows;

        // 1. 预处理 (严格对齐 Python 的 INTER_AREA 和 Letterbox)
        float scale = std::min(192.f / w, 192.f / h);
        int new_w = std::round(w * scale);
        int new_h = std::round(h * scale);

        cv::Mat resized;
        cv::resize(bgr_img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);

        int dw = (192 - new_w) / 2;
        int dh = (192 - new_h) / 2;
        int top = dh, bottom = 192 - new_h - top;
        int left = dw, right = 192 - new_w - left;

        cv::Mat letterboxed;
        cv::copyMakeBorder(resized, letterboxed, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        // 2. 转换为 ncnn::Mat 并归一化 (YOLOv8 默认 / 255.0)
        ncnn::Mat in = ncnn::Mat::from_pixels(letterboxed.data, ncnn::Mat::PIXEL_BGR2RGB, 192, 192);
        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        in.substract_mean_normalize(0, norm_vals);

        // 3. 执行推理
        ncnn::Extractor ex = yolo.create_extractor();
        ex.input("in0", in);    // 默认 PNNX 输入节点名
        ncnn::Mat out;
        ex.extract("out0", out); // 默认 PNNX 输出节点名

        // 4. 解析输出 Tensor [22, 2880] (C, W) -> YOLOv8 格式: bbox(4) + cls(6) + kpts(12)
        std::vector<Object> proposals;
        int num_anchors = out.w;
        int num_channels = out.h; // 应该是 22

        for (int i = 0; i < num_anchors; i++) {
            // 找最大分类得分
            int best_class_id = -1;
            float max_score = -1.0f;
            for (int c = 0; c < 6; c++) {
                float score = out.row(4 + c)[i];
                if (score > max_score) {
                    max_score = score;
                    best_class_id = c;
                }
            }

            if (max_score > conf_thresh) {
                Object obj;
                obj.label = best_class_id;
                obj.prob = max_score;

                // 还原 BBox
                float cx = out.row(0)[i];
                float cy = out.row(1)[i];
                float bw = out.row(2)[i];
                float bh = out.row(3)[i];
                obj.rect.x = cx - bw / 2.0f;
                obj.rect.y = cy - bh / 2.0f;
                obj.rect.width = bw;
                obj.rect.height = bh;

                // 还原关键点 (4个点，每个点 x, y, visible)
                obj.kpts.resize(4);
                for (int k = 0; k < 4; k++) {
                    float kpt_x = out.row(10 + k * 3)[i];     // 10 是起始偏置 (4box + 6cls)
                    float kpt_y = out.row(10 + k * 3 + 1)[i];
                    
                    // 映射回原图尺寸
                    obj.kpts[k].x = (kpt_x - left) / scale;
                    obj.kpts[k].y = (kpt_y - top) / scale;
                }
                proposals.push_back(obj);
            }
        }

        if (proposals.empty()) return false;

        // 5. 按置信度排序并进行 NMS
        std::sort(proposals.begin(), proposals.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, 0.5f);
        
        best_result = proposals[picked[0]]; // 只取最高置信度的目标

        // 6. 亚像素精修 (SubPix)
        cv::Mat gray;
        cv::cvtColor(bgr_img, gray, cv::COLOR_BGR2GRAY);
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 50, 0.0001);
        cv::cornerSubPix(gray, best_result.kpts, cv::Size(9, 9), cv::Size(-1, -1), criteria);

        // 7. 几何规则过滤
        if (!validate_convex_quadrilateral(best_result.kpts)) return false;

        // 8. 顺时针排序角点
        best_result.kpts = sort_corners(best_result.kpts);

        return true;
    }

    std::vector<std::string> class_names;

private:
    ncnn::Net yolo;
    int input_size;
};

// ==========================================
// 测试主函数
// ==========================================
int main() {
    // 替换为你端侧实际的模型路径
    DocumentDetectorNCNN detector("model.ncnn.param", "model.ncnn.bin");

    cv::Mat image = cv::imread("5.jpg");
    if (image.empty()) {
        std::cerr << "图像读取失败!" << std::endl;
        return -1;
    }

    Object result;
    bool found = detector.detect(image, result, 0.5f);

    if (found) {
        std::cout << "检测类别: " << detector.class_names[result.label] << ", 置信度: " << result.prob << std::endl;
        const char* corner_names[] = {"TL", "TR", "BR", "BL"};
        
        for (int i = 0; i < 4; i++) {
            std::cout << corner_names[i] << ": (" << result.kpts[i].x << ", " << result.kpts[i].y << ")" << std::endl;
            // 画圆
            cv::circle(image, result.kpts[i], 8, cv::Scalar(0, 0, 255), -1);
            cv::putText(image, corner_names[i], cv::Point(result.kpts[i].x + 15, result.kpts[i].y), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
        }
        
        // 画连线
        std::vector<cv::Point> pts;
        for (const auto& p : result.kpts) pts.push_back(p);
        cv::polylines(image, pts, true, cv::Scalar(0, 255, 0), 3);

        cv::imwrite("result_cpp.jpg", image);
        std::cout << "已保存至 result_cpp.jpg" << std::endl;
    } else {
        std::cout << "未检测到有效文档。" << std::endl;
    }

    return 0;
}