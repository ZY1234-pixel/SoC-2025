#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include "net.h"
#include "opencv2/opencv.hpp"

// 预处理参数（与训练时 preprocess_input 一致：ImageNet 标准）
constexpr float mean_vals[3] = {0.485f, 0.456f, 0.406f};
constexpr float norm_vals[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};

// 模型输入尺寸
constexpr int INPUT_W = 256;
constexpr int INPUT_H = 256;

/**
 * 预处理：resize + letterbox + 归一化
 * 完全复现训练时 utils/utils.py 中 resize_image 的逻辑
 */
ncnn::Mat preprocess(const cv::Mat& bgr_img, float& scale, int& pad_left, int& pad_top) {
    // 1. BGR -> RGB
    cv::Mat rgb_img;
    cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);

    // 2. 计算缩放和 letterbox 参数
    int iw = rgb_img.cols;
    int ih = rgb_img.rows;
    scale = std::min(static_cast<float>(INPUT_W) / iw, static_cast<float>(INPUT_H) / ih);
    int nw = static_cast<int>(iw * scale);
    int nh = static_cast<int>(ih * scale);

    // 3. 缩放
    cv::Mat resized;
    cv::resize(rgb_img, resized, cv::Size(nw, nh), 0, 0, cv::INTER_CUBIC);

    // 4. 创建灰底并粘贴缩放后的图
    pad_left = (INPUT_W - nw) / 2;
    pad_top  = (INPUT_H - nh) / 2;
    cv::Mat padded(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    resized.copyTo(padded(cv::Rect(pad_left, pad_top, nw, nh)));

    // === 5. 务必验证粘贴是否成功（调试用，稍后可注释） ===
    cv::imwrite("padded_debug.jpg", padded);

    // 6. 手动归一化并填入 ncnn::Mat
    ncnn::Mat in(INPUT_W, INPUT_H, 3);
    in.fill(0.0f);
    float* ch0 = in.channel(0);   // R
    float* ch1 = in.channel(1);   // G
    float* ch2 = in.channel(2);   // B

for (int y = 0; y < INPUT_H; ++y) {
    const cv::Vec3b* row = padded.ptr<cv::Vec3b>(y);
    for (int x = 0; x < INPUT_W; ++x) {
        ch0[y * INPUT_W + x] = row[x][0] / 255.0f;
        ch1[y * INPUT_W + x] = row[x][1] / 255.0f;
        ch2[y * INPUT_W + x] = row[x][2] / 255.0f;
    }
}
    return in;
}

/**
 * 从热力图解码角点坐标（逐通道 argmax，带亚像素精修）
 */
std::vector<cv::Point2f> decode_keypoints(const ncnn::Mat& heatmap) {
    std::vector<cv::Point2f> keypoints;
    const int H = heatmap.h;
    const int W = heatmap.w;
    const int C = heatmap.c;  // 应为 4

    for (int c = 0; c < C; ++c) {
        float max_val = -1.0f;
        int max_x = 0, max_y = 0;

        const float* channel_data = heatmap.channel(c);
        for (int y = 0; y < H; ++y) {
            const float* row = channel_data + y * W;
            for (int x = 0; x < W; ++x) {
                if (row[x] > max_val) {
                    max_val = row[x];
                    max_x = x;
                    max_y = y;
                }
            }
        }

        float fine_x = static_cast<float>(max_x);
        float fine_y = static_cast<float>(max_y);
        // 亚像素精修
        if (max_x > 0 && max_x < W - 1 && max_y > 0 && max_y < H - 1) {
            const float* row_m1 = channel_data + (max_y - 1) * W;
            const float* row_0  = channel_data + max_y * W;
            const float* row_p1 = channel_data + (max_y + 1) * W;
            float v_center = row_0[max_x];
            float v_left   = row_0[max_x - 1];
            float v_right  = row_0[max_x + 1];
            float v_top    = row_m1[max_x];
            float v_bottom = row_p1[max_x];

            float dx = 0.5f * (v_left - v_right) / (v_left + v_right - 2.0f * v_center + 1e-8f);
            float dy = 0.5f * (v_top - v_bottom) / (v_top + v_bottom - 2.0f * v_center + 1e-8f);
            fine_x += dx;
            fine_y += dy;
        }
        keypoints.emplace_back(fine_x, fine_y);
    }
    return keypoints;
}

/**
 * 将 256x256 输入图上的坐标映射回原图坐标
 */
std::vector<cv::Point2f> map_to_original(const std::vector<cv::Point2f>& kpts_256,
                                          float scale, int pad_left, int pad_top,
                                          int orig_w, int orig_h) {
    std::vector<cv::Point2f> kpts_orig;
    for (const auto& pt : kpts_256) {
        float x = (pt.x - pad_left) / scale;
        float y = (pt.y - pad_top) / scale;
        x = std::clamp(x, 0.0f, static_cast<float>(orig_w - 1));
        y = std::clamp(y, 0.0f, static_cast<float>(orig_h - 1));
        kpts_orig.emplace_back(x, y);
    }
    return kpts_orig;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // 1. 加载 ncnn 模型
    ncnn::Net net;
    net.opt.use_fp16_arithmetic = true;
net.opt.num_threads = 1;
    if (net.load_param("kpt_model_256x256.ncnn.param") != 0 ||
        net.load_model("kpt_model_256x256.ncnn.bin") != 0) {
        std::cerr << "Failed to load ncnn model!" << std::endl;
        return -1;
    }

    // 2. 读入图像
    cv::Mat bgr_img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (bgr_img.empty()) {
        std::cerr << "Failed to read image: " << argv[1] << std::endl;
        return -1;
    }
    int orig_w = bgr_img.cols;
    int orig_h = bgr_img.rows;

    // 3. 预处理
    float scale;
    int pad_left, pad_top;
    ncnn::Mat in = preprocess(bgr_img, scale, pad_left, pad_top);

    // 4. 推理
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    ex.extract("out0", out);  // 输出维度: (H, W, C) = (256, 256, 4)

    // 5. 解码角点（256x256 坐标）
    auto kpts_256 = decode_keypoints(out);

    // 6. 打印 256x256 峰值坐标（调试用）
    std::cout << "256x256 峰值坐标:" << std::endl;
    for (size_t i = 0; i < kpts_256.size(); ++i) {
        std::cout << "  角点" << i + 1 << ": (" << kpts_256[i].x << ", " << kpts_256[i].y << ")" << std::endl;
    }

    // 7. 映射回原图
    auto kpts_orig = map_to_original(kpts_256, scale, pad_left, pad_top, orig_w, orig_h);

    // 8. 打印原图坐标
    std::cout << "预测角点坐标 (原图尺寸):" << std::endl;
    for (size_t i = 0; i < kpts_orig.size(); ++i) {
        std::cout << "  角点" << i + 1 << ": (" << kpts_orig[i].x << ", " << kpts_orig[i].y << ")" << std::endl;
    }

    // 9. 打印预处理参数（调试用）
    std::cout << "预处理参数: scale=" << scale << ", pad_left=" << pad_left << ", pad_top=" << pad_top << std::endl;

    // 10. 可视化并保存
    cv::Mat vis_img = bgr_img.clone();
    for (size_t i = 0; i < kpts_orig.size(); ++i) {
        cv::circle(vis_img, cv::Point(static_cast<int>(kpts_orig[i].x), static_cast<int>(kpts_orig[i].y)),
                   10, cv::Scalar(0, 0, 255), -1);
        cv::putText(vis_img, "KP" + std::to_string(i + 1),
                    cv::Point(static_cast<int>(kpts_orig[i].x) + 15, static_cast<int>(kpts_orig[i].y) - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }
    cv::imwrite("kpt_result.jpg", vis_img);
    std::cout << "可视化结果已保存为 kpt_result.jpg" << std::endl;

    return 0;
}