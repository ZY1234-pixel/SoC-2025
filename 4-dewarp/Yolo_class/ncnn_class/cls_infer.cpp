#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include "net.h"
#include "opencv2/opencv.hpp"

// 模型输入尺寸
constexpr int INPUT_W = 256;
constexpr int INPUT_H = 256;

// 类别映射（按训练时文件夹名的字母顺序：double_page_book, newspaper_poster, receipt, screen, single_page, unclassified）
const std::vector<std::string> CLASS_NAMES = {
    "double_page_book",
    "newspaper_poster",
    "receipt",
    "screen",
    "single_page",
    "unclassified"
};

/**
 * 预处理：resize + letterbox + 归一化（仅除以255，与训练完全一致）
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

    // 4. 灰边填充
    pad_left = (INPUT_W - nw) / 2;
    pad_top  = (INPUT_H - nh) / 2;
    cv::Mat padded(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    resized.copyTo(padded(cv::Rect(pad_left, pad_top, nw, nh)));

    // 5. 归一化并填入 ncnn::Mat（仅除以255）
    ncnn::Mat in(INPUT_W, INPUT_H, 3);
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // 1. 加载 ncnn 模型
    ncnn::Net net;
    net.opt.use_fp16_arithmetic = false;   
    net.opt.num_threads = 1; 

    if (net.load_param("best.ncnn.param") != 0 ||
        net.load_model("best.ncnn.bin") != 0) {
        std::cerr << "Failed to load ncnn model!" << std::endl;
        return -1;
    }

    // 2. 读入图像
    cv::Mat bgr_img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (bgr_img.empty()) {
        std::cerr << "Failed to read image: " << argv[1] << std::endl;
        return -1;
    }

    // 3. 预处理
    float scale;
    int pad_left, pad_top;
    ncnn::Mat in = preprocess(bgr_img, scale, pad_left, pad_top);

    // 4. 推理
    ncnn::Extractor ex = net.create_extractor();
    // 注意：输入输出 blob 名可能不是 in0/out0，请检查 .param 文件
    ex.input("in0", in);

    ncnn::Mat out;
    ex.extract("out0", out);
std::cout << "Output shape: w=" << out.w << " h=" << out.h << " c=" << out.c << std::endl;
    // 5. 解析输出（根据实际形状）
    std::vector<float> probs(6);
    if (out.c == 6 && out.h == 1 && out.w == 1) {
        // 形状 (1,1,6) 即每个通道一个值
        for (int i = 0; i < 6; ++i)
            probs[i] = out.channel(i)[0];
    } else if (out.w == 6 && out.h == 1 && out.c == 1) {
        // 形状 (6,1,1) 即 w 方向排列
        for (int i = 0; i < 6; ++i)
            probs[i] = out[i];
    } else if (out.c == 1 && out.h == 1 && out.w == 1 && out.dims > 1) {
        // 极少数情况，可能被压平为 1x1 但实际有 6 个元素，尝试从数据指针读
        const float* data = (const float*)out.data;
        for (int i = 0; i < 6; ++i)
            probs[i] = data[i];
    } else {
        std::cerr << "Unexpected output shape: w=" << out.w 
                  << " h=" << out.h << " c=" << out.c << std::endl;
        return -1;
    }

    // 找到最大概率的类别
    int class_id = std::max_element(probs.begin(), probs.end()) - probs.begin();
    float confidence = probs[class_id];

    // 6. 打印结果
    std::cout << "类别: " << CLASS_NAMES[class_id] 
              << " (置信度: " << confidence * 100 << "%)" << std::endl;

    // 打印所有类别概率（可选）
    std::cout << "各分类概率:" << std::endl;
    for (size_t i = 0; i < CLASS_NAMES.size(); ++i) {
        std::cout << "  " << CLASS_NAMES[i] << ": " << probs[i] * 100 << "%" << std::endl;
    }

    // 7. 可视化（在原图上写分类结果）
    cv::Mat vis_img = bgr_img.clone();
    std::string label = CLASS_NAMES[class_id] + " " + std::to_string(confidence * 100).substr(0, 4) + "%";
    cv::putText(vis_img, label, cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    cv::imwrite("cls_result.jpg", vis_img);
    std::cout << "可视化结果已保存为 cls_result.jpg" << std::endl;

    return 0;
}