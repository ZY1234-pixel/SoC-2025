#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "net.h"
#include "opencv2/opencv.hpp"

// ==========================================
// heatmap_v12_512_aug 角点检测 NCNN 推理
// 输入: in0  (1,3,512,512) RGB, /255
// 输出: out0 (1,6) 分类 logits
//       out1 (1,4,64,64) 角点热力图
// 预处理/解码与 Python 端 predict.py 一致
// ==========================================

static constexpr int INPUT_SIZE = 512;
static constexpr int NUM_KPT = 4;

static const char* CLASS_NAMES[] = {
    "double_page_book", "newspaper_poster", "receipt",
    "screen", "single_page", "unclassified"};

// Python round() 是银行家舍入（half to even），保持一致
static int pyround(float x) {
    float f = std::floor(x);
    float diff = x - f;
    if (diff < 0.5f) return (int)f;
    if (diff > 0.5f) return (int)f + 1;
    return ((int)f % 2 == 0) ? (int)f : (int)f + 1;
}

struct LetterboxInfo {
    float scale = 1.f;
    float dw = 0.f;
    float dh = 0.f;
};

// 复刻 Python predict.py 的 letterbox：等比例缩放 + 居中灰边(114)
static cv::Mat letterbox(const cv::Mat& rgb, int size, LetterboxInfo& info) {
    int ih = rgb.rows, iw = rgb.cols;
    float r = std::min((float)size / ih, (float)size / iw);
    int nw = pyround(iw * r);
    int nh = pyround(ih * r);
    float dw = (size - nw) / 2.0f;
    float dh = (size - nh) / 2.0f;

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    int top = pyround(dh - 0.1f);
    int bottom = pyround(dh + 0.1f);
    int left = pyround(dw - 0.1f);
    int right = pyround(dw + 0.1f);

    cv::Mat padded(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(left, top, nw, nh)));

    info.scale = r;
    info.dw = dw;
    info.dh = dh;
    return padded;
}

// 从热力图解码归一化坐标（argmax + 抛物线亚像素精修）
static std::vector<cv::Point2f> decodeHeatmap(const ncnn::Mat& hm) {
    std::vector<cv::Point2f> kpts(NUM_KPT);
    int H = hm.h, W = hm.w;
    for (int c = 0; c < NUM_KPT; ++c) {
        const float* data = hm.channel(c);
        float maxv = -FLT_MAX;
        int mx = 0, my = 0;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float v = data[y * W + x];
                if (v > maxv) {
                    maxv = v;
                    mx = x;
                    my = y;
                }
            }
        }
        float fx = (float)mx, fy = (float)my;
        if (mx > 0 && mx < W - 1 && my > 0 && my < H - 1) {
            float vc = data[my * W + mx];
            float vl = data[my * W + mx - 1];
            float vr = data[my * W + mx + 1];
            float vt = data[(my - 1) * W + mx];
            float vb = data[(my + 1) * W + mx];
            float dx = -0.5f * (vr - vl) / (vl - 2.0f * vc + vr + 1e-6f);
            float dy = -0.5f * (vb - vt) / (vt - 2.0f * vc + vb + 1e-6f);
            if (std::fabs(dx) < 1.0f) fx += dx;
            if (std::fabs(dy) < 1.0f) fy += dy;
        }
        // 归一化回 0~1（与 Python 一致：x/(W-1), y/(H-1)）
        kpts[c] = cv::Point2f(fx / (W - 1), fy / (H - 1));
    }
    return kpts;
}

static std::vector<cv::Point2f> toOriginal(const std::vector<cv::Point2f>& norm,
                                           int size, const LetterboxInfo& info) {
    std::vector<cv::Point2f> pts(NUM_KPT);
    for (int i = 0; i < NUM_KPT; ++i) {
        float xp = norm[i].x * size;
        float yp = norm[i].y * size;
        pts[i] = cv::Point2f((xp - info.dw) / info.scale, (yp - info.dh) / info.scale);
    }
    return pts;
}

// 读取 YOLO pose 标注：class cx cy w h 然后 4×(x y v)，返回归一化角点
static bool loadGt(const std::filesystem::path& labelPath, std::vector<cv::Point2f>& gt) {
    std::ifstream f(labelPath);
    if (!f.is_open()) return false;
    std::string line;
    std::getline(f, line);
    std::istringstream iss(line);
    int cls;
    float tmp;
    iss >> cls;
    for (int i = 0; i < 4; ++i) iss >> tmp;  // bbox cx cy w h
    gt.clear();
    for (int i = 0; i < NUM_KPT; ++i) {
        float x, y, v;
        iss >> x >> y >> v;
        gt.emplace_back(x, y);
    }
    return true;
}

static void saveVis(const cv::Mat& bgr, const std::vector<cv::Point2f>& pred,
                    const std::vector<cv::Point2f>& gt, const std::string& clsName,
                    const std::string& outPath) {
    cv::Mat vis = bgr.clone();
    std::vector<cv::Point> ppts, gpts;
    for (int i = 0; i < NUM_KPT; ++i) {
        ppts.emplace_back((int)std::lround(pred[i].x), (int)std::lround(pred[i].y));
    }
    cv::polylines(vis, std::vector<std::vector<cv::Point>>{ppts}, true, cv::Scalar(0, 0, 255), 2);
    for (int i = 0; i < NUM_KPT; ++i) {
        cv::circle(vis, ppts[i], 6, cv::Scalar(0, 0, 255), -1);
        cv::putText(vis, "p" + std::to_string(i), cv::Point(ppts[i].x + 10, ppts[i].y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }
    if (!gt.empty()) {
        for (int i = 0; i < NUM_KPT; ++i) {
            gpts.emplace_back((int)std::lround(gt[i].x), (int)std::lround(gt[i].y));
        }
        cv::polylines(vis, std::vector<std::vector<cv::Point>>{gpts}, true, cv::Scalar(0, 255, 0), 2);
        for (int i = 0; i < NUM_KPT; ++i) {
            cv::circle(vis, gpts[i], 5, cv::Scalar(0, 255, 0), -1);
            cv::putText(vis, "g" + std::to_string(i), cv::Point(gpts[i].x + 10, gpts[i].y + 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }
    }
    cv::putText(vis, "ncnn: " + clsName, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(255, 0, 255), 2);
    cv::imwrite(outPath, vis);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_or_dir> [out_dir] [label_dir]" << std::endl;
        return -1;
    }
    std::string outDir = argc > 2 ? argv[2] : "ncnn_results";
    namespace fs = std::filesystem;

    // 收集图片（支持文件夹或单张图片）
    std::vector<std::string> imgs;
    fs::path input(argv[1]);
    if (fs::is_directory(input)) {
        for (auto& entry : fs::directory_iterator(input)) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                imgs.push_back(entry.path().string());
            }
        }
    } else {
        imgs.push_back(argv[1]);
    }
    std::sort(imgs.begin(), imgs.end());
    if (imgs.empty()) {
        std::cerr << "no image found" << std::endl;
        return -1;
    }
    fs::create_directories(outDir);

    // 加载 ncnn 模型（注意：不要显式设置 num_threads=0，部分版本会崩溃）
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_arithmetic = false;
    if (net.load_param("heatmap_v12_512_aug.ncnn.param") != 0 ||
        net.load_model("heatmap_v12_512_aug.ncnn.bin") != 0) {
        std::cerr << "failed to load ncnn model" << std::endl;
        return -1;
    }

    std::vector<std::vector<double>> errs(NUM_KPT);
    int n = 0;
    for (const auto& ip : imgs) {
        cv::Mat bgr = cv::imread(ip, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            std::cerr << "cannot read: " << ip << std::endl;
            continue;
        }
        int ow = bgr.cols, oh = bgr.rows;
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        LetterboxInfo info;
        cv::Mat padded = letterbox(rgb, INPUT_SIZE, info);

        ncnn::Mat in = ncnn::Mat::from_pixels(padded.data, ncnn::Mat::PIXEL_RGB,
                                              INPUT_SIZE, INPUT_SIZE);
        const float norm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
        in.substract_mean_normalize(0, norm);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in);
        ncnn::Mat cls_mat, hm_mat;
        ex.extract("out0", cls_mat);
        ex.extract("out1", hm_mat);

        int best = 0;
        float bestv = -FLT_MAX;
        for (int c = 0; c < 6; ++c) {
            float v = cls_mat.channel(c)[0];
            if (v > bestv) {
                bestv = v;
                best = c;
            }
        }
        std::vector<cv::Point2f> norm_pts = decodeHeatmap(hm_mat);
        std::vector<cv::Point2f> pred = toOriginal(norm_pts, INPUT_SIZE, info);

        std::vector<cv::Point2f> gt;
        std::wstring stemW = fs::path(ip).stem().wstring();
        fs::path lbl;
        if (argc > 3) {
            lbl = fs::path(argv[3]) / (stemW + L".txt");
        } else {
            lbl = fs::path(L"D:/奔图/deeplabv3p_zzh/YoloV8_Pose/data/val/labels") / (stemW + L".txt");
            if (!fs::exists(lbl)) {
                lbl = fs::path(L"D:/奔图/deeplabv3p_zzh/YoloV8_Pose/data/0722WKX/Data/labels") / (stemW + L".txt");
            }
        }
        bool hasGt = fs::exists(lbl) && loadGt(lbl, gt);
        for (int i = 0; i < NUM_KPT; ++i) {
            if (hasGt) {
                double ex_ = pred[i].x - gt[i].x * ow;
                double ey_ = pred[i].y - gt[i].y * oh;
                errs[i].push_back(std::sqrt(ex_ * ex_ + ey_ * ey_));
            }
        }

        std::string outPath = outDir + "/" + fs::path(ip).stem().string() + "_ncnn.jpg";
        saveVis(bgr, pred, hasGt ? gt : std::vector<cv::Point2f>(), CLASS_NAMES[best], outPath);
        ++n;
        if (n % 100 == 0 || n == (int)imgs.size()) {
            std::cout << "[" << n << "/" << imgs.size() << "] " << fs::path(ip).stem().string() << std::endl;
        }
    }

    // 汇总
    std::cout << "\n===== ncnn 推理汇总 (" << n << " 张) =====" << std::endl;
    const char* kname[NUM_KPT] = {"kpt0(左上)", "kpt1(右上)", "kpt2(右下)", "kpt3(左下)"};
    for (int i = 0; i < NUM_KPT; ++i) {
        if (errs[i].empty()) continue;
        auto e = errs[i];
        std::sort(e.begin(), e.end());
        double mean = 0;
        for (double v : e) mean += v;
        mean /= e.size();
        double median = e.size() % 2 ? e[e.size() / 2]
                                     : (e[e.size() / 2 - 1] + e[e.size() / 2]) / 2.0;
        std::cout << kname[i] << ": mean=" << mean << "px  median=" << median << "px"
                  << "  n=" << e.size() << std::endl;
    }
    std::cout << "可视化结果保存在: " << outDir << std::endl;
    return 0;
}
