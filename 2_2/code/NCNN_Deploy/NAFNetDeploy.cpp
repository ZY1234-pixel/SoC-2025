// NAFNetDeploy.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "ncnn/net.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <windows.h>
#include <psapi.h>


using namespace std;
using namespace cv;

long get_process_memory_kb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (long)(pmc.WorkingSetSize / 1024);
    return 0;
}

class NAFNetDenoiser {
private:
    ncnn::Net net;
    bool use_gpu;
    const int MULTIPLE = 32;

    const int TILE_SIZE = 1024;
    const int OVERLAP = 32;

public:
    double infer_time_ms = 0;
    long memory_increase_mb = 0;
    int tile_count = 0;

    NAFNetDenoiser(const string& param, const string& bin, bool use_gpu = true)
        : use_gpu(use_gpu)
    {

        net.opt.use_vulkan_compute = use_gpu;
        net.opt.use_packing_layout = true;
        net.opt.num_threads = 8;

        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;

        net.opt.use_int8_inference = true;
        net.opt.use_int8_storage = true;

        if (use_gpu) net.set_vulkan_device(0);
        net.load_param(param.c_str());
        net.load_model(bin.c_str());
    }

    Mat denoise(const Mat& input) {
        long mem_before = get_process_memory_kb();
        auto t1 = chrono::high_resolution_clock::now();

        int W = input.cols;
        int H = input.rows;

        Mat output(H, W, CV_32FC3, Scalar(0, 0, 0));
        Mat weight(H, W, CV_32FC1, Scalar(0));
        tile_count = 0;

        for (int y = 0; y < H; y += (TILE_SIZE - OVERLAP)) {
            for (int x = 0; x < W; x += (TILE_SIZE - OVERLAP)) {

                int x2 = min(x + TILE_SIZE, W);
                int y2 = min(y + TILE_SIZE, H);
                int w = x2 - x;
                int h = y2 - y;

                int pw = (w + MULTIPLE - 1) / MULTIPLE * MULTIPLE;
                int ph = (h + MULTIPLE - 1) / MULTIPLE * MULTIPLE;

                Mat tile_img = input(Rect(x, y, w, h));
                Mat pad;
                copyMakeBorder(tile_img, pad, 0, ph - h, 0, pw - w, BORDER_REFLECT_101);

                Mat rgb;
                cvtColor(pad, rgb, COLOR_BGR2RGB);
                ncnn::Mat in = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB, pw, ph);

                const float mean[] = { 0,0,0 };
                const float norm[] = { 1.0f / 255, 1.0f / 255, 1.0f / 255 };
                in.substract_mean_normalize(mean, norm);

                ncnn::Extractor ex = net.create_extractor();
                ex.input("in0", in);
                ncnn::Mat out;
                ex.extract("out0", out);

                Mat res(ph, pw, CV_32FC3);
                for (int c = 0; c < 3; c++) {
                    const float* ptr = out.channel(c);
                    for (int hh = 0; hh < ph; hh++) {
                        for (int ww = 0; ww < pw; ww++) {
                            res.at<Vec3f>(hh, ww)[c] = ptr[hh * pw + ww];
                        }
                    }
                }

                res = res(Rect(0, 0, w, h));

                for (int yy = 0; yy < h; yy++) {
                    for (int xx = 0; xx < w; xx++) {
                        output.at<Vec3f>(y + yy, x + xx) += res.at<Vec3f>(yy, xx);
                        weight.at<float>(y + yy, x + xx) += 1.0f;
                    }
                }
                tile_count++;
            }
        }

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float w = weight.at<float>(y, x);
                if (w > 0)
                    output.at<Vec3f>(y, x) /= w;
            }
        }

        output *= 255.0f;
        Mat result8;
        output.convertTo(result8, CV_8UC3);
        cvtColor(result8, result8, COLOR_RGB2BGR);

        auto t2 = chrono::high_resolution_clock::now();
        infer_time_ms = chrono::duration<double, milli>(t2 - t1).count();

        long mem_after = get_process_memory_kb();
        memory_increase_mb = (mem_after - mem_before) / 1024;

        cout << "\n====== 推理信息 ======\n";
        cout << "图像: " << W << "x" << H << "\n";
        cout << "块大小: " << TILE_SIZE << "\n";
        cout << "总块数: " << tile_count << "\n";
        cout << "推理时间: " << infer_time_ms << " ms\n";
        cout << "内存增量: " << memory_increase_mb << " MB\n";
        cout << "========================\n";

        return result8;
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " input.jpg output.jpg\n";
        return -1;
    }

    Mat img = imread(argv[1]);
    if (img.empty()) {
        cerr << "无法读取图像\n";
        return -1;
    }

    NAFNetDenoiser model(
        "nafnet_model.ncnn.param",
        "nafnet_model.ncnn.bin",
        true
    );

    Mat out = model.denoise(img);
    imwrite(argv[2], out);

    return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
