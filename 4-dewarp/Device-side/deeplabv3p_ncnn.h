#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include <ncnn/net.h>

class DeeplabV3_NCNN {
public:
    static constexpr const char* kParamPath =
        "deeplabv3p.ncnn.param";
    static constexpr const char* kBinPath =
        "deeplabv3p.ncnn.bin";
    static constexpr const char* kDefaultInputPath = "img/";
    static constexpr const char* kDefaultSavePath = "img_out/";

    DeeplabV3_NCNN();

    void set_book_threshold(float threshold);
    void set_output_type(int output_type);
    void set_corner_lost_process(bool enabled);
    cv::Mat detect_image(const cv::Mat& image);
    cv::Mat detect_image(const cv::Mat& image, cv::Mat* filled_image);

private:
    std::unique_ptr<ncnn::Net> net;

    int input_w = 640;
    int input_h = 640;
    float book_threshold = 0.65f;
    int output_type = 0;  // 0: blended overlay, 1: 0/255 mask
    bool enable_corner_lost_process = false;  // keep default inference aligned with Python postprocess

    cv::Mat preprocess(const cv::Mat& img, int& new_w, int& new_h);
    cv::Mat refine_mask(const cv::Mat& mask);
    void configure_net();
    void load_net();
};
