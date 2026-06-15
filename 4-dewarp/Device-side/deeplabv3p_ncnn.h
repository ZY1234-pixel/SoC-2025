#pragma once
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include "runtime_config.h"

class DeeplabV3_NCNN {
public:
    struct KeypointResult {
        cv::Point2f point;
        float score = 0.0f;
    };

    static constexpr const char* kParamPath =
        "deeplabv3p.fp16.ncnn.param";
    static constexpr const char* kBinPath =
        "deeplabv3p.fp16.ncnn.bin";
    static constexpr const char* kDefaultInputPath = "img/";
    static constexpr const char* kDefaultSavePath = "img_out/";

    DeeplabV3_NCNN();

    void set_book_threshold(float threshold);
    void set_output_type(int output_type);
    void set_corner_lost_process(bool enabled);
    cv::Mat detect_image(const cv::Mat& image);
    cv::Mat detect_image(const cv::Mat& image, cv::Mat* filled_image);
    cv::Mat detect_image(
        const cv::Mat& image,
        cv::Mat* filled_image,
        std::vector<KeypointResult>* keypoints,
        cv::Mat* mask_out = nullptr,
        cv::Mat* fill_mask_out = nullptr
    );

private:
    std::unique_ptr<ncnn::Net> net;

    int input_w = RuntimeConfig::kInputWidth;
    int input_h = RuntimeConfig::kInputHeight;
    float book_threshold = RuntimeConfig::kBookThreshold;
    int output_type = RuntimeConfig::kSaveVisualization ? 0 : 1;  // 0: blended overlay, 1: 0/255 mask
    bool enable_corner_lost_process = RuntimeConfig::kEnableCornerLostProcess;

    cv::Mat preprocess(const cv::Mat& img, int& new_w, int& new_h);
    cv::Mat refine_mask(const cv::Mat& mask);
    void configure_net();
    void load_net();
};
