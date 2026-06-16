#include "deeplabv3p_ncnn.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <vector>

#include "CornerLostProcess.cpp"

namespace {
cv::Mat to_binary_255(const cv::Mat& mask)
{
    cv::Mat bin;
    mask.convertTo(bin, CV_8U);
    cv::threshold(bin, bin, 127, 255, cv::THRESH_BINARY);
    return bin;
}

bool is_corner_fill_reasonable(const cv::Mat& original_mask, const ProcessResult& result)
{
    if (original_mask.empty() || result.mask.empty() || result.fill.empty()) {
        std::cout << "Reject corner lost fill: empty input/result/fill" << std::endl;
        return false;
    }

    cv::Mat original_bin = to_binary_255(original_mask);
    cv::Mat fill_bin = to_binary_255(result.fill);

    int original_area = cv::countNonZero(original_bin);
    int fill_area = cv::countNonZero(fill_bin);
    if (original_area <= 0 || fill_area <= 0) {
        std::cout << "Reject corner lost fill: original_area=" << original_area
                  << " fill_area=" << fill_area
                  << std::endl;
        return false;
    }

    int canvas_h = fill_bin.rows;
    int canvas_w = fill_bin.cols;
    int border = std::min(3, std::min(canvas_h, canvas_w));
    if (border > 0) {
        int border_pixels = 0;
        border_pixels += cv::countNonZero(fill_bin(cv::Rect(0, 0, canvas_w, border)));
        border_pixels += cv::countNonZero(fill_bin(cv::Rect(0, canvas_h - border, canvas_w, border)));
        border_pixels += cv::countNonZero(fill_bin(cv::Rect(0, 0, border, canvas_h)));
        border_pixels += cv::countNonZero(fill_bin(cv::Rect(canvas_w - border, 0, border, canvas_h)));
        if (border_pixels > 0) {
            std::cout << "Reject corner lost fill: touches expanded canvas border"
                      << " border_pixels=" << border_pixels
                      << " fill_area=" << fill_area
                      << std::endl;
            return false;
        }
    }

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(fill_bin, labels, stats, centroids, 8);
    if (num_labels <= 1) {
        return false;
    }

    int largest = 1;
    for (int i = 2; i < num_labels; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(largest, cv::CC_STAT_AREA)) {
            largest = i;
        }
    }

    int w = stats.at<int>(largest, cv::CC_STAT_WIDTH);
    int h = stats.at<int>(largest, cv::CC_STAT_HEIGHT);
    int largest_area = stats.at<int>(largest, cv::CC_STAT_AREA);
    if (w <= 0 || h <= 0 || largest_area <= 0) {
        return false;
    }

    double fill_ratio = fill_area / static_cast<double>(original_area);
    double compactness = largest_area / (static_cast<double>(w) * h);
    double aspect = std::max(w / static_cast<double>(h), h / static_cast<double>(w));

    if (fill_ratio > 0.80 || compactness < 0.12 || aspect > 15.0) {
        std::cout << "Reject corner lost fill: fill_area=" << fill_area
                  << " original_area=" << original_area
                  << " fill_ratio=" << fill_ratio
                  << " bbox=" << w << "x" << h
                  << " compactness=" << compactness
                  << " aspect=" << aspect
                  << std::endl;
        return false;
    }

    return true;
}

cv::Mat build_filled_image(const cv::Mat& image, const ProcessResult& result)
{
    if (image.empty() || result.mask.empty() || result.fill.empty()) {
        return cv::Mat();
    }

    cv::Mat filled = cv::Mat::zeros(result.mask.size(), image.type());
    cv::Rect image_roi(result.offset.x, result.offset.y, image.cols, image.rows);
    cv::Rect canvas_roi(0, 0, filled.cols, filled.rows);
    image_roi &= canvas_roi;
    if (image_roi.width <= 0 || image_roi.height <= 0) {
        return cv::Mat();
    }

    cv::Rect src_roi(
        image_roi.x - result.offset.x,
        image_roi.y - result.offset.y,
        image_roi.width,
        image_roi.height
    );
    image(src_roi).copyTo(filled(image_roi));

    cv::Mat fill_binary = to_binary_255(result.fill);
    if (fill_binary.size() != filled.size()) {
        return cv::Mat();
    }

    if (filled.channels() == 1) {
        filled.setTo(cv::Scalar(255), fill_binary);
    } else {
        filled.setTo(cv::Scalar(255, 255, 255), fill_binary);
    }
    return filled;
}

cv::Mat build_segmentation_overlay(const cv::Mat& image, const cv::Mat& mask_255)
{
    cv::Mat mask = to_binary_255(mask_255);
    cv::Mat seg_img = cv::Mat::zeros(image.size(), image.type());
    seg_img.setTo(cv::Scalar(0, 0, 128), mask);

    cv::Mat blended;
    cv::addWeighted(image, 0.30, seg_img, 0.70, 0.0, blended);
    return blended;
}

float sigmoid(float x)
{
    if (x >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    float e = std::exp(x);
    return e / (1.0f + e);
}

std::vector<DeeplabV3_NCNN::KeypointResult> decode_keypoints(
    const ncnn::Mat& heatmaps,
    int left,
    int top,
    int new_w,
    int new_h,
    int original_w,
    int original_h
) {
    std::vector<DeeplabV3_NCNN::KeypointResult> keypoints;
    if (heatmaps.empty() || heatmaps.c <= 0 || heatmaps.w <= 0 || heatmaps.h <= 0 ||
        new_w <= 0 || new_h <= 0 || original_w <= 0 || original_h <= 0) {
        return keypoints;
    }

    keypoints.reserve(static_cast<size_t>(heatmaps.c));
    int w = heatmaps.w;
    int h = heatmaps.h;

    for (int ch = 0; ch < heatmaps.c; ++ch) {
        ncnn::Mat hm_channel = heatmaps.channel(ch);
        const float* hm = hm_channel;
        int max_idx = 0;
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < w * h; ++i) {
            if (hm[i] > max_logit) {
                max_logit = hm[i];
                max_idx = i;
            }
        }

        float x = static_cast<float>(max_idx % w);
        float y = static_cast<float>(max_idx / w);
        float score = sigmoid(max_logit);

        if (x >= 1.0f && x < static_cast<float>(w - 1) &&
            y >= 1.0f && y < static_cast<float>(h - 1)) {
            int ix = static_cast<int>(x);
            int iy = static_cast<int>(y);
            auto value = [&](int yy, int xx) -> float {
                return sigmoid(hm[yy * w + xx]);
            };

            float center = value(iy, ix);
            float denom_x = value(iy, ix - 1) + value(iy, ix + 1) - 2.0f * center + 1e-8f;
            float denom_y = value(iy - 1, ix) + value(iy + 1, ix) - 2.0f * center + 1e-8f;
            x += 0.5f * (value(iy, ix - 1) - value(iy, ix + 1)) / denom_x;
            y += 0.5f * (value(iy - 1, ix) - value(iy + 1, ix)) / denom_y;
        }

        float mapped_x = (x - static_cast<float>(left)) * (static_cast<float>(original_w) / new_w);
        float mapped_y = (y - static_cast<float>(top)) * (static_cast<float>(original_h) / new_h);
        mapped_x = std::clamp(mapped_x, 0.0f, static_cast<float>(original_w - 1));
        mapped_y = std::clamp(mapped_y, 0.0f, static_cast<float>(original_h - 1));

        keypoints.push_back({cv::Point2f(mapped_x, mapped_y), score});
    }

    return keypoints;
}

void draw_keypoints(cv::Mat& image, const std::vector<DeeplabV3_NCNN::KeypointResult>& keypoints)
{
    if (image.empty() || image.channels() != 3 || keypoints.empty()) {
        return;
    }

    if (keypoints.size() >= 2) {
        cv::line(image, keypoints[0].point, keypoints[1].point, cv::Scalar(0, 180, 255), 4, cv::LINE_AA);
    }

    for (size_t i = 0; i < keypoints.size(); ++i) {
        const cv::Point2f& p = keypoints[i].point;
        cv::circle(image, p, 7, cv::Scalar(40, 40, 255), cv::FILLED, cv::LINE_AA);
        cv::circle(image, p, 7, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        char label[32];
        std::snprintf(label, sizeof(label), "%zu:%.2f", i, keypoints[i].score);
        cv::putText(
            image,
            label,
            cv::Point(static_cast<int>(p.x) + 10, static_cast<int>(p.y) - 10),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(255, 255, 255),
            2,
            cv::LINE_AA
        );
    }
}

cv::Mat apply_corner_lost_process(
    const cv::Mat& mask_255,
    const cv::Mat& image,
    cv::Mat* filled_image,
    cv::Mat* fill_mask_out
)
{
    if (filled_image != nullptr) {
        filled_image->release();
    }
    if (fill_mask_out != nullptr) {
        fill_mask_out->release();
    }

    cv::Mat bin_mask = to_binary_255(mask_255);
    if (cv::countNonZero(bin_mask) == 0) {
        return bin_mask;
    }

    int expand_margin = static_cast<int>(std::max(bin_mask.rows, bin_mask.cols) * 0.15);
    BookMaskRestorer restorer(0.92, expand_margin, cv::Size(15, 15));
    ProcessResult result = restorer.process(bin_mask);

    cv::Mat fill_binary = to_binary_255(result.fill);
    if (cv::countNonZero(fill_binary) == 0) {
        return bin_mask;
    }

    if (RuntimeConfig::kEnableCornerLostProcess && !is_corner_fill_reasonable(bin_mask, result)) {
        return bin_mask;
    }

    if (filled_image != nullptr) {
        *filled_image = build_filled_image(image, result);
    }
    if (fill_mask_out != nullptr) {
        *fill_mask_out = fill_binary;
    }

    return to_binary_255(result.mask);
}
}

void DeeplabV3_NCNN::configure_net()
{
    net->opt.num_threads = RuntimeConfig::kNumThreads;
    net->opt.openmp_blocktime = RuntimeConfig::kOpenMPBlockTime;
    net->opt.lightmode = RuntimeConfig::kLightMode;
    net->opt.use_local_pool_allocator = RuntimeConfig::kUseLocalPoolAllocator;
    net->opt.use_packing_layout = RuntimeConfig::kUsePackingLayout;
    net->opt.use_fp16_packed = RuntimeConfig::kUseFP16Packed;
    net->opt.use_fp16_storage = RuntimeConfig::kUseFP16Storage;
    net->opt.use_fp16_arithmetic = RuntimeConfig::kUseFP16Arithmetic;
    net->opt.use_vulkan_compute = RuntimeConfig::kUseVulkanCompute;
}

void DeeplabV3_NCNN::load_net()
{
    net = std::make_unique<ncnn::Net>();
    configure_net();

    int ret = net->load_param(kParamPath);
    if (ret != 0) std::cerr << "load_param failed: " << ret << " path: " << kParamPath << std::endl;

    ret = net->load_model(kBinPath);
    if (ret != 0) std::cerr << "load_model failed: " << ret << " path: " << kBinPath << std::endl;
}

DeeplabV3_NCNN::DeeplabV3_NCNN()
{
    load_net();
}

void DeeplabV3_NCNN::set_book_threshold(float threshold)
{
    book_threshold = std::clamp(threshold, 0.0f, 1.0f);
}

void DeeplabV3_NCNN::set_output_type(int type)
{
    output_type = type == 1 ? 1 : 0;
}

void DeeplabV3_NCNN::set_corner_lost_process(bool enabled)
{
    enable_corner_lost_process = enabled;
}

// preprocess: keep-aspect resize + pad, return BGR CV_8UC3 canvas, new_w/new_h are scaled dims
cv::Mat DeeplabV3_NCNN::preprocess(const cv::Mat& img, int& new_w, int& new_h)
{
    cv::Mat image = img;
    if (image.channels() == 1) cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

    int w = image.cols;
    int h = image.rows;
    float scale = std::min(input_w / (float)w, input_h / (float)h);
    new_w = std::max(1, int(w * scale));
    new_h = std::max(1, int(h * scale));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(input_h, input_w, CV_8UC3, cv::Scalar(128,128,128));
    int left = (input_w - new_w) / 2;
    int top  = (input_h - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));

    if (!canvas.isContinuous()) canvas = canvas.clone();
    return canvas;
}

// refine_mask: mirror of Python refine (focus on class 1)
// input: binary mask CV_8U where foreground is 255 (or >0). output: CV_8U (0 or 255)
cv::Mat DeeplabV3_NCNN::refine_mask(const cv::Mat& pr)
{
    cv::Mat mask = (pr > 0);
    mask.convertTo(mask, CV_8U, 255);

    const int min_side = std::min(mask.rows, mask.cols);
    if (min_side >= 1000) {
        cv::Mat erode_kernel = cv::Mat::ones(cv::Size(3, 3), CV_8U);
        cv::erode(mask, mask, erode_kernel, cv::Point(-1, -1), 1);
    }

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    if (num_labels > 1) {
        int largest = 1;
        for (int i = 2; i < num_labels; ++i) {
            if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(largest, cv::CC_STAT_AREA)) {
                largest = i;
            }
        }
        cv::compare(labels, largest, mask, cv::CMP_EQ);
    }

    cv::Mat mask_bin = mask;

    // Small output maps use a lighter kernel to avoid eating the book contour.
    int morph_size = min_side >= 1000 ? 7 : 3;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_size, morph_size));
    cv::morphologyEx(mask_bin, mask_bin, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask_bin, mask_bin, cv::MORPH_OPEN, kernel);

    mask_bin = (mask_bin > 0);
    mask_bin.convertTo(mask_bin, CV_8U, 255);
    return mask_bin;
}

cv::Mat DeeplabV3_NCNN::detect_image(const cv::Mat& image_in)
{
    return detect_image(image_in, nullptr);
}

cv::Mat DeeplabV3_NCNN::detect_image(const cv::Mat& image_in, cv::Mat* filled_image)
{
    return detect_image(image_in, filled_image, nullptr, nullptr, nullptr);
}

cv::Mat DeeplabV3_NCNN::detect_image(
    const cv::Mat& image_in,
    cv::Mat* filled_image,
    std::vector<KeypointResult>* keypoints,
    cv::Mat* mask_out,
    cv::Mat* fill_mask_out
)
{
    if (image_in.empty()) return cv::Mat();
    if (filled_image != nullptr) {
        filled_image->release();
    }
    if (keypoints != nullptr) {
        keypoints->clear();
    }
    if (mask_out != nullptr) {
        mask_out->release();
    }
    if (fill_mask_out != nullptr) {
        fill_mask_out->release();
    }

    cv::Mat image = image_in.clone();
    if (image.channels() == 1)
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

    int nw, nh;
    cv::Mat input = preprocess(image, nw, nh);

    ncnn::Mat in(input.cols, input.rows, 3);
    for (int y = 0; y < input.rows; ++y) {
        const cv::Vec3b* row = input.ptr<cv::Vec3b>(y);
        float* r = in.channel(0);
        float* g = in.channel(1);
        float* b = in.channel(2);
        for (int x = 0; x < input.cols; ++x) {
            int idx = y * input.cols + x;
            r[idx] = row[x][2] / 255.0f;
            g[idx] = row[x][1] / 255.0f;
            b[idx] = row[x][0] / 255.0f;
        }
    }

    ncnn::Extractor ex = net->create_extractor();
    ex.set_light_mode(false);
    ex.input("in0", in);

    ncnn::Mat out;
    int ret = ex.extract("out0", out);
    if (ret != 0 || out.empty()) {
        std::cerr << "extract out0 failed: " << ret << std::endl;
        return cv::Mat();
    }

    ncnn::Mat kpt_out;
    ret = ex.extract("out1", kpt_out);
    if (ret != 0 || kpt_out.empty()) {
        std::cerr << "extract out1 failed: " << ret << std::endl;
        return cv::Mat();
    }

    int w = out.w;
    int h = out.h;
    int c = out.c;   // class number

    // -----------------------------
    // 1. two-class logits -> book-vs-background logit diff
    // -----------------------------
    cv::Mat book_diff(h, w, CV_32F);
    ncnn::Mat bg_channel;
    ncnn::Mat book_channel;
    const float* bg_base = nullptr;
    const float* book_base = nullptr;
    if (c > 0) {
        bg_channel = out.channel(0);
        bg_base = bg_channel;
    }
    if (c > 1) {
        book_channel = out.channel(1);
        book_base = book_channel;
    }

    for (int y = 0; y < h; y++)
    {
        const float* bg_ptr = bg_base != nullptr ? bg_base + y * w : nullptr;
        const float* book_ptr = book_base != nullptr ? book_base + y * w : nullptr;
        float* diff_ptr = book_diff.ptr<float>(y);

        for (int x = 0; x < w; x++)
        {
            float diff = -std::numeric_limits<float>::infinity();
            if (book_ptr != nullptr && bg_ptr != nullptr) {
                diff = book_ptr[x] - bg_ptr[x];
            }
            diff_ptr[x] = diff;
        }
    }

    // -----------------------------
    // 2. crop padding区域
    // -----------------------------
    int left = (input_w - nw) / 2;
    int top  = (input_h - nh) / 2;

    std::vector<KeypointResult> decoded_keypoints = decode_keypoints(
        kpt_out,
        left,
        top,
        nw,
        nh,
        image.cols,
        image.rows
    );
    if (keypoints != nullptr) {
        *keypoints = decoded_keypoints;
    }

    float sx = w / (float)input_w;
    float sy = h / (float)input_h;

    int x0 = (int)(left * sx);
    int y0 = (int)(top * sy);
    int cw = (int)(nw * sx);
    int ch = (int)(nh * sy);

    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    cw = std::min(w - x0, cw);
    ch = std::min(h - y0, ch);

    if (cw <= 0 || ch <= 0) {
        std::cerr << "Invalid segmentation crop: x=" << x0
                  << " y=" << y0
                  << " w=" << cw
                  << " h=" << ch
                  << " image=" << image.cols << "x" << image.rows
                  << std::endl;
        return cv::Mat();
    }

    cv::Mat diff_crop = book_diff(cv::Rect(x0, y0, cw, ch)).clone();
    int finite_count = 0;
    double min_diff = std::numeric_limits<double>::infinity();
    double max_diff = -std::numeric_limits<double>::infinity();
    for (int y = 0; y < diff_crop.rows; ++y) {
        float* row = diff_crop.ptr<float>(y);
        for (int x = 0; x < diff_crop.cols; ++x) {
            if (std::isfinite(row[x])) {
                finite_count++;
                min_diff = std::min(min_diff, static_cast<double>(row[x]));
                max_diff = std::max(max_diff, static_cast<double>(row[x]));
            } else {
                row[x] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    // -----------------------------
    // 3. build binary mask on the cropped output map.
    // p_book >= t is equivalent to book_logit - bg_logit >= log(t / (1 - t)).
    // -----------------------------
    if (finite_count == 0 || std::fabs(max_diff - min_diff) < 1e-6) {
        std::cerr << "Collapsed logit output: diff min=" << min_diff
                  << " max=" << max_diff
                  << " finite_count=" << finite_count
                  << " image=" << image.cols << "x" << image.rows
                  << std::endl;
        return cv::Mat();
    }

    auto probability_threshold_to_logit = [](float threshold) -> float {
        threshold = std::clamp(threshold, 1e-4f, 1.0f - 1e-4f);
        return std::log(threshold / (1.0f - threshold));
    };

    auto build_mask_with_threshold = [&](float threshold) -> cv::Mat {
        const float logit_threshold = probability_threshold_to_logit(threshold);
        cv::Mat bin(diff_crop.rows, diff_crop.cols, CV_8U);
        for (int y = 0; y < diff_crop.rows; y++)
        {
            const float* diff_ptr = diff_crop.ptr<float>(y);
            uchar* bin_ptr = bin.ptr<uchar>(y);
            for (int x = 0; x < diff_crop.cols; x++)
            {
                bin_ptr[x] = diff_ptr[x] >= logit_threshold ? 255 : 0;
            }
        }

        int raw_fg = cv::countNonZero(bin);

        cv::Mat mask_after_morph = refine_mask(bin);
        int morph_fg = cv::countNonZero(mask_after_morph);
        if (morph_fg == 0 && raw_fg > 0) {
            mask_after_morph = bin;
        }
        return mask_after_morph;
    };

    std::vector<float> thresholds = {
        book_threshold,
        std::min(book_threshold, 0.55f),
        0.50f,
        0.45f,
        0.40f
    };

    cv::Mat refined_mask;
    int refined_foreground = 0;
    for (float threshold : thresholds) {
        cv::Mat candidate = build_mask_with_threshold(threshold);
        int candidate_fg = cv::countNonZero(candidate);
        if (refined_mask.empty() || candidate_fg > refined_foreground) {
            refined_mask = candidate;
            refined_foreground = candidate_fg;
        }
        if (candidate_fg > 0) {
            break;
        }
    }

    refined_foreground = cv::countNonZero(refined_mask);
    if (refined_foreground == 0) {
        std::cout << "Empty mask: diff min=" << min_diff
                  << " max=" << max_diff
                  << " image=" << image.cols << "x" << image.rows
                  << std::endl;
    }

    cv::Mat mask_resized;
    cv::resize(refined_mask, mask_resized, image.size(), 0, 0, cv::INTER_LINEAR);

    cv::Mat mask_255;
    cv::threshold(mask_resized, mask_255, 127, 255, cv::THRESH_BINARY);

    cv::Mat output_mask = mask_255;
    if (enable_corner_lost_process) {
        output_mask = apply_corner_lost_process(mask_255, image, filled_image, fill_mask_out);
    }
    if (mask_out != nullptr) {
        *mask_out = output_mask.clone();
    }

    if (output_type == 1) {
        return output_mask;
    }

    cv::Mat result = build_segmentation_overlay(image, output_mask);
    draw_keypoints(result, decoded_keypoints);
    return result;
}
