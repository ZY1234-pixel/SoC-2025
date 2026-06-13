#include "deeplabv3p_ncnn.h"
#include <algorithm>
#include <cmath>
#include <iostream>
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

cv::Mat apply_corner_lost_process(const cv::Mat& mask_255, const cv::Mat& image, cv::Mat* filled_image)
{
    if (filled_image != nullptr) {
        filled_image->release();
    }

    cv::Mat bin_mask = to_binary_255(mask_255);
    if (cv::countNonZero(bin_mask) == 0) {
        return bin_mask;
    }

    int expand_margin = static_cast<int>(std::max(bin_mask.rows, bin_mask.cols) * 0.15);
    BookMaskRestorer restorer(0.92, expand_margin, cv::Size(15, 15));
    ProcessResult result = restorer.process(bin_mask, "", "");

    if (!is_corner_fill_reasonable(bin_mask, result)) {
        return bin_mask;
    }

    if (filled_image != nullptr) {
        *filled_image = build_filled_image(image, result);
    }

    return to_binary_255(result.mask);
}
}

void DeeplabV3_NCNN::configure_net()
{
    net->opt.num_threads = 8;
    net->opt.use_packing_layout = false;
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

// preprocess: keep-aspect resize + pad, return RGB CV_8UC3 canvas, new_w/new_h are scaled dims
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
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    cv::Mat canvas(input_h, input_w, CV_8UC3, cv::Scalar(128,128,128));
    int left = (input_w - new_w) / 2;
    int top  = (input_h - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));

    // convert to RGB because ncnn::Mat::from_pixels PIXEL_RGB expects RGB ordering
    cv::Mat canvas_rgb;
    cv::cvtColor(canvas, canvas_rgb, cv::COLOR_BGR2RGB);

    if (!canvas_rgb.isContinuous()) canvas_rgb = canvas_rgb.clone();
    return canvas_rgb;
}

// refine_mask: mirror of Python refine (focus on class 1)
// input: binary mask CV_8U where foreground is 255 (or >0). output: CV_8U (0 or 255)
cv::Mat DeeplabV3_NCNN::refine_mask(const cv::Mat& pr)
{
    cv::Mat mask = (pr > 0);
    mask.convertTo(mask, CV_8U, 255);

    cv::Mat erode_kernel = cv::Mat::ones(cv::Size(3, 3), CV_8U);
    cv::erode(mask, mask, erode_kernel, cv::Point(-1, -1), 1);

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

    // fill holes via floodFill from border
    int h = mask.rows, w = mask.cols;
    cv::Mat flood = mask.clone();
    cv::Mat tmp = cv::Mat::zeros(h+2, w+2, CV_8U);
    cv::floodFill(flood, tmp, cv::Point(0,0), 255);
    cv::Mat flood_inv;
    cv::bitwise_not(flood, flood_inv);
    cv::Mat mask_bin;
    cv::bitwise_or(mask, flood_inv, mask_bin);

    // morphology smoothing
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));
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
    if (image_in.empty()) return cv::Mat();
    if (filled_image != nullptr) {
        filled_image->release();
    }

    cv::Mat image = image_in.clone();
    if (image.channels() == 1)
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

    int nw, nh;
    cv::Mat input = preprocess(image, nw, nh);

    ncnn::Mat in = ncnn::Mat::from_pixels(
        input.data,
        ncnn::Mat::PIXEL_RGB,
        input.cols,
        input.rows,
        input.step
    );

    const float norm_vals[3] = {1.f/255.f, 1.f/255.f, 1.f/255.f};
    in.substract_mean_normalize(nullptr, norm_vals);

    ncnn::Extractor ex = net->create_extractor();
    ex.set_light_mode(false);
    ex.input("in0", in);

    ncnn::Mat out;
    int ret = ex.extract("out0", out);
    if (ret != 0 || out.empty()) {
        std::cerr << "extract out0 failed: " << ret << std::endl;
        return cv::Mat();
    }

    int w = out.w;
    int h = out.h;
    int c = out.c;   // class number

    // -----------------------------
    // 1. two-class logits -> book probability only
    // -----------------------------
    cv::Mat book_prob(h, w, CV_32F);
    const float* bg_base = nullptr;
    const float* book_base = nullptr;
    if (c > 0) bg_base = out.channel(0);
    if (c > 1) book_base = out.channel(1);

    for (int y = 0; y < h; y++)
    {
        const float* bg_ptr = bg_base != nullptr ? bg_base + y * w : nullptr;
        const float* book_ptr = book_base != nullptr ? book_base + y * w : nullptr;
        float* prob_ptr = book_prob.ptr<float>(y);

        for (int x = 0; x < w; x++)
        {
            float p = 0.f;
            if (book_ptr != nullptr && bg_ptr != nullptr) {
                float diff = book_ptr[x] - bg_ptr[x];
                if (diff >= 0.f) {
                    p = 1.0f / (1.0f + std::exp(-diff));
                } else {
                    float e = std::exp(diff);
                    p = e / (1.0f + e);
                }
            }
            prob_ptr[x] = p;
        }
    }

    // -----------------------------
    // 2. crop padding区域
    // -----------------------------
    int left = (input_w - nw) / 2;
    int top  = (input_h - nh) / 2;

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

    // -----------------------------
    // 3. resize book probability back to original size
    // -----------------------------
    cv::Mat book_resized;
    cv::Mat crop = book_prob(cv::Rect(x0, y0, cw, ch)).clone();
    cv::resize(crop, book_resized, image.size(), 0, 0, cv::INTER_LINEAR);

    // -----------------------------
    // 4. build binary label by foreground probability threshold.
    // For two-class segmentation, argmax equals p_book > 0.5. A higher
    // threshold suppresses boundary/background overflow at the cost of holes.
    // -----------------------------
    double min_prob = 0.0;
    double max_prob = 0.0;
    cv::minMaxLoc(book_resized, &min_prob, &max_prob);
    if (max_prob < 0.10 || std::fabs(max_prob - min_prob) < 1e-6) {
        std::cerr << "Collapsed probability output: p_book min=" << min_prob
                  << " max=" << max_prob
                  << " image=" << image.cols << "x" << image.rows
                  << std::endl;
        return cv::Mat();
    }

    // 5. refine + build mask
    auto build_mask_with_threshold = [&](float threshold) -> cv::Mat {
        cv::Mat label(image.rows, image.cols, CV_8U);
        for (int y = 0; y < image.rows; y++)
        {
            const float* prob_ptr = book_resized.ptr<float>(y);
            uchar* label_ptr = label.ptr<uchar>(y);
            for (int x = 0; x < image.cols; x++)
            {
                label_ptr[x] = prob_ptr[x] >= threshold ? 1 : 0;
            }
        }

        cv::Mat bin;
        cv::compare(label, 1, bin, cv::CMP_EQ);
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
        std::cout << "Empty mask: p_book min=" << min_prob
                  << " max=" << max_prob
                  << " image=" << image.cols << "x" << image.rows
                  << std::endl;
    }

    cv::Mat mask_255;
    refined_mask.convertTo(mask_255, CV_8U);
    cv::threshold(mask_255, mask_255, 127, 255, cv::THRESH_BINARY);

    cv::Mat output_mask = mask_255;
    if (enable_corner_lost_process) {
        output_mask = apply_corner_lost_process(mask_255, image, filled_image);
    }

    if (output_type == 1) {
        return output_mask;
    }

    return build_segmentation_overlay(image, mask_255);
}
