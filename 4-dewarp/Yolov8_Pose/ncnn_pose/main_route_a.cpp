#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "net.h"

namespace {

constexpr int kInputSize = 512;
constexpr int kTopK = 20;
constexpr float kToleranceRatio = 0.03f;
constexpr float kConfidenceThreshold = 0.40f;
constexpr float kLetterboxPad = 114.0f;

const std::array<std::array<int, 4>, 24> kPermutations = {{
    {{0, 1, 2, 3}}, {{0, 1, 3, 2}}, {{0, 2, 1, 3}}, {{0, 2, 3, 1}},
    {{0, 3, 1, 2}}, {{0, 3, 2, 1}}, {{1, 0, 2, 3}}, {{1, 0, 3, 2}},
    {{1, 2, 0, 3}}, {{1, 2, 3, 0}}, {{1, 3, 0, 2}}, {{1, 3, 2, 0}},
    {{2, 0, 1, 3}}, {{2, 0, 3, 1}}, {{2, 1, 0, 3}}, {{2, 1, 3, 0}},
    {{2, 3, 0, 1}}, {{2, 3, 1, 0}}, {{3, 0, 1, 2}}, {{3, 0, 2, 1}},
    {{3, 1, 0, 2}}, {{3, 1, 2, 0}}, {{3, 2, 0, 1}}, {{3, 2, 1, 0}},
}};

using Quad = std::array<cv::Point2f, 4>;

struct RouteAResult {
    Quad quad;
    int class_id = -1;
    float confidence = 0.0f;
};

struct Detection {
    int class_id = -1;
    float confidence = 0.0f;
    Quad corners;
    bool corner_clipped = false;
};

static std::array<int, 4> best_permutation(const Quad& target, const Quad& candidate) {
    std::array<int, 4> best = kPermutations[0];
    double best_error = std::numeric_limits<double>::infinity();
    for (const auto& permutation : kPermutations) {
        double error = 0.0;
        for (int point = 0; point < 4; ++point) {
            error += static_cast<double>(cv::norm(candidate[permutation[point]] - target[point]));
        }
        if (error < best_error) {
            best_error = error;
            best = permutation;
        }
    }
    return best;
}

static double quad_distance(const Quad& first, const Quad& second) {
    const auto permutation = best_permutation(first, second);
    double distance = 0.0;
    for (int point = 0; point < 4; ++point) {
        distance += static_cast<double>(cv::norm(second[permutation[point]] - first[point]));
    }
    return distance / 4.0;
}

// 当前三分类 YOLOv8-Pose 输出：
// out0 = [box(4), class(3), keypoints(12)] x anchors。
// 有意不使用 NMS，而是选取 Top-K 原始 anchor，
// 计算几何 medoid，并融合几何一致的候选。
static RouteAResult select_route_a(const ncnn::Mat& output) {
    if (output.h < 16 || output.w <= 0) {
        throw std::runtime_error("unexpected out0 shape");
    }
    const int channels = output.h;
    const int anchor_count = output.w;
    const int class_count = channels - 16;
    const int keypoint_offset = 4 + class_count;

    std::vector<float> scores(anchor_count);
    for (int anchor = 0; anchor < anchor_count; ++anchor) {
        float best_score = -1.0f;
        for (int class_id = 0; class_id < class_count; ++class_id) {
            best_score = std::max(best_score, output.row(4 + class_id)[anchor]);
        }
        scores[anchor] = best_score;
    }

    std::vector<int> order(anchor_count);
    std::iota(order.begin(), order.end(), 0);
    const int k = std::min(kTopK, anchor_count);
    std::partial_sort(
        order.begin(), order.begin() + k, order.end(),
        [&scores](int first, int second) { return scores[first] > scores[second]; }
    );

    std::vector<Quad> quads(k);
    std::vector<float> selected_scores(k);
    for (int index = 0; index < k; ++index) {
        const int anchor = order[index];
        selected_scores[index] = scores[anchor];
        for (int point = 0; point < 4; ++point) {
            quads[index][point] = cv::Point2f(
                output.row(keypoint_offset + point * 3)[anchor],
                output.row(keypoint_offset + point * 3 + 1)[anchor]
            );
        }
    }

    std::vector<std::vector<double>> distances(k, std::vector<double>(k, 0.0));
    for (int first = 0; first < k; ++first) {
        for (int second = first + 1; second < k; ++second) {
            const double distance = quad_distance(quads[first], quads[second]);
            distances[first][second] = distance;
            distances[second][first] = distance;
        }
    }

    int medoid = 0;
    double best_total = std::numeric_limits<double>::infinity();
    for (int candidate = 0; candidate < k; ++candidate) {
        const double total = std::accumulate(
            distances[candidate].begin(), distances[candidate].end(), 0.0
        );
        if (total < best_total) {
            best_total = total;
            medoid = candidate;
        }
    }

    const double tolerance_px =
        static_cast<double>(kToleranceRatio) * kInputSize * std::sqrt(2.0);
    std::vector<int> kept;
    double score_sum = 0.0;
    for (int candidate = 0; candidate < k; ++candidate) {
        if (distances[medoid][candidate] <= tolerance_px) {
            kept.push_back(candidate);
            score_sum += selected_scores[candidate];
        }
    }
    if (kept.empty()) {
        kept.push_back(medoid);
        score_sum = selected_scores[medoid];
    }

    Quad stabilized = {{{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}}};
    for (int candidate : kept) {
        const float weight = static_cast<float>(selected_scores[candidate] / score_sum);
        const auto permutation = best_permutation(quads[medoid], quads[candidate]);
        for (int point = 0; point < 4; ++point) {
            stabilized[point] += weight * quads[candidate][permutation[point]];
        }
    }

    int class_id = 0;
    const int best_anchor = order.front();
    float best_class_score = -1.0f;
    for (int candidate_class = 0; candidate_class < class_count; ++candidate_class) {
        const float score = output.row(4 + candidate_class)[best_anchor];
        if (score > best_class_score) {
            best_class_score = score;
            class_id = candidate_class;
        }
    }

    return {stabilized, class_id, scores[best_anchor]};
}

static float polygon_area(const std::vector<cv::Point2f>& polygon) {
    float twice_area = 0.0f;
    for (size_t index = 0; index < polygon.size(); ++index) {
        const cv::Point2f& current = polygon[index];
        const cv::Point2f& next = polygon[(index + 1) % polygon.size()];
        twice_area += current.x * next.y - current.y * next.x;
    }
    return std::abs(twice_area) * 0.5f;
}

static std::vector<cv::Point2f> clip_polygon(
    const std::vector<cv::Point2f>& polygon,
    float width,
    float height
) {
    enum class Boundary { Left, Right, Top, Bottom };
    const Boundary boundaries[] = {
        Boundary::Left, Boundary::Right, Boundary::Top, Boundary::Bottom
    };
    auto inside = [width, height](const cv::Point2f& point, Boundary boundary) {
        switch (boundary) {
            case Boundary::Left: return point.x >= 0.0f;
            case Boundary::Right: return point.x <= width;
            case Boundary::Top: return point.y >= 0.0f;
            case Boundary::Bottom: return point.y <= height;
        }
        return false;
    };
    auto intersection = [](const cv::Point2f& first, const cv::Point2f& second,
                           Boundary boundary, float width, float height) {
        float delta = 0.0f;
        float limit = 0.0f;
        bool vertical = false;
        switch (boundary) {
            case Boundary::Left: delta = second.x - first.x; limit = 0.0f; vertical = true; break;
            case Boundary::Right: delta = second.x - first.x; limit = width; vertical = true; break;
            case Boundary::Top: delta = second.y - first.y; limit = 0.0f; break;
            case Boundary::Bottom: delta = second.y - first.y; limit = height; break;
        }
        const float source = vertical ? first.x : first.y;
        const float interpolation = std::abs(delta) < 1e-6f
            ? 0.0f : (limit - source) / delta;
        return first + interpolation * (second - first);
    };

    std::vector<cv::Point2f> current = polygon;
    for (Boundary boundary : boundaries) {
        if (current.empty()) {
            return {};
        }
        std::vector<cv::Point2f> next;
        for (size_t index = 0; index < current.size(); ++index) {
            const cv::Point2f& first = current[index];
            const cv::Point2f& second = current[(index + 1) % current.size()];
            const bool first_inside = inside(first, boundary);
            const bool second_inside = inside(second, boundary);
            if (first_inside != second_inside) {
                next.push_back(intersection(first, second, boundary, width, height));
            }
            if (second_inside) {
                next.push_back(second);
            }
        }
        if (!next.empty() && cv::norm(next.front() - next.back()) <= 1e-6f) {
            next.pop_back();
        }
        current = std::move(next);
    }
    return current;
}

static Quad largest_inscribed_quad(std::vector<cv::Point2f> polygon) {
    if (polygon.size() < 4) {
        throw std::runtime_error("visible polygon has fewer than four vertices");
    }
    if (polygon.size() == 4) {
        return {polygon[0], polygon[1], polygon[2], polygon[3]};
    }
    float best_area = -1.0f;
    Quad best{};
    for (size_t first = 0; first < polygon.size(); ++first) {
        for (size_t second = first + 1; second < polygon.size(); ++second) {
            for (size_t third = second + 1; third < polygon.size(); ++third) {
                for (size_t fourth = third + 1; fourth < polygon.size(); ++fourth) {
                    const std::vector<cv::Point2f> candidate = {
                        polygon[first], polygon[second], polygon[third], polygon[fourth]
                    };
                    const float area = polygon_area(candidate);
                    if (area > best_area) {
                        best_area = area;
                        best = {candidate[0], candidate[1], candidate[2], candidate[3]};
                    }
                }
            }
        }
    }
    return best;
}

static Quad reorder_corners(const Quad& quad) {
    cv::Point2f center(0.0f, 0.0f);
    for (const auto& point : quad) {
        center += point;
    }
    center *= 0.25f;

    std::array<int, 4> order = {0, 1, 2, 3};
    std::sort(order.begin(), order.end(), [&quad, center](int first, int second) {
        return std::atan2(quad[first].y - center.y, quad[first].x - center.x) <
               std::atan2(quad[second].y - center.y, quad[second].x - center.x);
    });

    float minimum_sum = std::numeric_limits<float>::infinity();
    for (int index : order) {
        minimum_sum = std::min(minimum_sum, quad[index].x + quad[index].y);
    }
    int start = 0;
    for (int index = 0; index < 4; ++index) {
        if (std::abs(quad[order[index]].x + quad[order[index]].y - minimum_sum) <= 1e-6f &&
            quad[order[index]].y < quad[order[start]].y) {
            start = index;
        }
    }

    Quad result{};
    for (int index = 0; index < 4; ++index) {
        result[index] = quad[order[(start + index) % 4]];
    }
    return result;
}

static std::pair<Quad, bool> visible_quad(
    const Quad& quad,
    int image_width,
    int image_height
) {
    const std::vector<cv::Point2f> clipped = clip_polygon(
        {quad[0], quad[1], quad[2], quad[3]},
        static_cast<float>(image_width),
        static_cast<float>(image_height)
    );
    if (clipped.size() == 4) {
        return {{clipped[0], clipped[1], clipped[2], clipped[3]}, false};
    }
    if (clipped.size() > 4) {
        return {largest_inscribed_quad(clipped), true};
    }
    return {quad, true};
}

class RouteADocumentDetector {
public:
    RouteADocumentDetector(const std::string& param_path, const std::string& bin_path) {
        network_.opt.use_vulkan_compute = false;
        if (network_.load_param(param_path.c_str()) != 0 ||
            network_.load_model(bin_path.c_str()) != 0) {
            throw std::runtime_error("failed to load ncnn model");
        }
        class_names_ = {"book", "non_book", "card"};
    }

    bool detect(const cv::Mat& image, Detection& detection) const {
        const int image_width = image.cols;
        const int image_height = image.rows;
        const float scale = std::min(
            static_cast<float>(kInputSize) / image_width,
            static_cast<float>(kInputSize) / image_height
        );
        const int resized_width = static_cast<int>(std::round(image_width * scale));
        const int resized_height = static_cast<int>(std::round(image_height * scale));

        cv::Mat resized;
        cv::resize(
            image, resized, cv::Size(resized_width, resized_height),
            0.0, 0.0, cv::INTER_AREA
        );

        const float dw = (kInputSize - resized_width) / 2.0f;
        const float dh = (kInputSize - resized_height) / 2.0f;
        const int left = static_cast<int>(std::round(dw - 0.1f));
        const int top = static_cast<int>(std::round(dh - 0.1f));
        cv::Mat letterboxed;
        cv::copyMakeBorder(
            resized, letterboxed, top, kInputSize - resized_height - top,
            left, kInputSize - resized_width - left,
            cv::BORDER_CONSTANT, cv::Scalar(kLetterboxPad, kLetterboxPad, kLetterboxPad)
        );

        ncnn::Mat input = ncnn::Mat::from_pixels(
            letterboxed.data, ncnn::Mat::PIXEL_BGR2RGB, kInputSize, kInputSize
        );
        const float normalization[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        input.substract_mean_normalize(nullptr, normalization);

        ncnn::Extractor extractor = network_.create_extractor();
        if (extractor.input("in0", input) != 0) {
            throw std::runtime_error("failed to set ncnn input");
        }
        ncnn::Mat output;
        if (extractor.extract("out0", output) != 0) {
            throw std::runtime_error("failed to extract ncnn out0");
        }

        RouteAResult route_a = select_route_a(output);
        if (route_a.confidence < kConfidenceThreshold) {
            return false;
        }

        for (auto& point : route_a.quad) {
            point.x = (point.x - left) / scale;
            point.y = (point.y - top) / scale;
        }
        auto visible = visible_quad(route_a.quad, image_width, image_height);
        detection.class_id = route_a.class_id;
        detection.confidence = route_a.confidence;
        detection.corners = reorder_corners(visible.first);
        detection.corner_clipped = visible.second;
        return true;
    }

    const std::vector<std::string>& class_names() const {
        return class_names_;
    }

private:
    ncnn::Net network_;
    std::vector<std::string> class_names_;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0]
                  << " <model.param> <model.bin> <input.jpg> <output.jpg>\n";
        return 1;
    }

    try {
        RouteADocumentDetector detector(argv[1], argv[2]);
        cv::Mat image = cv::imread(argv[3], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "failed to read image: " << argv[3] << "\n";
            return 2;
        }

        Detection detection;
        if (!detector.detect(image, detection)) {
            std::cout << "no document above confidence threshold\n";
            return 3;
        }

        const char* corner_names[] = {"TL", "TR", "BR", "BL"};
        std::cout << "class=" << detector.class_names().at(detection.class_id)
                  << " confidence=" << detection.confidence
                  << " corner_clipped=" << detection.corner_clipped << "\n";
        std::cout << std::setprecision(9);

        std::vector<cv::Point> polygon;
        for (int index = 0; index < 4; ++index) {
            const cv::Point2f& point = detection.corners[index];
            std::cout << corner_names[index] << "=(" << point.x << ", "
                      << point.y << ")\n";
            polygon.emplace_back(cvRound(point.x), cvRound(point.y));
            cv::circle(image, point, 8, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
            cv::putText(
                image, corner_names[index], point + cv::Point2f(12.0f, -10.0f),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2, cv::LINE_AA
            );
        }
        cv::polylines(image, polygon, true, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        if (!cv::imwrite(argv[4], image)) {
            std::cerr << "failed to write image: " << argv[4] << "\n";
            return 4;
        }
        std::cout << "output=" << argv[4] << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "route-a inference failed: " << error.what() << "\n";
        return 5;
    }
}
