#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <optional>
#include <functional>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

struct LineEq {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
};

struct Polynomial {
    // 按降幂存储，例如：
    // 2次：{a, b, c} 表示 a*x^2 + b*x + c
    // 1次：{m, c} 表示 m*x + c
    std::vector<double> coeffs;
};

struct ProcessResult {
    cv::Mat mask;
    cv::Point offset;
    cv::Mat fill;
};

class BookMaskRestorer {
public:
    BookMaskRestorer(double convex_threshold = 0.92,
        int expand_margin = 500,
        cv::Size morph_kernel = cv::Size(15, 15))
        : convex_threshold_(convex_threshold),
        expand_margin_(expand_margin),
        morph_kernel_(cv::getStructuringElement(cv::MORPH_RECT, morph_kernel)),
        NEIGHBOR_COUNT(200),
        BORDER_DIST(5) {}

    // ===================== 缺陷判断 =====================
    std::map<std::string, bool> check_boundary_touch(const cv::Mat& mask) {
        cv::Mat erode;
        cv::erode(mask, erode, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 1);

        int h = erode.rows;
        int w = erode.cols;

        auto anyPositive = [&](const cv::Rect& roi) -> bool {
            cv::Mat sub = erode(roi);
            return cv::countNonZero(sub > 0) > 0;
        };

        return {
            {"top",    anyPositive(cv::Rect(0, 0, w, std::min(10, h)))},
            {"bottom", anyPositive(cv::Rect(0, std::max(0, h - 10), w, std::min(10, h)))},
            {"left",   anyPositive(cv::Rect(0, 0, std::min(10, w), h))},
            {"right",  anyPositive(cv::Rect(std::max(0, w - 10), 0, std::min(10, w), h))}
        };
    }

    bool check_convexity_defect(const cv::Mat& mask) {
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(mask.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (cnts.empty()) return true;

        auto it = std::max_element(cnts.begin(), cnts.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return std::fabs(cv::contourArea(a)) < std::fabs(cv::contourArea(b));
            });
        double cnt_area = std::fabs(cv::contourArea(*it));
        if (cnt_area <= 1e-9) return true;

        std::vector<cv::Point> hull;
        cv::convexHull(*it, hull);
        double hull_area = std::fabs(cv::contourArea(hull));
        if (hull_area <= 1e-9) return true;

        return (cnt_area / hull_area) < convex_threshold_;
    }

    std::pair<bool, std::map<std::string, bool>> is_mask_incomplete(const cv::Mat& mask) {
        auto touch = check_boundary_touch(mask);
        bool incomplete = false;
        for (const auto& kv : touch) {
            if (kv.second) {
                incomplete = true;
                break;
            }
        }
        if (!incomplete) {
            incomplete = check_convexity_defect(mask);
        }
        return { incomplete, touch };
    }

    // ===================== 线段交点 =====================
    std::optional<cv::Point2f> seg_intersect(const cv::Point2f& a1,
        const cv::Point2f& a2,
        const cv::Point2f& b1,
        const cv::Point2f& b2) {
        float d = (a2.x - a1.x) * (b2.y - b1.y) - (a2.y - a1.y) * (b2.x - b1.x);
        if (std::fabs(d) < 1e-6f) return std::nullopt;

        float s = ((b1.x - a1.x) * (b2.y - b1.y) - (b1.y - a1.y) * (b2.x - b1.x)) / d;
        if (s < -1e-6f || s > 1.0f + 1e-6f) return std::nullopt;

        float t = ((b1.x - a1.x) * (a2.y - a1.y) - (b1.y - a1.y) * (a2.x - a1.x)) / d;
        if (t < -1e-6f || t > 1.0f + 1e-6f) return std::nullopt;

        return cv::Point2f(a1.x + s * (a2.x - a1.x), a1.y + s * (a2.y - a1.y));
    }

    // ===================== 获取缺角边 =====================
    std::vector<std::pair<std::string, std::vector<cv::Point2f>>> get_defect_edges(const std::vector<cv::Point>& cnt,
        int h, int w) {
        std::vector<std::pair<std::string, std::pair<cv::Point2f, cv::Point2f>>> edges = {
            {"top",    {{0.f, 0.f}, {static_cast<float>(w - 1), 0.f}}},
            {"bottom", {{0.f, static_cast<float>(h - 1)}, {static_cast<float>(w - 1), static_cast<float>(h - 1)}}},
            {"left",   {{0.f, 0.f}, {0.f, static_cast<float>(h - 1)}}},
            {"right",  {{static_cast<float>(w - 1), 0.f}, {static_cast<float>(w - 1), static_cast<float>(h - 1)}}}
        };

        std::map<std::string, std::vector<cv::Point2f>> inters;
        for (const auto& e : edges) inters[e.first] = {};

        int n = static_cast<int>(cnt.size());
        for (int i = 0; i < n; ++i) {
            cv::Point2f p1(static_cast<float>(cnt[i].x), static_cast<float>(cnt[i].y));
            cv::Point2f p2(static_cast<float>(cnt[(i + 1) % n].x), static_cast<float>(cnt[(i + 1) % n].y));

            for (const auto& e : edges) {
                const std::string& side = e.first;
                const cv::Point2f& b1 = e.second.first;
                const cv::Point2f& b2 = e.second.second;
                auto pt = seg_intersect(p1, p2, b1, b2);
                if (pt.has_value()) {
                    inters[side].push_back(*pt);
                }
            }
        }

        std::vector<std::pair<std::string, std::vector<cv::Point2f>>> result;
        for (const auto& e : edges) {
            const std::string& side = e.first;
            auto pts = inters[side];
            if (pts.size() < 2) {
                continue;
            }

            if (side == "left" || side == "right") {
                std::sort(pts.begin(), pts.end(),
                    [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
                if (std::fabs(pts.back().y - pts.front().y) >= 20.0f) {
                    result.push_back({ side, {pts.front(), pts.back()} });
                }
            }
            else {
                std::sort(pts.begin(), pts.end(),
                    [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
                if (std::fabs(pts.back().x - pts.front().x) >= 20.0f) {
                    result.push_back({ side, {pts.front(), pts.back()} });
                }
            }
        }
        return result;
    }

    // ===================== 方向规则 =====================
    void get_side_rule(const std::string& side,
        std::function<bool(const cv::Point2f&, const cv::Point2f&)>& sort_key,
        std::array<std::string, 2>& dirs) {
        if (side == "left") {
            sort_key = [](const cv::Point2f& p1, const cv::Point2f& p2) { return p1.y < p2.y; };
            dirs = { "cw", "ccw" };
        }
        else if (side == "right") {
            sort_key = [](const cv::Point2f& p1, const cv::Point2f& p2) { return p1.y < p2.y; };
            dirs = { "ccw", "cw" };
        }
        else if (side == "bottom") {
            sort_key = [](const cv::Point2f& p1, const cv::Point2f& p2) { return p1.x < p2.x; };
            dirs = { "cw", "ccw" };
        }
        else if (side == "top") {
            sort_key = [](const cv::Point2f& p1, const cv::Point2f& p2) { return p1.x < p2.x; };
            dirs = { "ccw", "cw" };
        }
    }

    // ===================== 取轮廓点 =====================
    std::vector<cv::Point2f> get_contour_points(const std::vector<cv::Point2f>& cnt_pts,
        const cv::Point2f& target_pt,
        const std::string& direction,
        int img_h, int img_w) {
        double min_dist = std::numeric_limits<double>::max();
        int idx = 0;
        for (int i = 0; i < static_cast<int>(cnt_pts.size()); ++i) {
            double dx = cnt_pts[i].x - target_pt.x;
            double dy = cnt_pts[i].y - target_pt.y;
            double d = std::hypot(dx, dy);
            if (d < min_dist) {
                min_dist = d;
                idx = i;
            }
        }

        int n = static_cast<int>(cnt_pts.size());
        int step = (direction == "cw") ? 1 : -1;

        std::vector<cv::Point2f> points;
        int current_idx = idx;

        int visited = 0;
        int max_visit = std::max(n * 3, NEIGHBOR_COUNT * 3);
        while (static_cast<int>(points.size()) < NEIGHBOR_COUNT && visited < max_visit) {
            current_idx = (current_idx + step + n) % n;
            ++visited;
            float x = cnt_pts[current_idx].x;
            float y = cnt_pts[current_idx].y;

            bool is_img_border =
                (x < BORDER_DIST ||
                    x > img_w - BORDER_DIST ||
                    y < BORDER_DIST ||
                    y > img_h - BORDER_DIST);

            if (!is_img_border) {
                points.push_back(cnt_pts[current_idx]);
            }
        }

        return points;
    }

    // ===================== 直线/曲线判断 =====================
    bool is_straight(const std::vector<cv::Point2f>& pts) {
        std::vector<double> x, y;
        x.reserve(pts.size());
        y.reserve(pts.size());
        for (const auto& p : pts) {
            x.push_back(p.x);
            y.push_back(p.y);
        }

        if (variance(x) < 5.0) return true;
        if (variance(y) < 5.0) return true;

        // y = m*x + c
        double sumx = 0.0, sumy = 0.0, sumxx = 0.0, sumxy = 0.0;
        int n = static_cast<int>(x.size());
        for (int i = 0; i < n; ++i) {
            sumx += x[i];
            sumy += y[i];
            sumxx += x[i] * x[i];
            sumxy += x[i] * y[i];
        }

        double det = n * sumxx - sumx * sumx;
        if (std::fabs(det) < 1e-9) return true;

        double m = (n * sumxy - sumx * sumy) / det;
        double c = (sumy - m * sumx) / n;

        double err = 0.0;
        for (int i = 0; i < n; ++i) {
            double diff = y[i] - m * x[i] - c;
            err += diff * diff;
        }
        err /= n;

        return err < 8.0;
    }

    // ===================== 拟合函数 =====================
    LineEq fit_line(const std::vector<cv::Point2f>& pts) {
        if (pts.size() < 2) {
            return { 0.0, -1.0, 0.0 };
        }

        cv::Vec4f line;
        cv::fitLine(pts, line, cv::DIST_L2, 0.0, 0.01, 0.01);
        double vx = line[0];
        double vy = line[1];
        double x0 = line[2];
        double y0 = line[3];

        double a = vy;
        double b = -vx;
        double cc = vx * y0 - vy * x0;
        double norm = std::hypot(a, b) + 1e-6;
        return { a / norm, b / norm, cc / norm };
    }

    Polynomial fit_curve(const std::vector<cv::Point2f>& pts) {
        std::vector<double> x, y;
        x.reserve(pts.size());
        y.reserve(pts.size());
        for (const auto& p : pts) {
            x.push_back(p.x);
            y.push_back(p.y);
        }

        if (variance(x) < 2.0) {
            LineEq line = fit_line(pts);
            return line_to_polynomial(line);
        }

        // 二次拟合：y = a*x^2 + b*x + cint main() {
//     std::string input_img = "E:/VSTest/CornerLostProcess/corner-lost/input/ori-corner-lost4.jpg";
//     std::string input_mask = "E:/VSTest/CornerLostProcess/corner-lost/input/corner-lost4.png";
//     std::string out_path = "E:/VSTest/CornerLostProcess/corner-lost/output";

//     std::filesystem::create_directories(out_path);

//     cv::Mat img = cv::imread(input_img, cv::IMREAD_COLOR);
//     cv::Mat mask = cv::imread(input_mask, cv::IMREAD_GRAYSCALE);

//     if (img.empty()) {
//         std::cerr << "Failed to read input image: " << input_img << std::endl;
//         return 1;
//     }
//     if (mask.empty()) {
//         std::cerr << "Failed to read input mask: " << input_mask << std::endl;
//         return 1;
//     }

//     std::string filename = std::filesystem::path(input_img).filename().string();
//     int h_ori = mask.rows;
//     int w_ori = mask.cols;

//     int expand_margin = static_cast<int>(std::max(h_ori, w_ori) * 0.15);

//     BookMaskRestorer restorer(0.92, expand_margin, cv::Size(15, 15));
//     ProcessResult res = restorer.process(mask, filename, out_path);

//     int x0 = res.offset.x;
//     int y0 = res.offset.y;

//     cv::Mat result = cv::Mat::zeros(res.mask.rows, res.mask.cols, CV_8UC3);
//     img.copyTo(result(cv::Rect(x0, y0, mask.cols, mask.rows)));

//     cv::Mat fill_binary;
//     cv::compare(res.fill, 255, fill_binary, cv::CMP_EQ);
//     result.setTo(cv::Scalar(255, 255, 255), fill_binary);

//     cv::imwrite((std::filesystem::path(out_path) / ("final_mask_" + filename)).string(), res.mask);
//     cv::imwrite((std::filesystem::path(out_path) / ("final_filled_" + filename)).string(), result);

//     std::cout << "修复完成" << std::endl;
//     return 0;
// }
        double Sx = 0.0, Sx2 = 0.0, Sx3 = 0.0, Sx4 = 0.0;
        double Sy = 0.0, Sxy = 0.0, Sx2y = 0.0;
        int n = static_cast<int>(x.size());

        for (int i = 0; i < n; ++i) {
            double xi = x[i];
            double yi = y[i];
            double xi2 = xi * xi;
            Sx += xi;
            Sx2 += xi2;
            Sx3 += xi2 * xi;
            Sx4 += xi2 * xi2;
            Sy += yi;
            Sxy += xi * yi;
            Sx2y += xi2 * yi;
        }

        // 3x3 高斯消元
        double A[3][4] = {
            {Sx4, Sx3, Sx2, Sx2y},
            {Sx3, Sx2, Sx,  Sxy},
            {Sx2, Sx,  static_cast<double>(n), Sy}
        };

        bool ok = gaussian_elimination_3x3(A);
        if (!ok) {
            LineEq line = fit_line(pts);
            return line_to_polynomial(line);
        }

        Polynomial poly;
        poly.coeffs = { A[0][3], A[1][3], A[2][3] };
        return poly;
    }

    // ===================== 求所有交点 =====================
    std::optional<cv::Point2f> get_extend_intersection(const LineEq& f1, bool is_s1,
        const LineEq& f2, bool is_s2,
        int w, int h, const std::string& side) {
        // 1. 两条直线相交
        if (is_s1 && is_s2) {
            double a1 = f1.a, b1 = f1.b, c1 = f1.c;
            double a2 = f2.a, b2 = f2.b, c2 = f2.c;

            double det = a1 * b2 - a2 * b1;
            if (std::fabs(det) < 1e-6) {
                return std::nullopt;
            }

            cv::Point2f pt(
                static_cast<float>((b1 * c2 - b2 * c1) / det),
                static_cast<float>((a2 * c1 - a1 * c2) / det)
            );

            return select_correct_intersection({ pt }, w, h, side);
        }

        // 2. 解析求所有实数交点
        const LineEq& line = is_s1 ? f1 : f2;
        const Polynomial& curve = is_s1 ? curve2_ : curve1_;

        double a = line.a, b = line.b, c = line.c;

        auto line_y = [&](double x) -> double {
            return (-a * x - c) / (b + 1e-8);
        };

        // curve(x) - line_y(x) = 0
        Polynomial line_poly;
        line_poly.coeffs = { -a / b, -c / b };
        Polynomial poly = subtract_polynomial(curve, line_poly);

        std::vector<cv::Point2f> intersections = solve_real_roots_and_intersections(poly, line_y, w, h);

        return select_correct_intersection(intersections, w, h, side);
    }

    std::optional<cv::Point2f> get_curve_curve_intersection(int w, int h, const std::string& side) {
        if (curve1_.coeffs.empty() || curve2_.coeffs.empty()) {
            return std::nullopt;
        }

        auto eval_poly = [](const Polynomial& poly, double x) -> double {
            double y = 0.0;
            for (double coeff : poly.coeffs) {
                y = y * x + coeff;
            }
            return y;
        };

        Polynomial poly = subtract_polynomial(curve1_, curve2_);
        auto curve_y = [&](double x) -> double {
            return eval_poly(curve1_, x);
        };

        std::vector<cv::Point2f> intersections = solve_real_roots_and_intersections(poly, curve_y, w, h);
        return select_correct_intersection(intersections, w, h, side);
    }

    struct CornerRule {
        std::string name;
        std::string side_a;
        std::string side_b;
        cv::Point2f anchor;
    };

    struct FitResult {
        bool straight = true;
        LineEq line;
        Polynomial curve;
    };

    std::map<std::string, std::vector<cv::Point2f>> get_border_intersections(const std::vector<cv::Point>& cnt,
        int h, int w) {
        std::vector<std::pair<std::string, std::pair<cv::Point2f, cv::Point2f>>> edges = {
            {"top",    {{0.f, 0.f}, {static_cast<float>(w - 1), 0.f}}},
            {"bottom", {{0.f, static_cast<float>(h - 1)}, {static_cast<float>(w - 1), static_cast<float>(h - 1)}}},
            {"left",   {{0.f, 0.f}, {0.f, static_cast<float>(h - 1)}}},
            {"right",  {{static_cast<float>(w - 1), 0.f}, {static_cast<float>(w - 1), static_cast<float>(h - 1)}}}
        };

        std::map<std::string, std::vector<cv::Point2f>> inters;
        for (const auto& e : edges) inters[e.first] = {};

        auto push_unique = [](std::vector<cv::Point2f>& pts, const cv::Point2f& p) {
            for (const auto& old : pts) {
                if (cv::norm(old - p) < 2.0f) return;
            }
            pts.push_back(p);
        };

        int n = static_cast<int>(cnt.size());
        for (int i = 0; i < n; ++i) {
            cv::Point2f p1(static_cast<float>(cnt[i].x), static_cast<float>(cnt[i].y));
            cv::Point2f p2(static_cast<float>(cnt[(i + 1) % n].x), static_cast<float>(cnt[(i + 1) % n].y));

            if (p1.x <= 1.0f) push_unique(inters["left"], p1);
            if (p1.x >= static_cast<float>(w - 2)) push_unique(inters["right"], p1);
            if (p1.y <= 1.0f) push_unique(inters["top"], p1);
            if (p1.y >= static_cast<float>(h - 2)) push_unique(inters["bottom"], p1);

            for (const auto& e : edges) {
                auto pt = seg_intersect(p1, p2, e.second.first, e.second.second);
                if (pt.has_value()) {
                    push_unique(inters[e.first], *pt);
                }
            }
        }

        return inters;
    }

    std::optional<cv::Point2f> select_corner_border_point(const std::vector<cv::Point2f>& pts,
        const std::string& side,
        const CornerRule& rule) {
        if (pts.empty()) return std::nullopt;

        bool prefer_max = false;
        if (rule.name == "top_left") {
            prefer_max = true;
        }
        else if (rule.name == "top_right") {
            prefer_max = (side == "right");
        }
        else if (rule.name == "bottom_left") {
            prefer_max = (side == "bottom");
        }
        else if (rule.name == "bottom_right") {
            prefer_max = false;
        }

        auto value = [&](const cv::Point2f& p) {
            if (side == "left" || side == "right") {
                return p.y;
            }
            return p.x;
        };

        return *std::min_element(pts.begin(), pts.end(),
            [&](const cv::Point2f& a, const cv::Point2f& b) {
                return prefer_max ? value(a) > value(b) : value(a) < value(b);
            });
    }

    std::vector<cv::Point2f> get_hull_corners(const std::vector<cv::Point>& cnt) {
        std::vector<cv::Point> hull;
        cv::convexHull(cnt, hull);
        if (hull.empty()) return {};

        double peri = cv::arcLength(hull, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(hull, approx, std::max(3.0, peri * 0.01), true);

        std::vector<cv::Point2f> corners;
        corners.reserve(approx.size());
        for (const auto& p : approx) {
            corners.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
        }
        return corners;
    }

    std::optional<cv::Point2f> select_visible_corner(const std::vector<cv::Point2f>& corners,
        const std::string& side,
        const CornerRule& rule,
        int h,
        int w) {
        if (corners.empty()) return std::nullopt;

        float tol = std::max(30.0f, static_cast<float>(std::max(h, w)) * 0.04f);
        std::vector<cv::Point2f> candidates;
        for (const auto& p : corners) {
            bool near_side = false;
            if (side == "left") {
                near_side = p.x <= tol;
            }
            else if (side == "right") {
                near_side = p.x >= static_cast<float>(w - 1) - tol;
            }
            else if (side == "top") {
                near_side = p.y <= tol;
            }
            else if (side == "bottom") {
                near_side = p.y >= static_cast<float>(h - 1) - tol;
            }

            if (near_side) {
                candidates.push_back(p);
            }
        }

        if (candidates.empty()) return std::nullopt;

        bool prefer_max = false;
        if (rule.name == "top_left") {
            prefer_max = true;
        }
        else if (rule.name == "top_right") {
            prefer_max = (side == "right");
        }
        else if (rule.name == "bottom_left") {
            prefer_max = (side == "bottom");
        }
        else if (rule.name == "bottom_right") {
            prefer_max = false;
        }

        auto value = [&](const cv::Point2f& p) {
            if (side == "left" || side == "right") {
                return p.y;
            }
            return p.x;
        };

        return *std::min_element(candidates.begin(), candidates.end(),
            [&](const cv::Point2f& a, const cv::Point2f& b) {
                return prefer_max ? value(a) > value(b) : value(a) < value(b);
            });
    }

    FitResult fit_edge(const std::vector<cv::Point2f>& pts) {
        FitResult fit;
        fit.straight = is_straight(pts);
        fit.line = fit_line(pts);
        if (!fit.straight) {
            fit.curve = fit_curve(pts);
        }
        return fit;
    }

    std::vector<cv::Point2f> intersect_line_curve(const LineEq& line,
        const Polynomial& curve,
        int w,
        int h) {
        std::vector<cv::Point2f> intersections;

        if (curve.coeffs.empty()) {
            return intersections;
        }

        if (std::fabs(line.b) < 1e-8) {
            if (std::fabs(line.a) < 1e-8) return intersections;

            double x = -line.c / line.a;
            double y = eval_polynomial(curve, x);
            if (std::fabs(x) <= w * 10.0 && std::fabs(y) <= h * 10.0) {
                intersections.emplace_back(static_cast<float>(x), static_cast<float>(y));
            }
            return intersections;
        }

        auto line_y = [&](double x) -> double {
            return (-line.a * x - line.c) / line.b;
        };

        Polynomial line_poly;
        line_poly.coeffs = { -line.a / line.b, -line.c / line.b };
        Polynomial poly = subtract_polynomial(curve, line_poly);
        return solve_real_roots_and_intersections(poly, line_y, w, h);
    }

    std::vector<cv::Point2f> get_fit_intersections(const FitResult& f1,
        const FitResult& f2,
        int w,
        int h) {
        std::vector<cv::Point2f> intersections;

        if (f1.straight && f2.straight) {
            double det = f1.line.a * f2.line.b - f2.line.a * f1.line.b;
            if (std::fabs(det) < 1e-6) return intersections;

            intersections.emplace_back(
                static_cast<float>((f1.line.b * f2.line.c - f2.line.b * f1.line.c) / det),
                static_cast<float>((f2.line.a * f1.line.c - f1.line.a * f2.line.c) / det)
            );
            return intersections;
        }

        if (f1.straight && !f2.straight) {
            return intersect_line_curve(f1.line, f2.curve, w, h);
        }

        if (!f1.straight && f2.straight) {
            return intersect_line_curve(f2.line, f1.curve, w, h);
        }

        Polynomial poly = subtract_polynomial(f1.curve, f2.curve);
        auto curve_y = [&](double x) -> double {
            return eval_polynomial(f1.curve, x);
        };
        return solve_real_roots_and_intersections(poly, curve_y, w, h);
    }

    std::vector<cv::Point2f> get_line_intersections(const LineEq& line1, const LineEq& line2) {
        std::vector<cv::Point2f> intersections;
        double det = line1.a * line2.b - line2.a * line1.b;
        if (std::fabs(det) < 1e-6) return intersections;

        intersections.emplace_back(
            static_cast<float>((line1.b * line2.c - line2.b * line1.c) / det),
            static_cast<float>((line2.a * line1.c - line1.a * line2.c) / det)
        );
        return intersections;
    }

    bool is_outside_corner(const cv::Point2f& p, const CornerRule& rule, int h, int w) {
        if (rule.name == "top_left") {
            return p.x < 0.0f && p.y < 0.0f;
        }
        if (rule.name == "top_right") {
            return p.x > static_cast<float>(w - 1) && p.y < 0.0f;
        }
        if (rule.name == "bottom_left") {
            return p.x < 0.0f && p.y > static_cast<float>(h - 1);
        }
        if (rule.name == "bottom_right") {
            return p.x > static_cast<float>(w - 1) && p.y > static_cast<float>(h - 1);
        }
        return false;
    }

    bool is_in_expand_range(const cv::Point2f& p, int h, int w) {
        return p.x >= -expand_margin_ &&
            p.y >= -expand_margin_ &&
            p.x <= static_cast<float>(w - 1 + expand_margin_) &&
            p.y <= static_cast<float>(h - 1 + expand_margin_);
    }

    cv::Point2f clamp_to_expand_range(const cv::Point2f& p, int h, int w) {
        return cv::Point2f(
            std::clamp(p.x, -static_cast<float>(expand_margin_), static_cast<float>(w - 1 + expand_margin_)),
            std::clamp(p.y, -static_cast<float>(expand_margin_), static_cast<float>(h - 1 + expand_margin_))
        );
    }

    std::optional<cv::Point2f> find_corner_intersection(const std::vector<cv::Point2f>& cnt_pts,
        const cv::Point2f& p1,
        const cv::Point2f& p2,
        const CornerRule& rule,
        int h,
        int w) {
        std::optional<cv::Point2f> best;
        double best_score = std::numeric_limits<double>::max();
        const std::array<std::string, 2> dirs = { "cw", "ccw" };

        for (const auto& dir1 : dirs) {
            std::vector<cv::Point2f> pts1 = get_contour_points(cnt_pts, p1, dir1, h, w);
            if (pts1.size() < 8) continue;

            for (const auto& dir2 : dirs) {
                std::vector<cv::Point2f> pts2 = get_contour_points(cnt_pts, p2, dir2, h, w);
                if (pts2.size() < 8) continue;

                FitResult fit1 = fit_edge(pts1);
                FitResult fit2 = fit_edge(pts2);
                std::vector<cv::Point2f> crosses = get_fit_intersections(fit1, fit2, w, h);
                std::vector<cv::Point2f> line_crosses = get_line_intersections(fit1.line, fit2.line);
                crosses.insert(crosses.end(), line_crosses.begin(), line_crosses.end());

                for (const auto& cross : crosses) {
                    if (!is_outside_corner(cross, rule, h, w)) continue;

                    double score = cv::norm(cross - rule.anchor);
                    if (score < best_score) {
                        best_score = score;
                        best = cross;
                    }
                }
            }
        }

        return best;
    }

    std::optional<std::pair<cv::Point2f, cv::Point2f>> select_side_segment(const std::vector<cv::Point2f>& pts,
        const std::string& side) {
        if (pts.size() < 2) return std::nullopt;

        auto value = [&](const cv::Point2f& p) {
            if (side == "left" || side == "right") {
                return p.y;
            }
            return p.x;
        };

        auto min_it = std::min_element(pts.begin(), pts.end(),
            [&](const cv::Point2f& a, const cv::Point2f& b) {
                return value(a) < value(b);
            });
        auto max_it = std::max_element(pts.begin(), pts.end(),
            [&](const cv::Point2f& a, const cv::Point2f& b) {
                return value(a) < value(b);
            });

        if (std::fabs(value(*max_it) - value(*min_it)) < 20.0f) {
            return std::nullopt;
        }

        return std::make_pair(*min_it, *max_it);
    }

    bool is_outside_side(const cv::Point2f& p, const std::string& side, int h, int w) {
        if (side == "left") {
            return p.x < 0.0f;
        }
        if (side == "right") {
            return p.x > static_cast<float>(w - 1);
        }
        if (side == "top") {
            return p.y < 0.0f;
        }
        if (side == "bottom") {
            return p.y > static_cast<float>(h - 1);
        }
        return false;
    }

    double distance_to_side_endpoint_axis(const cv::Point2f& p,
        const cv::Point2f& p1,
        const cv::Point2f& p2,
        const std::string& side) {
        if (side == "left" || side == "right") {
            return std::min(std::fabs(p.y - p1.y), std::fabs(p.y - p2.y));
        }
        return std::min(std::fabs(p.x - p1.x), std::fabs(p.x - p2.x));
    }

    double side_segment_length_axis(const cv::Point2f& p1,
        const cv::Point2f& p2,
        const std::string& side) {
        if (side == "left" || side == "right") {
            return std::fabs(p2.y - p1.y);
        }
        return std::fabs(p2.x - p1.x);
    }

    double outside_side_distance(const cv::Point2f& p, const std::string& side, int h, int w) {
        if (side == "left") {
            return std::fabs(std::min(p.x, 0.0f));
        }
        if (side == "right") {
            return std::fabs(std::max(p.x - static_cast<float>(w - 1), 0.0f));
        }
        if (side == "top") {
            return std::fabs(std::min(p.y, 0.0f));
        }
        if (side == "bottom") {
            return std::fabs(std::max(p.y - static_cast<float>(h - 1), 0.0f));
        }
        return 0.0;
    }

    std::optional<cv::Point2f> find_single_side_intersection(const std::vector<cv::Point2f>& cnt_pts,
        const cv::Point2f& p1,
        const cv::Point2f& p2,
        const std::string& side,
        int h,
        int w) {
        std::optional<cv::Point2f> best;
        double best_score = std::numeric_limits<double>::max();
        double side_len = side_segment_length_axis(p1, p2, side);
        double endpoint_threshold = std::max(120.0, side_len * 0.25);
        const std::array<std::string, 2> dirs = { "cw", "ccw" };

        for (const auto& dir1 : dirs) {
            std::vector<cv::Point2f> pts1 = get_contour_points(cnt_pts, p1, dir1, h, w);
            if (pts1.size() < 8) continue;

            for (const auto& dir2 : dirs) {
                std::vector<cv::Point2f> pts2 = get_contour_points(cnt_pts, p2, dir2, h, w);
                if (pts2.size() < 8) continue;

                FitResult fit1 = fit_edge(pts1);
                FitResult fit2 = fit_edge(pts2);
                std::vector<cv::Point2f> crosses = get_fit_intersections(fit1, fit2, w, h);
                std::vector<cv::Point2f> line_crosses = get_line_intersections(fit1.line, fit2.line);
                crosses.insert(crosses.end(), line_crosses.begin(), line_crosses.end());

                for (const auto& cross : crosses) {
                    if (!is_outside_side(cross, side, h, w)) continue;
                    if (!is_in_expand_range(cross, h, w)) continue;

                    double endpoint_dist = distance_to_side_endpoint_axis(cross, p1, p2, side);
                    if (endpoint_dist > endpoint_threshold) continue;

                    double score = endpoint_dist + outside_side_distance(cross, side, h, w) * 0.1;
                    if (score < best_score) {
                        best_score = score;
                        best = cross;
                    }
                }
            }
        }

        return best;
    }

    double distance_to_axis_range(const cv::Point2f& p,
        const cv::Point2f& p1,
        const cv::Point2f& p2,
        const std::string& side) {
        double v = (side == "left" || side == "right") ? p.y : p.x;
        double a = (side == "left" || side == "right") ? p1.y : p1.x;
        double b = (side == "left" || side == "right") ? p2.y : p2.x;
        double lo = std::min(a, b);
        double hi = std::max(a, b);
        if (v < lo) return lo - v;
        if (v > hi) return v - hi;
        return 0.0;
    }

    std::optional<cv::Point2f> find_side_intersection_between_corners(const std::vector<cv::Point2f>& cnt_pts,
        const cv::Point2f& p1,
        const cv::Point2f& p2,
        const std::string& side,
        int h,
        int w) {
        std::optional<cv::Point2f> best;
        double best_score = std::numeric_limits<double>::max();
        const std::array<std::string, 2> dirs = { "cw", "ccw" };

        for (const auto& dir1 : dirs) {
            std::vector<cv::Point2f> pts1 = get_contour_points(cnt_pts, p1, dir1, h, w);
            if (pts1.size() < 8) continue;

            for (const auto& dir2 : dirs) {
                std::vector<cv::Point2f> pts2 = get_contour_points(cnt_pts, p2, dir2, h, w);
                if (pts2.size() < 8) continue;

                FitResult fit1 = fit_edge(pts1);
                FitResult fit2 = fit_edge(pts2);
                std::vector<cv::Point2f> crosses = get_fit_intersections(fit1, fit2, w, h);
                std::vector<cv::Point2f> line_crosses = get_line_intersections(fit1.line, fit2.line);
                crosses.insert(crosses.end(), line_crosses.begin(), line_crosses.end());

                for (const auto& cross : crosses) {
                    if (!is_outside_side(cross, side, h, w)) continue;

                    double axis_dist = distance_to_axis_range(cross, p1, p2, side);
                    double score = axis_dist + outside_side_distance(cross, side, h, w) * 0.1;
                    if (score < best_score) {
                        best_score = score;
                        best = cross;
                    }
                }
            }
        }

        return best;
    }

    std::optional<cv::Point2f> intersect_line_with_expand_side(const LineEq& line,
        const std::string& side,
        int h,
        int w) {
        if (side == "left" || side == "right") {
            double x = (side == "left")
                ? -static_cast<double>(expand_margin_)
                : static_cast<double>(w - 1 + expand_margin_);
            if (std::fabs(line.b) < 1e-8) return std::nullopt;

            double y = (-line.a * x - line.c) / line.b;
            return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
        }

        double y = (side == "top")
            ? -static_cast<double>(expand_margin_)
            : static_cast<double>(h - 1 + expand_margin_);
        if (std::fabs(line.a) < 1e-8) return std::nullopt;

        double x = (-line.b * y - line.c) / line.a;
        return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    }

    std::optional<cv::Point2f> extend_corner_to_expand_side(const std::vector<cv::Point2f>& cnt_pts,
        const cv::Point2f& corner,
        const std::string& side,
        int h,
        int w) {
        std::optional<cv::Point2f> best;
        double best_score = std::numeric_limits<double>::max();
        double local_threshold = std::max(80.0, static_cast<double>(std::max(h, w)) * 0.08);
        const std::array<std::string, 2> dirs = { "cw", "ccw" };

        for (const auto& dir : dirs) {
            std::vector<cv::Point2f> pts = get_contour_points(cnt_pts, corner, dir, h, w);
            if (pts.size() < 8) continue;
            if (cv::norm(pts.front() - corner) > local_threshold) continue;

            LineEq line = fit_line(pts);
            std::optional<cv::Point2f> pt = intersect_line_with_expand_side(line, side, h, w);
            if (!pt.has_value()) continue;
            if (!is_outside_side(*pt, side, h, w)) continue;

            double axis_dist = distance_to_side_endpoint_axis(*pt, corner, corner, side);
            double score = axis_dist + cv::norm(*pt - corner) * 0.01;
            if (score < best_score) {
                best_score = score;
                best = pt;
            }
        }

        return best;
    }

    // ===================== 补全逻辑 =====================
    ProcessResult restore_mask(const cv::Mat& mask, const std::string& filename, const std::string& out_path) {
        int h = mask.rows;
        int w = mask.cols;

        cv::Mat mask_m;
        cv::morphologyEx(mask, mask_m, cv::MORPH_CLOSE, morph_kernel_);

        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(mask_m.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        if (cnts.empty()) {
            return { mask.clone(), cv::Point(0, 0), cv::Mat::zeros(mask.size(), mask.type()) };
        }

        auto it = std::max_element(cnts.begin(), cnts.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return std::fabs(cv::contourArea(a)) < std::fabs(cv::contourArea(b));
            });
        const std::vector<cv::Point>& cnt = *it;

        std::vector<cv::Point2f> cnt_pts;
        cnt_pts.reserve(cnt.size());
        for (const auto& p : cnt) {
            cnt_pts.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
        }

        auto touch = check_boundary_touch(mask);
        std::map<std::string, std::vector<cv::Point2f>> border_points = get_border_intersections(cnt, h, w);
        std::vector<cv::Point2f> hull_corners = get_hull_corners(cnt);

        curve1_ = Polynomial{};
        curve2_ = Polynomial{};
        std::vector<std::vector<cv::Point2f>> fill_polys_original;

        std::vector<CornerRule> rules = {
            {"top_left", "left", "top", cv::Point2f(0.f, 0.f)},
            {"top_right", "right", "top", cv::Point2f(static_cast<float>(w - 1), 0.f)},
            {"bottom_left", "left", "bottom", cv::Point2f(0.f, static_cast<float>(h - 1))},
            {"bottom_right", "right", "bottom", cv::Point2f(static_cast<float>(w - 1), static_cast<float>(h - 1))}
        };

        for (const auto& rule : rules) {
            if (!touch[rule.side_a] || !touch[rule.side_b]) {
                continue;
            }

            auto p1 = select_visible_corner(hull_corners, rule.side_a, rule, h, w);
            auto p2 = select_visible_corner(hull_corners, rule.side_b, rule, h, w);
            if (!p1.has_value()) {
                p1 = select_corner_border_point(border_points[rule.side_a], rule.side_a, rule);
            }
            if (!p2.has_value()) {
                p2 = select_corner_border_point(border_points[rule.side_b], rule.side_b, rule);
            }
            if (!p1.has_value() || !p2.has_value()) {
                continue;
            }

            std::optional<cv::Point2f> cross_a = extend_corner_to_expand_side(cnt_pts, *p1, rule.side_a, h, w);
            std::optional<cv::Point2f> cross_b = extend_corner_to_expand_side(cnt_pts, *p2, rule.side_b, h, w);
            if (!cross_a.has_value()) {
                cross_a = find_side_intersection_between_corners(cnt_pts, *p1, *p2, rule.side_a, h, w);
            }
            if (!cross_b.has_value()) {
                cross_b = find_side_intersection_between_corners(cnt_pts, *p1, *p2, rule.side_b, h, w);
            }
            if (!cross_a.has_value() || !cross_b.has_value()) {
                continue;
            }

            cv::Point2f c1 = clamp_to_expand_range(*p1, h, w);
            cv::Point2f c2 = clamp_to_expand_range(*cross_a, h, w);
            cv::Point2f c3 = clamp_to_expand_range(*cross_b, h, w);
            cv::Point2f c4 = clamp_to_expand_range(*p2, h, w);

            std::vector<cv::Point2f> poly = { c1, c2, c3, c4 };
            double area = std::fabs(cv::contourArea(poly));
            if (area < 50.0) {
                continue;
            }

            fill_polys_original.push_back(poly);
        }

        int touch_count = 0;
        std::string single_side;
        for (const auto& side : std::array<std::string, 4>{ "left", "right", "top", "bottom" }) {
            if (touch[side]) {
                ++touch_count;
                single_side = side;
            }
        }

        if (touch_count == 1) {
            auto segment = select_side_segment(border_points[single_side], single_side);
            if (segment.has_value()) {
                cv::Point2f p1 = segment->first;
                cv::Point2f p2 = segment->second;
                std::optional<cv::Point2f> cross = find_single_side_intersection(cnt_pts, p1, p2, single_side, h, w);
                if (cross.has_value()) {
                    cv::Point2f c1 = clamp_to_expand_range(p1, h, w);
                    cv::Point2f c2 = clamp_to_expand_range(p2, h, w);
                    cv::Point2f c3 = clamp_to_expand_range(*cross, h, w);

                    double area = std::fabs((c1.x * (c2.y - c3.y) + c2.x * (c3.y - c1.y) + c3.x * (c1.y - c2.y)) * 0.5);
                    if (area >= 50.0) {
                        fill_polys_original.push_back({
                            c1,
                            c2,
                            c3
                        });
                    }
                }
            }
        }

        if (fill_polys_original.empty()) {
            return { mask.clone(), cv::Point(0, 0), cv::Mat::zeros(mask.size(), mask.type()) };
        }

        double min_x = 0.0;
        double min_y = 0.0;
        double max_x = static_cast<double>(w - 1);
        double max_y = static_cast<double>(h - 1);
        for (const auto& poly : fill_polys_original) {
            for (const auto& p : poly) {
                min_x = std::min(min_x, static_cast<double>(p.x));
                min_y = std::min(min_y, static_cast<double>(p.y));
                max_x = std::max(max_x, static_cast<double>(p.x));
                max_y = std::max(max_y, static_cast<double>(p.y));
            }
        }

        int min_ix = static_cast<int>(std::floor(min_x));
        int min_iy = static_cast<int>(std::floor(min_y));
        int max_ix = static_cast<int>(std::ceil(max_x));
        int max_iy = static_cast<int>(std::ceil(max_y));

        int x0 = -min_ix;
        int y0 = -min_iy;
        int canvas_w = max_ix - min_ix + 1;
        int canvas_h = max_iy - min_iy + 1;

        cv::Mat new_canvas = cv::Mat::zeros(canvas_h, canvas_w, CV_8UC1);
        mask.copyTo(new_canvas(cv::Rect(x0, y0, w, h)));

        cv::Mat fill_mask = cv::Mat::zeros(new_canvas.size(), CV_8UC1);
        for (const auto& poly_original : fill_polys_original) {
            std::vector<cv::Point> poly;
            poly.reserve(poly_original.size());
            for (const auto& p : poly_original) {
                poly.emplace_back(
                    static_cast<int>(std::round(p.x + x0)),
                    static_cast<int>(std::round(p.y + y0))
                );
            }

            std::vector<std::vector<cv::Point>> polys = { poly };
            cv::fillPoly(fill_mask, polys, cv::Scalar(255));
        }

        cv::Mat final_mask;
        cv::bitwise_or(new_canvas, fill_mask, final_mask);

        return { final_mask, cv::Point(x0, y0), fill_mask };
    }

    ProcessResult process(const cv::Mat& mask, const std::string& filename, const std::string& out_path) {
        cv::Mat bin_mask;
        cv::threshold(mask, bin_mask, 127, 255, cv::THRESH_BINARY);

        auto incomplete = is_mask_incomplete(bin_mask);
        if (incomplete.first) {
            return restore_mask(bin_mask, filename, out_path);
        }

        return { bin_mask.clone(), cv::Point(0, 0), cv::Mat::zeros(bin_mask.size(), bin_mask.type()) };
    }

private:
    double convex_threshold_;
    int expand_margin_;
    cv::Mat morph_kernel_;

public:
    const int NEIGHBOR_COUNT;
    const int BORDER_DIST;

private:
    LineEq line1_;
    LineEq line2_;
    Polynomial curve1_;
    Polynomial curve2_;

private:
    static double variance(const std::vector<double>& vals) {
        if (vals.empty()) return 0.0;
        double sum = 0.0;
        for (double v : vals) sum += v;
        double mean = sum / vals.size();

        double var = 0.0;
        for (double v : vals) {
            double d = v - mean;
            var += d * d;
        }
        return var / vals.size();
    }

    static bool gaussian_elimination_3x3(double A[3][4]) {
        for (int i = 0; i < 3; ++i) {
            int pivot = i;
            for (int r = i + 1; r < 3; ++r) {
                if (std::fabs(A[r][i]) > std::fabs(A[pivot][i])) pivot = r;
            }
            if (std::fabs(A[pivot][i]) < 1e-12) return false;
            if (pivot != i) {
                for (int c = i; c < 4; ++c) std::swap(A[i][c], A[pivot][c]);
            }

            double div = A[i][i];
            for (int c = i; c < 4; ++c) A[i][c] /= div;

            for (int r = 0; r < 3; ++r) {
                if (r == i) continue;
                double factor = A[r][i];
                for (int c = i; c < 4; ++c) {
                    A[r][c] -= factor * A[i][c];
                }
            }
        }
        return true;
    }

    static Polynomial line_to_polynomial(const LineEq& line) {
        // y = (-a/b)x + (-c/b)
        double m = -line.a / (line.b + 1e-8);
        double c = -line.c / (line.b + 1e-8);
        return Polynomial{ {m, c} };
    }

    static Polynomial subtract_polynomial(const Polynomial& p1, const Polynomial& p2) {
        int n1 = static_cast<int>(p1.coeffs.size());
        int n2 = static_cast<int>(p2.coeffs.size());
        int n = std::max(n1, n2);

        std::vector<double> a(n, 0.0), b(n, 0.0);
        for (int i = 0; i < n1; ++i) a[n - n1 + i] = p1.coeffs[i];
        for (int i = 0; i < n2; ++i) b[n - n2 + i] = p2.coeffs[i];

        std::vector<double> c(n, 0.0);
        for (int i = 0; i < n; ++i) c[i] = a[i] - b[i];

        // 去掉前导 0
        while (c.size() > 1 && std::fabs(c.front()) < 1e-12) {
            c.erase(c.begin());
        }
        return Polynomial{ c };
    }

    static double eval_polynomial(const Polynomial& poly, double x) {
        double y = 0.0;
        for (double coeff : poly.coeffs) {
            y = y * x + coeff;
        }
        return y;
    }

    static std::vector<cv::Point2f> solve_real_roots_and_intersections(const Polynomial& poly,
        const std::function<double(double)>& line_y,
        int w, int h) {
        std::vector<cv::Point2f> intersections;
        std::vector<double> c = poly.coeffs;

        while (c.size() > 1 && std::fabs(c.front()) < 1e-12) {
            c.erase(c.begin());
        }

        if (c.empty()) return intersections;

        if (c.size() == 1) {
            return intersections;
        }
        else if (c.size() == 2) {
            double a = c[0];
            double b = c[1];
            if (std::fabs(a) < 1e-12) return intersections;
            double x = -b / a;
            double y = line_y(x);
            if (std::fabs(x) <= w * 10.0 && std::fabs(y) <= h * 10.0) {
                intersections.push_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
            }
        }
        else if (c.size() == 3) {
            double a = c[0], b = c[1], d = c[2];
            double disc = b * b - 4.0 * a * d;
            if (disc < -1e-12) return intersections;
            if (disc < 0.0) disc = 0.0;

            double sqrt_disc = std::sqrt(disc);
            double x1 = (-b + sqrt_disc) / (2.0 * a);
            double x2 = (-b - sqrt_disc) / (2.0 * a);

            auto push_if_valid = [&](double x) {
                double y = line_y(x);
                if (std::fabs(x) <= w * 10.0 && std::fabs(y) <= h * 10.0) {
                    intersections.emplace_back(static_cast<float>(x), static_cast<float>(y));
                }
            };

            push_if_valid(x1);
            if (std::fabs(x2 - x1) > 1e-12) {
                push_if_valid(x2);
            }
        }

        return intersections;
    }

    static std::optional<cv::Point2f> select_correct_intersection(const std::vector<cv::Point2f>& points,
        int w, int h, const std::string& side) {
        if (points.empty()) return std::nullopt;

        std::vector<cv::Point2f> pts = points;
        std::vector<float> xs, ys;
        xs.reserve(pts.size());
        ys.reserve(pts.size());
        for (const auto& p : pts) {
            xs.push_back(p.x);
            ys.push_back(p.y);
        }

        if (side == "left") {
            std::vector<cv::Point2f> valid;
            for (const auto& p : pts) if (p.x < 0) valid.push_back(p);
            if (valid.empty()) return std::nullopt;
            return *std::min_element(valid.begin(), valid.end(),
                [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
        }
        else if (side == "right") {
            std::vector<cv::Point2f> valid;
            for (const auto& p : pts) if (p.x > w) valid.push_back(p);
            if (valid.empty()) return std::nullopt;
            return *std::max_element(valid.begin(), valid.end(),
                [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
        }
        else if (side == "top") {
            std::vector<cv::Point2f> valid;
            for (const auto& p : pts) if (p.y < 0) valid.push_back(p);
            if (valid.empty()) return std::nullopt;
            return *std::min_element(valid.begin(), valid.end(),
                [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
        }
        else if (side == "bottom") {
            std::vector<cv::Point2f> valid;
            for (const auto& p : pts) if (p.y > h) valid.push_back(p);
            if (valid.empty()) return std::nullopt;
            return *std::max_element(valid.begin(), valid.end(),
                [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
        }

        return pts.front();
    }

    static void print_points(const std::vector<cv::Point2f>& pts) {
        std::cout << "[";
        for (size_t i = 0; i < pts.size(); ++i) {
            std::cout << "[" << pts[i].x << " " << pts[i].y << "]";
            if (i + 1 != pts.size()) std::cout << "\n ";
        }
        std::cout << "]" << std::endl;
    }
};

#ifdef CORNER_LOST_PROCESS_STANDALONE
// int main() {
//     std::string input_img = "E:/VSTest/CornerLostProcess/corner-lost/input/ori-corner-lost4.jpg";
//     std::string input_mask = "E:/VSTest/CornerLostProcess/corner-lost/input/corner-lost4.png";
//     std::string out_path = "E:/VSTest/CornerLostProcess/corner-lost/output";

//     std::filesystem::create_directories(out_path);

//     cv::Mat img = cv::imread(input_img, cv::IMREAD_COLOR);
//     cv::Mat mask = cv::imread(input_mask, cv::IMREAD_GRAYSCALE);

//     if (img.empty()) {
//         std::cerr << "Failed to read input image: " << input_img << std::endl;
//         return 1;
//     }
//     if (mask.empty()) {
//         std::cerr << "Failed to read input mask: " << input_mask << std::endl;
//         return 1;
//     }

//     std::string filename = std::filesystem::path(input_img).filename().string();
//     int h_ori = mask.rows;
//     int w_ori = mask.cols;

//     int expand_margin = static_cast<int>(std::max(h_ori, w_ori) * 0.15);

//     BookMaskRestorer restorer(0.92, expand_margin, cv::Size(15, 15));
//     ProcessResult res = restorer.process(mask, filename, out_path);

//     int x0 = res.offset.x;
//     int y0 = res.offset.y;

//     cv::Mat result = cv::Mat::zeros(res.mask.rows, res.mask.cols, CV_8UC3);
//     img.copyTo(result(cv::Rect(x0, y0, mask.cols, mask.rows)));

//     cv::Mat fill_binary;
//     cv::compare(res.fill, 255, fill_binary, cv::CMP_EQ);
//     result.setTo(cv::Scalar(255, 255, 255), fill_binary);

//     cv::imwrite((std::filesystem::path(out_path) / ("final_mask_" + filename)).string(), res.mask);
//     cv::imwrite((std::filesystem::path(out_path) / ("final_filled_" + filename)).string(), result);

//     std::cout << "修复完成" << std::endl;
//     return 0;
// }
#endif
