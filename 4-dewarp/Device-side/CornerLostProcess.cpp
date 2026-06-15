#include <opencv2/opencv.hpp>
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

            if (p1.x <= 1.0f) inters["left"].push_back(p1);
            if (p1.x >= static_cast<float>(w - 2)) inters["right"].push_back(p1);
            if (p1.y <= 1.0f) inters["top"].push_back(p1);
            if (p1.y >= static_cast<float>(h - 2)) inters["bottom"].push_back(p1);

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

        // 二次拟合：y = a*x^2 + b*x + c
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

    // ===================== 补全逻辑 =====================
    ProcessResult restore_mask(const cv::Mat& mask) {
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

        auto defect_edges = get_defect_edges(cnt, h, w);
        if (defect_edges.empty()) {
            return { mask.clone(), cv::Point(0, 0), cv::Mat::zeros(mask.size(), mask.type()) };
        }

        int x0 = expand_margin_;
        int y0 = expand_margin_;

        cv::Mat new_canvas = cv::Mat::zeros(h + 2 * y0, w + 2 * x0, CV_8UC1);
        mask.copyTo(new_canvas(cv::Rect(x0, y0, w, h)));

        cv::Mat fill_mask = cv::Mat::zeros(new_canvas.size(), CV_8UC1);

        curve1_ = Polynomial{};
        curve2_ = Polynomial{};

        for (const auto& side_item : defect_edges) {
            const std::string& side = side_item.first;
            const std::vector<cv::Point2f>& inter_pts = side_item.second;

            std::function<bool(const cv::Point2f&, const cv::Point2f&)> sort_key;
            std::array<std::string, 2> dirs;
            get_side_rule(side, sort_key, dirs);

            std::vector<cv::Point2f> sorted = inter_pts;
            std::sort(sorted.begin(), sorted.end(), sort_key);

            cv::Point2f p1 = sorted[0];
            cv::Point2f p2 = sorted[1];

            std::vector<cv::Point2f> pts1 = get_contour_points(cnt_pts, p1, dirs[0], h, w);
            std::vector<cv::Point2f> pts2 = get_contour_points(cnt_pts, p2, dirs[1], h, w);

            bool s1 = is_straight(pts1);
            bool s2 = is_straight(pts2);

            line1_ = fit_line(pts1);
            line2_ = fit_line(pts2);

            if (!s1) {
                curve1_ = fit_curve(pts1);
            }
            if (!s2) {
                curve2_ = fit_curve(pts2);
            }

            std::optional<cv::Point2f> cross;
            if (s1 && s2) {
                cross = get_extend_intersection(line1_, true, line2_, true, w, h, side);
            }
            else if (s1 && !s2) {
                cross = get_extend_intersection(line1_, true, line2_, false, w, h, side);
            }
            else if (!s1 && s2) {
                cross = get_extend_intersection(line1_, false, line2_, true, w, h, side);
            }
            else {
                cross = get_curve_curve_intersection(w, h, side);
            }

            if (!cross.has_value() && !(s1 && s2)) {
                cross = get_extend_intersection(line1_, true, line2_, true, w, h, side);
            }

            if (!cross.has_value()) {
                continue;
            }

            std::vector<cv::Point> poly = {
                cv::Point(static_cast<int>(std::round(p1.x + x0)), static_cast<int>(std::round(p1.y + y0))),
                cv::Point(static_cast<int>(std::round(p2.x + x0)), static_cast<int>(std::round(p2.y + y0))),
                cv::Point(static_cast<int>(std::round(cross->x + x0)), static_cast<int>(std::round(cross->y + y0)))
            };

            std::vector<std::vector<cv::Point>> polys = { poly };
            cv::fillPoly(fill_mask, polys, cv::Scalar(255));
        }

        cv::Mat final_mask;
        cv::bitwise_or(new_canvas, fill_mask, final_mask);

        return { final_mask, cv::Point(x0, y0), fill_mask };
    }

    ProcessResult process(const cv::Mat& mask) {
        cv::Mat bin_mask;
        cv::threshold(mask, bin_mask, 127, 255, cv::THRESH_BINARY);

        auto incomplete = is_mask_incomplete(bin_mask);
        if (incomplete.first) {
            return restore_mask(bin_mask);
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

};
