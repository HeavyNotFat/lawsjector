#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define LS_EXPORT extern "C" __declspec(dllexport)
#else
#define LS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct Taps {
    std::vector<int> offset;
    std::vector<int> index;
    std::vector<float> weight;
};

Taps build_taps(int src, int dst) {
    Taps taps;
    taps.offset.resize(static_cast<size_t>(dst) + 1, 0);
    const double scale = static_cast<double>(src) / static_cast<double>(dst);
    const double support = std::max(1.0, scale);
    std::vector<float> weights;
    std::vector<int> indices;
    for (int i = 0; i < dst; ++i) {
        const double center = (i + 0.5) * scale - 0.5;
        const int left = static_cast<int>(std::floor(center - support));
        const int right = static_cast<int>(std::ceil(center + support));
        weights.clear();
        indices.clear();
        double total = 0.0;
        for (int j = left; j <= right; ++j) {
            const double dist = std::fabs(center - static_cast<double>(j));
            if (dist >= support) continue;
            const double w = 1.0 - dist / support;
            weights.push_back(static_cast<float>(w));
            indices.push_back(std::min(src - 1, std::max(0, j)));
            total += w;
        }
        if (total <= 0.0) {
            weights = {1.0f};
            indices = {std::min(src - 1, std::max(0, static_cast<int>(center + 0.5)))};
            total = 1.0;
        }
        for (size_t k = 0; k < weights.size(); ++k) {
            taps.index.push_back(indices[k]);
            taps.weight.push_back(static_cast<float>(weights[k] / total));
        }
        taps.offset[static_cast<size_t>(i) + 1] = static_cast<int>(taps.index.size());
    }
    return taps;
}

template <typename Fn>
void parallel_rows(int rows, Fn fn) {
    if (rows <= 0) return;
    const unsigned hw = std::thread::hardware_concurrency();
    const int threads = static_cast<int>(std::max(1u, std::min(hw == 0 ? 4u : hw, 8u)));
    const int chunk = (rows + threads - 1) / threads;
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        const int begin = t * chunk;
        const int end = std::min(rows, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back([begin, end, &fn] { fn(begin, end); });
    }
    for (auto& th : pool) th.join();
}

/**
 * @brief 三角形滤波缩放，上采样等价双线性、下采样接近面积平均。
 * @param src 源图 RGB888
 * @param sw 源宽
 * @param sh 源高
 * @param dst 输出缓冲
 * @param dw 目标宽
 * @param dh 目标高
 */
void resize_rgb(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    const Taps x_taps = build_taps(sw, dw);
    const Taps y_taps = build_taps(sh, dh);
    std::vector<float> tmp(static_cast<size_t>(dw) * sh * 3);
    // 小图不值得开线程，线程创建开销比计算还大
    const bool big = static_cast<long long>(dw) * dh * 3 > 150000;
    auto run_rows = [&](int rows, auto&& fn) {
        if (big) parallel_rows(rows, fn);
        else fn(0, rows);
    };

    run_rows(sh, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            const uint8_t* row = src + static_cast<size_t>(y) * sw * 3;
            float* out = tmp.data() + static_cast<size_t>(y) * dw * 3;
            for (int x = 0; x < dw; ++x) {
                float r = 0.0f, g = 0.0f, b = 0.0f;
                for (int k = x_taps.offset[x]; k < x_taps.offset[x + 1]; ++k) {
                    const int idx = x_taps.index[k] * 3;
                    const float w = x_taps.weight[k];
                    r += w * row[idx];
                    g += w * row[idx + 1];
                    b += w * row[idx + 2];
                }
                out[x * 3] = r;
                out[x * 3 + 1] = g;
                out[x * 3 + 2] = b;
            }
        }
    });

    run_rows(dh, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            uint8_t* out = dst + static_cast<size_t>(y) * dw * 3;
            const int begin = y_taps.offset[y];
            const int end = y_taps.offset[y + 1];
            for (int x = 0; x < dw * 3; ++x) {
                float acc = 0.0f;
                for (int k = begin; k < end; ++k) {
                    acc += y_taps.weight[k] * tmp[static_cast<size_t>(y_taps.index[k]) * dw * 3 + x];
                }
                const float v = acc < 0.0f ? 0.0f : (acc > 255.0f ? 255.0f : acc);
                out[x] = static_cast<uint8_t>(v + 0.5f);
            }
        }
    });
}

inline float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

float bilinear_at(const uint8_t* rgb, int w, int h, float x, float y, int ch) {
    x = std::min(std::max(x, 0.0f), static_cast<float>(w - 1));
    y = std::min(std::max(y, 0.0f), static_cast<float>(h - 1));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(w - 1, x0 + 1);
    const int y1 = std::min(h - 1, y0 + 1);
    const float fx = x - x0;
    const float fy = y - y0;
    const float p00 = rgb[(static_cast<size_t>(y0) * w + x0) * 3 + ch];
    const float p01 = rgb[(static_cast<size_t>(y0) * w + x1) * 3 + ch];
    const float p10 = rgb[(static_cast<size_t>(y1) * w + x0) * 3 + ch];
    const float p11 = rgb[(static_cast<size_t>(y1) * w + x1) * 3 + ch];
    const float top = p00 + (p01 - p00) * fx;
    const float bottom = p10 + (p11 - p10) * fx;
    return top + (bottom - top) * fy;
}

/**
 * @brief 把四边形区域采样成 dst_w x dst_h 的图，用于识别行裁剪。
 * @param rgb 源图 RGB888
 * @param w 源宽
 * @param h 源高
 * @param quad 四角点，顺序为左上、右上、右下、左下
 * @param dst 输出缓冲
 * @param dst_w 输出宽
 * @param dst_h 输出高
 */
void warp_quad(const uint8_t* rgb, int w, int h, const float quad[4][2], uint8_t* dst, int dst_w, int dst_h) {
    const float dx1 = quad[1][0] - quad[0][0];
    const float dy1 = quad[1][1] - quad[0][1];
    const float dx3 = quad[2][0] - quad[3][0];
    const float dy3 = quad[2][1] - quad[3][1];
    for (int y = 0; y < dst_h; ++y) {
        const float v = (dst_h <= 1) ? 0.0f : (y + 0.5f) / dst_h;
        const float top_x = quad[0][0] + (quad[3][0] - quad[0][0]) * v;
        const float top_y = quad[0][1] + (quad[3][1] - quad[0][1]) * v;
        const float step_x = dx1 + (dx3 - dx1) * v;
        const float step_y = dy1 + (dy3 - dy1) * v;
        uint8_t* out_row = dst + static_cast<size_t>(y) * dst_w * 3;
        for (int x = 0; x < dst_w; ++x) {
            const float u = (dst_w <= 1) ? 0.0f : (x + 0.5f) / dst_w;
            float sx = top_x + step_x * u;
            float sy = top_y + step_y * u;
            sx = sx < 0.0f ? 0.0f : (sx > w - 1.0f ? static_cast<float>(w - 1) : sx);
            sy = sy < 0.0f ? 0.0f : (sy > h - 1.0f ? static_cast<float>(h - 1) : sy);
            const int x0 = static_cast<int>(sx);
            const int y0 = static_cast<int>(sy);
            const int x1 = std::min(w - 1, x0 + 1);
            const int y1 = std::min(h - 1, y0 + 1);
            const float fx = sx - x0;
            const float fy = sy - y0;
            const uint8_t* r0 = rgb + static_cast<size_t>(y0) * w * 3;
            const uint8_t* r1 = rgb + static_cast<size_t>(y1) * w * 3;
            uint8_t* out = out_row + x * 3;
            for (int c = 0; c < 3; ++c) {
                const float p00 = r0[x0 * 3 + c];
                const float p01 = r0[x1 * 3 + c];
                const float p10 = r1[x0 * 3 + c];
                const float p11 = r1[x1 * 3 + c];
                const float top = p00 + (p01 - p00) * fx;
                const float bottom = p10 + (p11 - p10) * fx;
                out[c] = static_cast<uint8_t>(top + (bottom - top) * fy + 0.5f);
            }
        }
    }
}

struct Box {
    float pts[4][2];
    float score;
    float x;
    float y;
};

struct Rect {
    float cx, cy, hw, hh, angle;
};

std::vector<std::array<float, 2>> convex_hull(std::vector<std::array<float, 2>> pts) {
    std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) {
        return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
    });
    pts.erase(std::unique(pts.begin(), pts.end(), [](const auto& a, const auto& b) {
                  return a[0] == b[0] && a[1] == b[1];
              }),
              pts.end());
    if (pts.size() < 3) return pts;
    const size_t n = pts.size();
    std::vector<std::array<float, 2>> hull(2 * n);
    size_t k = 0;
    auto cross = [](const std::array<float, 2>& o, const std::array<float, 2>& a,
                    const std::array<float, 2>& b) {
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    };
    for (size_t i = 0; i < n; ++i) {
        while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) --k;
        hull[k++] = pts[i];
    }
    for (size_t i = n - 1, t = k + 1; i > 0; --i) {
        const size_t idx = i - 1;
        while (k >= t && cross(hull[k - 2], hull[k - 1], pts[idx]) <= 0) --k;
        hull[k++] = pts[idx];
    }
    hull.resize(k - 1);
    return hull;
}

Rect min_area_rect(const std::vector<std::array<float, 2>>& hull) {
    Rect best{0, 0, 0, 0, 0};
    float best_area = 1e30f;
    for (size_t i = 0; i < hull.size(); ++i) {
        const auto& p = hull[i];
        const auto& q = hull[(i + 1) % hull.size()];
        const float ex = q[0] - p[0];
        const float ey = q[1] - p[1];
        const float len = std::sqrt(ex * ex + ey * ey);
        if (len < 1e-6f) continue;
        const float ux = ex / len;
        const float uy = ey / len;
        float min_u = 1e30f, max_u = -1e30f, min_v = 1e30f, max_v = -1e30f;
        for (const auto& pt : hull) {
            const float du = pt[0] * ux + pt[1] * uy;
            const float dv = -pt[0] * uy + pt[1] * ux;
            min_u = std::min(min_u, du);
            max_u = std::max(max_u, du);
            min_v = std::min(min_v, dv);
            max_v = std::max(max_v, dv);
        }
        const float area = (max_u - min_u) * (max_v - min_v);
        if (area < best_area) {
            best_area = area;
            const float cu = (min_u + max_u) * 0.5f;
            const float cv = (min_v + max_v) * 0.5f;
            best.cx = cu * ux - cv * uy;
            best.cy = cu * uy + cv * ux;
            best.hw = (max_u - min_u) * 0.5f;
            best.hh = (max_v - min_v) * 0.5f;
            best.angle = std::atan2(uy, ux);
        }
    }
    if (best.hh > best.hw) {
        std::swap(best.hw, best.hh);
        best.angle += kPi / 2.0f;
    }
    while (best.angle > kPi / 4.0f) best.angle -= kPi / 2.0f;
    while (best.angle < -kPi / 4.0f) best.angle += kPi / 2.0f;
    return best;
}

void rect_corners(const Rect& r, float out[4][2]) {
    const float c = std::cos(r.angle);
    const float s = std::sin(r.angle);
    const float us[4] = {-r.hw, r.hw, r.hw, -r.hw};
    const float vs[4] = {-r.hh, -r.hh, r.hh, r.hh};
    for (int i = 0; i < 4; ++i) {
        out[i][0] = r.cx + us[i] * c - vs[i] * s;
        out[i][1] = r.cy + us[i] * s + vs[i] * c;
    }
}

float polygon_mean(const float* pred, int w, int h, const float pts[4][2]) {
    float min_x = pts[0][0], max_x = pts[0][0], min_y = pts[0][1], max_y = pts[0][1];
    for (int i = 1; i < 4; ++i) {
        min_x = std::min(min_x, pts[i][0]);
        max_x = std::max(max_x, pts[i][0]);
        min_y = std::min(min_y, pts[i][1]);
        max_y = std::max(max_y, pts[i][1]);
    }
    const int x0 = std::max(0, static_cast<int>(std::floor(min_x)));
    const int x1 = std::min(w - 1, static_cast<int>(std::ceil(max_x)));
    const int y0 = std::max(0, static_cast<int>(std::floor(min_y)));
    const int y1 = std::min(h - 1, static_cast<int>(std::ceil(max_y)));
    double sum = 0.0;
    int count = 0;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            bool inside = true;
            for (int i = 0; i < 4 && inside; ++i) {
                const float ax = pts[i][0];
                const float ay = pts[i][1];
                const float bx = pts[(i + 1) % 4][0];
                const float by = pts[(i + 1) % 4][1];
                const float cross = (bx - ax) * (y + 0.5f - ay) - (by - ay) * (x + 0.5f - ax);
                if (cross < 0.0f) inside = false;
            }
            if (inside) {
                sum += pred[static_cast<size_t>(y) * w + x];
                ++count;
            }
        }
    }
    return count > 0 ? static_cast<float>(sum / count) : 0.0f;
}

/**
 * @brief DB 后处理：阈值、2x2 膨胀、8 连通域、最小外接矩形、外扩、分数过滤。
 * @param pred 概率图
 * @param w 概率图宽
 * @param h 概率图高
 * @param scale_x 概率图到原图的横轴缩放
 * @param scale_y 概率图到原图的纵轴缩放
 * @param orig_w 原图宽
 * @param orig_h 原图高
 * @return 文本框列表
 */
std::vector<Box> db_postprocess(const float* pred, int w, int h, float scale_x, float scale_y,
                                int orig_w, int orig_h) {
    std::vector<uint8_t> mask(static_cast<size_t>(w) * h, 0);
    for (size_t i = 0; i < mask.size(); ++i) mask[i] = pred[i] > 0.3f ? 1 : 0;

    std::vector<uint8_t> dil(mask.size(), 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            dil[i] = mask[i] || (x > 0 && mask[i - 1]) || (y > 0 && mask[i - w]) ||
                     (x > 0 && y > 0 && mask[i - w - 1]);
        }
    }

    std::vector<int> label(mask.size(), -1);
    std::vector<Box> boxes;
    std::vector<int> stack;
    std::vector<std::array<float, 2>> pixels;
    for (int sy = 0; sy < h; ++sy) {
        for (int sx = 0; sx < w; ++sx) {
            const size_t start = static_cast<size_t>(sy) * w + sx;
            if (!dil[start] || label[start] >= 0) continue;
            pixels.clear();
            stack.clear();
            stack.push_back(static_cast<int>(start));
            label[start] = static_cast<int>(boxes.size());
            while (!stack.empty()) {
                const int cur = stack.back();
                stack.pop_back();
                const int cy = cur / w;
                const int cx = cur % w;
                pixels.push_back({static_cast<float>(cx), static_cast<float>(cy)});
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = cx + dx;
                        const int ny = cy + dy;
                        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                        const size_t ni = static_cast<size_t>(ny) * w + nx;
                        if (!dil[ni] || label[ni] >= 0) continue;
                        label[ni] = label[start];
                        stack.push_back(static_cast<int>(ni));
                    }
                }
            }
            if (pixels.size() < 3) continue;

            const auto hull = convex_hull(pixels);
            if (hull.size() < 3) continue;
            Rect rect = min_area_rect(hull);
            if (std::min(rect.hw, rect.hh) * 2.0f < 3.0f) continue;

            float corners[4][2];
            rect_corners(rect, corners);
            const float score = polygon_mean(pred, w, h, corners);
            if (score < 0.5f) continue;

            const float area = 4.0f * rect.hw * rect.hh;
            const float perimeter = 4.0f * (rect.hw + rect.hh);
            const float distance = area * 1.6f / std::max(1e-6f, perimeter);
            Rect expanded = rect;
            expanded.hw += distance;
            expanded.hh += distance;
            if (std::min(expanded.hw, expanded.hh) * 2.0f < 5.0f) continue;

            Box box;
            box.score = score;
            rect_corners(expanded, box.pts);
            box.x = 1e30f;
            box.y = 1e30f;
            for (int i = 0; i < 4; ++i) {
                box.pts[i][0] = std::min(std::max(box.pts[i][0] * scale_x, 0.0f), orig_w - 1.0f);
                box.pts[i][1] = std::min(std::max(box.pts[i][1] * scale_y, 0.0f), orig_h - 1.0f);
                box.x = std::min(box.x, box.pts[i][0]);
                box.y = std::min(box.y, box.pts[i][1]);
            }
            boxes.push_back(box);
        }
    }
    return boxes;
}

struct LsSession {
    Ort::Session session;
    std::string in_name;
    std::string out_name;
    std::vector<const char*> in_ptr;
    std::vector<const char*> out_ptr;

    LsSession(Ort::Env& env, const std::wstring& path, int threads)
        : session(env, path.c_str(), make_options(threads)) {
        Ort::AllocatorWithDefaultOptions allocator;
        auto in_alloc = session.GetInputNameAllocated(0, allocator);
        auto out_alloc = session.GetOutputNameAllocated(0, allocator);
        in_name = in_alloc.get();
        out_name = out_alloc.get();
        in_ptr = {in_name.c_str()};
        out_ptr = {out_name.c_str()};
    }

    static Ort::SessionOptions make_options(int threads) {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        return options;
    }
};

struct OcrEngine {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "lawsocr"};
    std::unique_ptr<LsSession> det;
    std::unique_ptr<LsSession> rec;
    std::vector<std::string> dict;
    int det_limit = 960;
    int threads = 4;
};

std::vector<std::string> load_dict(const char* utf8) {
    std::vector<std::string> dict;
    dict.emplace_back();
    std::string text(utf8);
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find('\n', start);
        std::string line = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) dict.push_back(line);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    dict.emplace_back(" ");
    return dict;
}

/**
 * @brief 检测输入预处理：限长边 960、32 对齐、归一化、NCHW。
 * @param rgb 原图 RGB888
 * @param w 原图宽
 * @param h 原图高
 * @param out_h 输出张量高
 * @param out_w 输出张量宽
 * @return 归一化后的张量数据
 */
std::vector<float> det_preprocess(const uint8_t* rgb, int w, int h, int& out_h, int& out_w) {
    float ratio = 1.0f;
    const int limit = 960;
    const int max_side = std::max(w, h);
    if (max_side > limit) ratio = static_cast<float>(limit) / static_cast<float>(max_side);
    int resize_h = static_cast<int>(h * ratio);
    int resize_w = static_cast<int>(w * ratio);
    resize_h = std::max(32, static_cast<int>(std::round(resize_h / 32.0)) * 32);
    resize_w = std::max(32, static_cast<int>(std::round(resize_w / 32.0)) * 32);
    out_h = resize_h;
    out_w = resize_w;

    std::vector<uint8_t> resized(static_cast<size_t>(resize_w) * resize_h * 3);
    resize_rgb(rgb, w, h, resized.data(), resize_w, resize_h);

    std::vector<float> tensor(static_cast<size_t>(resize_w) * resize_h * 3);
    const size_t plane = static_cast<size_t>(resize_w) * resize_h;
    for (size_t i = 0; i < plane; ++i) {
        tensor[i] = resized[i * 3] / 127.5f - 1.0f;
        tensor[plane + i] = resized[i * 3 + 1] / 127.5f - 1.0f;
        tensor[plane * 2 + i] = resized[i * 3 + 2] / 127.5f - 1.0f;
    }
    return tensor;
}

}  // namespace

/** @brief 返回原生库版本号。 */
LS_EXPORT const char* ls_version() { return "lawsjector_native 2.0"; }

/**
 * @brief 缩放 RGB 图，供 Python 侧兜底路径使用。
 * @param src 源数据
 * @param src_w 源宽
 * @param src_h 源高
 * @param dst 输出缓冲
 * @param dst_w 目标宽
 * @param dst_h 目标高
 * @return 0 成功，负数失败
 */
LS_EXPORT int ls_resize_rgb(const unsigned char* src, int src_w, int src_h,
                            unsigned char* dst, int dst_w, int dst_h) {
    if (src == nullptr || dst == nullptr) return -1;
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return -2;
    resize_rgb(src, src_w, src_h, dst, dst_w, dst_h);
    return 0;
}

/**
 * @brief 按阅读顺序排序文本行：先按 round(y/10) 分行，再按 x。
 * @param xy 每行的 (x, y)
 * @param count 行数
 * @param order 输出排序下标
 * @return 0 成功，负数失败
 */
LS_EXPORT int ls_order_lines(const float* xy, int count, int* order) {
    if (xy == nullptr || order == nullptr || count <= 0) return -1;
    std::iota(order, order + count, 0);
    std::stable_sort(order, order + count, [xy](int a, int b) {
        const double ya = std::nearbyint(static_cast<double>(xy[a * 2 + 1]) / 10.0);
        const double yb = std::nearbyint(static_cast<double>(xy[b * 2 + 1]) / 10.0);
        if (ya != yb) return ya < yb;
        return xy[a * 2] < xy[b * 2];
    });
    return 0;
}

/**
 * @brief 创建 OCR 引擎并加载检测、识别模型与字符字典。
 * @param det_path 检测模型路径
 * @param rec_path 识别模型路径
 * @param keys_utf8 字符字典内容，UTF-8，每行一个字
 * @param threads ONNX Runtime 线程数，<=0 时用 4
 * @return 引擎句柄，失败返回 nullptr
 */
LS_EXPORT void* lsocr_create(const wchar_t* det_path, const wchar_t* rec_path, const char* keys_utf8, int threads) {
    try {
        auto engine = std::make_unique<OcrEngine>();
        engine->threads = threads > 0 ? threads : 4;
        engine->det = std::make_unique<LsSession>(engine->env, det_path, engine->threads);
        engine->rec = std::make_unique<LsSession>(engine->env, rec_path, engine->threads);
        engine->dict = load_dict(keys_utf8);
        return engine.release();
    } catch (...) {
        return nullptr;
    }
}

/**
 * @brief 销毁引擎。
 * @param handle 引擎句柄
 */
LS_EXPORT void lsocr_destroy(void* handle) {
    delete static_cast<OcrEngine*>(handle);
}

/**
 * @brief 对一张 RGB 图执行检测 + 识别。
 * @param handle 引擎句柄
 * @param rgb 图像数据，RGB888
 * @param w 图像宽
 * @param h 图像高
 * @param out_len 输出文本字节数，可为 nullptr
 * @return 每行 x	y	score	text 的缓冲，失败返回 nullptr，需用 lsocr_free 释放
 */
LS_EXPORT const char* lsocr_run(void* handle, const unsigned char* rgb, int w, int h, int* out_len) {
    if (out_len != nullptr) *out_len = 0;
    auto* engine = static_cast<OcrEngine*>(handle);
    if (engine == nullptr || rgb == nullptr || w <= 0 || h <= 0) return nullptr;
    try {
        const auto t_begin = std::chrono::steady_clock::now();
        // PaddleOCR 按 BGR 训练，彩色文字（蓝链、绿答案）必须先换通道再推理
        std::vector<uint8_t> bgr(static_cast<size_t>(w) * h * 3);
        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
            bgr[i * 3] = rgb[i * 3 + 2];
            bgr[i * 3 + 1] = rgb[i * 3 + 1];
            bgr[i * 3 + 2] = rgb[i * 3];
        }
        const uint8_t* pixels = bgr.data();
        int det_h = 0;
        int det_w = 0;
        const std::vector<float> input = det_preprocess(pixels, w, h, det_h, det_w);

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 4> det_shape{1, 3, det_h, det_w};
        Ort::Value det_tensor = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(input.data()), input.size(), det_shape.data(), 4);
        auto det_out = engine->det->session.Run(Ort::RunOptions{nullptr}, engine->det->in_ptr.data(),
                                                &det_tensor, 1, engine->det->out_ptr.data(), 1);
        auto det_info = det_out[0].GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> det_dims = det_info.GetShape();
        const int out_w = static_cast<int>(det_dims.back());
        const int out_h = static_cast<int>(det_dims[det_dims.size() - 2]);
        const float* prob = det_out[0].GetTensorMutableData<float>();
        const float scale_x = static_cast<float>(w) / static_cast<float>(out_w);
        const float scale_y = static_cast<float>(h) / static_cast<float>(out_h);

        std::vector<Box> boxes = db_postprocess(prob, out_w, out_h, scale_x, scale_y, w, h);
        const auto t_det = std::chrono::steady_clock::now();
        if (std::getenv("LSOCR_DEBUG")) {
            std::fprintf(stderr, "[lsocr] det %dx%d boxes=%zu img=%dx%d\n", out_w, out_h, boxes.size(), w, h);
            if (!boxes.empty()) {
                std::fprintf(stderr, "[lsocr] box0=(%.1f,%.1f)-(%.1f,%.1f) score=%.3f\n",
                             boxes[0].pts[0][0], boxes[0].pts[0][1], boxes[0].pts[2][0], boxes[0].pts[2][1], boxes[0].score);
            }
        }

        std::string text_out;
        if (boxes.empty()) {
            char* empty = static_cast<char*>(std::malloc(1));
            if (empty != nullptr) empty[0] = '\0';
            return empty;
        }

        constexpr int kBatch = 6;
        const int dict_size = static_cast<int>(engine->dict.size());
        std::vector<std::string> lines(boxes.size());
        std::vector<float> scores(boxes.size(), 0.0f);

        std::vector<std::vector<uint8_t>> crops(boxes.size());
        std::vector<std::pair<int, int>> crop_dims(boxes.size());
        std::vector<float> ratios(boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i) {
            const Box& box = boxes[i];
            const float w0 = std::sqrt((box.pts[1][0] - box.pts[0][0]) * (box.pts[1][0] - box.pts[0][0]) +
                                       (box.pts[1][1] - box.pts[0][1]) * (box.pts[1][1] - box.pts[0][1]));
            const float w1 = std::sqrt((box.pts[2][0] - box.pts[3][0]) * (box.pts[2][0] - box.pts[3][0]) +
                                       (box.pts[2][1] - box.pts[3][1]) * (box.pts[2][1] - box.pts[3][1]));
            const float h0 = std::sqrt((box.pts[3][0] - box.pts[0][0]) * (box.pts[3][0] - box.pts[0][0]) +
                                       (box.pts[3][1] - box.pts[0][1]) * (box.pts[3][1] - box.pts[0][1]));
            const float h1 = std::sqrt((box.pts[2][0] - box.pts[1][0]) * (box.pts[2][0] - box.pts[1][0]) +
                                       (box.pts[2][1] - box.pts[1][1]) * (box.pts[2][1] - box.pts[1][1]));
            const int crop_w = std::max(2, static_cast<int>(std::max(w0, w1)));
            const int crop_h = std::max(2, static_cast<int>(std::max(h0, h1)));
            crops[i].resize(static_cast<size_t>(crop_w) * crop_h * 3);
            warp_quad(pixels, w, h, box.pts, crops[i].data(), crop_w, crop_h);
            crop_dims[i] = {crop_w, crop_h};
            ratios[i] = static_cast<float>(crop_w) / static_cast<float>(crop_h);
        }

        // 按宽高比排序后分批，避免一条长行把整批都拉宽
        std::vector<int> order(boxes.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return ratios[a] < ratios[b]; });

        for (size_t begin = 0; begin < order.size(); begin += kBatch) {
            const size_t end = std::min(order.size(), begin + kBatch);
            const int batch = static_cast<int>(end - begin);
            float max_ratio = 320.0f / 48.0f;
            for (size_t i = begin; i < end; ++i) max_ratio = std::max(max_ratio, ratios[order[i]]);
            const int img_w = static_cast<int>(48.0f * max_ratio);
            std::vector<float> rec_input(static_cast<size_t>(batch) * 3 * 48 * img_w, 0.0f);
            for (int b = 0; b < batch; ++b) {
                const int idx = order[begin + b];
                const int resized_w = std::min(img_w, static_cast<int>(std::ceil(48.0f * ratios[idx])));
                std::vector<uint8_t> resized(static_cast<size_t>(resized_w) * 48 * 3);
                resize_rgb(crops[idx].data(), crop_dims[idx].first, crop_dims[idx].second,
                           resized.data(), resized_w, 48);
                const size_t plane = static_cast<size_t>(img_w) * 48;
                float* dst = rec_input.data() + static_cast<size_t>(b) * 3 * plane;
                for (int y = 0; y < 48; ++y) {
                    for (int x = 0; x < resized_w; ++x) {
                        const size_t si = (static_cast<size_t>(y) * resized_w + x) * 3;
                        const size_t di = static_cast<size_t>(y) * img_w + x;
                        dst[di] = resized[si] / 127.5f - 1.0f;
                        dst[plane + di] = resized[si + 1] / 127.5f - 1.0f;
                        dst[plane * 2 + di] = resized[si + 2] / 127.5f - 1.0f;
                    }
                }
            }

            std::array<int64_t, 4> rec_shape{batch, 3, 48, img_w};
            Ort::Value rec_tensor = Ort::Value::CreateTensor<float>(
                mem, rec_input.data(), rec_input.size(), rec_shape.data(), 4);
            auto rec_out = engine->rec->session.Run(Ort::RunOptions{nullptr}, engine->rec->in_ptr.data(),
                                                    &rec_tensor, 1, engine->rec->out_ptr.data(), 1);
            auto rec_info = rec_out[0].GetTensorTypeAndShapeInfo();
            const std::vector<int64_t> rec_dims = rec_info.GetShape();
            const int steps = static_cast<int>(rec_dims[1]);
            const int classes = static_cast<int>(rec_dims[2]);
            const float* logits = rec_out[0].GetTensorMutableData<float>();

            for (int b = 0; b < batch; ++b) {
                const int idx = order[begin + b];
                const float* row = logits + static_cast<size_t>(b) * steps * classes;
                std::string text;
                double conf_sum = 0.0;
                int conf_count = 0;
                int prev = -1;
                for (int t = 0; t < steps; ++t) {
                    const float* step = row + static_cast<size_t>(t) * classes;
                    int best = 0;
                    float max_v = step[0];
                    for (int c = 1; c < classes; ++c) {
                        if (step[c] > max_v) {
                            max_v = step[c];
                            best = c;
                        }
                    }
                    // rec 模型输出已经是 softmax 概率，这里直接取最大值
                    const float prob = clamp01(max_v);
                    if (best != 0 && best != prev) {
                        if (best < dict_size) text += engine->dict[best];
                        conf_sum += prob;
                        ++conf_count;
                    }
                    prev = best;
                }
                lines[idx] = text;
                scores[idx] = conf_count > 0 ? static_cast<float>(conf_sum / conf_count) : 0.0f;
            }
            if (std::getenv("LSOCR_DEBUG")) {
                std::fprintf(stderr, "[lsocr] rec batch=%d img_w=%d first_len=%zu\n", batch, img_w, lines[order[begin]].size());
            }
        }
        const auto t_rec = std::chrono::steady_clock::now();
        if (std::getenv("LSOCR_DEBUG")) {
            const auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            std::fprintf(stderr, "[lsocr] det %.1f ms, rec %.1f ms, total %.1f ms\n",
                         ms(t_begin, t_det), ms(t_det, t_rec), ms(t_begin, t_rec));
        }
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (lines[i].empty() || scores[i] < 0.5f) continue;
            char head[96];
            std::snprintf(head, sizeof(head), "%.1f\t%.1f\t%.4f\t", boxes[i].x, boxes[i].y, scores[i]);
            text_out += head;
            text_out += lines[i];
            text_out += '\n';
        }

        char* result = static_cast<char*>(std::malloc(text_out.size() + 1));
        if (result == nullptr) return nullptr;
        std::memcpy(result, text_out.c_str(), text_out.size() + 1);
        if (out_len != nullptr) *out_len = static_cast<int>(text_out.size());
        return result;
    } catch (...) {
        return nullptr;
    }
}

/**
 * @brief 释放 lsocr_run 返回的缓冲。
 * @param text 缓冲指针
 */
LS_EXPORT void lsocr_free(const char* text) { std::free(const_cast<char*>(text)); }
