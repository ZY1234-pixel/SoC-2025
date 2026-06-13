#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include "deeplabv3p_ncnn.h"

namespace fs = std::filesystem;

static bool is_image_file(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

static uint16_t read_u16(const std::vector<unsigned char>& data, size_t offset, bool little_endian) {
    if (offset + 1 >= data.size()) return 0;
    if (little_endian) {
        return static_cast<uint16_t>(data[offset] | (data[offset + 1] << 8));
    }
    return static_cast<uint16_t>((data[offset] << 8) | data[offset + 1]);
}

static uint32_t read_u32(const std::vector<unsigned char>& data, size_t offset, bool little_endian) {
    if (offset + 3 >= data.size()) return 0;
    if (little_endian) {
        return static_cast<uint32_t>(data[offset] |
            (data[offset + 1] << 8) |
            (data[offset + 2] << 16) |
            (data[offset + 3] << 24));
    }
    return static_cast<uint32_t>((data[offset] << 24) |
        (data[offset + 1] << 16) |
        (data[offset + 2] << 8) |
        data[offset + 3]);
}

static int read_exif_orientation(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return 1;

    unsigned char soi[2] = {0, 0};
    file.read(reinterpret_cast<char*>(soi), 2);
    if (soi[0] != 0xff || soi[1] != 0xd8) return 1;

    while (file) {
        unsigned char marker_prefix = 0;
        file.read(reinterpret_cast<char*>(&marker_prefix), 1);
        if (marker_prefix != 0xff) return 1;

        unsigned char marker = 0;
        file.read(reinterpret_cast<char*>(&marker), 1);
        while (marker == 0xff && file) {
            file.read(reinterpret_cast<char*>(&marker), 1);
        }

        if (marker == 0xda || marker == 0xd9) return 1;

        unsigned char len_bytes[2] = {0, 0};
        file.read(reinterpret_cast<char*>(len_bytes), 2);
        int segment_len = (len_bytes[0] << 8) | len_bytes[1];
        if (segment_len < 2) return 1;

        std::vector<unsigned char> segment(static_cast<size_t>(segment_len - 2));
        file.read(reinterpret_cast<char*>(segment.data()), static_cast<std::streamsize>(segment.size()));
        if (!file) return 1;

        if (marker != 0xe1 || segment.size() < 14) {
            continue;
        }
        if (!(segment[0] == 'E' && segment[1] == 'x' && segment[2] == 'i' &&
              segment[3] == 'f' && segment[4] == 0 && segment[5] == 0)) {
            continue;
        }

        const size_t tiff = 6;
        bool little_endian = false;
        if (segment[tiff] == 'I' && segment[tiff + 1] == 'I') {
            little_endian = true;
        } else if (segment[tiff] == 'M' && segment[tiff + 1] == 'M') {
            little_endian = false;
        } else {
            return 1;
        }

        if (read_u16(segment, tiff + 2, little_endian) != 42) return 1;
        uint32_t ifd_offset = read_u32(segment, tiff + 4, little_endian);
        size_t ifd = tiff + ifd_offset;
        if (ifd + 2 > segment.size()) return 1;

        uint16_t entry_count = read_u16(segment, ifd, little_endian);
        size_t entry = ifd + 2;
        for (uint16_t i = 0; i < entry_count; ++i, entry += 12) {
            if (entry + 12 > segment.size()) break;
            uint16_t tag = read_u16(segment, entry, little_endian);
            if (tag == 0x0112) {
                uint16_t orientation = read_u16(segment, entry + 8, little_endian);
                if (orientation >= 1 && orientation <= 8) {
                    return orientation;
                }
                return 1;
            }
        }
    }

    return 1;
}

static void apply_exif_orientation(cv::Mat& image, int orientation) {
    switch (orientation) {
        case 2:
            cv::flip(image, image, 1);
            break;
        case 3:
            cv::rotate(image, image, cv::ROTATE_180);
            break;
        case 4:
            cv::flip(image, image, 0);
            break;
        case 6:
            cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
            break;
        case 8:
            cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
        default:
            break;
    }
}

static fs::path make_filled_save_path(const fs::path& mask_save_path) {
    fs::path filled_path = mask_save_path;
    filled_path.replace_filename(
        mask_save_path.stem().string() + "_filled" + mask_save_path.extension().string()
    );
    return filled_path;
}

static fs::path current_executable_path(const char* argv0) {
    char path_buf[4096] = {0};
    ssize_t len = readlink("/proc/self/exe", path_buf, sizeof(path_buf) - 1);
    if (len > 0) {
        path_buf[len] = '\0';
        return fs::path(path_buf);
    }
    return fs::absolute(argv0);
}

static bool process_one(const fs::path& input_path, const fs::path& save_path) {
    cv::Mat image = cv::imread(input_path.string(), cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << input_path << std::endl;
        return false;
    }
    apply_exif_orientation(image, read_exif_orientation(input_path));

    auto start = std::chrono::high_resolution_clock::now();
    DeeplabV3_NCNN deeplab;
    cv::Mat filled_image;
    cv::Mat result = deeplab.detect_image(image, &filled_image);
    auto end = std::chrono::high_resolution_clock::now();

    if (result.empty()) {
        std::cerr << "Inference returned empty result: " << input_path << std::endl;
        return false;
    }

    fs::path actual_save_path = save_path;
    if (result.channels() == 1) {
        std::string ext = actual_save_path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".jpg" || ext == ".jpeg") {
            actual_save_path.replace_extension(".png");
        }
    }

    if (!fs::exists(actual_save_path.parent_path())) {
        fs::create_directories(actual_save_path.parent_path());
    }

    if (!cv::imwrite(actual_save_path.string(), result)) {
        std::cerr << "Failed to save image: " << actual_save_path << std::endl;
        return false;
    }

    fs::path filled_save_path;
    if (!filled_image.empty()) {
        filled_save_path = make_filled_save_path(actual_save_path);
        if (!cv::imwrite(filled_save_path.string(), filled_image)) {
            std::cerr << "Failed to save filled image: " << filled_save_path << std::endl;
            return false;
        }
    }

    double infer_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << input_path.filename().string()
              << " | inference time: "
              << infer_time_ms << " ms"
              << " | saved: " << actual_save_path;
    if (!filled_image.empty()) {
        std::cout << " | filled: " << filled_save_path;
    }
    std::cout << std::endl;

    return true;
}

static bool process_one_in_child(const fs::path& exe_path, const fs::path& input_path, const fs::path& save_path) {
    if (!fs::exists(save_path.parent_path())) {
        fs::create_directories(save_path.parent_path());
    }

    std::string exe = exe_path.string();
    std::string input = input_path.string();
    std::string output = save_path.string();

    constexpr int max_attempts = 3;
    for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        pid_t pid = fork();
        if (pid < 0) {
            std::cerr << "Failed to fork for image: " << input_path << std::endl;
            return false;
        }

        if (pid == 0) {
            execl(exe.c_str(), exe.c_str(), input.c_str(), output.c_str(), static_cast<char*>(nullptr));
            std::cerr << "Failed to exec child process: " << exe_path
                      << " error=" << std::strerror(errno) << std::endl;
            _exit(127);
        }

        int status = 0;
        if (waitpid(pid, &status, 0) < 0) {
            std::cerr << "Failed to wait child process: " << input_path << std::endl;
            return false;
        }

        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            return true;
        }

        std::cerr << "Child inference failed: " << input_path
                  << " attempt=" << attempt
                  << " status=" << status << std::endl;
    }

    return false;
}

int main(int argc, char** argv) {
    std::cout << "main start" << std::endl;
    fs::path exe_path = current_executable_path(argv[0]);
    fs::path dir_origin_path = DeeplabV3_NCNN::kDefaultInputPath;
    fs::path dir_save_path   = DeeplabV3_NCNN::kDefaultSavePath;

    if (argc >= 2) {
        fs::path input_path = argv[1];
        if (fs::is_directory(input_path)) {
            fs::path output_dir = argc >= 3 ? fs::path(argv[2]) : dir_save_path;
            fs::create_directories(output_dir);

            int processed = 0;
            int failed = 0;
            for (const auto& entry : fs::directory_iterator(input_path)) {
                if (!entry.is_regular_file() || !is_image_file(entry.path())) {
                    continue;
                }
                fs::path save_path = output_dir / entry.path().filename();
                if (process_one_in_child(exe_path, entry.path(), save_path)) {
                    processed++;
                } else {
                    failed++;
                }
            }
            std::cout << "Done. processed=" << processed << " failed=" << failed << std::endl;
            return failed == 0 ? 0 : 1;
        }

        fs::path save_path = argc >= 3 ? fs::path(argv[2]) : (dir_save_path / input_path.filename());
        bool ok = process_one(input_path, save_path);
        return ok ? 0 : 1;
    }

    if (!fs::exists(dir_save_path)) {
        fs::create_directories(dir_save_path);
    }

    int processed = 0;
    int failed = 0;
    for (const auto& entry : fs::directory_iterator(dir_origin_path)) {
        if (!entry.is_regular_file() || !is_image_file(entry.path())) {
            continue;
        }

        std::string filename = entry.path().filename().string();
        fs::path save_path = fs::path(dir_save_path) / filename;
        if (process_one_in_child(exe_path, entry.path(), save_path)) {
            processed++;
        } else {
            failed++;
        }
    }

    std::cout << "Done. processed=" << processed << " failed=" << failed << std::endl;

    return 0;
}
