cmake_minimum_required(VERSION 3.12)
project(Doc4CornerDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# ====== 路径配置 ======
set(NCNN_DIR "D:/BT/ncnn/x64/lib/cmake/ncnn" CACHE PATH "ncnn cmake config path")
set(OpenCV_DIR "D:/BT/opencv/opencv/build/x64/vc16/lib" CACHE PATH "opencv cmake config path")

# 查找依赖库
find_package(ncnn REQUIRED CONFIG PATHS ${NCNN_DIR})
find_package(OpenCV REQUIRED CONFIG PATHS ${OpenCV_DIR})

message(STATUS " OpenCV Version found: ${OpenCV_VERSION}")
message(STATUS " NCNN library found.")

# ====== 编译配置 ======
add_executable(doc_detector main.cpp)

# 包含头文件路径 (确保 OpenCV 的头文件能被找到)
target_include_directories(doc_detector PRIVATE ${OpenCV_INCLUDE_DIRS})

# 链接依赖
target_link_libraries(doc_detector ncnn ${OpenCV_LIBS})