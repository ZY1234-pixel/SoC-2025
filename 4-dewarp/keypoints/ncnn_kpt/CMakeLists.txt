cmake_minimum_required(VERSION 3.12)
project(KptInfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ====== ncnn 配置 ======
set(NCNN_DIR "D:/BT/ncnn/x64/lib/cmake/ncnn")
find_package(ncnn REQUIRED CONFIG PATHS ${NCNN_DIR})

# ====== OpenCV 配置 ======
set(OpenCV_DIR "D:/BT/opencv/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED CONFIG PATHS ${OpenCV_DIR})

# ====== 可执行文件 ======
add_executable(kpt_infer kpt_infer.cpp)
target_link_libraries(kpt_infer ncnn ${OpenCV_LIBS})