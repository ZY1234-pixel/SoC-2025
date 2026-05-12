cmake_minimum_required(VERSION 3.12)
project(YOLOv8ClsInfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ====== 路径配置（请按实际路径修改） ======
set(NCNN_DIR "D:/BT/ncnn/x64/lib/cmake/ncnn")
set(OpenCV_DIR "D:/BT/opencv/opencv/build/x64/vc16/lib")

find_package(ncnn REQUIRED CONFIG PATHS ${NCNN_DIR})
find_package(OpenCV REQUIRED CONFIG PATHS ${OpenCV_DIR})

add_executable(cls_infer cls_infer.cpp)
target_link_libraries(cls_infer ncnn ${OpenCV_LIBS})