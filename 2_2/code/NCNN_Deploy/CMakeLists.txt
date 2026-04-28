cmake_minimum_required(VERSION 3.10)
project(NAFNetDeploy LANGUAGES CXX)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加源文件
add_executable(NAFNetDeploy NAFNetDeploy.cpp)

# -------------------- ncnn 配置 --------------------
set(NCNN_DIR "D:/BT/ncnn/x64" CACHE PATH "Path to ncnn")
target_include_directories(NAFNetDeploy PRIVATE ${NCNN_DIR}/include)
target_link_directories(NAFNetDeploy PRIVATE ${NCNN_DIR}/lib)

# -------------------- OpenCV 配置 --------------------
set(OpenCV_DIR "D:/BT/opencv/opencv/build" CACHE PATH "Path to OpenCV")
target_include_directories(NAFNetDeploy PRIVATE ${OpenCV_DIR}/include)
target_link_directories(NAFNetDeploy PRIVATE ${OpenCV_DIR}/x64/vc16/lib)

# -------------------- 链接库 --------------------
target_link_libraries(NAFNetDeploy
    ncnn
    opencv_world4130
)

# -------------------- 预处理器定义 --------------------
target_compile_definitions(NAFNetDeploy PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
    _CONSOLE
)

# -------------------- 运行时库设置（Windows） --------------------
if(MSVC)
    set_property(TARGET NAFNetDeploy PROPERTY
        MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
    )
endif()