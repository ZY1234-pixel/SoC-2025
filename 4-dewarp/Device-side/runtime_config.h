#pragma once

namespace RuntimeConfig {

// ==================== 运行开关集中配置 ====================
// 当前默认：fp32 推理，关闭 packing 和内存池，保留 SGEMM/Winograd 加速。

// 输出相关
static constexpr bool kSaveVisualization = false;          // false: 只保存 0-255 mask；true: 保存可视化图，测试时使用。
static constexpr bool kEnableCornerLostProcess = true;    // false: 不做缺角补绘；true: 开启缺角补绘。
static constexpr bool kCheckCornerFillReasonable = false; // false: 不过滤补绘结果，直接输出扩边后的新 mask。
static constexpr bool kSaveFilledRGB = true;              // true: 保存补绘后的 RGB 图。
static constexpr float kBookThreshold = 0.65f;             // 书本区域阈值，越大 mask 越保守。

// 输入尺寸
static constexpr int kInputWidth = 640;                    // 模型输入宽度，需和 pnnx 导出一致。
static constexpr int kInputHeight = 640;                   // 模型输入高度，需和 pnnx 导出一致。

// CPU 线程
static constexpr int kNumThreads = 8;                      // ncnn 推理线程数。
static constexpr int kOpenMPBlockTime = 0;                 // 0: 线程做完尽快休眠。

// ncnn 内存相关
static constexpr bool kLightMode = false;                  // false: 中间结果保留更保守，双输出模型更稳。
static constexpr bool kUseLocalPoolAllocator = false;      // 必须 false：打开后出现空输出/不稳定。

// ncnn 数据布局
static constexpr bool kUsePackingLayout = false;           // 必须 false：打开后 mask 有条带/半张缺失。

// 低精度相关
static constexpr bool kUseInt8Inference = false;           // false: 模型不是 int8 量化模型。
static constexpr bool kUseInt8Packed = false;              // false: 关闭 int8 packed。
static constexpr bool kUseInt8Storage = false;             // false: 关闭 int8 存储。
static constexpr bool kUseInt8Arithmetic = false;          // false: 关闭 int8 计算。
static constexpr bool kUseBF16Storage = false;             // false: 保持 fp32 精度。
static constexpr bool kUseBF16Packed = false;              // false: 关闭 bf16 packed。
static constexpr bool kUseFP16Packed = false;              // false: 关闭 fp16 packed。
static constexpr bool kUseFP16Storage = false;             // false: 当前使用 fp32 权重。
static constexpr bool kUseFP16Arithmetic = false;          // false: 关闭 fp16 计算，关键点更稳。

// GPU
static constexpr bool kUseVulkanCompute = false;           // false: 只用 CPU，避免 GPU 驱动差异。

} // namespace RuntimeConfig
