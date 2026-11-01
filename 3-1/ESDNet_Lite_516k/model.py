"""NCNN模型加载和512像素块预测。

由infer.py调用；本模块不读取图片、不拼图，也不包含训练依赖。
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

class ESDNetLiteNCNN:
    """持有模型与外部分配器，为每块建立独立提取器。

    权重固定使用in0/out0接口；提取器的轻量模式允许及时回收中间特征。
    不启用共享注意力、阶段性特征提取或其他未采用的实验分支。
    """

    def __init__(self, param: Path, binary: Path, threads: int = 8):
        """先设置运行选项和分配器，再加载配套的param与bin。"""
        import ncnn

        self.ncnn = ncnn
        self.net = ncnn.Net()
        self.blob_pool = ncnn.PoolAllocator()
        self.workspace_pool = ncnn.PoolAllocator()
        self._closed = False
        self.context = "tile"
        self.native_calls = 0
        self.preview_calls = 0
        # 保持原有精度路径，防止整理代码时无意启用另一种浮点实现。
        option = self.net.opt
        option.num_threads = threads
        option.use_vulkan_compute = False
        option.use_fp16_storage = False
        option.use_fp16_packed = False
        option.use_fp16_arithmetic = False
        option.use_bf16_storage = False
        option.use_int8_inference = True
        # 比率为0允许复用尺寸更大的空闲块；每块结束仍清池，沿用交付内存策略。
        self.blob_pool.set_size_compare_ratio(0.0)
        self.workspace_pool.set_size_compare_ratio(0.0)
        option.blob_allocator = self.blob_pool
        option.workspace_allocator = self.workspace_pool
        try:
            for path in (param, binary):
                if not path.is_file():
                    raise FileNotFoundError(f"缺少模型文件：{path}")
            if (
                self.net.load_param(str(param)) != 0
                or self.net.load_model(str(binary)) != 0
            ):
                raise RuntimeError("NCNN模型加载失败，请检查param与bin是否配套")
        except Exception:
            self.close()
            raise

    def prepare_image(self, image):
        """开始一张新图的统计；每块自行计算注意力，不额外跑预览网络。"""
        if self._closed:
            raise RuntimeError("模型已关闭，请重新创建实例")
        self.native_calls = 0

    def clear_pools(self):
        """释放不再被张量引用的缓冲；只能在当前块输出复制完成后调用。"""
        self.blob_pool.clear()
        self.workspace_pool.clear()

    def infer_chw(self, source: np.ndarray) -> np.ndarray:
        """接收RGB、float32、3×512×512输入，返回同形状恢复图。

        必须先把NCNN输出复制到NumPy，再销毁提取器或清池，避免悬空内存。
        """
        if self._closed:
            raise RuntimeError("模型已关闭，请重新创建实例")
        if source.shape != (3, 512, 512):
            raise ValueError(f"网络输入必须是3×512×512，实际为{source.shape}")
        extractor = self.net.create_extractor()
        output = None
        try:
            with extractor:
                extractor.set_light_mode(True)
                value = np.ascontiguousarray(source, dtype=np.float32)
                if extractor.input("in0", self.ncnn.Mat(value).clone()) != 0:
                    raise RuntimeError("NCNN输入设置失败")
                code, output = extractor.extract("out0")
                if code != 0:
                    raise RuntimeError("NCNN输出提取失败")
                result = np.array(output, dtype=np.float32)
            self.native_calls += 1
            return result
        finally:
            # 网络输出仍可能持有池内内存，先释放引用，再清理提取器和池。
            output = None
            extractor.clear()
            self.clear_pools()

    def close(self):
        """先清网络，再释放外部分配器，顺序不能倒置；允许重复关闭。"""
        if not self._closed:
            self.net.clear()
            self.clear_pools()
            self._closed = True

