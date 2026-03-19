"""适配器层：将各引擎的专有输出转换为 v2.0 标准格式。

本包提供版面分析引擎与下游恢复管线之间的解耦边界。
每个适配器将特定引擎的原始输出转换为引擎无关的 v2.0 JSON 模式，
供 :class:`~docflow.pipeline.RecoveryPipeline` 使用。

已支持的适配器：
  - :class:`PaddleAdapter` —— PaddleOCR / ppstructure
  - （未来）可通过继承 :class:`BaseAdapter` 添加其他引擎
"""

from docflow.adapters.base_adapter import BaseAdapter

# PaddleAdapter 需要 cv2 和 numpy —— 延迟导入以避免未安装时的硬依赖
try:
    from docflow.adapters.paddle_adapter import PaddleAdapter
except ImportError:
    PaddleAdapter = None  # type: ignore[assignment,misc]

__all__ = ["BaseAdapter", "PaddleAdapter"]
