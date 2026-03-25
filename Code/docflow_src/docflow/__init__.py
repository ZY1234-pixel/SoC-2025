"""DocFlow —— 版面恢复工具：将版面分析结果转换为格式化文档。"""
from docflow.pipeline import RecoveryPipeline
from docflow.config import RecoveryConfig

__version__ = "0.3.0"
__all__ = ["RecoveryPipeline", "RecoveryConfig", "__version__"]
