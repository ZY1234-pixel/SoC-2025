"""渲染器基类接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from docflow.config import RecoveryConfig


class BaseRenderer(ABC):
    """文档渲染器抽象基类。"""

    def __init__(self, config: Optional["RecoveryConfig"] = None) -> None:
        # 延迟导入以避免模块级循环依赖
        from docflow.config import RecoveryConfig
        self.config: RecoveryConfig = config or RecoveryConfig()

    @abstractmethod
    def render(self, document, output_path: str, **options) -> None:
        """将文档渲染到指定输出路径。"""

    @abstractmethod
    def render_bytes(self, document, **options) -> bytes:
        """渲染为内存中的字节数据。"""
