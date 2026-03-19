"""适配器基类接口。

每个版面分析引擎的适配器都须继承 :class:`BaseAdapter`
并实现 :meth:`convert` 方法，返回符合 DocFlow v2.0 JSON 模式的字典。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAdapter(ABC):
    """版面分析引擎适配器的抽象基类。

    子类将引擎特有的分析结果转换为
    恢复管线所使用的 v2.0 标准 JSON 格式。
    """

    @abstractmethod
    def convert(self, results: Any, image: Any, **kwargs) -> dict:
        """将引擎特有结果转换为 v2.0 标准 JSON。

        参数
        ----------
        results:
            版面分析引擎的原始输出。
        image:
            源页面图像（格式因引擎而异）。
        **kwargs:
            引擎特有选项（如 ``img_idx``、``run_sorting``）。

        返回
        -------
        符合 DocFlow v2.0 JSON 模式的字典。
        """
