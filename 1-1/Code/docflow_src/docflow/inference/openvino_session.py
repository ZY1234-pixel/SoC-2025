"""为现有 ONNX 风格调用代码提供最小 OpenVINO 兼容接口。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openvino as ov


def _cpu_precision(requested: str, capabilities) -> str:
    return "bf16" if str(requested).lower() == "bf16" and "BF16" in capabilities else "f32"


class _TensorInfo:
    def __init__(self, port) -> None:
        self.name = port.get_any_name()
        self.shape = [None if dim.is_dynamic else dim.get_length() for dim in port.get_partial_shape()]


class OpenVINOInferSession:
    """兼容项目中 `run()` 和列表输入两种既有推理调用。"""

    def __init__(self, config, inference_precision: str = "f32") -> None:
        if isinstance(config, dict):
            model_path = config.get("model_path")
            inference_precision = config.get("inference_precision", inference_precision)
        else:
            model_path = config
        self.model_path = Path(model_path or "")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"OpenVINO model not found: {self.model_path}")

        core = ov.Core()
        self.inference_precision = _cpu_precision(
            inference_precision,
            core.get_property("CPU", "OPTIMIZATION_CAPABILITIES"),
        )
        precision = ov.Type.bf16 if self.inference_precision == "bf16" else ov.Type.f32
        self.model = core.read_model(self.model_path)
        self.compiled = core.compile_model(
            self.model,
            "CPU",
            {
                "PERFORMANCE_HINT": "LATENCY",
                ov.properties.hint.inference_precision: precision,
            },
        )
        self.session = self
        self._inputs = [_TensorInfo(port) for port in self.model.inputs]
        self._outputs = [_TensorInfo(port) for port in self.model.outputs]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_input_names(self):
        return [item.name for item in self._inputs]

    def get_output_names(self):
        return [item.name for item in self._outputs]

    def get_output_name(self, output_idx=0):
        return self._outputs[output_idx].name

    def run(self, output_names, input_feed):
        normalized = {}
        input_ranks = {
            item.name: len(self.model.input(item.name).get_partial_shape())
            for item in self._inputs
        }
        for name, value in input_feed.items():
            value = np.asarray(value)
            if value.ndim == input_ranks[name] + 1 and value.shape[0] == 1:
                value = value[0]
            normalized[name] = value
        result = self.compiled(normalized)
        names = output_names or self.get_output_names()
        return [np.asarray(result[self.compiled.output(name)]) for name in names]

    def __call__(self, input_content):
        if isinstance(input_content, dict):
            input_feed = input_content
        else:
            values = input_content if isinstance(input_content, (list, tuple)) else [input_content]
            input_feed = dict(zip(self.get_input_names(), values))
        outputs = self.run(None, input_feed)

        # LORE 原调用按六个位置解包，但紧凑模型已删除完全未使用的 st 分支。
        if self.get_output_names() == ["hm", "wh", "ax", "cr", "reg"]:
            outputs.insert(1, np.empty((0,), dtype=np.float32))
        return outputs

    def have_key(self, key="character"):
        return False

    def get_character_list(self, key="character"):
        return []

    def get_metadata(self, key=None):
        return {} if key is None else []

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map={})
