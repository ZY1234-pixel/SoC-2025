from docflow.inference.openvino_session import _cpu_precision


def test_bf16_falls_back_to_fp32_on_unsupported_cpu() -> None:
    assert _cpu_precision("bf16", ["FP32", "INT8"]) == "f32"
    assert _cpu_precision("bf16", ["FP32", "BF16"]) == "bf16"
