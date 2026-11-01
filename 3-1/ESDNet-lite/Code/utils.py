"""Shared utilities for training and inference."""

from __future__ import annotations

import ctypes
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

DEFAULT_TILE = 512
DEFAULT_OVERLAP = 128
DEFAULT_NCNN_THREADS = 4
MAX_SAFE_TILE = 1024

_BLEND_CACHE_TORCH = {}
_BLEND_CACHE_NP = {}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint_state_dict(model_path: str):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        return checkpoint
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    for key in ('params_ema', 'params'):
        if key in checkpoint:
            return checkpoint[key]
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise ValueError(f'Unrecognized checkpoint format: {list(checkpoint.keys())}')


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_tensorrt_engine_candidates() -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    project_root = repo_root.parent
    candidates: list[Path] = []
    preferred = [
        repo_root / 'Code' / 'weights' / 'uhdm_lite_s_best_fp16.plan',
        repo_root / 'Code' / 'weights' / 'esdnet_lite_fp16.plan',
        project_root / 'NAFNet' / 'output_model' / 'esdnet_lite_fp16.plan',
        project_root / 'NAFNet' / 'output_model' / 'uhdm_lite_s_best_fp16.plan',
    ]
    search_roots = [
        repo_root / 'Code' / 'weights',
        project_root / 'NAFNet' / 'output_model',
    ]
    for path in preferred:
        if path.is_file() and path not in candidates:
            candidates.append(path)
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob('*.plan')):
            if path.is_file() and path not in candidates:
                candidates.append(path)
    return candidates


def resolve_tensorrt_engine_path(engine_path: str | None) -> Path:
    if engine_path:
        path = Path(engine_path)
        if path.is_file():
            return path
        candidates = find_tensorrt_engine_candidates()
        if candidates:
            print(f'[WARN] 指定 TensorRT engine 不存在: {path}')
            print(f'[WARN] 自动回退到候选 engine: {candidates[0]}')
            return candidates[0]
        raise FileNotFoundError(f'TensorRT engine 不存在: {path}')

    candidates = find_tensorrt_engine_candidates()
    if candidates:
        print(f'[i] 未显式指定 --trt_engine，自动使用: {candidates[0]}')
        return candidates[0]
    raise FileNotFoundError(
        '未找到可用 TensorRT engine。请传入 --trt_engine，'
        '或先在 Code/weights/ / NAFNet/output_model/ 下准备 .plan 文件。'
    )


def configure_tensorrt_runtime_library_path() -> None:
    py_ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
    site_packages = Path(sys.prefix) / 'lib' / py_ver / 'site-packages'
    lib_dirs: list[str] = []
    candidates = [
        site_packages / 'tensorrt_libs',
        site_packages / 'nvidia' / 'cudnn' / 'lib',
        site_packages / 'nvidia' / 'cublas' / 'lib',
        site_packages / 'nvidia' / 'cuda_runtime' / 'lib',
        Path('/tmp/trt86_runtime') / 'nvidia' / 'cudnn' / 'lib',
    ]
    extra_dirs = os.environ.get('TRT_EXTRA_LIB_DIRS', '')
    for item in extra_dirs.split(':'):
        item = item.strip()
        if item:
            candidates.append(Path(item))

    for path in candidates:
        if path.is_dir():
            lib_dirs.append(str(path))

    old_ld = os.environ.get('LD_LIBRARY_PATH', '')
    merged = lib_dirs + ([old_ld] if old_ld else [])
    if merged:
        os.environ['LD_LIBRARY_PATH'] = ':'.join(merged)

    preload_names = [
        'libcudnn.so.8',
        'libcublas.so.12',
        'libcublasLt.so.12',
        'libnvinfer.so.8',
        'libnvinfer_plugin.so.8',
        'libnvonnxparser.so.8',
    ]
    for lib_dir in lib_dirs:
        for name in preload_names:
            lib_path = Path(lib_dir) / name
            if lib_path.exists():
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def import_tensorrt_bindings():
    configure_tensorrt_runtime_library_path()
    try:
        import tensorrt as trt
    except ImportError:
        try:
            import tensorrt_bindings as trt
        except ImportError as exc:
            raise RuntimeError(
                '未安装可用的 TensorRT Python 绑定，请先安装 TensorRT 8.6 GA 对应 bindings。'
            ) from exc
    return trt


def trt_dtype_to_torch(trt, dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f'暂不支持的 TensorRT dtype: {dtype}')
    return mapping[dtype]



def _make_blend_vector_torch(size: int, overlap: int) -> torch.Tensor:
    vector = torch.ones(size, dtype=torch.float32)
    if overlap <= 0:
        return vector
    coord = torch.arange(overlap, dtype=torch.float32)
    ramp = 0.5 - 0.5 * torch.cos(torch.pi * (coord + 1.0) / (overlap + 1.0))
    vector[:overlap] = ramp
    vector[-overlap:] = torch.flip(ramp, dims=[0])
    return vector


def get_blend_window_torch(size: int, overlap: int) -> torch.Tensor:
    key = (size, overlap)
    if key not in _BLEND_CACHE_TORCH:
        vector = _make_blend_vector_torch(size, overlap)
        window = vector.unsqueeze(1) * vector.unsqueeze(0)
        _BLEND_CACHE_TORCH[key] = window.unsqueeze(0).unsqueeze(0)
    return _BLEND_CACHE_TORCH[key]


def _make_blend_vector_np(size: int, overlap: int) -> np.ndarray:
    vector = np.ones(size, dtype=np.float32)
    if overlap <= 0:
        return vector
    coord = np.arange(overlap, dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.pi * (coord + 1.0) / (overlap + 1.0))
    vector[:overlap] = ramp
    vector[-overlap:] = ramp[::-1]
    return vector


def get_blend_window_np(size: int, overlap: int) -> np.ndarray:
    key = (size, overlap)
    if key not in _BLEND_CACHE_NP:
        vector = _make_blend_vector_np(size, overlap)
        _BLEND_CACHE_NP[key] = np.outer(vector, vector).astype(np.float32)
    return _BLEND_CACHE_NP[key]


def pad_tile_tensor(tile: torch.Tensor, tile_size: int) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = tile.shape
    if height == tile_size and width == tile_size:
        return tile, height, width
    pad_right = tile_size - width
    pad_bottom = tile_size - height
    pad_mode = 'reflect' if (width > 1 and height > 1 and pad_right < width and pad_bottom < height) else 'replicate'
    padded = torch.nn.functional.pad(tile, (0, pad_right, 0, pad_bottom), mode=pad_mode)
    return padded, height, width


def pad_tile_array(tile: np.ndarray, tile_size: int) -> tuple[np.ndarray, int, int]:
    _, height, width = tile.shape
    if height == tile_size and width == tile_size:
        return np.ascontiguousarray(tile, dtype=np.float32), height, width
    pad_right = tile_size - width
    pad_bottom = tile_size - height
    pad_mode = 'reflect' if (width > 1 and height > 1 and pad_right < width and pad_bottom < height) else 'edge'
    padded = np.pad(tile, ((0, 0), (0, pad_bottom), (0, pad_right)), mode=pad_mode)
    return np.ascontiguousarray(padded, dtype=np.float32), height, width


class NcnnModel:
    """Thin wrapper around the Python NCNN binding."""

    def __init__(self, param_path: str, bin_path: str, num_threads: int = DEFAULT_NCNN_THREADS, use_vulkan: bool = False):
        import ncnn

        self._ncnn = ncnn
        self.padder_size = 8
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = use_vulkan
        if not use_vulkan and num_threads > 0:
            self.net.opt.num_threads = num_threads
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

    def infer_chw(self, image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(image, dtype=np.float32)
        with self.net.create_extractor() as extractor:
            extractor.input('in0', self._ncnn.Mat(image).clone())
            _, output = extractor.extract('out0')
        return np.array(output, dtype=np.float32)


class TensorRTModel:
    """Thin wrapper around a serialized TensorRT engine."""

    def __init__(self, engine_path: str, device: str = 'cuda:0'):
        if not torch.cuda.is_available():
            raise RuntimeError('TensorRT 推理需要可用的 CUDA 设备')

        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError(f'TensorRT backend 仅支持 CUDA 设备，当前为: {device}')
        if self.device.index is None:
            self.device = torch.device(f'cuda:{torch.cuda.current_device()}')

        self.engine_path = str(engine_path)
        self.trt = import_tensorrt_bindings()
        self.logger = self.trt.Logger(self.trt.Logger.ERROR)
        self.runtime = self.trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f'反序列化 TensorRT engine 失败: {engine_path}')

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f'创建 TensorRT execution context 失败: {engine_path}')

        self.input_name = None
        self.output_name = None
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == self.trt.TensorIOMode.OUTPUT:
                self.output_name = name

        if self.input_name is None or self.output_name is None:
            raise RuntimeError('TensorRT engine 缺少输入或输出 tensor')

        self.input_shape = tuple(int(v) for v in self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(int(v) for v in self.engine.get_tensor_shape(self.output_name))
        if len(self.input_shape) != 4 or self.input_shape[1] != 3:
            raise ValueError(f'当前仅支持 NCHW 三通道输入，实际 shape: {self.input_shape}')

        if self.input_shape[2] <= 0 or self.input_shape[3] <= 0:
            raise ValueError(f'当前推理入口只支持固定 shape engine，实际输入 shape: {self.input_shape}')
        if self.input_shape[2] != self.input_shape[3]:
            raise ValueError(f'当前推理入口只支持方形 tile engine，实际输入 shape: {self.input_shape}')

        self.tile_size = int(self.input_shape[2])
        self.padder_size = 8
        self.input_torch_dtype = trt_dtype_to_torch(self.trt, self.engine.get_tensor_dtype(self.input_name))
        self.output_torch_dtype = trt_dtype_to_torch(self.trt, self.engine.get_tensor_dtype(self.output_name))

    def infer_chw(self, image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(image, dtype=np.float32)
        expected_shape = (3, self.tile_size, self.tile_size)
        if tuple(image.shape) != expected_shape:
            raise ValueError(
                f'TensorRT engine 固定输入为 {expected_shape}，实际收到 {tuple(image.shape)}'
            )

        input_shape = (1, *expected_shape)
        with torch.cuda.device(self.device):
            input_tensor = torch.from_numpy(image).unsqueeze(0).to(
                device=self.device,
                dtype=self.input_torch_dtype,
            ).contiguous()
            if hasattr(self.context, 'set_input_shape'):
                self.context.set_input_shape(self.input_name, input_shape)
            output_shape = tuple(int(v) for v in self.context.get_tensor_shape(self.output_name))
            output_tensor = torch.empty(output_shape, device=self.device, dtype=self.output_torch_dtype)

            self.context.set_tensor_address(self.input_name, int(input_tensor.data_ptr()))
            self.context.set_tensor_address(self.output_name, int(output_tensor.data_ptr()))
            stream = torch.cuda.current_stream(device=self.device)
            ok = self.context.execute_async_v3(stream.cuda_stream)
            if not ok:
                raise RuntimeError(f'TensorRT 执行失败: {self.engine_path}')
            stream.synchronize()
            return output_tensor.squeeze(0).detach().float().cpu().numpy()
