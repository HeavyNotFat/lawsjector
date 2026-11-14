from __future__ import annotations

import ctypes
import os
from pathlib import Path

from PIL import Image

from .paths import NATIVE_DIR

_DLL_NAME = "lawsjector_native.dll"
_ORT_NAME = "onnxruntime.dll"
_ort_loaded: object | None = None

def _load_library() -> ctypes.CDLL:
    """加载 DLL，并先把 ONNX Runtime 放进搜索路径。

    :returns: 已加载的 ctypes 库
    :raises OSError: DLL 或依赖加载失败
    """
    global _ort_loaded
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(NATIVE_DIR))
    ort = NATIVE_DIR / _ORT_NAME
    if _ort_loaded is None and ort.exists():
        _ort_loaded = ctypes.WinDLL(str(ort))  # 保持引用，避免被卸载
    return ctypes.CDLL(str(NATIVE_DIR / _DLL_NAME))

class NativeOCR:
    """原生 OCR 引擎：检测 + 识别全部在 C++ 里完成。"""

    def __init__(self, det_model: Path, rec_model: Path, keys: Path, threads: int) -> None:
        """创建引擎并加载两个 ONNX 模型。

        :param det_model: 检测模型路径
        :param rec_model: 识别模型路径
        :param keys: 字符字典路径
        :param threads: ONNX Runtime 线程数
        :raises OSError: 动态库加载失败
        :raises RuntimeError: 模型初始化失败
        """
        lib = _load_library()
        lib.lsocr_create.restype = ctypes.c_void_p
        lib.lsocr_create.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        lib.lsocr_run.restype = ctypes.c_void_p
        lib.lsocr_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.lsocr_free.restype = None
        lib.lsocr_free.argtypes = [ctypes.c_void_p]
        lib.lsocr_destroy.restype = None
        lib.lsocr_destroy.argtypes = [ctypes.c_void_p]

        self._lib = lib
        keys_text = keys.read_text(encoding="utf-8").encode("utf-8")
        self._handle = lib.lsocr_create(
            str(det_model), str(rec_model), keys_text, int(threads)
        )
        if not self._handle:
            raise RuntimeError("原生 OCR 引擎初始化失败")

    def run(self, image: Image.Image) -> list[tuple[str, float, float, float]] | None:
        """识别一张图片。

        :param image: PIL 图像
        :returns: (文字, 置信度, x, y) 列表，失败时返回 None
        """
        rgb = image.convert("RGB").tobytes()
        length = ctypes.c_int(0)
        ptr = self._lib.lsocr_run(
            self._handle, rgb, image.width, image.height, ctypes.byref(length)
        )
        if not ptr:
            return None
        try:
            data = ctypes.string_at(ptr, length.value)
        finally:
            self._lib.lsocr_free(ptr)
        lines: list[tuple[str, float, float, float]] = []
        for raw in data.decode("utf-8", "replace").splitlines():
            parts = raw.split("\t", 3)
            if len(parts) != 4:
                continue
            x, y, score, text = parts
            lines.append((text, float(score), float(x), float(y)))
        return lines

    def close(self) -> None:
        """释放引擎。"""
        if self._handle:
            self._lib.lsocr_destroy(self._handle)
            self._handle = None
