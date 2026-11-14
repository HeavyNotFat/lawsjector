from __future__ import annotations

import ctypes
from pathlib import Path

from PIL import Image

from .paths import RESOURCE_DIR

DLL_NAME = "lawsjector_native.dll"

class NativePrep:
    """封装 lawsjector_native.dll 的缩放与行排序。"""

    def __init__(self, dll_path: Path) -> None:
        """加载 DLL 并声明函数签名。

        :param dll_path: DLL 路径
        :raises OSError: DLL 无法加载时
        """
        self._lib = ctypes.CDLL(str(dll_path))
        self._lib.ls_version.restype = ctypes.c_char_p
        self._lib.ls_version.argtypes = []
        self._lib.ls_resize_rgb.restype = ctypes.c_int
        self._lib.ls_resize_rgb.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.ls_order_lines.restype = ctypes.c_int
        self._lib.ls_order_lines.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]

    @property
    def version(self) -> str:
        """:returns: DLL 版本字符串"""
        return self._lib.ls_version().decode("utf-8", "replace")

    def resize_rgb(self, image: Image.Image, size: tuple[int, int]) -> Image.Image | None:
        """用原生三角滤波缩放 RGB 图像。

        :param image: 源图
        :param size: 目标宽高
        :returns: 缩放后的图像，失败时返回 None
        """
        width, height = image.size
        dst_w, dst_h = size
        src = image.tobytes()
        buf = ctypes.create_string_buffer(dst_w * dst_h * 3)
        rc = self._lib.ls_resize_rgb(src, width, height, buf, dst_w, dst_h)
        if rc != 0:
            return None
        return Image.frombytes("RGB", (dst_w, dst_h), buf.raw)

    def order_lines(self, points: list[tuple[float, float]]) -> list[int]:
        """按阅读顺序返回行索引。

        :param points: 每行的 (x, y)
        :returns: 排序后的索引列表，失败时返回原顺序
        """
        count = len(points)
        if count <= 1:
            return list(range(count))
        xy = (ctypes.c_float * (count * 2))()
        for i, (x, y) in enumerate(points):
            xy[i * 2] = x
            xy[i * 2 + 1] = y
        order = (ctypes.c_int * count)()
        if self._lib.ls_order_lines(xy, count, order) != 0:
            return list(range(count))
        return list(order)

_native: NativePrep | None = None
_loaded = False

def get_native() -> NativePrep | None:
    """获取全局 NativePrep 实例。

    :returns: 实例，DLL 不存在或加载失败时返回 None
    """
    global _native, _loaded
    if _loaded:
        return _native
    _loaded = True
    dll = RESOURCE_DIR / "native" / DLL_NAME
    if not dll.exists():
        return None
    try:
        _native = NativePrep(dll)
    except OSError:
        _native = None
    return _native
