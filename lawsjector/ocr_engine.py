from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

from .bank import polish_text, smart_join, strip_watermark
from .native import get_native
from .paths import MODELS_DIR, NATIVE_DIR

@dataclass(slots=True)
class OCRLine:
    text: str
    confidence: float = 1.0
    x: float = 0.0
    y: float = 0.0

@dataclass(slots=True)
class OCRResult:
    lines: tuple[OCRLine, ...]
    engine: str
    elapsed_ms: float

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text)

_lock = threading.Lock()
_rapid_engine = None
_rapid_error: BaseException | None = None
_native_engine = None
_native_error: BaseException | None = None
_warmed_up = False

_NOISE_RE = re.compile(
    r"^(?:"
    r"关闭(?:\s*Esc)?|"
    r"题号\s*Q?\d+.*|"
    r"本地题库.*|"
    r"水印.*|"
    r"使用说明.*|"
    r"法治题库.*|"
    r"智能查询.*|"
    r"粘贴或输入.*|"
    r"搜索结果.*|"
    r"题目详情.*|"
    r"复制本题.*|"
    r"查询答案.*|"
    r"全部题型.*|"
    r"匹配|题型|题目|选项|答案|解析|搜索|结果|"
    r"\d+\s*道题.*|"
    r"(?:单选题|多选题|判断题)\s*[·:：].*|"
    r"(?:单选题|多选题|判断题)\s*[·:：]?|"
    r"答案\s*[·:：].*|"
    r"解析\s*[·:：].*|"
    r"\d{1,3}\s*[%％].*"
    r")$"
)
_OPTION_RE = re.compile(r"^[A-Ea-e]\s*[.．、]\s*")
_WATERMARK_HINT_RE = re.compile(
    r"2026\s*年度.*?复习题|法治理论知识考试复习题|2026\s*年.*?考试.*?复习"
)

def _prepare(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    if w <= 0 or h <= 0:
        return image
    scale = 1.0
    if w < 900:
        scale = min(1.6, 900 / w)
    elif w > 1900:
        scale = 1900 / w
    if h * scale > 2200:
        scale = 2200 / h
    if abs(scale - 1.0) > 0.02:
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        native = get_native()
        resized = native.resize_rgb(image, size) if native is not None else None
        image = resized if resized is not None else image.resize(size, Image.LANCZOS)
    return image

def _model_dir() -> Path:
    """返回 ONNX 模型目录，环境变量优先。

    :returns: 模型所在目录
    """
    override = os.environ.get("LAWSJECTOR_OCR_ONNX_DIR", "").strip()
    return Path(override) if override else MODELS_DIR

def _custom_models() -> dict[str, str]:
    """读取 det/rec/字典三个文件，齐全时返回 RapidOCR 参数。

    :returns: RapidOCR 的模型参数，缺文件时返回空字典
    """
    base = _model_dir()
    det, rec, keys = base / "det.onnx", base / "rec.onnx", base / "ppocr_keys_v1.txt"
    if det.exists() and rec.exists() and keys.exists():
        return {
            "det_model_path": str(det),
            "rec_model_path": str(rec),
            "rec_keys_path": str(keys),
        }
    return {}

def has_native() -> bool:
    """检查原生引擎所需的模型与 DLL 是否齐全。

    :returns: 齐全返回 True
    """
    base = _model_dir()
    dll = NATIVE_DIR / "lawsjector_native.dll"
    return (
        dll.exists()
        and (base / "det.onnx").exists()
        and (base / "rec.onnx").exists()
        and (base / "ppocr_keys_v1.txt").exists()
    )

def _get_native_engine():
    global _native_engine, _native_error
    if _native_engine is not None:
        return _native_engine
    if _native_error is not None:
        raise _native_error
    try:
        from .native_ocr import NativeOCR

        base = _model_dir()
        det = base / "det.onnx"
        rec = base / "rec.onnx"
        keys = base / "ppocr_keys_v1.txt"
        if not (det.exists() and rec.exists() and keys.exists()):
            raise FileNotFoundError(f"{base} 下缺少 det.onnx / rec.onnx / ppocr_keys_v1.txt")
        threads = min(8, max(1, os.cpu_count() or 4))
        _native_engine = NativeOCR(det, rec, keys, threads)
        return _native_engine
    except Exception as exc:
        _native_error = exc
        raise

def _native_ocr(image: Image.Image) -> OCRResult:
    engine = _get_native_engine()
    items = engine.run(_prepare(image))
    if items is None:
        raise RuntimeError("原生 OCR 引擎返回失败")
    lines = tuple(OCRLine(text=text, confidence=score, x=x, y=y) for text, score, x, y in items)
    label = "ONNX CPU · C++"
    if _model_dir() != MODELS_DIR:
        label += " · 自训练"
    return OCRResult(lines=lines, engine=label, elapsed_ms=0.0)

def _get_rapid_engine():
    global _rapid_engine, _rapid_error
    if _rapid_engine is not None:
        return _rapid_engine
    if _rapid_error is not None:
        raise _rapid_error
    try:
        # 动态导入，打包时不必带 rapidocr_onnxruntime（体积大），只在源码环境下兜底
        rapidocr = importlib.import_module("rapidocr_onnxruntime")

        _rapid_engine = rapidocr.RapidOCR(
            det_limit_side_len=960,
            det_limit_type="max",
            **_custom_models(),
        )
        return _rapid_engine
    except Exception as exc:
        _rapid_error = exc
        raise

def _rapid_ocr(image: Image.Image) -> OCRResult:
    np = importlib.import_module("numpy")

    engine = _get_rapid_engine()
    prepared = _prepare(image)
    array = np.asarray(prepared)[:, :, ::-1].copy()
    result, _ = engine(array)
    lines: list[OCRLine] = []
    if result:
        for item in result:
            try:
                box, text, confidence = item[0], str(item[1]).strip(), float(item[2])
            except (TypeError, ValueError, IndexError):
                continue
            if not text:
                continue
            try:
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                x, y = min(xs), min(ys)
            except Exception:
                x = y = 0.0
            lines.append(OCRLine(text=text, confidence=confidence, x=x, y=y))

    native = get_native()
    if native is not None and len(lines) > 1:
        order = native.order_lines([(line.x, line.y) for line in lines])
        lines = [lines[i] for i in order]
    else:
        lines.sort(key=lambda line: (round(line.y / 10), line.x))

    label = "RapidOCR · ONNX CPU"
    if _custom_models():
        label += " · 自训练"
    if native is not None:
        label += " · native"
    return OCRResult(lines=tuple(lines), engine=label, elapsed_ms=0.0)

async def _windows_ocr_async(path: str) -> list[str]:
    from winsdk.windows.globalization import Language
    from winsdk.windows.graphics.imaging import BitmapDecoder
    from winsdk.windows.media.ocr import OcrEngine
    from winsdk.windows.storage import FileAccessMode, StorageFile

    file = await StorageFile.get_file_from_path_async(path)
    stream = await file.open_async(FileAccessMode.READ)
    decoder = await BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()

    tags = []
    try:
        tags = [str(lang.language_tag) for lang in OcrEngine.available_recognizer_languages]
    except Exception:
        pass
    engine = None
    for tag in ("zh-Hans-CN", "zh-CN", "en-US"):
        try:
            engine = OcrEngine.try_create_from_language(Language(tag))
        except Exception:
            engine = None
        if engine is not None:
            break
    if engine is None:
        engine = OcrEngine.try_create_from_user_profile_languages()
    if engine is None:
        raise RuntimeError(f"系统没有可用的 OCR 语言包（已安装：{tags}）")

    result = await engine.recognize_async(bitmap)
    return [line.text for line in result.lines]

def _windows_ocr(image: Image.Image) -> OCRResult:
    prepared = _prepare(image)
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            temp_path = fh.name
        prepared.save(temp_path, format="PNG")
        texts = asyncio.run(_windows_ocr_async(temp_path))
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
    lines = tuple(
        OCRLine(text=text.strip(), confidence=1.0, x=0.0, y=float(i * 10))
        for i, text in enumerate(texts)
        if text.strip()
    )
    return OCRResult(lines=lines, engine="Windows OCR · CPU", elapsed_ms=0.0)

def warmup() -> None:
    """在后台预热 OCR，避免第一次点击时承担模型初始化开销。
    """
    global _warmed_up
    if _warmed_up:
        return
    try:
        from PIL import ImageDraw, ImageFont

        image = Image.new("RGB", (640, 120), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 26)
        except Exception:
            font = ImageFont.load_default()
        draw.text((20, 35), "法治题库 RapidOCR 123", font=font, fill="black")
        with _lock:
            if _warmed_up:
                return
            _native_ocr(image)
            _warmed_up = True
    except Exception:
        pass

def recognize(image: Image.Image) -> OCRResult:
    """识别 PIL 图片，第一次调用会加载模型。

    :param image: PIL 图像
    :returns: 识别结果
    :raises ValueError: 截图区域小于 2 像素
    """
    if image.width < 2 or image.height < 2:
        raise ValueError("截图区域太小")

    started = time.perf_counter()
    errors: list[BaseException] = []
    win_available = importlib.util.find_spec("winsdk") is not None

    with _lock:
        try:
            result = _native_ocr(image)
            result.elapsed_ms = (time.perf_counter() - started) * 1000
            return result
        except Exception as exc:
            errors.append(exc)

        try:
            result = _rapid_ocr(image)
            result.elapsed_ms = (time.perf_counter() - started) * 1000
            return result
        except Exception as exc:
            errors.append(exc)

        if win_available:
            try:
                result = _windows_ocr(image)
                result.elapsed_ms = (time.perf_counter() - started) * 1000
                return result
            except Exception as exc:
                errors.append(exc)

    detail = str(errors[-1]) if errors else "未知错误"
    raise RuntimeError(f"OCR 识别失败：{detail}")

def _clean_line(text: str) -> str:
    text = _WATERMARK_HINT_RE.sub(" ", text)
    text = polish_text(text, blank=" ")
    if not text:
        return ""
    if "关键词匹配" in text and ("%" in text or "％" in text):
        return ""
    if re.fullmatch(r"[·:：\-\s]+", text):
        return ""
    return text

def build_queries(result: OCRResult) -> tuple[str, str]:
    """从 OCR 结果生成题干查询与全文查询。

    :param result: OCR 结果
    :returns: (题干查询, 全文查询)
    """
    lines: list[str] = []
    for line in result.lines:
        if line.confidence < 0.32:
            continue
        cleaned = _clean_line(line.text)
        if not cleaned:
            continue
        if cleaned in ("选项", "答案", "解析"):
            break
        if _OPTION_RE.match(cleaned):
            break
        if _NOISE_RE.match(cleaned):
            continue
        lines.append(cleaned)

    stem = ""
    for line in lines:
        stem = smart_join(stem, line)
    stem = strip_watermark(stem).strip()
    stem = re.sub(r"^\s*(?:题目|题干)\s*[：:]\s*", "", stem)
    stem = re.sub(r"(?<=[，。；：！？、])\s+(?=[\u3400-\u9fff])", "", stem)

    all_lines: list[str] = []
    for line in result.lines:
        if line.confidence < 0.32:
            continue
        cleaned = _clean_line(line.text)
        if not cleaned:
            continue
        if cleaned in ("答案", "解析"):
            break
        if _NOISE_RE.match(cleaned):
            continue
        all_lines.append(cleaned)
    full = ""
    for line in all_lines:
        full = smart_join(full, line)
    full = strip_watermark(full).strip()
    full = re.sub(r"(?<=[，。；：！？、])\s+(?=[\u3400-\u9fff])", "", full)

    if len(re.sub(r"\W", "", stem)) < 5:
        stem = full
    return stem, full
