from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from shapely.geometry import Polygon

from lawsjector.native_ocr import NativeOCR

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "resources" / "models"


def edit_distance(a: str, b: str) -> int:
    """计算 Levenshtein 距离。

    :param a: 字符串 a
    :param b: 字符串 b
    :returns: 编辑距离
    """
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def iou(p, q) -> float:
    """计算两个多边形的 IoU。

    :param p: 多边形 p
    :param q: 多边形 q
    :returns: IoU
    """
    try:
        a = Polygon(p).buffer(0)
        b = Polygon(q).buffer(0)
        if a.is_empty or b.is_empty:
            return 0.0
        return float(a.intersection(b).area / max(1e-9, a.union(b).area))
    except Exception:
        return 0.0


def score_page(results: list[tuple[str, float, float, float]], gt: list[tuple[str, list]], max_dist: float = 80.0):
    """按左上角距离匹配行，统计一页的字符错误。

    :param results: 引擎输出 (文字, 置信度, x, y)
    :param gt: 真值 (文字, 四点)
    :param max_dist: 匹配的最大曼哈顿距离
    :returns: (错误数, 总字数)
    """
    pairs = []
    for i, (_, points) in enumerate(gt):
        gx = min(p[0] for p in points)
        gy = min(p[1] for p in points)
        for j, item in enumerate(results):
            pairs.append((abs(gx - item[2]) + abs(gy - item[3]), i, j))
    pairs.sort()

    used_gt = set()
    used_res = set()
    err = 0
    total = 0
    for dist, i, j in pairs:
        if dist > max_dist:
            break
        if i in used_gt or j in used_res:
            continue
        used_gt.add(i)
        used_res.add(j)
        text = gt[i][0]
        err += edit_distance(results[j][0], text)
        total += len(text)
    for i, (text, _) in enumerate(gt):
        if i not in used_gt:
            err += len(text)
            total += len(text)
    return err, total

def run_rapidocr(engine: RapidOCR, rows, limit: int) -> tuple[float, float]:
    """跑 RapidOCR 引擎并统计页面 CER 与耗时。

    :param engine: RapidOCR 实例
    :param rows: 页面与真值
    :param limit: 页数
    :returns: (CER, 每页毫秒)
    """
    err = total = 0
    t0 = time.perf_counter()
    for path, label in rows[:limit]:
        img = cv2.imread(str(path))
        result, _ = engine(img)
        lines = []
        for item in result or []:
            box = np.array(item[0], dtype=np.float32).reshape(-1, 2)
            lines.append((str(item[1]), float(item[2]), float(box[:, 0].min()), float(box[:, 1].min())))
        gt = [(item["transcription"], item["points"]) for item in json.loads(label)]
        e, n = score_page(lines, gt)
        err += e
        total += n
    elapsed = time.perf_counter() - t0
    return err / max(1, total), elapsed / max(1, limit) * 1000


def run_native(engine: NativeOCR, rows, limit: int) -> tuple[float, float]:
    """跑原生引擎并统计页面 CER 与耗时。

    :param engine: NativeOCR 实例
    :param rows: 页面与真值
    :param limit: 页数
    :returns: (CER, 每页毫秒)
    """
    err = total = 0
    t0 = time.perf_counter()
    for path, label in rows[:limit]:
        img = Image.open(path).convert("RGB")
        lines = engine.run(img) or []
        gt = [(item["transcription"], item["points"]) for item in json.loads(label)]
        e, n = score_page(lines, gt)
        err += e
        total += n
    elapsed = time.perf_counter() - t0
    return err / max(1, total), elapsed / max(1, limit) * 1000


def main() -> int:
    """对比官方、Python 自训练、原生引擎的 CER 与耗时。

    :returns: 进程退出码
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--models", default=str(MODELS))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    val = ROOT / "rapidocr_train" / "data" / "det" / "val.txt"
    rows = []
    for line in val.read_text(encoding="utf-8").splitlines():
        if line.strip():
            name, _, label = line.partition("\t")
            rows.append((val.parent / name, label))

    engines = [
        ("官方 PP-OCRv4（Python）", lambda: RapidOCR(det_limit_side_len=960, det_limit_type="max")),
        (
            "自训练（Python 路径）",
            lambda: RapidOCR(
                det_limit_side_len=960,
                det_limit_type="max",
                det_model_path=str(MODELS / "det.onnx"),
                rec_model_path=str(MODELS / "rec.onnx"),
                rec_keys_path=str(MODELS / "ppocr_keys_v1.txt"),
            ),
        ),
    ]
    print(f"验证集 {len(rows[:args.limit])} 页\n")
    print("| 引擎 | 页面 CER | ms/页 |")
    print("| --- | --- | --- |")
    for name, factory in engines:
        cer, ms = run_rapidocr(factory(), rows, args.limit)
        print(f"| {name} | {cer:.4f} | {ms:.0f} |")

    models = Path(args.models)
    native = NativeOCR(models / "det.onnx", models / "rec.onnx", models / "ppocr_keys_v1.txt", args.threads)
    cer, ms = run_native(native, rows, args.limit)
    print(f"| 全原生 C++（{args.threads} 线程） | {cer:.4f} | {ms:.0f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
