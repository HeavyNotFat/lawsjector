from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent
ONNX_DIR = ROOT / "output" / "onnx"


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
    """计算两个四点多边形的 IoU。

    :param p: 多边形 p
    :param q: 多边形 q
    :returns: IoU，异常时返回 0
    """
    try:
        a = Polygon(p).buffer(0)
        b = Polygon(q).buffer(0)
        if a.is_empty or b.is_empty:
            return 0.0
        return float(a.intersection(b).area / max(1e-9, a.union(b).area))
    except Exception:
        return 0.0


def page_cer(engine: RapidOCR, rows: list[tuple[Path, str]], match_iou: float = 0.3) -> dict[str, float]:
    """统计整页端到端 CER，漏检的行按全错计算。

    :param engine: RapidOCR 实例
    :param rows: 整页图与检测标签
    :param match_iou: 判定框匹配的 IoU 阈值
    :returns: 含 page_cer / ms_per_page / pages 的字典
    """
    total_err = total_len = 0
    pages = 0
    t0 = time.time()
    for img_path, label in rows:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt = [(item["transcription"], np.array(item["points"], dtype=np.float32)) for item in json.loads(label)]
        result, _ = engine(img)
        preds = [] if not result else [(str(it[1]), np.array(it[0], dtype=np.float32).reshape(-1, 2)) for it in result]

        used = set()
        for gt_text, gt_box in gt:
            best, best_j = 0.0, -1
            for j, (_, pb) in enumerate(preds):
                if j in used:
                    continue
                v = iou(gt_box, pb)
                if v > best:
                    best, best_j = v, j
            if best >= match_iou and best_j >= 0:
                used.add(best_j)
                total_err += edit_distance(preds[best_j][0], gt_text)
            else:
                total_err += len(gt_text)
            total_len += max(1, len(gt_text))
        pages += 1
    return {
        "page_cer": total_err / max(1, total_len),
        "ms_per_page": (time.time() - t0) / max(1, pages) * 1000,
        "pages": pages,
    }


def build(rec_only: bool, args) -> RapidOCR:
    """构造待评估的 RapidOCR 引擎。

    :param rec_only: 只替换 rec，检测用官方模型
    :param args: 命令行参数
    :returns: RapidOCR 实例
    """
    kwargs = {"det_limit_side_len": 960, "det_limit_type": "max"}
    if not rec_only:
        kwargs["det_model_path"] = args.det_model
    kwargs["rec_model_path"] = args.rec_model
    kwargs["rec_keys_path"] = args.rec_keys
    return RapidOCR(**kwargs)


def main() -> int:
    """对比官方模型与自训练模型的整页 CER。

    :returns: 进程退出码
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=str(ROOT / "data" / "det" / "val.txt"))
    ap.add_argument("--rec-model", default=str(ONNX_DIR / "rec.onnx"))
    ap.add_argument("--det-model", default=str(ONNX_DIR / "det.onnx"))
    ap.add_argument("--rec-keys", default=str(ONNX_DIR / "ppocr_keys_v1.txt"))
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--rec-only", action="store_true", help="只替换 rec，检测仍用官方模型")
    args = ap.parse_args()

    path = Path(args.val)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            name, _, label = line.partition("\t")
            rows.append((path.parent / name, label))
    rows = rows[: args.limit]

    variants = [("官方 PP-OCRv4（det+rec）", RapidOCR(det_limit_side_len=960, det_limit_type="max"))]
    if args.rec_only:
        variants.append(("官方 det + 自训练 rec", build(True, args)))
    else:
        variants.append(("自训练 det + rec", build(False, args)))

    print(f"整页验证集：{len(rows)} 页\n")
    print("| 方案 | 页面 CER | 平均耗时 ms/页 |")
    print("| --- | --- | --- |")
    for name, engine in variants:
        m = page_cer(engine, rows)
        print(f"| {name} | {m['page_cer']:.4f} | {m['ms_per_page']:.0f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
