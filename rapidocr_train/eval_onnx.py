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
KEYS = ONNX_DIR / "ppocr_keys_v1.txt"


def edit_distance(a: str, b: str) -> int:
    """计算 Levenshtein 距离。

    :param a: 字符串 a
    :param b: 字符串 b
    :returns: 编辑距离
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def read_label_file(path: Path) -> list[tuple[Path, str]]:
    """读取 `图片<TAB>标签` 格式的标签文件。

    :param path: 标签文件路径
    :returns: (图片绝对路径, 标签) 列表
    """
    rows: list[tuple[Path, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, _, label = line.partition("\t")
        rows.append((path.parent / name, label))
    return rows


def rec_cer(engine: RapidOCR, rows: list[tuple[Path, str]]) -> tuple[float, float]:
    """统计识别字符错误率与整行准确率。

    :param engine: RapidOCR 实例
    :param rows: 行图与真值
    :returns: (CER, 整行准确率)
    """
    total_err = 0
    total_len = 0
    hit = 0
    for img_path, gt in rows:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        res, _ = engine.text_rec([img])
        pred = res[0][0] if res else ""
        total_err += edit_distance(pred, gt)
        total_len += max(1, len(gt))
        hit += int(pred == gt)
    return total_err / max(1, total_len), hit / max(1, len(rows))


def box_iou(p: np.ndarray, q: np.ndarray) -> float:
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
        inter = a.intersection(b).area
        union = a.union(b).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def det_pr(engine: RapidOCR, rows: list[tuple[Path, str]], iou_thr: float = 0.5) -> dict[str, float]:
    """统计文本框的 precision / recall / F1。

    :param engine: RapidOCR 实例
    :param rows: 整页图与检测标签
    :param iou_thr: 判定匹配的 IoU 阈值
    :returns: 含 precision/recall/f1/tp/fp/fn 的字典
    """
    tp = fp = fn = 0
    for img_path, label in rows:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt_boxes = [np.array(item["points"], dtype=np.float32) for item in json.loads(label)]
        pred, _ = engine.text_det(img)
        pred_boxes = [] if pred is None else [np.array(b, dtype=np.float32).reshape(-1, 2) for b in pred]

        used = set()
        for gt in gt_boxes:
            best_iou, best_j = 0.0, -1
            for j, pb in enumerate(pred_boxes):
                if j in used:
                    continue
                iou = box_iou(gt, pb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0:
                used.add(best_j)
                tp += 1
            else:
                fn += 1
        fp += len(pred_boxes) - len(used)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def pipeline(engine: RapidOCR, rows: list[tuple[Path, str]], limit: int = 10) -> tuple[float, float]:
    """统计整页推理耗时与平均识别行数。

    :param engine: RapidOCR 实例
    :param rows: 整页图
    :param limit: 参与统计的页数
    :returns: (每页毫秒数, 平均行数)
    """
    total = 0.0
    lines = 0
    warm = cv2.imread(str(rows[0][0])) if rows else None
    if warm is not None:
        engine(warm)  # 预热，避免首次 session 初始化计入耗时
    for img_path, _ in rows[:limit]:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        t0 = time.time()
        result, _ = engine(img)
        total += time.time() - t0
        lines += len(result or [])
    n = max(1, min(limit, len(rows)))
    return total / n * 1000, lines / n


def main() -> int:
    """对比官方模型与自训练模型的 CER / P-R / 耗时。

    :returns: 进程退出码
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-val", default=str(ROOT / "data" / "rec" / "val.txt"))
    ap.add_argument("--det-val", default=str(ROOT / "data" / "det" / "val.txt"))
    ap.add_argument("--rec-model", default=str(ONNX_DIR / "rec.onnx"))
    ap.add_argument("--det-model", default=str(ONNX_DIR / "det.onnx"))
    ap.add_argument("--rec-keys", default=str(KEYS))
    ap.add_argument("--limit-rec", type=int, default=400)
    ap.add_argument("--limit-det", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "output" / "eval_report.md"))
    args = ap.parse_args()

    rec_rows = read_label_file(Path(args.rec_val))[: args.limit_rec]
    det_rows = read_label_file(Path(args.det_val))[: args.limit_det]

    print("加载官方模型（baseline）...")
    base = RapidOCR()
    print("加载自训练模型（finetune）...")
    custom = RapidOCR(
        det_model_path=args.det_model,
        rec_model_path=args.rec_model,
        rec_keys_path=args.rec_keys,
    )

    report: dict[str, dict[str, float]] = {}
    print(f"\n评估 rec CER（{len(rec_rows)} 行）...")
    base_cer, base_exact = rec_cer(base, rec_rows)
    ft_cer, ft_exact = rec_cer(custom, rec_rows)
    report["rec"] = {
        "baseline_cer": base_cer,
        "baseline_exact": base_exact,
        "finetune_cer": ft_cer,
        "finetune_exact": ft_exact,
    }

    print(f"评估 det P/R/F1（{len(det_rows)} 页）...")
    base_det = det_pr(base, det_rows)
    ft_det = det_pr(custom, det_rows)
    report["det"] = {f"baseline_{k}": v for k, v in base_det.items()}
    report["det"].update({f"finetune_{k}": v for k, v in ft_det.items()})

    print("评估整页端到端耗时...")
    base_ms, base_lines = pipeline(base, det_rows, 10)
    ft_ms, ft_lines = pipeline(custom, det_rows, 10)
    report["pipeline"] = {
        "baseline_ms": base_ms,
        "baseline_lines": base_lines,
        "finetune_ms": ft_ms,
        "finetune_lines": ft_lines,
    }

    lines = [
        "# RapidOCR 微调前后对比",
        "",
        f"- 识别验证集：`{args.rec_val}`（{len(rec_rows)} 行）",
        f"- 检测验证集：`{args.det_val}`（{len(det_rows)} 页）",
        "",
        "| 指标 | 官方 PP-OCRv4 | 自训练 | 变化 |",
        "| --- | --- | --- | --- |",
        f"| rec CER（越低越好） | {base_cer:.4f} | {ft_cer:.4f} | {(ft_cer - base_cer) * 100:+.2f} pp |",
        f"| rec 整行准确率 | {base_exact:.2%} | {ft_exact:.2%} | {(ft_exact - base_exact) * 100:+.2f} pp |",
        f"| det Precision | {base_det['precision']:.3f} | {ft_det['precision']:.3f} | {ft_det['precision'] - base_det['precision']:+.3f} |",
        f"| det Recall | {base_det['recall']:.3f} | {ft_det['recall']:.3f} | {ft_det['recall'] - base_det['recall']:+.3f} |",
        f"| det F1 | {base_det['f1']:.3f} | {ft_det['f1']:.3f} | {ft_det['f1'] - base_det['f1']:+.3f} |",
        f"| 整页耗时（ms/页） | {base_ms:.0f} | {ft_ms:.0f} | {ft_ms - base_ms:+.0f} |",
        f"| 平均识别行数 | {base_lines:.1f} | {ft_lines:.1f} | — |",
        "",
        "> 合成数据的 CER 只反映领域字体/水印/版式适配程度，真实截图请用 `data/real` 自建验证集复测。",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out.parent / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"\n报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
