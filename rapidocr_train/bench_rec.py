from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from rapidocr_onnxruntime import RapidOCR

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


def read_rows(path: Path) -> list[tuple[Path, str]]:
    """读取识别标签文件。

    :param path: `图片<TAB>文字` 文件
    :returns: (图片路径, 文字) 列表
    """
    rows: list[tuple[Path, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, _, text = line.partition("\t")
        img = Path(name)
        rows.append((img if img.is_absolute() else path.parent / name, text))
    return rows


def evaluate(engine: RapidOCR, rows: list[tuple[Path, str]], batch: int) -> dict[str, float]:
    """统计 CER、整行准确率和单行耗时。

    :param engine: RapidOCR 实例
    :param rows: 行图与真值
    :param batch: 批量大小，1 表示逐行调用
    :returns: 含 cer/exact/ms_per_line 的字典
    """
    images = []
    texts = []
    for img_path, text in rows:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
            texts.append(text)

    err = total = hit = 0
    elapsed = 0.0
    for begin in range(0, len(images), batch):
        chunk = images[begin : begin + batch]
        t0 = time.perf_counter()
        results, _ = engine.text_rec(chunk)
        elapsed += time.perf_counter() - t0
        for (pred, _score), gt in zip(results, texts[begin : begin + batch]):
            err += edit_distance(pred, gt)
            total += max(1, len(gt))
            hit += int(pred == gt)
    return {
        "cer": err / max(1, total),
        "exact": hit / max(1, len(images)),
        "ms_per_line": elapsed / max(1, len(images)) * 1000,
        "lines": float(len(images)),
    }


def main() -> int:
    """对比官方与自训练识别模型的速度和精度。

    :returns: 进程退出码
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=str(ROOT))
    ap.add_argument("--limit", type=int, default=0, help="每个验证集最多测多少行，0 表示全部")
    ap.add_argument("--batch", type=int, default=6, help="批量大小，默认与 RapidOCR 一致")
    args = ap.parse_args()

    onnx_dir = Path(args.train_dir) / "output" / "onnx"
    keys = onnx_dir / "ppocr_keys_v1.txt"
    datasets = {
        "行图 data/rec": read_rows(Path(args.train_dir) / "data" / "rec" / "val.txt"),
        "整页裁剪 data/rec_page": read_rows(Path(args.train_dir) / "data" / "rec_page" / "val.txt"),
    }
    if args.limit > 0:
        datasets = {k: v[: args.limit] for k, v in datasets.items()}

    engines = {
        "官方 PP-OCRv4": RapidOCR(),
        "自训练（行图）": RapidOCR(
            rec_model_path=str(onnx_dir / "rec_lines_only.onnx"), rec_keys_path=str(keys)
        ),
        "自训练（行图+整页裁剪）": RapidOCR(rec_model_path=str(onnx_dir / "rec.onnx"), rec_keys_path=str(keys)),
    }

    print(f"批量大小：{args.batch}\n")
    print("| 数据集 | 模型 | CER | 整行准确率 | ms/行 | 行/秒 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, rows in datasets.items():
        for model_name, engine in engines.items():
            m = evaluate(engine, rows, args.batch)
            print(
                f"| {name} | {model_name} | {m['cer']:.4f} | {m['exact']:.2%} | "
                f"{m['ms_per_line']:.2f} | {1000 / max(1e-9, m['ms_per_line']):.1f} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
