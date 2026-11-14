from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent


def load_pages(label_file: Path) -> list[tuple[Path, list[dict]]]:
    """读取检测标签文件。

    :param label_file: 形如 `图片<TAB>JSON` 的标签文件
    :returns: (图片路径, 文本框列表) 的列表
    """
    rows: list[tuple[Path, list[dict]]] = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, _, label = line.partition("\t")
        rows.append((label_file.parent / name, json.loads(label)))
    return rows


def crop_box(page: Image.Image, points: list[list[int]], pad: int) -> Image.Image:
    """按四点外接矩形裁剪，并向外扩 pad 像素。

    :param page: 整页图像
    :param points: 四点坐标
    :param pad: 外扩像素
    :returns: 行图
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(page.width, int(max(xs)) + pad)
    y1 = min(page.height, int(max(ys)) + pad)
    return page.crop((x0, y0, x1, y1))


def build_split(
    split: str,
    pages: list[tuple[Path, list[dict]]],
    out: Path,
    rng: random.Random,
    pad: int,
    max_len: int,
) -> int:
    """裁剪一个切分的行图并写标签文件。

    :param split: train 或 val
    :param pages: load_pages 的结果
    :param out: 输出根目录
    :param rng: 随机数生成器
    :param pad: 外扩像素
    :param max_len: 允许的最大字数，超过则跳过
    :returns: 写出的行图数量
    """
    rows: list[str] = []
    idx = 0
    for img_path, items in pages:
        page = Image.open(img_path).convert("RGB")
        for item in items:
            text = item.get("transcription", "")
            if not text or len(text) > max_len:
                continue
            crop = crop_box(page, item["points"], pad)
            if crop.width < 8 or crop.height < 8:
                continue
            dst = out / "images" / split / f"{split}_{idx:05d}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            crop.save(dst, "JPEG", quality=rng.choice([80, 88, 92, 95]))
            rows.append(f"{dst.resolve()}\t{text}")
            idx += 1
    (out / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[{split}] {len(rows)} 张行图 -> {out / (split + '.txt')}")
    return len(rows)


def main() -> int:
    """从整页截图按检测标注裁出识别行图。

    :returns: 进程退出码
    """
    ap = argparse.ArgumentParser(description="从整页截图裁剪 rec 行图")
    ap.add_argument("--det", default=str(ROOT / "data" / "det"))
    ap.add_argument("--out", default=str(ROOT / "data" / "rec_page"))
    ap.add_argument("--pad", type=int, default=2, help="裁剪外扩像素")
    ap.add_argument("--max-len", type=int, default=25, help="超过该字数的框跳过")
    ap.add_argument("--seed", type=int, default=20261116)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    det_dir = Path(args.det)
    out = Path(args.out)
    train_pages = load_pages(det_dir / "train.txt")
    val_pages = load_pages(det_dir / "val.txt")
    build_split("train", train_pages, out, rng, args.pad, args.max_len)
    build_split("val", val_pages, out, rng, args.pad, args.max_len)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
