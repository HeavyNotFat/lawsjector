from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import synth_common as sc  # noqa: E402


def build_split(
    split: str,
    count: int,
    out: Path,
    rng: random.Random,
    fonts: list[str],
    corpus: dict[str, list[str]],
    charset: set[str],
) -> list[Path]:
    """渲染一个切分的整页截图并写检测标签。

    :param split: train 或 val
    :param count: 页数
    :param out: 数据集根目录
    :param rng: 随机数生成器
    :param fonts: 候选字体
    :param corpus: load_corpus 的结果
    :param charset: 字符字典
    :returns: 写出的图片路径列表
    """
    rows: list[str] = []
    written: list[Path] = []
    box_total = 0
    for i in range(count):
        page = sc.render_page(rng, fonts, corpus, charset)
        width, height = page.image.size
        items = []
        for box in page.boxes:
            if box.x0 < 0 or box.y0 < 0 or box.x1 > width or box.y1 > height:
                continue
            if box.x1 - box.x0 < 3 or box.y1 - box.y0 < 3:
                continue
            items.append({"transcription": box.text, "points": box.points()})
        if not items:
            continue
        box_total += len(items)
        rel = f"images/{split}/{split}_{i:05d}.jpg"
        path = out / rel
        sc.save_jpeg(page.image, path, rng)
        rows.append(f"{rel}\t{json.dumps(items, ensure_ascii=False)}")
        written.append(path)
    (out / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[{split}] {len(rows)} 张整页图，{box_total} 个文本框 -> {out / (split + '.txt')}")
    return written


def main() -> int:
    """生成 det 页级合成数据。

    :returns: 进程退出码
    """
    ap = argparse.ArgumentParser(description="生成 det 页级合成数据")
    ap.add_argument("--out", default=str(sc.ROOT / "data" / "det"))
    ap.add_argument("--train", type=int, default=240)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    corpus = sc.load_corpus()
    charset = sc.load_charset()
    fonts = sc.load_font_paths()
    out = Path(args.out)

    train_imgs = build_split("train", args.train, out, rng, fonts, corpus, charset)
    val_imgs = build_split("val", args.val, out, rng, fonts, corpus, charset)

    meta = {
        "seed": args.seed,
        "train_pages": len(train_imgs),
        "val_pages": len(val_imgs),
        "canvas": "1120~1360 x 820~1000",
        "labels": "每行一个文本框：transcription + 四点坐标；水印不参与标注",
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"统计：{meta}")

    if args.preview:
        sc.make_preview(val_imgs[:6], sc.ROOT / "data" / "preview_det.jpg", cols=2, thumb=680)
        print(f"预览图 -> {sc.ROOT / 'data' / 'preview_det.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
