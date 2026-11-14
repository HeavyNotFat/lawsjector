from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import synth_common as sc  # noqa: E402


def build_split(
    split: str,
    texts: list[str],
    out: Path,
    rng: random.Random,
    fonts: list[str],
) -> list[Path]:
    """渲染一个切分的行图并写标签文件。

    :param split: train 或 val
    :param texts: 行文本列表
    :param out: 数据集根目录
    :param rng: 随机数生成器
    :param fonts: 候选字体
    :returns: 写出的图片路径列表
    """
    rows: list[str] = []
    written: list[Path] = []
    for i, text in enumerate(texts):
        img = sc.render_line(text, rng, fonts)
        rel = f"images/{split}/{split}_{i:05d}.jpg"
        path = out / rel
        sc.save_jpeg(img, path, rng)
        rows.append(f"{rel}\t{text}")
        written.append(path)
    (out / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[{split}] {len(rows)} 张行图 -> {out / (split + '.txt')}")
    return written


def main() -> int:
    """生成 rec 行级合成数据。

    :returns: 进程退出码
    """
    ap = argparse.ArgumentParser(description="生成 rec 行级合成数据")
    ap.add_argument("--out", default=str(sc.ROOT / "data" / "rec"))
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--preview", action="store_true", help="额外输出一张预览拼图")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    corpus = sc.load_corpus()
    charset = sc.load_charset()
    fonts = sc.load_font_paths()
    print(f"语料：题目 {len(corpus['题目'])} / 选项 {len(corpus['选项'])} / 解析 {len(corpus['解析'])}，字体 {len(fonts)} 种")

    out = Path(args.out)
    total = args.train + args.val
    texts = sc.sample_texts(corpus, charset, rng, total)
    if len(texts) < total:
        print(f"警告：可用去重文本只有 {len(texts)} 条（目标 {total}）", file=sys.stderr)
    rng.shuffle(texts)
    train_texts = texts[: args.train]
    val_texts = texts[args.train :]

    build_split("train", train_texts, out, rng, fonts)
    val_imgs = build_split("val", val_texts, out, rng, fonts)

    meta = {
        "seed": args.seed,
        "train": len(train_texts),
        "val": len(val_texts),
        "max_text_len": max((len(t) for t in texts), default=0),
        "avg_text_len": round(sum(len(t) for t in texts) / max(1, len(texts)), 2),
        "fonts": [Path(f).name for f in fonts],
        "note": "使用 25 字以内的行，字符全部在 ppocr_keys_v1.txt 内",
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"统计：{meta}")

    if args.preview:
        sc.make_preview(val_imgs[:9], sc.ROOT / "data" / "preview_rec.jpg", cols=3, thumb=460)
        print(f"预览图 -> {sc.ROOT / 'data' / 'preview_rec.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
