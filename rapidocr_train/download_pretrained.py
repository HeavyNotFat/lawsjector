from __future__ import annotations

import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pretrained"

MODELS = {
    "ch_PP-OCRv4_det_train.tar": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar",
    "ch_PP-OCRv4_rec_train.tar": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar",
    "ch_ppocr_mobile_v2.0_cls_train.tar": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar",
}


def download(url: str, dst: Path) -> None:
    """下载单个文件，已存在则跳过。

    :param url: 下载地址
    :param dst: 目标路径
    """
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[跳过] 已存在 {dst.name} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")
        return
    print(f"[下载] {url}")
    start = time.time()

    def hook(blocks: int, block_size: int, total: int) -> None:
        done = blocks * block_size
        if total > 0 and (done % (5 * 1024 * 1024) < block_size or done >= total):
            pct = done * 100 / total
            speed = done / max(1e-6, time.time() - start) / 1024 / 1024
            sys.stdout.write(f"\r  {pct:5.1f}%  {done / 1024 / 1024:7.1f}/{total / 1024 / 1024:.1f} MB  {speed:5.1f} MB/s")
            sys.stdout.flush()

    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    sys.stdout.write("\n")
    tmp.replace(dst)
    print(f"[完成] {dst.name} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> int:
    """下载并解压 PP-OCRv4 训练权重。

    :returns: 进程退出码
    """
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        tar_path = OUT / name
        try:
            download(url, tar_path)
        except Exception as exc:
            print(f"[失败] {name}: {exc}", file=sys.stderr)
            return 1
        target_dir = OUT / name.replace(".tar", "")
        if not target_dir.exists():
            print(f"[解压] {name}")
            with tarfile.open(tar_path) as tf:
                tf.extractall(OUT)
        else:
            print(f"[跳过] 已解压 {target_dir.name}")

    print("\n可用权重：")
    for p in sorted(OUT.rglob("*.pdparams")):
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
