from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "resources" / "models"


def quantize(src: Path, dst: Path) -> None:
    """动态量化一个 ONNX 模型。

    :param src: 输入模型
    :param dst: 输出模型
    """
    if not src.exists():
        print(f"跳过（不存在）：{src}")
        return
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    src_mb = src.stat().st_size / 1024 / 1024
    dst_mb = dst.stat().st_size / 1024 / 1024
    print(f"{src.name}: {src_mb:.1f} MB -> {dst.name}: {dst_mb:.1f} MB")


def main() -> int:
    """把 det/rec 量化成 int8，输出到 output/onnx/q8。

    :returns: 进程退出码
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(ROOT / "output" / "onnx" / "q8"))
    args = ap.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    quantize(src / "det.onnx", out / "det.onnx")
    quantize(src / "rec.onnx", out / "rec.onnx")
    keys = src / "ppocr_keys_v1.txt"
    if keys.exists():
        (out / "ppocr_keys_v1.txt").write_bytes(keys.read_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
