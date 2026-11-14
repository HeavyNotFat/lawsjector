from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "third_party" / "PaddleOCR"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
KEYS = REPO / "ppocr" / "utils" / "ppocr_keys_v1.txt"
ONNX_DIR = ROOT / "output" / "onnx"

DEFAULT_CKPT = {
    "rec": ROOT / "output" / "rec_v4_finetune" / "best_accuracy.pdparams",
    "det": ROOT / "output" / "det_v4_finetune" / "best_accuracy.pdparams",
}
CONFIG = {
    "rec": ROOT / "configs" / "rec_finetune.yml",
    "det": ROOT / "configs" / "det_finetune.yml",
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=str(cwd or ROOT), check=True)


def export_one(kind: str, checkpoint: Path) -> Path:
    """导出一个模型并转换为 ONNX。

    :param kind: det 或 rec
    :param checkpoint: pdparams 路径
    :returns: 生成的 ONNX 路径
    :raises FileNotFoundError: 权重不存在时
    """
    if not checkpoint.exists():
        raise FileNotFoundError(f"找不到权重：{checkpoint}")
    infer_dir = ROOT / "output" / f"infer_{kind}"
    if infer_dir.exists():
        shutil.rmtree(infer_dir)
    print(f"导出 {kind} 推理模型")
    run(
        [
            PY,
            REPO / "tools" / "export_model.py",
            "-c",
            CONFIG[kind],
            "-o",
            f"Global.pretrained_model={checkpoint}",
            f"Global.save_inference_dir={infer_dir}",
            "Global.use_gpu=False",
        ],
        cwd=REPO,
    )

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ONNX_DIR / f"{kind}.onnx"
    raw_path = ONNX_DIR / f"{kind}_raw.onnx"
    run(
        [
            ROOT / ".venv" / "Scripts" / "paddle2onnx.exe",
            "--model_dir",
            infer_dir,
            "--model_filename",
            "inference.pdmodel",
            "--params_filename",
            "inference.pdiparams",
            "--save_file",
            raw_path,
            "--opset_version",
            "11",
            "--enable_onnx_checker",
            "True",
        ]
    )
    try:
        run([ROOT / ".venv" / "Scripts" / "onnxslim.exe", raw_path, onnx_path])
        raw_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"[warn] onnxslim 失败，直接使用原始 ONNX：{exc}")
        raw_path.replace(onnx_path)

    print(f"[完成] {onnx_path}  ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return onnx_path


def deploy(target: Path) -> Path:
    """把 ONNX 模型同步到应用资源目录。

    :param target: 目标目录，通常是 `..\\resources\\models`
    :returns: 目标目录
    """
    target.mkdir(parents=True, exist_ok=True)
    for name in ("det.onnx", "rec.onnx", "ppocr_keys_v1.txt"):
        src = ONNX_DIR / name
        if src.exists():
            shutil.copy2(src, target / name)
            print(f"[部署] {src.name} -> {target}")
    return target


def main() -> int:
    """导出 det/rec 模型到 ONNX 并部署到 resources。

    :returns: 进程退出码
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["det", "rec", "both"], default="both")
    ap.add_argument("--rec-checkpoint", default=None)
    ap.add_argument("--det-checkpoint", default=None)
    ap.add_argument("--no-deploy", action="store_true", help="只导出，不复制到 resources")
    args = ap.parse_args()

    kinds = ["rec", "det"] if args.kind == "both" else [args.kind]
    for kind in kinds:
        explicit = args.rec_checkpoint if kind == "rec" else args.det_checkpoint
        checkpoint = Path(explicit) if explicit else DEFAULT_CKPT[kind]
        export_one(kind, checkpoint)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(KEYS, ONNX_DIR / "ppocr_keys_v1.txt")
    print(f"\n字符字典：{ONNX_DIR / 'ppocr_keys_v1.txt'}")

    if not args.no_deploy:
        target = ROOT.parent / "resources" / "models"
        deploy(target)
        print(f"应用会从这里加载：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
