from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "third_party" / "PaddleOCR"
CONF_DIR = ROOT / "configs"
PRET = ROOT / "pretrained"


def _set(cfg: dict, dotted: str, value) -> None:
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def _dump(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False)
    path.write_text(text, encoding="utf-8")
    print(f"[写出] {path.relative_to(ROOT)}")


def build_det(epochs: int, batch_size: int, lr: float) -> dict:
    """基于官方 det 配置生成微调配置。

    :param epochs: 训练轮数
    :param batch_size: 单卡 batch
    :param lr: 初始学习率
    :returns: 配置字典
    """
    base = REPO / "configs" / "det" / "ch_PP-OCRv4" / "ch_PP-OCRv4_det_student.yml"
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))

    _set(cfg, "Global.epoch_num", epochs)
    _set(cfg, "Global.save_model_dir", str(ROOT / "output" / "det_v4_finetune"))
    _set(cfg, "Global.save_epoch_step", max(1, epochs // 5))
    _set(cfg, "Global.print_batch_step", 20)
    _set(cfg, "Global.eval_batch_step", [0, 60])
    _set(cfg, "Global.pretrained_model", str(PRET / "ch_PP-OCRv4_det_train" / "best_accuracy.pdparams"))
    _set(cfg, "Global.use_gpu", True)
    _set(cfg, "Global.distributed", False)
    _set(cfg, "Global.save_res_path", str(ROOT / "output" / "det_predicts.txt"))

    _set(cfg, "Optimizer.lr.learning_rate", lr)
    _set(cfg, "Optimizer.lr.warmup_epoch", min(2, max(1, epochs // 10)))

    # CopyPaste 依赖 ICDAR 的 ext_data，自建数据没有该字段
    transforms = [t for t in cfg["Train"]["dataset"]["transforms"] if "CopyPaste" not in t]
    for t in transforms:
        for name, kwargs in t.items():
            if isinstance(kwargs, dict) and "total_epoch" in kwargs:
                kwargs["total_epoch"] = epochs
    _set(cfg, "Train.dataset.transforms", transforms)

    _set(cfg, "Train.dataset.data_dir", str(ROOT / "data" / "det"))
    _set(cfg, "Train.dataset.label_file_list", [str(ROOT / "data" / "det" / "train.txt")])
    _set(cfg, "Train.dataset.ratio_list", [1.0])
    _set(cfg, "Train.loader.batch_size_per_card", batch_size)
    _set(cfg, "Train.loader.num_workers", 0)

    _set(cfg, "Eval.dataset.data_dir", str(ROOT / "data" / "det"))
    _set(cfg, "Eval.dataset.label_file_list", [str(ROOT / "data" / "det" / "val.txt")])
    _set(cfg, "Eval.loader.batch_size_per_card", 1)
    _set(cfg, "Eval.loader.num_workers", 0)
    return cfg


def build_rec(epochs: int, batch_size: int, lr: float) -> dict:
    """基于官方 rec 配置生成微调配置。

    :param epochs: 训练轮数
    :param batch_size: 单卡 batch
    :param lr: 初始学习率
    :returns: 配置字典
    """
    base = REPO / "configs" / "rec" / "PP-OCRv4" / "ch_PP-OCRv4_rec.yml"
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))

    _set(cfg, "Global.epoch_num", epochs)
    _set(cfg, "Global.save_model_dir", str(ROOT / "output" / "rec_v4_finetune"))
    _set(cfg, "Global.save_epoch_step", max(1, epochs // 5))
    _set(cfg, "Global.print_batch_step", 20)
    _set(cfg, "Global.eval_batch_step", [0, 250])
    _set(cfg, "Global.pretrained_model", str(PRET / "ch_PP-OCRv4_rec_train" / "student.pdparams"))
    _set(cfg, "Global.use_gpu", True)
    _set(cfg, "Global.distributed", False)
    _set(cfg, "Global.character_dict_path", str(REPO / "ppocr" / "utils" / "ppocr_keys_v1.txt"))
    _set(cfg, "Global.max_text_length", 25)
    _set(cfg, "Global.use_space_char", True)
    _set(cfg, "Global.save_res_path", str(ROOT / "output" / "rec_predicts.txt"))

    _set(cfg, "Optimizer.lr.learning_rate", lr)
    _set(cfg, "Optimizer.lr.warmup_epoch", min(2, max(1, epochs // 10)))

    train_labels = [str(ROOT / "data" / "rec" / "train.txt")]
    val_labels = [str(ROOT / "data" / "rec" / "val.txt")]
    page_train = ROOT / "data" / "rec_page" / "train.txt"
    page_val = ROOT / "data" / "rec_page" / "val.txt"
    if page_train.exists() and page_val.exists():
        train_labels.append(str(page_train))
        val_labels.append(str(page_val))

    _set(cfg, "Train.dataset.data_dir", str(ROOT / "data" / "rec"))
    _set(cfg, "Train.dataset.label_file_list", train_labels)
    _set(cfg, "Train.loader.num_workers", 0)
    _set(cfg, "Train.sampler.first_bs", batch_size)
    _set(cfg, "Train.sampler.fix_bs", True)

    _set(cfg, "Eval.dataset.data_dir", str(ROOT / "data" / "rec"))
    _set(cfg, "Eval.dataset.label_file_list", val_labels)
    _set(cfg, "Eval.loader.batch_size_per_card", 32)
    _set(cfg, "Eval.loader.num_workers", 0)
    return cfg


def main() -> int:
    """生成 det/rec 微调配置。

    :returns: 进程退出码
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30, help="det/rec 训练轮数")
    ap.add_argument("--det-batch", type=int, default=2, help="det 单卡 batch")
    ap.add_argument("--rec-batch", type=int, default=8, help="rec 单卡 batch")
    ap.add_argument("--det-lr", type=float, default=2e-4)
    ap.add_argument("--rec-lr", type=float, default=2e-4)
    ap.add_argument("--suffix", type=str, default="", help="输出文件名后缀，例如 _smoke")
    args = ap.parse_args()

    det_path = CONF_DIR / f"det_finetune{args.suffix}.yml"
    rec_path = CONF_DIR / f"rec_finetune{args.suffix}.yml"
    _dump(build_det(args.epochs, args.det_batch, args.det_lr), det_path)
    _dump(build_rec(args.epochs, args.rec_batch, args.rec_lr), rec_path)
    print(f"\n训练轮数={args.epochs}  det_bs={args.det_batch}  rec_bs={args.rec_batch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
