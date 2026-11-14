# rapidocr_train

用 PaddleOCR 微调 PP-OCRv4 的检测/识别模型，导出 ONNX 给 `resources/models`，由 C++ 引擎或 RapidOCR 加载。

## 环境

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

- Python 3.10 + PaddlePaddle-GPU 2.6.2（Windows 最后一个 GPU 版本，自带 CUDA 11.8 / cuDNN 8.1）
- PaddleOCR v2.10.0 源码克隆到 `third_party/PaddleOCR`
- `albumentations` 必须锁 1.3.1（1.4+ 依赖的 albucore 强制要 PyTorch）

验证 GPU：

```powershell
.venv\Scripts\python.exe -c "import paddle; print(paddle.__version__, paddle.is_compiled_with_cuda())"
```

## 数据

```powershell
.venv\Scripts\python.exe build_rec_dataset.py --train 4000 --val 400 --preview
.venv\Scripts\python.exe build_det_dataset.py --train 240 --val 30 --preview
.venv\Scripts\python.exe build_rec_from_pages.py
```

| 目录 | 内容 |
| --- | --- |
| `data/rec` | 由题库文本渲染的行图，8 种字体、20~44 号字、水印、旋转/模糊/JPEG |
| `data/det` | 整页「考试界面」截图 + 每行文本框 JSON |
| `data/rec_page` | 按检测框从整页截图裁出的行图（分布与推理一致） |

真实截图放 `data/real`，用 PPOCRLabel 标注后把标签文件加进配置的 `label_file_list`。

## 训练

```powershell
powershell -File train_rec.ps1    # 识别，约 4~7 分钟/epoch
powershell -File train_det.ps1    # 检测，约 20 秒/epoch
```

6GB 显存的安全参数：rec batch 8、det batch 2、lr 2e-4，配置由 `make_configs.py` 生成。

## 导出与部署

```powershell
.venv\Scripts\python.exe export_onnx.py --kind both
```

`export_model.py` → `paddle2onnx --opset_version 11` → `onnxslim`，最后自动复制到 `..\resources\models`。

## 评估

```powershell
.venv\Scripts\python.exe eval_onnx.py      # rec CER + det P/R
.venv\Scripts\python.exe eval_pages.py     # 整页端到端 CER
.venv\Scripts\python.exe bench_native.py   # 官方 / Python / C++ 三套引擎对比
```

## 踩过的坑

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `No module named torch` | albumentations 装成 1.4+ | 锁 `albumentations==1.3.1` |
| 显存不足 | batch 太大 | det 2→1、rec 8→4，lr 同比减半 |
| DataLoader 卡住 | Windows 多进程 | `num_workers: 0` |
| checkpoint 很慢 | 单次保存 260 MB | 调大 `save_epoch_step` |
| 真实截图英文/数字全错 | 合成语料几乎没有拉丁字符，微调后灾难性遗忘 | 补真实数据 + 英数字样本，lr 降到 5e-5 只跑 1~2 epoch |

合成数据只能解决冷启动，上线前必须用真实截图再训一轮，并且用真实截图评估。
