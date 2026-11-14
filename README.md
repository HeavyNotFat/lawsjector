# 法治题库 · 智能查询（LawsJector）

Tkinter 桌面版题库检索工具：截图 → 离线 OCR → 模糊匹配 → 显示原题/答案/解析。
纯本地运行，不联网、不上传数据。

## 快速开始

```powershell
git clone https://github.com/HeavyNotFat/lawsjector.git
cd lawsjector
python -m pip install -r requirements.txt
```

把 OCR 模型放进 `resources/models/`（`det.onnx`、`rec.onnx`、`ppocr_keys_v1.txt`），来源见 [resources/README.md](resources/README.md)，然后：

```powershell
python main.py
```

C++ 引擎需要先编译：

```powershell
powershell -File native\build.ps1
```

需要 Visual Studio（C++ 桌面开发）和 ONNX Runtime C++ SDK，详见 [native/README.md](native/README.md)。没有编译也不要紧，源码环境会自动退回 RapidOCR。

打包：

```powershell
python -m PyInstaller build_exe.spec --noconfirm
```

## 目录结构

```
LawsJector/
├─ main.py                 唯一入口
├─ lawsjector/             应用包
│  ├─ app.py               Tkinter 界面与主循环
│  ├─ bank.py              题库模型、检索、文本清洗
│  ├─ ocr_engine.py        OCR 调度：原生 C++ → RapidOCR → Windows OCR
│  ├─ native_ocr.py        C++ OCR 引擎的 ctypes 封装
│  ├─ native.py            C++ 图像预处理的 ctypes 封装（兜底路径用）
│  ├─ snipper.py           屏幕框选
│  ├─ build_bank.py        DOCX → 题库 TXT 构建器
│  └─ paths.py             资源路径（兼容 PyInstaller）
├─ native/
│  ├─ lawsjector_native.cpp    C++：缩放、行排序、DB 检测后处理、CTC 解码
│  ├─ build.ps1                用 MSVC 编译到 resources/native/
│  └─ third_party/onnxruntime-win-x64-1.20.1/   ONNX Runtime C++ SDK（只用于编译）
├─ resources/
│  ├─ 法治理论知识题库.txt
│  ├─ 法治题库.ico
│  ├─ models/              模型目录（内容不入库，见 resources/README.md）
│  └─ native/              编译产物（不入库）
├─ rapidocr_train/         训练工作区（不参与打包）
├─ build_exe.spec          PyInstaller 配置
└─ README.md
```

## 架构

```
截图(PIL) ─► C++ 引擎 ─► ONNX Runtime ─► 文本框/文字 ─► Python 只做匹配与 UI
                │
                ├─ det：缩放(限长边960) + 归一化 + 推理 + DB 后处理(连通域/最小外接矩形/unclip)
                └─ rec：透视裁剪 + 按宽高比分批 + 归一化 + 推理 + CTC 解码
```

计算全部在 `lawsjector_native.dll` 里，Python 只负责截图、传 RGB 字节、收 `文字/置信度/坐标`。
ONNX Runtime 线程数取 `min(8, CPU 核数)`，检测与识别共用两个 Session。

## 常用命令

```powershell
python main.py                       # 运行
python main.py --self-test           # 自检：题库 / 检索 / OCR / OCR 检索
python -m lawsjector.build_bank      # 重新生成题库 TXT
powershell -File native\build.ps1    # 编译 C++ DLL
python -m PyInstaller build_exe.spec --noconfirm   # 打包 exe
```

打包只包含 `lawsjector/` 与 `resources/`；`rapidocr_train/`、`rapidocr_onnxruntime`、`opencv`、`numpy`
都不进包（OCR 已由 C++ 接管），所以 exe 从 114 MB 降到 42 MB。

## OCR 模型

默认从 `resources/models/` 加载官方 PP-OCRv4（`det.onnx`、`rec.onnx`、`ppocr_keys_v1.txt`），
文件缺失时退回 RapidOCR 内置模型（仅源码环境）或 Windows OCR。

自训练模型放在 `resources/models/custom/`，用环境变量切换：

```powershell
$env:LAWSJECTOR_OCR_ONNX_DIR = "<仓库路径>\resources\models\custom"
python main.py
```

合成数据练出来的模型在真实截图上英文/数字退化明显，默认因此用官方模型；
如何用真实截图重训见 `rapidocr_train\README.md`。

## C++ 部分

`native/lawsjector_native.cpp` 导出：

| 函数 | 用途 |
| --- | --- |
| `lsocr_create` / `lsocr_destroy` | 创建/销毁引擎，加载 det、rec 模型与字符字典 |
| `lsocr_run` / `lsocr_free` | 输入 RGB 字节，返回 `x\ty\tscore\ttext` 行文本 |
| `ls_resize_rgb` / `ls_order_lines` | 缩放与行排序（RapidOCR 兜底路径使用） |

编译依赖 ONNX Runtime C++ SDK（`native/third_party/`，仅编译期需要），运行时只需要
`resources/native/` 下的两个 DLL。
## 文档

- [技术博客：从截图到答案](docs/blog.md)
- [训练与模型导出](rapidocr_train/README.md)
