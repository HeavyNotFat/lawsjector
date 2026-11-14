# native

`lawsjector_native.cpp` 编译成 `resources/native/lawsjector_native.dll`，提供缩放、行排序和整套 OCR 引擎（检测预处理、DB 后处理、识别裁剪与 CTC 解码）。

## 依赖

1. Visual Studio 2019/2022（含 C++ 桌面开发），`build.ps1` 会用 `vswhere` 自动找 `vcvars64.bat`。
2. ONNX Runtime C++ SDK（1.20.1 CPU 版）：

```powershell
# 下载后解压到 native/third_party/
# https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-win-x64-1.20.1.zip
```

目录结构应为：

```
native/third_party/onnxruntime-win-x64-1.20.1/
├─ include/onnxruntime_cxx_api.h
└─ lib/onnxruntime.lib
```

## 编译

```powershell
powershell -File native\build.ps1
```

产物：

- `resources/native/lawsjector_native.dll`
- `resources/native/onnxruntime.dll`

运行还需要 `msvcp140.dll`、`msvcp140_1.dll`、`vcruntime140.dll`、`vcruntime140_1.dll`（VC++ 运行库，打包前放进 `resources/native/`）。

## 接口

| 函数 | 用途 |
| --- | --- |
| `lsocr_create` / `lsocr_destroy` | 创建/销毁引擎，加载 det、rec 模型与字符字典 |
| `lsocr_run` / `lsocr_free` | 输入 RGB 字节，输出 `x\ty\tscore\ttext` 行文本 |
| `ls_resize_rgb` / `ls_order_lines` | 缩放与阅读顺序排序（Python 兜底路径用） |
