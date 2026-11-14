# 一个截图搜题工具：把 OCR 从 Python 搬进 C++，exe 从 114 MB 压到 42 MB

> 项目：法治题库 · 智能查询（LawsJector）
> 场景：截图 → 离线 OCR → 1291 道题的题库里模糊匹配 → 显示答案和解析
> 技术栈：Tkinter + C++（ONNX Runtime）+ PaddleOCR 训练 + PyInstaller

## 一、工具在干什么

题库来自复习资料，考试是机考，题干在屏幕上。人工复制去搜效率很低，于是：

1. 框选屏幕上的题目；
2. 本地 OCR 识别文字（不联网、不上传）；
3. 用识别结果在题库里模糊匹配；
4. 弹出原题、答案、解析。

## 二、整体架构

```
Tkinter UI（app.py）      截图、列表、详情
        │ PIL.Image
ocr_engine.py（胶水）     预处理、调度、兜底
        │ RGB bytes（ctypes）
lawsjector_native.dll     检测 + 识别 + 解码，全部 C++
  └─ onnxruntime.dll      ONNX Runtime C++ API
        │ 文本/置信度/坐标
bank.py                   二元组倒排 + difflib 相似度
```

Python 只做三件事：截图、传字节、拿结果做匹配。

## 三、一次识别怎么跑

### 预处理

宽度小于 900 的截图放大到 900（小字检测容易漏），超过 1900 的缩小，高度上限 2200。

### 检测

1. RGB → BGR（PaddleOCR 按 BGR 训练，不换通道彩色文字会认崩）；
2. 限长边 960、按 32 对齐；
3. 三角滤波缩放，归一化 `v / 127.5 - 1`，排成 NCHW；
4. ONNX Runtime 推理，得到 0~1 的概率图。

### 检测后处理

阈值 0.3 → 2×2 膨胀 → 8 连通域 → 凸包 → 最小外接矩形 → 矩形内概率均值 ≥ 0.5 → 外扩（unclip 1.6）→ 映射回原图。

unclip 距离用 DB 的公式：

```
distance = area * ratio / perimeter
```

矩形外扩 `distance` 等价于 pyclipper 圆角外扩再取最小外接矩形，所以加宽高即可。

### 识别

每个检测框做透视采样，按宽高比排序后分批（batch 6），缩放到高 48、宽度 `48 × max(6.67, 宽高比)`，归一化后推理。

识别模型输出已经是 softmax 概率，不要再做一次 softmax。

### CTC 解码

每帧取 argmax，跳过 blank(0) 和连续重复，查字典拼接。字典 = blank + 6623 字 + 空格 = 6625。置信度低于 0.5 的行丢弃。

### 匹配

行按 `(round(y/10), x)` 排成阅读顺序；先精确/包含匹配，没命中就查二元组倒排索引召回候选，再用 `difflib.SequenceMatcher` 排序。

## 四、为什么搬进 C++

| 方案 | 页面 CER | 耗时 |
| --- | --- | --- |
| 官方 PP-OCRv4（Python / RapidOCR） | 0.0823 | 3178 ms |
| 自训练（Python / RapidOCR） | 0.0686 | 2295 ms |
| 自训练（C++ 原生） | 0.0598 | 1697 ms |

省掉了 Python 与 cv2 之间的数据往返，检测后处理是手写的连续内存扫描。

## 五、C++ 怎么写

对外只暴露 C 接口，ctypes 好调、ABI 稳定：

```cpp
LS_EXPORT void* lsocr_create(const wchar_t* det_path, const wchar_t* rec_path,
                             const char* keys_utf8, int threads);
LS_EXPORT const char* lsocr_run(void* handle, const unsigned char* rgb,
                                int w, int h, int* out_len);
LS_EXPORT void lsocr_free(const char* text);
```

`lsocr_run` 返回 `x\ty\tscore\ttext` 行文本，Python 解析成列表。

编译只需要 ONNX Runtime SDK：

```powershell
cl /O2 /MT /EHsc /std:c++17 /utf-8 /LD lawsjector_native.cpp ^
   /I onnxruntime/include /link onnxruntime/lib/onnxruntime.lib
```

`/MT` 静态链接 C 运行库；`onnxruntime.dll` 依赖 `MSVCP140.dll`、`VCRUNTIME140.dll`，打包时一起放进 `resources/native/`。

完整代码见 `native/lawsjector_native.cpp`（780 行）与 `lawsjector/native_ocr.py`。

## 六、模型怎么练

1. 题库文本按标点切成 ≤25 字的行，渲染成截图风格（多字体、水印、模糊、JPEG）；
2. 再从整页截图按检测框裁一遍行图，让训练分布贴近推理；
3. PaddleOCR 微调 det（20 epoch）与 rec（8 epoch）；
4. `paddle2onnx` 导出 ONNX，`onnxslim` 精简，自动复制到 `resources/models`。

| 验证集 | 官方 | 自训练 |
| --- | --- | --- |
| 行图 CER | 0.0325 | 0.0133 |
| 行图整行准确率 | 68.00% | 86.00% |
| 整页裁剪 CER | 0.1113 | 0.0157 |
| 整页裁剪整行准确率 | 30.56% | 85.19% |

## 七、踩过的坑

| 坑 | 表现 | 修复 |
| --- | --- | --- |
| RGB / BGR | 彩色文字（蓝题号、绿答案、灰网址）全错 | 入口统一 RGB→BGR |
| 小截图不放大 | 700px 宽的真实截图大面积识别不到 | 推理前放大到 900 宽 |
| 合成数据遗忘 | 真实截图里英文、数字几乎全错 | 默认用官方模型，补真实数据重训 |
| checkpoint 过大 | 单次保存 260 MB，训练后期卡几分钟 | 调大 `save_epoch_step` |
| 依赖冲突 | albumentations 1.4+ 强制 PyTorch | 锁 1.3.1 |
| 打包臃肿 | 默认拖进 OpenCV / numpy / CUDA | OCR 走原生 DLL，spec 里 excludes |

## 八、分发包

```
法治题库.zip
├─ 法治题库.exe      42 MB，双击运行，无需 Python、无需联网
└─ 使用说明.txt      SmartScreen 怎么点、杀软误报怎么办
```

两个绕不开的提示：Windows SmartScreen（未签名程序）和杀软误报。彻底解决需要代码签名证书。

## 九、小结

- Python 管「人」：界面、截图、匹配；
- C++ 管「机器」：像素、矩阵、推理；
- 模型管「字」：训练、导出、部署。

三块各司其职，才有 1.7 秒一页、42 MB 一个 exe、双击就能用的结果。
