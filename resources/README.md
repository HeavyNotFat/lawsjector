# resources

运行所需的资源。仓库里只提交题库和图标，模型与编译产物不进版本库。

| 文件 | 说明 | 是否提交 |
| --- | --- | --- |
| `法治理论知识题库.txt` | 1291 道题，程序启动时读取 | 是 |
| `法治题库.ico` | 程序图标 | 是 |
| `models/det.onnx`、`rec.onnx`、`ppocr_keys_v1.txt` | OCR 模型与字典 | 否 |
| `native/*.dll` | C++ 引擎和 ONNX Runtime | 否 |

## 模型从哪来

1. **官方 PP-OCRv4**（推荐先用这个）：

   安装 `rapidocr-onnxruntime` 后，从包目录复制：

   ```
   site-packages/rapidocr_onnxruntime/models/ch_PP-OCRv4_det_infer.onnx  -> resources/models/det.onnx
   site-packages/rapidocr_onnxruntime/models/ch_PP-OCRv4_rec_infer.onnx  -> resources/models/rec.onnx
   site-packages/rapidocr_onnxruntime/ppocr_keys_v1.txt                  -> resources/models/ppocr_keys_v1.txt
   ```

2. **自己训练**：见 `rapidocr_train/README.md`，`export_onnx.py` 会自动把三个文件复制到这里。

模型文件缺失时，程序会退回源码环境的 RapidOCR 或 Windows OCR；打包版必须带模型，否则 OCR 不可用。

## 题库来源

`法治理论知识题库.txt` 由 `python -m lawsjector.build_bank` 从同名 docx 生成。docx/pdf 体积较大，未提交，重新生成时自行放入本目录即可。
