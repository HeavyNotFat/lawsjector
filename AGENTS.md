# LawsJector 项目约定

- Python 3.12（应用）、3.10（训练环境 `rapidocr_train\.venv`）。
- 入口只有 `main.py`；应用代码在 `lawsjector\`，资源在 `resources\`，训练在 `rapidocr_train\`。
- OCR 主路径是 C++：`native\lawsjector_native.cpp` 编译为 `resources\native\lawsjector_native.dll`，
  通过 `lawsjector\native_ocr.py` 调用；Python 只做 UI、截图和题库匹配。
- 改 C++ 后先 `powershell -File native\build.ps1`，再 `python main.py --self-test` 验证。
- 打包：`python -m PyInstaller build_exe.spec --noconfirm`，产物在 `dist\`，`rapidocr_train\` 不参与打包。

## 代码风格

- 注释只保留核心，解释为什么；不复述代码。
- 禁止 `===` / `---` 分割线，禁止文件头声明（模块 docstring、版权头、banner），文件直接从 `import` 开始。
- Python 函数 docstring 用 reStructuredText（`:param:` / `:returns:` / `:raises:`）。
- C++ 函数注释用 Doxygen（`@brief` / `@param` / `@return`）。
