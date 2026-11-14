from pathlib import Path

project = Path(SPECPATH)

datas = [
    (str(project / "resources" / "法治理论知识题库.txt"), "resources"),
    (str(project / "resources" / "models" / "det.onnx"), "resources/models"),
    (str(project / "resources" / "models" / "rec.onnx"), "resources/models"),
    (str(project / "resources" / "models" / "ppocr_keys_v1.txt"), "resources/models"),
    (str(project / "resources" / "native"), "resources/native"),
]
binaries = []

# OCR 走原生 C++（resources/native/lawsjector_native.dll + onnxruntime.dll），
# 不再打包 rapidocr_onnxruntime / opencv / numpy，exe 体积可以小一半。
excludes = [
    "rapidocr_onnxruntime",
    "cv2",
    "numpy",
    "onnxruntime",
    "shapely",
    "pyclipper",
    "yaml",
    "winsdk",
    "winrt",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "matplotlib",
    "pandas",
    "scipy",
    "torch",
    "tensorflow",
    "IPython",
    "jupyter",
    "notebook",
    "sklearn",
    "numba",
    "llvmlite",
    "tkinter.test",
    "test",
    "unittest",
]

a = Analysis(
    [str(project / "main.py")],
    pathex=[str(project)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="法治题库",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=True,
    icon=[str(project / "resources" / "法治题库.ico")],
)
