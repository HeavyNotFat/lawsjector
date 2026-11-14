from __future__ import annotations

import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))

# 打包后取 exe 所在目录，源码运行时取项目根目录
APP_DIR = (
    Path(sys.executable).resolve().parent
    if FROZEN
    else Path(__file__).resolve().parent.parent
)

RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR)) / "resources" if FROZEN else APP_DIR / "resources"

TXT_PATH = RESOURCE_DIR / "法治理论知识题库.txt"
SOURCE_DOCX = RESOURCE_DIR / "2026年法治理论知识学习考试综合题库.docx"
CLEAN_DOCX = RESOURCE_DIR / "2026年法治理论知识学习考试综合题库_去水印.docx"
ICON_PATH = RESOURCE_DIR / "法治题库.ico"

MODELS_DIR = RESOURCE_DIR / "models"
NATIVE_DIR = RESOURCE_DIR / "native"

ERROR_LOG = APP_DIR / "app_error.log"
SELFTEST_LOG = APP_DIR / "selftest.log"
