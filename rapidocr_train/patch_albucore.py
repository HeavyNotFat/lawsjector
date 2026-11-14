from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MARK = "# --- lawsjector-rapidocr compat patch ---"
PATCH = f'''

{MARK}
def __getattr__(name):
    if name == "preserve_channel_dim":
        from albucore.decorators import preserve_channel_dim as _func

        return _func
    raise AttributeError(f"module {{__name__!r}} has no attribute {{name!r}}")
'''


def main() -> int:
    """给 albucore 打补丁，让 albumentations 能延迟导入 preserve_channel_dim。

    :returns: 进程退出码
    """
    spec = importlib.util.find_spec("albucore")
    if spec is None or not spec.origin:
        print("未安装 albucore，跳过", file=sys.stderr)
        return 1
    utils_py = Path(spec.origin).parent / "utils.py"
    if not utils_py.exists():
        print(f"找不到 {utils_py}", file=sys.stderr)
        return 1

    text = utils_py.read_text(encoding="utf-8")
    idx = text.find(MARK)
    if idx != -1:
        text = text[:idx].rstrip() + "\n"
    utils_py.write_text(text + PATCH, encoding="utf-8")
    print(f"已写入兼容补丁：{utils_py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
