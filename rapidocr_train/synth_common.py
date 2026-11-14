from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "data" / "corpus" / "法治理论知识题库.txt"
CHAR_DICT_PATH = ROOT / "third_party" / "PaddleOCR" / "ppocr" / "utils" / "ppocr_keys_v1.txt"

WATERMARK = "2026年度法治理论知识考试复习题"

FONT_DIR = Path(r"C:\Windows\Fonts")
FONT_CANDIDATES = [
    "msyh.ttc",
    "msyhbd.ttc",
    "simhei.ttf",
    "simsun.ttc",
    "simkai.ttf",
    "Deng.ttf",
    "msjh.ttc",
    "STZHONGS.TTF",
]

LIGHT_THEMES = [
    ((255, 255, 255), (20, 20, 20)),
    ((247, 250, 252), (16, 42, 67)),
    ((255, 253, 246), (40, 40, 40)),
    ((240, 244, 248), (30, 30, 30)),
    ((250, 250, 250), (60, 60, 60)),
    ((243, 246, 250), (10, 34, 57)),
]
DARK_THEMES = [
    ((28, 30, 34), (235, 235, 235)),
    ((20, 26, 38), (220, 225, 235)),
]

BREAK_AFTER = "，。；：！？、）》”’…"
BREAK_BEFORE = "（《“‘"

FIELD_RE = re.compile(r"^【(题目|选项|解析|题型|分类|答案|备注)】(.*)$")
SPACE_RE = re.compile(r"[ \t\u3000]+")

TYPE_NAMES = ("单选题", "多选题", "判断题")


def load_corpus(path: Path | str = CORPUS_PATH) -> dict[str, list[str]]:
    """解析题库 TXT。

    :param path: 题库文本路径
    :returns: 字段名到文本列表的映射，字段为「题目/选项/解析/题型/分类/答案」
    """
    corpus: dict[str, list[str]] = {k: [] for k in ("题目", "选项", "解析", "题型", "分类", "答案")}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("====="):
            continue
        m = FIELD_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if key == "选项":
            value = re.sub(r"^[A-Ea-e]\s*[.．、]\s*", "", value).strip()
        if key in corpus and value:
            corpus[key].append(value)
    return corpus


def load_charset(path: Path | str = CHAR_DICT_PATH) -> set[str]:
    """读取 PaddleOCR 字符字典。

    :param path: ppocr_keys_v1.txt 路径
    :returns: 允许出现的字符集合（含空格）
    """
    chars = set(Path(path).read_text(encoding="utf-8").splitlines())
    chars.add(" ")
    return {c for c in chars if c}


def split_into_lines(text: str, max_len: int = 25) -> list[str]:
    """按中文标点把长句切成不超过 max_len 的行。

    :param text: 原始文本
    :param max_len: 单行最大字数，对应 PP-OCRv4 的 max_text_length
    :returns: 切分后的行列表
    """
    text = SPACE_RE.sub(" ", text).strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    lines: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if n - i <= max_len:
            lines.append(text[i:])
            break
        window = text[i : i + max_len]
        cut = -1
        for j in range(len(window) - 1, max(5, int(max_len * 0.55)) - 1, -1):
            if window[j] in BREAK_AFTER:
                cut = j + 1
                break
        if cut < 0:
            cut = max_len
            while cut > 6 and i + cut < n and text[i + cut] in BREAK_BEFORE:
                cut -= 1
        lines.append(text[i : i + cut])
        i += cut
    return [s.strip() for s in lines if len(s.strip()) >= 4]


def ui_lines(rng: random.Random, corpus: dict[str, list[str]]) -> list[str]:
    """生成题库界面里的固定短句。

    :param rng: 随机数生成器
    :param corpus: load_corpus 的结果
    :returns: 界面文案列表
    """
    qid = rng.randint(1, 1291)
    qtype = rng.choice(corpus["题型"] or list(TYPE_NAMES))
    category = rng.choice(corpus["分类"] or ["法治思想"])
    pct = rng.randint(60, 99)
    return [
        "关闭 Esc",
        f"题号 Q{qid:04d}",
        f"{qtype} · {category}",
        f"答案：{rng.choice('ABCD')}",
        "解析",
        f"关键词匹配 {pct}%",
        f"共 {rng.randint(1200, 1300)} 道题",
        "已复制到剪贴板",
        "未找到匹配的题目",
        "复制本题",
        "查询答案",
        "全部题型",
        "上一题",
        "下一题",
        "输入题目或关键词",
    ]


def sample_texts(
    corpus: dict[str, list[str]],
    charset: set[str],
    rng: random.Random,
    count: int,
    max_len: int = 25,
) -> list[str]:
    """从题库采样去重后的行文本。

    :param corpus: load_corpus 的结果
    :param charset: 字符字典
    :param rng: 随机数生成器
    :param count: 目标条数
    :param max_len: 单行最大字数
    :returns: 采样到的行文本列表
    """
    buckets = [
        (corpus["题目"], 0.42),
        (corpus["选项"], 0.26),
        (corpus["解析"], 0.22),
        (corpus["分类"], 0.05),
        (list(TYPE_NAMES), 0.02),
        (ui_lines(rng, corpus), 0.03),
    ]
    keys = [b[0] for b in buckets]
    weights = [b[1] for b in buckets]
    out: list[str] = []
    seen: set[str] = set()
    guard = 0
    while len(out) < count and guard < count * 40:
        guard += 1
        bucket = rng.choices(keys, weights=weights, k=1)[0]
        if not bucket:
            continue
        raw = rng.choice(bucket)
        if rng.random() < 0.22 and len(raw) > max_len:
            start = rng.randint(0, max(0, len(raw) - max_len - 1))
            raw = raw[start : start + rng.randint(6, max_len)]
        lines = split_into_lines(raw, max_len)
        if not lines:
            continue
        line = rng.choice(lines)
        if len(line) > max_len or not all(c in charset for c in line):
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def load_font_paths() -> list[str]:
    """列出本机可用的中文字体。

    :returns: 字体文件绝对路径列表
    :raises FileNotFoundError: 字体目录下没有候选字体时
    """
    paths = [str(FONT_DIR / name) for name in FONT_CANDIDATES if (FONT_DIR / name).exists()]
    if not paths:
        raise FileNotFoundError(f"{FONT_DIR} 下没有找到可用中文字体")
    return paths


def _text_size(font: ImageFont.FreeTypeFont, text: str) -> tuple[int, int, int, int]:
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    return probe.textbbox((0, 0), text, font=font)


def _theme(rng: random.Random) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if rng.random() < 0.07:
        return rng.choice(DARK_THEMES)
    return rng.choice(LIGHT_THEMES)


def add_watermark(
    img: Image.Image,
    rng: random.Random,
    fonts: list[str],
    *,
    alpha: tuple[int, int] = (26, 58),
) -> Image.Image:
    """叠加半透明斜向水印，水印不进入标注。

    :param img: 底图
    :param rng: 随机数生成器
    :param fonts: 候选字体
    :param alpha: 水印透明度范围
    :returns: 叠加水印后的图像
    """
    size = int(max(12, min(76, img.height * rng.uniform(0.7, 1.2))))
    font = ImageFont.truetype(rng.choice(fonts), size)
    l, t, r, b = _text_size(font, WATERMARK)
    tw, th = r - l, b - t
    layer = Image.new("RGBA", (tw + 8, th + 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    color = (110, 116, 128, rng.randint(*alpha))
    draw.text((4 - l, 4 - t), WATERMARK, font=font, fill=color)
    layer = layer.rotate(rng.uniform(-18, 18), expand=True, resample=Image.BICUBIC)
    x = rng.randint(-layer.width // 3, max(1, img.width - layer.width // 3))
    y = (img.height - layer.height) // 2 + rng.randint(-4, 4)
    out = img.convert("RGBA")
    out.paste(layer, (x, y), layer)
    return out.convert("RGB")


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """模拟截图常见的亮度/对比度抖动、轻微旋转和模糊。

    :param img: 输入图像
    :param rng: 随机数生成器
    :returns: 退化后的图像
    """
    bg = img.getpixel((0, 0))
    if rng.random() < 0.35:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.86, 1.14))
    if rng.random() < 0.35:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.86, 1.22))
    if rng.random() < 0.30:
        img = img.rotate(rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, fillcolor=bg)
    if rng.random() < 0.22:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    return img


LINE_SIZES = [20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 44]
LINE_WEIGHTS = [4, 8, 12, 16, 18, 16, 12, 8, 5, 2, 1]


def render_line(
    text: str,
    rng: random.Random,
    fonts: list[str],
    *,
    watermark_prob: float = 0.5,
) -> Image.Image:
    """渲染一行文字的识别行图。

    :param text: 行文本
    :param rng: 随机数生成器
    :param fonts: 候选字体
    :param watermark_prob: 叠加水印的概率
    :returns: 行图
    """
    size = rng.choices(LINE_SIZES, weights=LINE_WEIGHTS, k=1)[0]
    font = ImageFont.truetype(rng.choice(fonts), size)
    l, t, r, b = _text_size(font, text)
    tw, th = r - l, b - t
    pad_x = rng.randint(8, 22)
    pad_y = rng.randint(5, 14)
    width = max(8, tw + 2 * pad_x + rng.randint(0, 16))
    height = max(8, th + 2 * pad_y + rng.randint(0, 6))
    bg, fg = _theme(rng)
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x - l, pad_y - t), text, font=font, fill=fg)

    if rng.random() < watermark_prob:
        img = add_watermark(img, rng, fonts)
    img = degrade(img, rng)

    if rng.random() < 0.12:
        small = img.resize((max(1, img.width // 2), max(1, img.height // 2)), Image.BILINEAR)
        img = small.resize((small.width * 2, small.height * 2), Image.BICUBIC)
    return img


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    """按像素宽度折行，优先在标点处断开。

    :param draw: ImageDraw 对象
    :param text: 原文
    :param font: 字体
    :param max_width: 单行最大像素宽度
    :returns: 折行后的文本列表
    """
    lines: list[str] = []
    current = ""
    for ch in text:
        candidate = current + ch
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
            continue
        cut = -1
        for j in range(len(current) - 1, max(4, len(current) // 2) - 1, -1):
            if current[j] in BREAK_AFTER:
                cut = j + 1
                break
        if cut > 0:
            lines.append(current[:cut])
            current = current[cut:] + ch
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


@dataclass
class TextBox:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int

    def points(self) -> list[list[int]]:
        """返回四点坐标。

        :returns: 顺时针四点坐标
        """
        return [
            [self.x0, self.y0],
            [self.x1, self.y0],
            [self.x1, self.y1],
            [self.x0, self.y1],
        ]


@dataclass
class Page:
    image: Image.Image
    boxes: list[TextBox] = field(default_factory=list)


def _draw_boxed(
    draw: ImageDraw.ImageDraw,
    page_boxes: list[TextBox],
    text: str,
    font: ImageFont.FreeTypeFont,
    x: int,
    y: int,
    fill: tuple[int, int, int],
) -> tuple[int, int]:
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    draw.text((x, y), text, font=font, fill=fill)
    if text.strip():
        page_boxes.append(TextBox(text=text.strip(), x0=int(l), y0=int(t), x1=int(r), y1=int(b)))
    return int(r - l), int(b - t)


def render_page(
    rng: random.Random,
    fonts: list[str],
    corpus: dict[str, list[str]],
    charset: set[str],
) -> Page:
    """渲染一整页题库界面截图，并记录每行文字的外接框。

    :param rng: 随机数生成器
    :param fonts: 候选字体
    :param corpus: load_corpus 的结果
    :param charset: 字符字典，用于过滤标注
    :returns: 页面图像与文本框列表
    """
    width = rng.choice([1120, 1200, 1280, 1360])
    height = rng.choice([820, 880, 940, 1000])
    bg, fg = _theme(rng)
    if bg in [t[0] for t in DARK_THEMES]:
        bg, fg = LIGHT_THEMES[0]

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    boxes: list[TextBox] = []

    title_font = ImageFont.truetype(rng.choice(fonts), rng.choice([30, 32, 34]))
    meta_font = ImageFont.truetype(rng.choice(fonts), rng.choice([22, 24, 26]))
    body_size = rng.choice([26, 28, 30, 32])
    body_font = ImageFont.truetype(rng.choice(fonts), body_size)
    small_font = ImageFont.truetype(rng.choice(fonts), rng.choice([22, 24, 26]))

    margin = rng.choice([56, 64, 72])
    max_width = width - 2 * margin
    muted = (98, 125, 152)

    title = "2026年法治理论知识学习考试"
    l, t, r, b = draw.textbbox((0, 0), title, font=title_font)
    _draw_boxed(draw, boxes, title, title_font, (width - (r - l)) // 2, rng.randint(22, 40), fg)
    draw.line([(margin, 92), (width - margin, 92)], fill=(217, 226, 236), width=2)

    qid = rng.randint(1, 1291)
    qtype = rng.choice(corpus["题型"] or list(TYPE_NAMES))
    category = rng.choice(corpus["分类"] or ["习近平法治思想"])
    _draw_boxed(draw, boxes, f"题号 Q{qid:04d}", meta_font, margin, 116, muted)
    meta = f"{qtype} · {category}"
    l, t, r, b = draw.textbbox((0, 0), meta, font=meta_font)
    _draw_boxed(draw, boxes, meta, meta_font, width - margin - (r - l), 116, muted)

    stem = rng.choice(corpus["题目"]) if corpus["题目"] else "请选择正确答案"
    y = 168
    if rng.random() < 0.08:
        _draw_boxed(draw, boxes, "【题目】", meta_font, margin, y, muted)
        y += 40
    for line in wrap_text(draw, stem, body_font, max_width):
        _draw_boxed(draw, boxes, line, body_font, margin, y, fg)
        y += int(body_size * rng.uniform(1.55, 1.75))

    y += rng.randint(8, 24)
    option_pool = corpus["选项"]
    if option_pool:
        start = rng.randrange(0, max(1, len(option_pool) - 4))
        options = option_pool[start : start + 4]
    else:
        options = ["坚持科学立法", "坚持以人民为中心", "坚持依法行政", "坚持司法独立"]
    for idx, opt in enumerate(options):
        letter = "ABCD"[idx]
        opt_lines = wrap_text(draw, f"{letter}. {opt}", body_font, max_width)
        for opt_line in opt_lines:
            _draw_boxed(draw, boxes, opt_line, body_font, margin, y, fg)
            y += int(body_size * rng.uniform(1.5, 1.7))

    y += rng.randint(10, 28)
    draw.line([(margin, y), (width - margin, y)], fill=(217, 226, 236), width=1)
    y += 16
    answer = rng.choice("ABCD")
    _draw_boxed(draw, boxes, f"答案：{answer}", body_font, margin, y, (11, 107, 58))
    y += int(body_size * 1.6)
    explanation = rng.choice(corpus["解析"]) if corpus["解析"] else "本题考查法治理论知识。"
    for line in wrap_text(draw, explanation, small_font, max_width):
        if y > height - 90:
            break
        _draw_boxed(draw, boxes, line, small_font, margin, y, fg)
        y += int(small_font.size * rng.uniform(1.5, 1.7))

    footer_y = height - 58
    _draw_boxed(draw, boxes, "关闭 Esc", meta_font, margin, footer_y, muted)
    l, t, r, b = draw.textbbox((0, 0), "复制本题  ·  查询答案", font=meta_font)
    _draw_boxed(draw, boxes, "复制本题  ·  查询答案", meta_font, width - margin - (r - l), footer_y, muted)

    for _ in range(rng.choice([1, 1, 2])):
        img = add_watermark(img, rng, fonts)
    img = degrade(img, rng)

    valid = [b for b in boxes if b.text and all(c in charset for c in b.text)]
    return Page(image=img, boxes=valid)


def save_jpeg(img: Image.Image, path: Path, rng: random.Random) -> None:
    """以随机质量保存 JPEG。

    :param img: 图像
    :param path: 目标路径
    :param rng: 随机数生成器
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    quality = rng.choice([62, 70, 78, 85, 90, 95])
    img.save(path, format="JPEG", quality=quality, subsampling=rng.choice([0, 1, 2]))


def make_preview(paths: list[Path], out_path: Path, *, cols: int = 3, thumb: int = 520) -> None:
    """把若干张图拼成预览大图。

    :param paths: 图片路径列表
    :param out_path: 预览图输出路径
    :param cols: 每行张数
    :param thumb: 缩略宽度
    """
    if not paths:
        return
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        ratio = thumb / max(1, im.width)
        im = im.resize((thumb, max(1, int(im.height * ratio))), Image.BICUBIC)
        imgs.append(im)
    rows = (len(imgs) + cols - 1) // cols
    row_h = [0] * rows
    for i, im in enumerate(imgs):
        row_h[i // cols] = max(row_h[i // cols], im.height)
    sheet = Image.new("RGB", (cols * thumb, sum(row_h)), (245, 245, 245))
    y = 0
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if i < len(imgs):
                sheet.paste(imgs[i], (c * thumb, y))
        y += row_h[r]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
