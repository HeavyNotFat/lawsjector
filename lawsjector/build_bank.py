from __future__ import annotations

import argparse
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from .bank import (
    Question,
    WATERMARK,
    fix_source_noise,
    normalize_radicals,
    polish_text,
    smart_join,
    strip_watermark,
)
from .paths import CLEAN_DOCX, SOURCE_DOCX, TXT_PATH

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

_ALT_CONTENT_RE = re.compile(r"<mc:AlternateContent\b.*?</mc:AlternateContent>", re.S)
_DRAWING_RE = re.compile(r"<w:drawing\b.*?</w:drawing>", re.S)
_PICT_RE = re.compile(r"<w:pict\b.*?</w:pict>", re.S)
_SHAPE_RE = re.compile(r"<v:shape\b.*?</v:shape>", re.S)

@dataclass(slots=True)
class Para:
    index: int
    text: str  # 归一化后的原文，保留 \t
    check: str  # 用于判断的干净文本
    style: str
    eff: int | None  # 首行有效缩进（twips）

def _remove_blocks_with_watermark(xml: str) -> str:
    """删除包含水印文字的形状/文本框，保留其它图形。

    :param xml: document.xml 文本
    :returns: 清洗后的 XML
    """

    def repl(match: re.Match[str]) -> str:
        block = match.group(0)
        return "" if WATERMARK in block else block

    xml = _ALT_CONTENT_RE.sub(repl, xml)
    xml = _DRAWING_RE.sub(repl, xml)
    xml = _PICT_RE.sub(repl, xml)
    xml = _SHAPE_RE.sub(repl, xml)
    return xml

def _clean_document_xml(xml: str) -> str:
    cleaned = _remove_blocks_with_watermark(xml)
    if WATERMARK in cleaned:
        # 极少见：水印文字被拆到多个 run 里。至少保证正文提取干净；
        # 同时把剩余的 VML textpath 字符串替换为空。
        cleaned = re.sub(
            r'(<v:textpath\b[^>]*\bstring=")[^"]*(")',
            r"\1\2",
            cleaned,
        )
    if WATERMARK in cleaned:
        raise ValueError("无法完全删除文档中的水印文字，请检查源 DOCX。")
    ET.fromstring(cleaned.encode("utf-8"))
    return cleaned

def clean_docx(src: Path, dest: Path) -> str:
    """写出无形状水印的 DOCX。

    :param src: 源 DOCX
    :param dest: 输出 DOCX
    :returns: 清洗后的 document.xml
    """
    with zipfile.ZipFile(src, "r") as zin:
        try:
            xml = zin.read("word/document.xml").decode("utf-8")
        except KeyError as exc:
            raise ValueError("不是有效的 DOCX：缺少 word/document.xml") from exc
        cleaned_xml = _clean_document_xml(xml)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == "word/document.xml":
                    data = cleaned_xml.encode("utf-8")
                zout.writestr(info, data)
    return cleaned_xml

def _int_attr(element: ET.Element | None, name: str) -> int:
    if element is None:
        return 0
    value = element.get(W + name)
    if value is None and name == "left":
        value = element.get(W + "start")
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0

def _paragraph_style(p: ET.Element) -> str:
    ppr = p.find(W + "pPr")
    if ppr is None:
        return "Normal"
    style = ppr.find(W + "pStyle")
    if style is None:
        return "Normal"
    return style.get(W + "val") or "Normal"

def _paragraph_effective_indent(p: ET.Element) -> int | None:
    ppr = p.find(W + "pPr")
    if ppr is None:
        return None
    ind = ppr.find(W + "ind")
    if ind is None:
        return None
    left = _int_attr(ind, "left")
    hanging = _int_attr(ind, "hanging")
    first = _int_attr(ind, "firstLine")
    return left - hanging + first

def _paragraph_raw_text(p: ET.Element) -> str:
    parts: list[str] = []
    for node in p.iter():
        tag = node.tag
        if tag == W + "t":
            parts.append(node.text or "")
        elif tag == W + "tab":
            parts.append("\t")
        elif tag in (W + "br", W + "cr"):
            parts.append(" ")
    text = "".join(parts)
    if not text:
        return ""
    text = normalize_radicals(text)
    text = strip_watermark(text)
    text = fix_source_noise(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("\xa0", " ").replace("\u3000", " ")
    text = re.sub(r"[ \f\v]+", " ", text)
    return text.strip()

def extract_paragraphs(document_xml: str) -> list[Para]:
    root = ET.fromstring(document_xml.encode("utf-8"))
    body = root.find(W + "body")
    if body is None:
        raise ValueError("DOCX 中没有 body")

    def iter_body_paragraphs(element: ET.Element):
        """只遍历正文段落，跳过文本框/图形内部的段落。

        :param element: XML 根节点
        :returns: 正文段落迭代器
        """
        for child in element:
            if child.tag in (W + "drawing", W + "pict"):
                continue
            if child.tag == W + "p":
                yield child
                continue
            yield from iter_body_paragraphs(child)

    paragraphs: list[Para] = []
    idx = 0
    for p in iter_body_paragraphs(body):
        raw = _paragraph_raw_text(p)
        idx += 1
        if not raw:
            continue
        paragraphs.append(
            Para(
                index=idx,
                text=raw,
                check=polish_text(raw, blank=" "),
                style=_paragraph_style(p),
                eff=_paragraph_effective_indent(p),
            )
        )
    return paragraphs

def _is_section(text: str) -> str | None:
    if "单选题" in text and ("共" in text or "题" in text):
        return "单选题"
    if "多选题" in text and ("共" in text or "题" in text):
        return "多选题"
    if "判断题" in text and ("共" in text or "题" in text):
        return "判断题"
    return None

def _is_category(text: str) -> bool:
    return (
        text.startswith("【")
        and text.endswith("】")
        and not text.startswith(("【答案】", "【解析】"))
    )

def _is_answer(text: str) -> bool:
    return text.startswith("【答案】")

def _is_explanation(text: str) -> bool:
    return text.startswith("【解析】")

_EXPLANATION_TAIL_RE = re.compile(
    r"(?:综上|本题答案|故本题|故选|解析\s*[；;：:]|"
    r"因此\s*[A-E]{1,5}\s*(?:项)?\s*正确|"
    r"选项\s*[A-E]|[A-E]\s*项(?:不|错误|正确|属于|中))"
)

def _looks_like_explanation_tail(text: str) -> bool:
    """识别被排版成 ListParagraph 的解析尾巴。

    :param text: 段落文本
    :returns: 是否像解析尾部
    """
    return bool(_EXPLANATION_TAIL_RE.search(text))

def _find_stem_start(
    paras: list[Para],
    stem_end: int,
    floor: int,
) -> int:
    """从选项前一行开始往前找题干。

    :param paras: 段落列表
    :param stem_end: 选项起始下标
    :param floor: 向前搜索的下界
    :returns: 题干起始下标
    """
    j = stem_end
    seen_list = False
    collected = 0
    while j > floor:
        p = paras[j]
        t = p.check
        if not t:
            j -= 1
            continue
        if _is_answer(t) or _is_explanation(t) or _is_section(t) or _is_category(t):
            break
        if seen_list and p.style != "ListParagraph":
            break
        if seen_list and p.style == "ListParagraph" and _looks_like_explanation_tail(t):
            break
        # 没有 ListParagraph 时，最多向上收 4 段，避免吞掉上一题解析。
        if collected >= 4 and not seen_list:
            break
        j -= 1
        collected += 1
        if p.style == "ListParagraph" and (p.eff == 27 or p.eff is None):
            seen_list = True
    return j + 1

def _group_options(raw: list[Para]) -> list[str]:
    """把跨段的选项合并回一条。

    :param raw: 原始选项段落
    :returns: 合并后的选项列表
    """
    if not raw:
        return []
    has_list = any(p.style == "ListParagraph" for p in raw)
    groups: list[list[Para]] = []
    if has_list:
        for p in raw:
            if p.style == "ListParagraph" or not groups:
                groups.append([p])
            else:
                groups[-1].append(p)
    else:
        groups = [[p] for p in raw]

    options: list[str] = []
    for group in groups:
        text = ""
        for p in group:
            piece = polish_text(p.text, blank="____")
            text = smart_join(text, piece)
        text = re.sub(r"^[A-Ea-e]\s*[.．、]\s*", "", text).strip()
        if text:
            options.append(text)
    return options

def _parse_structure(paras: list[Para]) -> tuple[list[Question], dict[str, int]]:
    section_positions: list[tuple[int, str]] = []
    answer_positions: list[int] = []
    category_positions: list[tuple[int, str]] = []

    for i, p in enumerate(paras):
        text = p.check
        sec = _is_section(text)
        if sec:
            section_positions.append((i, sec))
        if _is_answer(text):
            answer_positions.append(i)
        elif _is_category(text):
            category_positions.append((i, text[1:-1].strip()))

    if not answer_positions:
        raise ValueError("没有解析到任何【答案】行，DOCX 结构可能已改变。")

    questions: list[Question] = []
    meta: list[dict[str, object]] = []
    prev_answer = -1
    section_idx = -1
    para_pos = {id(p): i for i, p in enumerate(paras)}

    for n, answer_i in enumerate(answer_positions, start=1):
        while section_idx + 1 < len(section_positions) and section_positions[section_idx + 1][0] < answer_i:
            section_idx += 1
        qtype = section_positions[section_idx][1] if section_idx >= 0 else "单选题"

        j = answer_i - 1
        raw_options: list[Para] = []
        while j > prev_answer:
            p = paras[j]
            if p.eff == 307:
                raw_options.append(p)
                j -= 1
            else:
                break
        raw_options.reverse()

        options = _group_options(raw_options)
        embedded_explanation = ""
        if options:
            # 极少数源文件把某题解析误排成一个选项。
            for oi, opt in enumerate(options):
                if re.match(r"^解析\s*[；;：:]", opt):
                    embedded_explanation = re.sub(r"^解析\s*[；;：:]\s*", "", opt)
                    options.pop(oi)
                    break

        opt_start_i = para_pos[id(raw_options[0])] if raw_options else answer_i
        stem_end_i = opt_start_i - 1 if raw_options else answer_i - 1
        floor = max(prev_answer, section_positions[section_idx][0] if section_idx >= 0 else -1)
        stem_start_i = _find_stem_start(paras, stem_end_i, floor)

        stem = ""
        for p in paras[stem_start_i : stem_end_i + 1]:
            if p.eff == 307:
                continue
            stem = smart_join(stem, polish_text(p.text, blank="____"))
        if not stem:
            stem = "（题干缺失）"

        answer_text = polish_text(paras[answer_i].text, blank=" ")
        answer_text = re.sub(r"^【答案】\s*", "", answer_text).strip()

        q = Question(
            number=n,
            qid=f"Q{n:04d}",
            qtype=qtype,
            category="",
            stem=stem,
            options=tuple(options),
            answer=answer_text,
            explanation=embedded_explanation,
        )
        questions.append(q)
        meta.append(
            {
                "answer_i": answer_i,
                "start_i": stem_start_i,
                "section_i": section_positions[section_idx][0] if section_idx >= 0 else -1,
                "embedded": embedded_explanation,
            }
        )
        prev_answer = answer_i

    # 分类：取本节内、题目开始前的最后一个【分类】标题
    cat_indices = [i for i, _ in category_positions]
    for q, m in zip(questions, meta):
        start_i = int(m["start_i"])
        section_i = int(m["section_i"])
        chosen = ""
        for idx, name in category_positions:
            if section_i < idx < start_i:
                chosen = name
        q.category = chosen

    # 解析：从本题答案之后，到下一题题干开始之前
    for qi, (q, m) in enumerate(zip(questions, meta)):
        answer_i = int(m["answer_i"])
        if qi + 1 < len(meta):
            end_i = int(meta[qi + 1]["start_i"])
        else:
            end_i = len(paras)

        parts: list[str] = []
        for p in paras[answer_i + 1 : end_i]:
            t = p.check
            if not t or _is_answer(t) or _is_section(t) or _is_category(t) or _is_explanation(t):
                if _is_explanation(t):
                    t = re.sub(r"^【解析】\s*", "", t)
                    if t:
                        parts.append(t)
                continue
            parts.append(polish_text(p.text, blank=" "))

        explanation = ""
        for part in parts:
            explanation = smart_join(explanation, part)
        explanation = re.sub(r"^【解析】\s*", "", explanation).strip()

        if not explanation and m["embedded"]:
            explanation = str(m["embedded"])
        q.explanation = explanation

    # 备注：选项不完整 / 答案字母超出选项
    for q in questions:
        notes: list[str] = []
        if q.qtype in ("单选题", "多选题") and len(q.options) < 4:
            notes.append(f"原题仅提供{len(q.options)}个选项")
        if q.qtype in ("单选题", "多选题"):
            max_letter = "ABCDE"[max(0, len(q.options) - 1)]
            missing = [ch for ch in q.answer_letters if ch > max_letter]
            if missing:
                notes.append("原题缺少选项" + "、".join(missing))
        elif q.qtype == "判断题" and q.answer not in ("正确", "错误"):
            notes.append("答案不是“正确/错误”")
        q.note = "；".join(notes)

    counts: dict[str, int] = {}
    for q in questions:
        counts[q.qtype] = counts.get(q.qtype, 0) + 1
    return questions, counts

def write_txt(questions: list[Question], path: Path, source_name: str) -> None:
    counts: dict[str, int] = {}
    for q in questions:
        counts[q.qtype] = counts.get(q.qtype, 0) + 1
    lines: list[str] = [
        "# ============================================================",
        "# 2026年法治理论知识学习考试综合题库（纯净文字版）",
        f"# 共 {len(questions)} 题 | 单选 {counts.get('单选题', 0)} | "
        f"多选 {counts.get('多选题', 0)} | 判断 {counts.get('判断题', 0)}",
        f"# 水印已去除；来源：{source_name}",
        "# 字段固定为：题号 / 题型 / 分类 / 题目 / 选项 / 答案 / 解析 / 备注",
        "# 可直接用记事本查看，也可由 app.py 自动读取。",
        "# ============================================================",
        "",
    ]
    for q in questions:
        lines.append(f"===== {q.qid} =====")
        lines.append(f"【题型】{q.qtype}")
        if q.category:
            lines.append(f"【分类】{q.category}")
        lines.append(f"【题目】{q.stem}")
        for i, opt in enumerate(q.options):
            letter = "ABCDE"[i]
            lines.append(f"【选项】{letter}. {opt}")
        lines.append(f"【答案】{q.answer}")
        lines.append(f"【解析】{q.explanation}")
        if q.note:
            lines.append(f"【备注】{q.note}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

def build_all(
    source: str | Path | None = None,
    *,
    clean_path: str | Path | None = None,
    txt_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    src = Path(source).resolve() if source else SOURCE_DOCX
    if not src.exists():
        raise FileNotFoundError(f"找不到源 DOCX：{src}")

    clean = Path(clean_path).resolve() if clean_path else CLEAN_DOCX
    txt = Path(txt_path).resolve() if txt_path else TXT_PATH

    if verbose:
        print(f"[1/3] 读取并去除水印：{src.name}")
    document_xml = clean_docx(src, clean)

    if verbose:
        print(f"[2/3] 解析题目结构 ...")
    paras = extract_paragraphs(document_xml)
    questions, counts = _parse_structure(paras)

    if len(questions) != sum(counts.values()) or not questions:
        raise ValueError("题目解析结果为空或不完整。")
    expected = 469 + 449 + 373
    if len(questions) != expected:
        print(
            f"提示：解析到 {len(questions)} 题，源文件标称 {expected} 题。",
            file=sys.stderr,
        )

    if verbose:
        print(f"[3/3] 写出 TXT：{txt.name}")
    write_txt(questions, txt, src.name)

    if verbose:
        print()
        print(f"完成：{len(questions)} 题")
        for name in ("单选题", "多选题", "判断题"):
            print(f"  {name}: {counts.get(name, 0)}")
        print(f"  去水印 DOCX: {clean}")
        print(f"  题库 TXT  : {txt}")

    return {
        "questions": questions,
        "counts": counts,
        "clean_docx": clean,
        "txt": txt,
    }

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 DOCX 生成去水印题库 TXT")
    parser.add_argument("--docx", help="源 DOCX 路径")
    parser.add_argument("--clean-docx", help="输出去水印 DOCX 路径")
    parser.add_argument("--txt", help="输出题库 TXT 路径")
    parser.add_argument("--quiet", action="store_true", help="不打印进度")
    args = parser.parse_args(argv)
    try:
        build_all(
            args.docx,
            clean_path=args.clean_docx,
            txt_path=args.txt,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"构建失败：{exc}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
