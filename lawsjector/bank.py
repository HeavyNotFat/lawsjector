from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Sequence

WATERMARK = "2026年度法治理论知识考试复习题"

# PDF 转 DOCX 时，某些字体会把汉字拆成“康熙部首/兼容字符”。
_RADICAL_RANGES = ((0x2E80, 0x2FDF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F))
_RADICAL_TRANS = {
    code: unicodedata.normalize("NFKC", chr(code))
    for start, end in _RADICAL_RANGES
    for code in range(start, end + 1)
    if len(unicodedata.normalize("NFKC", chr(code))) == 1
}
# Unicode 的 CJK Radicals Supplement 里有一小部分字符没有 NFKC 分解，
# 但在中文 PDF 转换件中会作为正常汉字出现。
_RADICAL_TRANS.update(
    {
        ord("⺠"): "民",
        ord("⻝"): "食",
        ord("⻛"): "风",
        ord("⻜"): "飞",
        ord("⻢"): "马",
        ord("⻥"): "鱼",
        ord("⻦"): "鸟",
        ord("⻩"): "黄",
        ord("⻬"): "齐",
        ord("⻮"): "齿",
        ord("⻰"): "龙",
        ord("⻳"): "龟",
        ord("⻄"): "西",
        ord("⻅"): "见",
        ord("⻉"): "贝",
        ord("⻋"): "车",
        ord("⻔"): "门",
        ord("⻘"): "青",
        ord("⻙"): "韦",
        ord("⻚"): "页",
        ord("⻑"): "長",
        ord("⻓"): "长",
    }
)

# 常见转换/OCR 噪声。只修正证据明确的词组，避免改坏法律原文。
_PHRASE_FIXES = (
    ("噪👉", "噪声"),
    ("👉明", "声明"),
    ("出又", "出口"),
    ("登记户又", "登记户口"),
    ("进又", "进口"),
    ("端又", "端口"),
    ("接又", "接口"),
    ("排污又", "排污口"),
    ("对又", "对口"),
    ("人又", "人口"),
    ("溪又", "溪口"),
    ("沟又", "沟口"),
    ("又头", "口头"),
    ("又述", "口述"),
    ("入又", "入口"),
)

_WATERMARK_RE = re.compile(
    r"2026\s*年\s*度\s*法\s*治\s*理\s*论\s*知\s*识\s*考\s*试\s*复\s*习\s*题"
)
_WS_RE = re.compile(r"[ \t\f\v]+")
_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_CJK_SPACE_RE = re.compile(r"(?<=[\u3400-\u9fff])[ \t]+(?=[\u3400-\u9fff])")
_PUNCT_SPACE_RE = re.compile(r"[ \t]+([，。；：！？、）】》”’])")
_OPEN_SPACE_RE = re.compile(r"([（【《“‘])[ \t]+")
_MULTI_BLANK_RE = re.compile(r"(?:_{2,}\s*){2,}")

_RECORD_RE = re.compile(r"^\s*=+\s*(Q\d+)\s*=+\s*$", re.I)
_FIELD_RE = re.compile(r"^【(题型|分类|题目|选项|答案|解析|备注)】(.*)$")
_OPT_LABEL_RE = re.compile(r"^[A-Ea-e]\s*[.．、]\s*")
_OPTION_SPLIT_RE = re.compile(r"(?:^|[\s。？?！!])([A-Ea-e])\s*[.．、]\s*", re.M)

def normalize_radicals(text: str) -> str:
    """把康熙部首等兼容字符还原成正常汉字。

    :param text: 原始文本
    :returns: 归一化后的文本
    """
    return text.translate(_RADICAL_TRANS)

def strip_watermark(text: str) -> str:
    return _WATERMARK_RE.sub("", text)

def fix_source_noise(text: str) -> str:
    for old, new in _PHRASE_FIXES:
        text = text.replace(old, new)
    return text

def polish_text(text: str, *, blank: str = " ") -> str:
    """清洗用于显示的文字。

    :param text: 原始文本
    :returns: 清洗后的文本
    """
    if not text:
        return ""
    text = normalize_radicals(text)
    text = strip_watermark(text)
    text = fix_source_noise(text)
    text = _INVISIBLE_RE.sub("", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("\xa0", " ").replace("\u3000", " ")
    if "\t" in text:
        text = text.replace("\t", blank)
    text = _WS_RE.sub(" ", text)
    text = _CJK_SPACE_RE.sub("", text)
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    text = _OPEN_SPACE_RE.sub(r"\1", text)
    text = text.replace(" ；", "；").replace(" ，", "，")
    if "____" in text:
        text = _MULTI_BLANK_RE.sub("____", text)
    return text.strip()

def smart_join(left: str, right: str) -> str:
    """把被 PDF 转换拆成多段的同一句话拼回去。

    :param left: 左侧文本
    :param right: 右侧文本
    :returns: 拼接结果
    """
    left, right = left.strip(), right.strip()
    if not left:
        return right
    if not right:
        return left
    if re.search(r"[\u3400-\u9fff]$", left) and re.match(r"[\u3400-\u9fff]", right):
        return left + right
    if re.search(r"[，。；：！？、]$", left) and re.match(r"[\u3400-\u9fff]", right):
        return left + right
    if left.endswith(("-", "—", "/", "（", "【", "《", "“", "‘")):
        return left + right
    if right.startswith(("，", "。", "；", "：", "！", "？", "、", "）", "】", "》", "”", "’")):
        return left + right
    if re.search(r"[A-Za-z0-9]$", left) and re.match(r"[\u3400-\u9fff]", right):
        return left + right
    return left + " " + right

def normalize_key(text: str) -> str:
    """检索用归一化：去标点/空白、统一全半角、转小写。

    :param text: 原始文本
    :returns: 归一化后的键
    """
    text = polish_text(text, blank=" ")
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if ch.isalnum())

def bigrams(text: str) -> frozenset[str]:
    if len(text) < 2:
        return frozenset()
    return frozenset(text[i : i + 2] for i in range(len(text) - 1))

def _strip_opt_label(text: str) -> str:
    return _OPT_LABEL_RE.sub("", text, count=1).strip()

def query_variants(raw: str) -> list[str]:
    """从粘贴内容生成若干检索式。

    :param raw: 粘贴或输入的内容
    :returns: 检索式列表
    """
    raw = polish_text(raw, blank=" ")
    if not raw:
        return []

    cleaned: list[str] = []

    def add(value: str) -> None:
        key = normalize_key(value)
        if len(key) >= 1 and key not in cleaned:
            cleaned.append(key)

    add(re.sub(r"^\s*(?:题目|题干|查询|问)\s*[：:]\s*", "", raw))
    add(re.sub(r"【答案】.*$", "", raw, flags=re.S))
    add(raw)

    m = _OPTION_SPLIT_RE.search(raw)
    if m and m.start() >= 8:
        add(raw[: m.start()])

    lines = [ln.strip() for ln in raw.splitlines() if len(normalize_key(ln)) >= 2]
    if lines:
        add(max(lines, key=lambda s: len(normalize_key(s))))

    cleaned.sort(key=len, reverse=True)
    return cleaned[:4]

def query_terms(raw: str) -> list[str]:
    """空格/竖线分隔的关键词查询。

    :param raw: 用户输入
    :returns: 关键词列表
    """
    if not raw:
        return []
    # 注意：不能用 polish_text，否则中文之间的空格会被当作排版空格删掉。
    text = normalize_radicals(strip_watermark(fix_source_noise(raw)))
    text = _INVISIBLE_RE.sub("", text)
    text = text.replace("\xa0", " ").replace("\u3000", " ").replace("\n", " ")
    parts = re.split(r"[\s|｜]+", text)
    terms: list[str] = []
    for part in parts:
        key = normalize_key(part)
        if len(key) >= 2 and key not in terms:
            terms.append(key)
    return terms[:6]

@dataclass(slots=True)
class Question:
    number: int
    qid: str
    qtype: str
    category: str
    stem: str
    options: tuple[str, ...]
    answer: str
    explanation: str = ""
    note: str = ""

    key: str = field(default="", repr=False)
    full_key: str = field(default="", repr=False)
    all_grams: frozenset[str] = field(default=frozenset(), repr=False)

    @property
    def answer_letters(self) -> set[str]:
        if self.qtype == "判断题":
            return set()
        return set(re.findall(r"[A-E]", self.answer.upper()))

    @property
    def correct_indexes(self) -> set[int]:
        letters = "ABCDE"
        return {letters.index(ch) for ch in self.answer_letters if ch in letters}

    def option_letter(self, index: int) -> str:
        return "ABCDE"[index] if 0 <= index < 5 else "?"

    def answer_display(self) -> str:
        if self.qtype == "判断题":
            return self.answer.strip() or "（未提供）"
        letters = "".join(sorted(self.answer_letters))
        if not letters:
            return self.answer.strip() or "（未提供）"
        return letters

    def plain_text(self) -> str:
        lines = [
            f"【{self.qtype}】{self.category}".strip(),
            self.stem,
            "",
        ]
        for i, opt in enumerate(self.options):
            mark = "  ✓" if i in self.correct_indexes else ""
            lines.append(f"{self.option_letter(i)}. {opt}{mark}")
        if self.options:
            lines.append("")
        lines.append(f"答案：{self.answer_display()}")
        if self.explanation:
            lines.extend(["", "解析：", self.explanation])
        if self.note:
            lines.extend(["", f"备注：{self.note}"])
        return "\n".join(lines).strip()

@dataclass(slots=True)
class SearchResult:
    question: Question
    score: float
    reason: str

    @property
    def percent(self) -> int:
        return max(0, min(100, int(round(self.score * 100))))

class QuestionBank:
    """内存题库 + 字符二元组倒排索引。
    """

    def __init__(self, questions: Sequence[Question]):
        self.questions: list[Question] = list(questions)
        self._index: dict[str, list[int]] = {}
        self._stem_exact: dict[str, list[int]] = {}
        self._full_exact: dict[str, list[int]] = {}
        self._build_index()

    def _build_index(self) -> None:
        index: dict[str, list[int]] = {}
        stem_exact: dict[str, list[int]] = {}
        full_exact: dict[str, list[int]] = {}
        for idx, q in enumerate(self.questions):
            q.key = normalize_key(q.stem)
            option_key = "".join(normalize_key(o) for o in q.options)
            q.full_key = q.key + option_key
            q.all_grams = bigrams(q.key) | bigrams(option_key)
            stem_exact.setdefault(q.key, []).append(idx)
            full_exact.setdefault(q.full_key, []).append(idx)
            for gram in q.all_grams:
                index.setdefault(gram, []).append(idx)
        self._index = index
        self._stem_exact = stem_exact
        self._full_exact = full_exact

    @property
    def count(self) -> int:
        return len(self.questions)

    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for q in self.questions:
            counts[q.qtype] = counts.get(q.qtype, 0) + 1
        return counts

    def search(
        self,
        raw_query: str,
        qtype: str = "全部题型",
        limit: int = 40,
    ) -> tuple[list[SearchResult], float]:
        """返回匹配结果与耗时。

        :param raw_query: 查询文本
        :param qtype: 题型过滤，None 表示全部
        :param limit: 最多返回条数
        :returns: (结果列表, 耗时毫秒)
        """
        started = time.perf_counter()
        variants = query_variants(raw_query)
        if not variants:
            return [], (time.perf_counter() - started) * 1000.0

        type_filter = None if qtype in ("", "全部", "全部题型") else qtype
        allowed = None
        if type_filter:
            allowed = {i for i, q in enumerate(self.questions) if q.qtype == type_filter}
            if not allowed:
                return [], (time.perf_counter() - started) * 1000.0

        terms = query_terms(raw_query)
        keyword_results = self._search_terms(terms, allowed) if len(terms) >= 2 else []

        if len(variants[0]) == 1:
            results = self._search_single_char(variants[0], allowed, limit)
            return results, (time.perf_counter() - started) * 1000.0

        variant_grams = [(v, bigrams(v)) for v in variants]
        counter: dict[int, float] = {}

        exact_candidates: set[int] = set()
        for key, _ in variant_grams:
            for idx in self._stem_exact.get(key, ()):
                if allowed is None or idx in allowed:
                    counter[idx] = counter.get(idx, 0) + 100.0
                    exact_candidates.add(idx)
            for idx in self._full_exact.get(key, ()):
                if allowed is None or idx in allowed:
                    counter[idx] = counter.get(idx, 0) + 50.0

        for key, grams in variant_grams:
            for gram in grams:
                for idx in self._index.get(gram, ()):
                    if allowed is not None and idx not in allowed:
                        continue
                    counter[idx] = counter.get(idx, 0) + 1.0

        if not counter:
            candidates = list(allowed) if allowed is not None else list(range(len(self.questions)))
        else:
            candidates = list(counter)

        def rough(idx: int) -> tuple[float, int]:
            q = self.questions[idx]
            if idx in exact_candidates:
                return (10.0, idx)
            best = 0.0
            for key, grams in variant_grams:
                if key and (key in q.full_key or q.key in key):
                    best = max(best, 3.0)
                    continue
                if grams:
                    overlap = len(grams & q.all_grams)
                    best = max(best, overlap / len(grams))
            return (best, idx)

        candidates.sort(key=rough, reverse=True)
        shortlist = candidates[: max(120, limit * 4)]

        scored: list[SearchResult] = []
        for idx in shortlist:
            q = self.questions[idx]
            best_score = 0.0
            best_reason = "模糊匹配"
            for key, grams in variant_grams:
                score, reason = self._score_one(q, key, grams)
                if score > best_score:
                    best_score, best_reason = score, reason
            scored.append(SearchResult(q, best_score, best_reason))

        if keyword_results:
            merged = {r.question.qid: r for r in scored}
            for kr in keyword_results:
                old = merged.get(kr.question.qid)
                if old is None or kr.score > old.score:
                    merged[kr.question.qid] = kr
            scored = list(merged.values())

        scored.sort(key=lambda r: (-r.score, r.question.number))
        if not scored:
            return [], (time.perf_counter() - started) * 1000.0
        best_score = scored[0].score
        if best_score < 0.22:
            return scored[: min(5, limit)], (time.perf_counter() - started) * 1000.0

        cutoff = max(0.16, best_score * 0.55)
        results = [r for r in scored if r.score >= cutoff]
        return results[:limit], (time.perf_counter() - started) * 1000.0

    def _search_terms(
        self,
        terms: Sequence[str],
        allowed: set[int] | None,
    ) -> list[SearchResult]:
        """空格分隔的多关键词查询，每个词都必须出现。

        :param terms: 关键词列表
        :param allowed: 允许的题号集合
        :returns: 命中的题号列表
        """
        n = len(terms)
        if n < 2:
            return []
        results: list[SearchResult] = []
        for idx, q in enumerate(self.questions):
            if allowed is not None and idx not in allowed:
                continue
            stem_hits = sum(1 for term in terms if term in q.key)
            full_hits = sum(1 for term in terms if term in q.full_key)
            if full_hits < n:
                continue
            score = 0.50 + 0.22 + 0.18 * (stem_hits / n) + 0.08
            if stem_hits == n:
                score += 0.04
                positions = [q.key.find(term) for term in terms]
                span = max(positions) - min(positions)
                score += max(0.0, 0.03 * (1.0 - span / max(1, len(q.key))))
            results.append(SearchResult(q, min(score, 0.985), "关键词匹配"))
        return results

    def _search_single_char(
        self,
        key: str,
        allowed: set[int] | None,
        limit: int,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        for idx, q in enumerate(self.questions):
            if allowed is not None and idx not in allowed:
                continue
            if key in q.key:
                pos = q.key.index(key)
                score = 0.78 + 0.18 * (1.0 - pos / max(1, len(q.key)))
                results.append(SearchResult(q, min(score, 0.96), "关键词匹配"))
            elif key in q.full_key:
                results.append(SearchResult(q, 0.68, "选项匹配"))
        results.sort(key=lambda r: (-r.score, r.question.number))
        return results[:limit]

    def _score_one(
        self,
        q: Question,
        key: str,
        key_grams: frozenset[str],
    ) -> tuple[float, str]:
        stem = q.key
        if not stem or not key:
            return 0.0, "模糊匹配"

        if key == stem:
            return 1.0, "完全匹配"

        if key in stem:
            ratio = len(key) / max(1, len(stem))
            return min(0.988, 0.935 + 0.05 * min(1.0, ratio * 3.0)), "题目包含"
        if stem in key:
            ratio = len(stem) / max(1, len(key))
            return min(0.972, 0.905 + 0.065 * min(1.0, ratio * 2.0)), "包含于题目"

        if key in q.full_key:
            return 0.90, "选项匹配"

        sm = SequenceMatcher(None, key, stem, autojunk=False)
        if sm.real_quick_ratio() < 0.35 or sm.quick_ratio() < 0.35:
            ratio = 0.0
        else:
            ratio = sm.ratio()

        overlap = len(key_grams & q.all_grams)
        coverage = overlap / max(1, len(key_grams))
        dice = (2.0 * overlap) / max(1, len(key_grams) + len(q.all_grams))

        # 短关键词更看重覆盖率；长题干更看重序列相似度。
        if len(key) <= 8:
            score = 0.52 * ratio + 0.38 * coverage + 0.10 * dice
        else:
            score = 0.66 * ratio + 0.26 * coverage + 0.08 * dice
        score = min(score, 0.925)
        reason = "关键词匹配" if len(key) <= 6 and coverage >= 0.6 else "模糊匹配"
        return score, reason

    @classmethod
    def load_txt(cls, path: str | Path) -> "QuestionBank":
        path = Path(path)
        text = path.read_text(encoding="utf-8-sig")
        records: list[dict[str, object]] = []
        current: dict[str, object] | None = None
        last_field: str | None = None

        def finish() -> None:
            nonlocal current
            if current is not None and current.get("题目"):
                records.append(current)
            current = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = _RECORD_RE.match(line)
            if m:
                finish()
                current = {"qid": m.group(1).upper(), "选项": []}
                last_field = None
                continue
            if current is None:
                continue
            fm = _FIELD_RE.match(line)
            if fm:
                name, value = fm.group(1), fm.group(2).strip()
                last_field = name
                if name == "选项":
                    options = current.setdefault("选项", [])
                    assert isinstance(options, list)
                    options.append(_strip_opt_label(value))
                else:
                    current[name] = value
                continue
            if last_field:
                if last_field == "选项":
                    options = current.get("选项", [])
                    if isinstance(options, list) and options:
                        options[-1] = smart_join(str(options[-1]), line)
                else:
                    old = str(current.get(last_field, ""))
                    current[last_field] = smart_join(old, line)

        finish()

        questions: list[Question] = []
        for i, rec in enumerate(records, start=1):
            qid = str(rec.get("qid") or f"Q{i:04d}")
            number = int(re.sub(r"\D", "", qid) or i)
            raw_options = rec.get("选项", [])
            options = tuple(str(x).strip() for x in raw_options if str(x).strip()) if isinstance(raw_options, list) else ()
            questions.append(
                Question(
                    number=number,
                    qid=qid,
                    qtype=str(rec.get("题型", "")).strip() or "单选题",
                    category=str(rec.get("分类", "")).strip(),
                    stem=str(rec.get("题目", "")).strip(),
                    options=options,
                    answer=str(rec.get("答案", "")).strip(),
                    explanation=str(rec.get("解析", "")).strip(),
                    note=str(rec.get("备注", "")).strip(),
                )
            )
        return cls(questions)

def format_question_for_clipboard(q: Question) -> str:
    return q.plain_text()

__all__ = [
    "Question",
    "QuestionBank",
    "SearchResult",
    "WATERMARK",
    "bigrams",
    "format_question_for_clipboard",
    "normalize_key",
    "normalize_radicals",
    "polish_text",
    "query_terms",
    "query_variants",
    "smart_join",
]
