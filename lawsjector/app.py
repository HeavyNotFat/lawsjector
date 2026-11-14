from __future__ import annotations

import os
import re
import sys
import time
import traceback
import queue
import threading
from typing import Optional

import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk

from .bank import QuestionBank, SearchResult, format_question_for_clipboard
from .paths import APP_DIR, ERROR_LOG, FROZEN, TXT_PATH

PLACEHOLDER = "输入题目或关键词"

NAVY = "#0B2239"
NAVY_2 = "#12395C"
ACCENT = "#1D4ED8"
ACCENT_DARK = "#1E40AF"
GOLD = "#D4A72C"
BG = "#F3F6FA"
WHITE = "#FFFFFF"
INK = "#102A43"
BODY = "#334E68"
MUTED = "#627D98"
LINE = "#D9E2EC"
SOFT = "#F7FAFC"
GREEN = "#0B6B3A"
GREEN_BG = "#E6F7EC"
AMBER = "#9A6700"
AMBER_BG = "#FFF4D6"
RED = "#B42318"

def _load_bank() -> QuestionBank:
    if not TXT_PATH.exists():
        # 首次运行自动生成，不要求用户先打开命令行。
        from .build_bank import build_all

        build_all(verbose=False)
    return QuestionBank.load_txt(TXT_PATH)

def _truncate(text: str, limit: int = 26) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"

class QuizApp:
    def __init__(self, root: tk.Tk, bank: QuestionBank):
        self.root = root
        self.bank = bank

        self.current_result: Optional[SearchResult] = None
        self.last_results: list[SearchResult] = []
        self.result_map: dict[str, SearchResult] = {}
        self._live_job: Optional[str] = None
        self._placeholder = False

        self.help_window: Optional[tk.Toplevel] = None

        self.screenshot_display = "Ctrl+Shift+S"
        self.screenshot_sequences = ["<Control-Shift-s>", "<Control-Shift-S>"]
        self._active_screenshot_sequences: list[str] = []
        self._binding_capture = False

        self._snip_active = False
        self._snipper = None
        self._ocr_in_progress = False
        self._ocr_queue: queue.Queue = queue.Queue()
        self._ocr_thread: Optional[threading.Thread] = None

        self._pick_fonts()
        self._setup_window()
        self._setup_styles()
        self._build_ui()
        self._bind_shortcuts()
        self._show_welcome()
        self.root.after(120, self._focus_search)
        self.root.after(500, self._warmup_ocr)

    def _pick_fonts(self) -> None:
        families = set(tkfont.families(self.root))
        family = "Microsoft YaHei UI"
        for candidate in ("Microsoft YaHei UI", "微软雅黑", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC"):
            if candidate in families:
                family = candidate
                break
        self.family = family

        def f(size: int, weight: str = "normal") -> tuple[str, int, str]:
            return (family, size, weight)

        self.f_tiny = f(9)
        self.f_small = f(9)
        self.f_body = f(10)
        self.f_body_mid = f(11)
        self.f_body_mid_bold = f(11, "bold")
        self.f_body_bold = f(10, "bold")
        self.f_input = f(12)
        self.f_section = f(10, "bold")
        self.f_stem = f(12)
        self.f_answer = f(18, "bold")
        self.f_heading = f(18, "bold")
        self.f_subheading = f(10)
        self.f_button = f(10, "bold")
        self.f_tree = f(9)
        self.f_tree_heading = f(9, "bold")

    def _setup_window(self) -> None:
        self.root.title("法治题库 · 智能查询")
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        w = min(1180, max(980, screen_w - 80))
        h = min(780, max(620, screen_h - 90))
        x = max(0, (screen_w - w) // 2)
        y = max(0, (screen_h - h) // 2 - 20)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(980, 620)
        self.root.configure(bg=BG)
        self._set_icon()

    def _set_icon(self) -> None:
        try:
            image = tk.PhotoImage(width=32, height=32)
            image.put(NAVY, to=(0, 0, 32, 32))
            image.put(GOLD, to=(6, 7, 8, 25))
            image.put("#FFFFFF", to=(11, 8, 25, 10))
            image.put("#D9E2EC", to=(11, 14, 25, 15))
            image.put("#D9E2EC", to=(11, 19, 21, 20))
            self._icon = image
            self.root.iconphoto(True, image)
        except Exception:
            pass

    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Results.Treeview",
            background=WHITE,
            fieldbackground=WHITE,
            foreground=BODY,
            borderwidth=0,
            relief="flat",
            rowheight=36,
            font=self.f_tree,
        )
        style.configure(
            "Results.Treeview.Heading",
            background="#EEF3F8",
            foreground=NAVY_2,
            borderwidth=0,
            relief="flat",
            padding=(8, 8),
            font=self.f_tree_heading,
        )
        style.map(
            "Results.Treeview.Heading",
            background=[("active", "#E3EBF4")],
        )
        style.map(
            "Results.Treeview",
            background=[("selected", "#DCEBFF")],
            foreground=[("selected", NAVY)],
        )
        style.configure(
            "Vertical.TScrollbar",
            background="#C7D3E0",
            troughcolor=WHITE,
            bordercolor=WHITE,
            arrowcolor=BODY,
            lightcolor="#C7D3E0",
            darkcolor="#C7D3E0",
            width=10,
        )
        style.map("Vertical.TScrollbar", background=[("active", "#AEBECD")])
        style.configure(
            "Ocr.Horizontal.TProgressbar",
            troughcolor="#E8EEF5",
            background=ACCENT,
            bordercolor="#E8EEF5",
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )

    def _build_ui(self) -> None:
        self._build_header()
        self._build_status()

        self.body = tk.Frame(self.root, bg=BG)
        self.body.pack(fill="both", expand=True, padx=22, pady=(16, 12))

        self._build_search_card()
        self._build_content()

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=NAVY, height=68)
        header.pack(fill="x")
        header.pack_propagate(False)

        inner = tk.Frame(header, bg=NAVY)
        inner.pack(fill="both", expand=True, padx=24)

        brand = tk.Frame(inner, bg=NAVY)
        brand.pack(side="left", pady=10)
        tk.Frame(brand, bg=GOLD, width=4, height=36).pack(side="left", padx=(0, 14))
        tk.Label(
            brand,
            text="法治题库",
            bg=NAVY,
            fg=WHITE,
            font=self.f_heading,
            anchor="w",
        ).pack(side="left")

        right = tk.Frame(inner, bg=NAVY)
        right.pack(side="right", pady=12)
        self._header_button(right, "说明", self.show_help).pack(side="left")

    def _header_button(self, parent: tk.Widget, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg="#14395D",
            fg=WHITE,
            activebackground="#1E4A73",
            activeforeground=WHITE,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=self.f_small,
            padx=12,
            pady=5,
            cursor="hand2",
        )

    def _build_search_card(self) -> None:
        card = tk.Frame(
            self.body,
            bg=WHITE,
            highlightbackground=LINE,
            highlightthickness=1,
        )
        card.pack(fill="x")
        card.grid_columnconfigure(0, weight=1)

        inner = tk.Frame(card, bg=WHITE)
        inner.pack(fill="x", padx=18, pady=14)
        inner.grid_columnconfigure(0, weight=1)

        row = tk.Frame(inner, bg=WHITE)
        row.grid(row=0, column=0, sticky="ew")
        row.grid_columnconfigure(0, weight=1)

        self.search_box = tk.Text(
            row,
            height=2,
            wrap="word",
            bg=SOFT,
            fg=INK,
            insertbackground=ACCENT,
            font=self.f_input,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=LINE,
            highlightcolor="#93B4F5",
            padx=12,
            pady=8,
            undo=True,
        )
        self.search_box.grid(row=0, column=0, sticky="ew")
        self.search_box.tag_configure("placeholder", foreground="#94A3B8")

        controls = tk.Frame(inner, bg=WHITE)
        controls.grid(row=1, column=0, sticky="ew", pady=(9, 0))

        type_left = tk.Frame(controls, bg=WHITE)
        type_left.pack(side="left")

        self.type_var = tk.StringVar(value="全部题型")
        self.type_buttons: dict[str, tk.Button] = {}
        type_shell = tk.Frame(type_left, bg=WHITE, highlightbackground=LINE, highlightthickness=1)
        type_shell.pack(side="left")
        for value, label in (
            ("全部题型", "全部"),
            ("单选题", "单选"),
            ("多选题", "多选"),
            ("判断题", "判断"),
        ):
            btn = tk.Button(
                type_shell,
                text=label,
                command=lambda v=value: self._set_type_filter(v),
                bg="#F0F4F8",
                fg=MUTED,
                activebackground="#E3EBF4",
                activeforeground=NAVY,
                relief="flat",
                bd=0,
                highlightthickness=0,
                font=self.f_small,
                padx=12,
                pady=4,
                cursor="hand2",
            )
            btn.pack(side="left")
            self.type_buttons[value] = btn
        self._update_type_buttons()

        actions = tk.Frame(controls, bg=WHITE)
        actions.pack(side="right")

        self.search_button = tk.Button(
            actions,
            text="查询答案",
            command=lambda: self.do_search(explicit=True),
            bg=ACCENT,
            fg=WHITE,
            activebackground=ACCENT_DARK,
            activeforeground=WHITE,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=self.f_button,
            padx=18,
            pady=6,
            cursor="hand2",
        )
        self.search_button.pack(side="right")

        self.snip_button = tk.Button(
            actions,
            text=f"截屏搜题  {self.screenshot_display}",
            command=self.start_screenshot,
            bg=GOLD,
            fg=NAVY,
            activebackground="#E3B93A",
            activeforeground=NAVY,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=self.f_button,
            padx=14,
            pady=6,
            cursor="hand2",
        )
        self.snip_button.pack(side="right", padx=(0, 8))
        self._link_button(actions, "清空", self.clear_search).pack(side="right", padx=(0, 10))
        self._link_button(actions, "改键", self.start_key_capture).pack(side="right", padx=(0, 2))

        # OCR 进度直接显示在当前页面，不弹二级窗口。
        self.ocr_status = tk.Frame(inner, bg=WHITE)
        self.ocr_status.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        tk.Label(
            self.ocr_status,
            text="正在识别截图…",
            bg=WHITE,
            fg=ACCENT,
            font=self.f_small,
        ).pack(side="left")
        self.ocr_bar = ttk.Progressbar(
            self.ocr_status,
            mode="indeterminate",
            length=220,
            style="Ocr.Horizontal.TProgressbar",
        )
        self.ocr_bar.pack(side="left", padx=(12, 0))
        self.ocr_status.grid_remove()

        self._set_placeholder()

    def _build_content(self) -> None:
        content = tk.Frame(self.body, bg=BG)
        content.pack(fill="both", expand=True, pady=(12, 0))
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=4, minsize=380, uniform="main")
        content.grid_columnconfigure(1, weight=6, minsize=520, uniform="main")

        left = tk.Frame(content, bg=WHITE, highlightbackground=LINE, highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        left_head = tk.Frame(left, bg=WHITE)
        left_head.pack(fill="x", padx=16, pady=(13, 8))
        tk.Label(left_head, text="搜索结果", bg=WHITE, fg=NAVY, font=(self.family, 11, "bold")).pack(side="left")
        self.count_var = tk.StringVar(value="")
        tk.Label(left_head, textvariable=self.count_var, bg=WHITE, fg=MUTED, font=self.f_small).pack(side="right")

        tree_frame = tk.Frame(left, bg=WHITE)
        tree_frame.pack(fill="both", expand=True, padx=(8, 4), pady=(0, 8))

        columns = ("score", "type", "stem")
        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
            style="Results.Treeview",
        )
        self.tree.heading("score", text="匹配")
        self.tree.heading("type", text="题型")
        self.tree.heading("stem", text="题目")
        self.tree.column("score", width=56, minwidth=50, anchor="center", stretch=False)
        self.tree.column("type", width=60, minwidth=54, anchor="center", stretch=False)
        self.tree.column("stem", width=280, minwidth=190, anchor="w", stretch=True)
        self.tree.tag_configure("odd", background=WHITE)
        self.tree.tag_configure("even", background="#FAFCFE")
        self.tree.tag_configure("best", font=self.f_body_bold)

        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview, style="Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        self.empty_overlay = tk.Label(
            left,
            text="",
            bg=WHITE,
            fg="#9AA5B1",
            font=self.f_small,
        )
        self.empty_overlay.place(relx=0.5, rely=0.6, anchor="center")

        right = tk.Frame(content, bg=WHITE, highlightbackground=LINE, highlightthickness=1)
        right.grid(row=0, column=1, sticky="nsew")

        detail_head = tk.Frame(right, bg=WHITE)
        detail_head.pack(fill="x", padx=18, pady=(13, 8))
        tk.Label(detail_head, text="题目详情", bg=WHITE, fg=NAVY, font=(self.family, 11, "bold")).pack(side="left")
        self.detail_meta = tk.Label(detail_head, text="", bg=WHITE, fg=MUTED, font=self.f_small)
        self.detail_meta.pack(side="left", padx=(12, 0), pady=(2, 0))
        self._link_button(detail_head, "复制本题", self.copy_current).pack(side="right")

        detail_wrap = tk.Frame(right, bg=WHITE)
        detail_wrap.pack(fill="both", expand=True, padx=(8, 4), pady=(0, 8))

        self.detail = tk.Text(
            detail_wrap,
            wrap="word",
            bg=WHITE,
            fg=BODY,
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=18,
            pady=12,
            font=self.f_body_mid,
            cursor="arrow",
            state="disabled",
            selectbackground="#CFE2FF",
            spacing1=1,
            spacing3=2,
        )
        detail_scroll = ttk.Scrollbar(
            detail_wrap,
            orient="vertical",
            command=self.detail.yview,
            style="Vertical.TScrollbar",
        )
        self.detail.configure(yscrollcommand=detail_scroll.set)
        self.detail.pack(side="left", fill="both", expand=True)
        detail_scroll.pack(side="right", fill="y")
        self._configure_detail_tags(self.detail)

    def _build_status(self) -> None:
        bar = tk.Frame(self.root, bg=BG, height=30)
        bar.pack(fill="x", side="bottom", padx=24, pady=(0, 8))
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="就绪")
        tk.Label(
            bar,
            textvariable=self.status_var,
            bg=BG,
            fg=MUTED,
            font=self.f_tiny,
            anchor="w",
        ).pack(side="left")

    def _link_button(self, parent: tk.Widget, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=WHITE,
            fg=MUTED,
            activebackground="#EEF3F8",
            activeforeground=NAVY,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=self.f_tiny,
            padx=7,
            pady=3,
            cursor="hand2",
        )

    def _set_placeholder(self) -> None:
        self.search_box.delete("1.0", "end")
        self.search_box.insert("1.0", PLACEHOLDER, ("placeholder",))
        self.search_box.configure(insertwidth=0)
        self._placeholder = True

    def _clear_placeholder(self) -> None:
        if self._placeholder:
            self.search_box.delete("1.0", "end")
            self.search_box.tag_remove("placeholder", "1.0", "end")
            self.search_box.configure(insertwidth=2)
            self._placeholder = False

    def _focus_search(self) -> None:
        self.search_box.focus_set()
        self.search_box.mark_set("insert", "1.0")

    def get_query(self) -> str:
        if self._placeholder:
            return ""
        return self.search_box.get("1.0", "end-1c").strip()

    def _set_query(self, text: str, *, focus: bool = True) -> None:
        self._placeholder = False
        self.search_box.tag_remove("placeholder", "1.0", "end")
        self.search_box.delete("1.0", "end")
        self.search_box.insert("1.0", text)
        self.search_box.configure(insertwidth=2)
        if focus:
            self._focus_search()

    def _on_key_press(self, event: tk.Event) -> None:
        if event.state & 0x0004:  # Ctrl 组合键交给快捷键处理
            return
        if self._placeholder:
            if event.keysym in {
                "Left",
                "Right",
                "Up",
                "Down",
                "Home",
                "End",
                "Tab",
                "Escape",
                "Shift_L",
                "Shift_R",
                "Control_L",
                "Control_R",
                "Alt_L",
                "Alt_R",
            }:
                return
            if event.keysym == "Return":
                return
            self._clear_placeholder()

    def _on_paste(self, _event=None) -> None:
        if self._placeholder:
            self.search_box.delete("1.0", "end")
            self._placeholder = False
            self.search_box.configure(insertwidth=2)
        self.root.after(20, self._schedule_search)

    def _on_return(self, event: tk.Event) -> str | None:
        if event.state & 0x0001:  # Shift+Enter
            return None
        self.do_search(explicit=True)
        return "break"

    def _on_focus_out(self, _event=None) -> None:
        if not self.get_query():
            self._set_placeholder()

    def _on_focus_in(self, _event=None) -> None:
        pass

    def _schedule_search(self) -> None:
        if self._placeholder:
            return
        if self._live_job is not None:
            try:
                self.root.after_cancel(self._live_job)
            except Exception:
                pass
        self._live_job = self.root.after(180, self._live_search)

    def _live_search(self) -> None:
        self._live_job = None
        if self._placeholder:
            return
        query = self.get_query()
        if not query:
            self._show_welcome()
            self.count_var.set("")
            self.status_var.set("就绪")
            return
        self.do_search(explicit=False)

    def _set_type_filter(self, value: str) -> None:
        self.type_var.set(value)
        self._update_type_buttons()
        if self.get_query():
            self.do_search(explicit=False)

    def _update_type_buttons(self) -> None:
        selected = self.type_var.get()
        for value, button in self.type_buttons.items():
            if value == selected:
                button.configure(
                    bg=WHITE,
                    fg=ACCENT,
                    font=(self.family, 9, "bold"),
                    activebackground=WHITE,
                )
            else:
                button.configure(
                    bg="#F0F4F8",
                    fg=MUTED,
                    font=self.f_small,
                    activebackground="#E3EBF4",
                )

    def do_search(self, *, explicit: bool) -> None:
        if self._placeholder:
            query = ""
        else:
            query = self.get_query()
        if not query:
            if explicit:
                self.status_var.set("请输入题目或关键词")
                self._focus_search()
            return

        if self._live_job is not None:
            try:
                self.root.after_cancel(self._live_job)
            except Exception:
                pass
            self._live_job = None

        started = time.perf_counter()
        try:
            results, _ = self.bank.search(
                query,
                qtype=self.type_var.get(),
                limit=30,
            )
        except Exception as exc:
            self._log_exception(exc)
            messagebox.showerror("查询失败", f"查询时出现错误：\n{exc}")
            return
        total_ms = (time.perf_counter() - started) * 1000
        self.last_results = results
        self._fill_results(results)

        if not results:
            self.current_result = None
            self.count_var.set("0 条")
            self.empty_overlay.place(relx=0.5, rely=0.6, anchor="center")
            self.empty_overlay.lift()
            self._show_no_result(query)
            self.status_var.set(f"未找到相关题目  ·  {total_ms:.1f} ms")
            return

        self.empty_overlay.place_forget()
        best = results[0]
        self.count_var.set(f"{len(results)} 条")
        self.status_var.set(
            f"最佳匹配 {best.percent}%  ·  {best.reason}  ·  用时 {total_ms:.1f} ms  ·  "
            f"题型：{self.type_var.get()}"
        )
        self._select_result(str(0))

        if explicit and best.score < 0.30:
            self.status_var.set(
                f"未找到高相关题目，以下为最接近的结果  ·  "
                f"最佳 {best.percent}%  ·  {total_ms:.1f} ms"
            )

    def _fill_results(self, results: list[SearchResult]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.result_map.clear()

        for i, result in enumerate(results):
            iid = str(i)
            self.result_map[iid] = result
            tags = ["even" if i % 2 else "odd"]
            if i == 0:
                tags.append("best")
            self.tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    f"{result.percent}%",
                    result.question.qtype.replace("题", ""),
                    _truncate(result.question.stem, 22),
                ),
                tags=tuple(tags),
            )
        if results:
            self.tree.selection_set("0")
            self.tree.focus("0")
            self.tree.see("0")

    def _selected_result(self) -> Optional[SearchResult]:
        selection = self.tree.selection()
        if not selection:
            return None
        return self.result_map.get(selection[0])

    def _select_result(self, iid: str) -> None:
        if iid in self.result_map:
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self.tree.see(iid)
            self._show_result(self.result_map[iid])

    def _on_tree_select(self, _event=None) -> None:
        result = self._selected_result()
        if result:
            self._show_result(result)

    def _configure_detail_tags(self, widget: tk.Text) -> None:
        widget.tag_configure("meta", foreground=MUTED, font=self.f_small, spacing3=7)
        widget.tag_configure("stem", foreground=INK, font=self.f_stem, spacing1=5, spacing3=8, lmargin2=2)
        widget.tag_configure(
            "warning",
            foreground="#C62828",
            background="#FFF1F0",
            font=self.f_body_mid_bold,
            spacing1=8,
            spacing3=8,
        )
        widget.tag_configure("section", foreground=NAVY_2, font=self.f_section, spacing1=16, spacing3=6)
        widget.tag_configure("option", foreground=BODY, font=self.f_body_mid, lmargin1=8, lmargin2=8, spacing1=2)
        widget.tag_configure(
            "correct",
            foreground=GREEN,
            background=GREEN_BG,
            font=self.f_body_mid_bold,
            lmargin1=8,
            lmargin2=8,
            spacing1=2,
        )
        widget.tag_configure(
            "answer",
            foreground=GREEN,
            background=GREEN_BG,
            font=self.f_answer,
            spacing1=8,
            spacing3=8,
        )
        widget.tag_configure("explain", foreground="#52606D", font=self.f_body_mid, spacing1=4, spacing3=4)
        widget.tag_configure("note", foreground=AMBER, background=AMBER_BG, font=self.f_small, spacing1=10, spacing3=3)
        widget.tag_configure("footer", foreground="#9AA5B1", font=self.f_tiny, spacing1=18)
        widget.tag_configure("welcome_title", foreground=NAVY, font=(self.family, 16, "bold"), spacing3=8)
        widget.tag_configure("welcome_sub", foreground=MUTED, font=self.f_body_mid, spacing3=12)
        widget.tag_configure("bullet", foreground=ACCENT, font=self.f_body_bold, spacing1=4)
        widget.tag_configure("welcome_body", foreground=BODY, font=self.f_body_mid, spacing3=3, lmargin2=4)
        widget.tag_configure("center_hint", foreground="#9AA5B1", font=self.f_body, justify="center", spacing1=18)

    def _enable_and_clear(self, widget: tk.Text) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")

    def _finish_text(self, widget: tk.Text) -> None:
        widget.configure(state="disabled")
        widget.yview_moveto(0)

    def _show_welcome(self) -> None:
        self.detail_meta.configure(text="")
        self._enable_and_clear(self.detail)
        self.detail.insert("end", "输入题目开始查询", "center_hint")
        self._finish_text(self.detail)

    def _show_no_result(self, query: str) -> None:
        self.detail_meta.configure(text="")
        self._enable_and_clear(self.detail)
        self.detail.insert("end", "没有找到相关题目\n", "welcome_title")
        self.detail.insert("end", "换个关键词试试。", "welcome_sub")
        self._finish_text(self.detail)

    def _show_result(self, result: SearchResult) -> None:
        self.current_result = result
        q = result.question
        self.detail_meta.configure(
            text=f"{q.qtype} · {q.category or '未分类'} · {result.percent}% {result.reason}"
        )
        self._render_question(self.detail, result)

    def _render_question(self, widget: tk.Text, result: SearchResult) -> None:
        self._enable_and_clear(widget)
        q = result.question
        w = widget

        w.insert("end", q.stem + "\n", "stem")
        w.insert("end", "仔细看答案，选项可能是乱序。\n", "warning")

        if q.options:
            w.insert("end", "选项\n", "section")
            for i, opt in enumerate(q.options):
                letter = q.option_letter(i)
                if i in q.correct_indexes:
                    w.insert("end", f"{letter}. {opt}   ✓\n", "correct")
                else:
                    w.insert("end", f"{letter}. {opt}\n", "option")

        w.insert("end", "答案\n", "section")
        answer = q.answer_display()
        w.insert("end", f"  {answer}  \n", "answer")

        if q.note:
            w.insert("end", f"⚠ {q.note}\n", "note")

        if q.explanation:
            w.insert("end", "解析\n", "section")
            w.insert("end", q.explanation + "\n", "explain")

        self._finish_text(w)

    def _center_toplevel(self, win: tk.Toplevel, width: int, height: int) -> None:
        self.root.update_idletasks()
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - width) // 2)
        y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - height) // 2)
        win.geometry(f"{width}x{height}+{x}+{y}")

    def _warmup_ocr(self) -> None:
        if getattr(self, "_ocr_warmup_started", False):
            return
        self._ocr_warmup_started = True

        def run() -> None:
            try:
                from . import ocr_engine

                ocr_engine.warmup()
            except Exception:
                pass

        threading.Thread(target=run, daemon=True, name="ocr-warmup").start()

    def start_screenshot(self) -> None:
        if self._binding_capture or self._snip_active or self._ocr_in_progress:
            return
        try:
            from PIL import ImageGrab  # noqa: F401
        except Exception as exc:
            messagebox.showerror(
                "缺少截图组件",
                "截屏搜题需要 Pillow。\n\n请先安装：\n"
                "python -m pip install pillow\n\n"
                f"详细信息：{exc}",
                parent=self.root,
            )
            return

        import importlib.util

        from . import ocr_engine

        has_ocr = ocr_engine.has_native() or any(
            importlib.util.find_spec(name) is not None
            for name in ("rapidocr_onnxruntime", "winsdk")
        )
        if not has_ocr:
            messagebox.showerror(
                "缺少 OCR 模型",
                "截屏搜题需要 OCR 模型：把 det.onnx / rec.onnx / ppocr_keys_v1.txt 放到 resources\\models，\n"
                "或者安装 Windows OCR 兜底：\npython -m pip install winsdk",
                parent=self.root,
            )
            return

        self._snip_active = True
        self._close_help()
        self.status_var.set("准备截屏…")
        # 不触发窗口最小化/淡出动画：直接让主窗口透明后再抓屏。
        try:
            self.root.attributes("-alpha", 0.0)
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            pass
        self.root.after(150, self._capture_for_snip)

    def _capture_for_snip(self) -> None:
        if not self._snip_active:
            return
        try:
            from PIL import ImageGrab

            screenshot = ImageGrab.grab()
        except Exception as exc:
            self._restore_after_snip()
            self._log_exception(exc)
            messagebox.showerror("截屏失败", f"无法抓取屏幕：\n{exc}", parent=self.root)
            return

        try:
            from .snipper import RegionSnipper

            self._snipper = RegionSnipper(
                self.root,
                screenshot,
                self._on_snip_selected,
                self._on_snip_cancelled,
            )
        except Exception as exc:
            self._restore_after_snip()
            self._log_exception(exc)
            messagebox.showerror("截屏失败", f"无法打开框选窗口：\n{exc}", parent=self.root)

    def _restore_after_snip(self) -> None:
        self._snipper = None
        self._snip_active = False
        try:
            self.root.attributes("-alpha", 1.0)
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass

    def _on_snip_cancelled(self) -> None:
        self._restore_after_snip()
        self.status_var.set("已取消截屏")

    def _on_snip_selected(self, image) -> None:
        self._snipper = None
        self._snip_active = False
        try:
            self.root.attributes("-alpha", 1.0)
        except Exception:
            pass
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self._start_ocr(image)

    def _start_ocr(self, image) -> None:
        self._ocr_in_progress = True
        self._set_snip_button_busy(True)
        self.status_var.set("正在识别…")
        self._show_ocr_progress()

        self._ocr_queue = queue.Queue()

        def worker() -> None:
            try:
                from . import ocr_engine

                result = ocr_engine.recognize(image)
                self._ocr_queue.put(("ok", result))
            except BaseException as exc:
                self._ocr_queue.put(("error", exc))

        self._ocr_thread = threading.Thread(target=worker, daemon=True, name="ocr-worker")
        self.root.after(50, self._ocr_thread.start)
        self.root.after(100, self._poll_ocr)

    def _set_snip_button_busy(self, busy: bool) -> None:
        if not hasattr(self, "snip_button"):
            return
        if busy:
            self.snip_button.configure(
                text="识别中…",
                state="disabled",
                bg="#C7D3E0",
                fg=MUTED,
                cursor="arrow",
            )
        else:
            self.snip_button.configure(
                text=f"截屏搜题  {self.screenshot_display}",
                state="normal",
                bg=GOLD,
                fg=NAVY,
                cursor="hand2",
            )

    def _show_ocr_progress(self) -> None:
        self.ocr_status.grid()
        self.ocr_bar.start(12)

    def _hide_ocr_progress(self) -> None:
        try:
            self.ocr_bar.stop()
        except Exception:
            pass
        self.ocr_status.grid_remove()

    def _poll_ocr(self) -> None:
        if not self._ocr_in_progress:
            return
        try:
            kind, payload = self._ocr_queue.get_nowait()
        except queue.Empty:
            self.root.after(120, self._poll_ocr)
            return

        self._hide_ocr_progress()
        self._ocr_in_progress = False
        self._set_snip_button_busy(False)

        if kind == "error":
            self._log_exception(payload)
            self.status_var.set("OCR 识别失败")
            messagebox.showerror(
                "OCR 识别失败",
                f"{payload}\n\n"
                "截屏搜题需要以下任一离线 OCR 引擎：\n"
                "  · resources\\models 下的 det.onnx / rec.onnx / ppocr_keys_v1.txt\n"
                "  · winsdk（Windows 自带 OCR）\n\n"
                "模型可用 rapidocr_train\\export_onnx.py 生成并自动复制",
                parent=self.root,
            )
            return

        self._handle_ocr_result(payload)

    def _handle_ocr_result(self, result) -> None:
        try:
            from .ocr_engine import build_queries

            stem, full = build_queries(result)
        except Exception as exc:
            self._log_exception(exc)
            messagebox.showerror("OCR 结果处理失败", str(exc), parent=self.root)
            return

        candidates: list[str] = []
        for value in (stem, full):
            value = (value or "").strip()
            if value and value not in candidates:
                candidates.append(value)

        if not candidates:
            messagebox.showinfo(
                "没有识别到文字",
                "截图里没有识别到可用文字。\n请重新框选题干，尽量只包含题目部分。",
                parent=self.root,
            )
            self.status_var.set("OCR 未识别到文字")
            return

        best_choice = candidates[0]
        best_result = None
        for candidate in candidates:
            results, _ = self.bank.search(candidate, qtype=self.type_var.get(), limit=5)
            if results and (best_result is None or results[0].score > best_result.score):
                best_choice = candidate
                best_result = results[0]

        self._set_query(best_choice)
        self.do_search(explicit=True)

        ocr_seconds = max(0.0, result.elapsed_ms / 1000.0)
        prefix = f"OCR {ocr_seconds:.1f}s · 识别 {len(result.lines)} 行 · "
        if best_result is not None and best_result.score < 0.30:
            prefix = "识别结果可能不准确，以下为最接近题目 · " + prefix
        self.status_var.set(prefix + self.status_var.get())

    def _copy_result(self, result: SearchResult) -> None:
        text = format_question_for_clipboard(result.question)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()
        self.status_var.set("已复制题目和答案到剪贴板")

    def copy_current(self) -> None:
        if self.current_result:
            self._copy_result(self.current_result)
        else:
            self.status_var.set("当前没有可复制的题目")

    def clear_search(self) -> None:
        if self._live_job is not None:
            try:
                self.root.after_cancel(self._live_job)
            except Exception:
                pass
            self._live_job = None
        self.search_box.delete("1.0", "end")
        self._placeholder = False
        self._set_placeholder()
        self._focus_search()
        self.current_result = None
        self.last_results = []
        self.result_map.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.empty_overlay.place(relx=0.5, rely=0.6, anchor="center")
        self.empty_overlay.lift()
        self.count_var.set("")
        self.detail_meta.configure(text="")
        self.status_var.set("就绪")
        self._show_welcome()

    def show_help(self) -> None:
        if self.help_window is not None and self.help_window.winfo_exists():
            self.help_window.lift()
            return
        win = tk.Toplevel(self.root)
        self.help_window = win
        win.title("说明 · 法治题库")
        win.configure(bg=WHITE)
        win.transient(self.root)
        win.minsize(560, 480)
        width = min(700, max(560, self.root.winfo_screenwidth() - 80))
        height = min(600, max(480, self.root.winfo_screenheight() - 100))
        self._center_toplevel(win, width, height)
        win.protocol("WM_DELETE_WINDOW", self._close_help)

        header = tk.Frame(win, bg=NAVY, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text="说明",
            bg=NAVY,
            fg=WHITE,
            font=(self.family, 15, "bold"),
        ).pack(side="left", padx=22, pady=16)
        tk.Button(
            header,
            text="关闭  Esc",
            command=self._close_help,
            bg="#14395D",
            fg=WHITE,
            activebackground="#1E4A73",
            activeforeground=WHITE,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=self.f_small,
            padx=14,
            pady=6,
            cursor="hand2",
        ).pack(side="right", padx=18)

        text = tk.Text(
            win,
            wrap="word",
            bg=WHITE,
            fg=BODY,
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=24,
            pady=20,
            font=self.f_body_mid,
            spacing1=2,
            spacing3=5,
        )
        scroll = ttk.Scrollbar(win, orient="vertical", command=text.yview, style="Vertical.TScrollbar")
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        text.tag_configure("h", foreground=NAVY, font=(self.family, 12, "bold"), spacing1=16, spacing3=6)
        text.tag_configure("h2", foreground=NAVY_2, font=(self.family, 11, "bold"), spacing1=12, spacing3=4)
        text.tag_configure("credit", foreground=NAVY, font=(self.family, 11, "bold"), spacing1=2)
        text.tag_configure("thanks", foreground=BODY, font=self.f_body_mid, spacing3=4)
        text.tag_configure("bullet", foreground=ACCENT, font=self.f_body_mid)
        text.tag_configure("step", foreground=BODY, font=self.f_body_mid, lmargin2=16)
        text.tag_configure("note", foreground=MUTED, font=self.f_small, spacing1=6)
        text.tag_configure("muted", foreground=MUTED, font=self.f_small)

        text.insert("end", "开发者：高2025级 张嘉祺\n", "credit")
        text.insert("end", "感谢 梁恩钊老师 提供题库和答案\n", "thanks")

        text.insert("end", "使用方法\n", "h")

        text.insert("end", "方法 1：截屏搜题\n", "h2")
        steps = [
            f"点击页面上的金色“截屏搜题”按钮，或按 {self.screenshot_display}。",
            "主窗口会暂时隐藏；按住鼠标左键拖动，框选题目文字，也可以框选选项。",
            "松开鼠标后自动识别并查询，结果显示在右侧“题目详情”。",
            "识别不准确时，可直接修改输入框中的文字后重新查询。",
        ]
        for step in steps:
            text.insert("end", "• ", "bullet")
            text.insert("end", step + "\n", "step")
        text.insert("end", "点击“改键”可临时修改截屏快捷键；重启程序后恢复默认，不会保存。\n", "note")

        text.insert("end", "方法 2：手动输入\n", "h2")
        steps = [
            "在输入框输入完整题目，或输入 2-6 个关键词；多个关键词用空格分开。",
            "点击“查询答案”，或按 Enter。",
            "在右侧“题目详情”查看题目、选项、答案和解析。",
            "可用“全部 / 单选 / 多选 / 判断”筛选题型；错字、漏字会自动匹配最接近的题。",
        ]
        for step in steps:
            text.insert("end", "• ", "bullet")
            text.insert("end", step + "\n", "step")

        text.configure(state="disabled")
        win.bind("<Escape>", lambda _e: self._close_help())

    def _close_help(self) -> None:
        if self.help_window is not None:
            try:
                self.help_window.destroy()
            except Exception:
                pass
        self.help_window = None

    def _bind_shortcuts(self) -> None:
        self.search_box.bind("<KeyPress>", self._on_key_press, add="+")
        self.search_box.bind("<Return>", self._on_return)
        self.search_box.bind("<Control-Return>", lambda _e: (self.do_search(explicit=True), "break")[1])
        self.search_box.bind("<KeyRelease>", lambda _e: self._schedule_search(), add="+")
        self.search_box.bind("<<Paste>>", self._on_paste, add="+")
        self.search_box.bind("<<Cut>>", lambda _e: self.root.after(20, self._schedule_search), add="+")
        self.search_box.bind("<Control-a>", self._select_all_search)
        self.search_box.bind("<FocusIn>", self._on_focus_in)
        self.search_box.bind("<FocusOut>", self._on_focus_out)

        self.root.bind("<F1>", lambda _e: self.show_help())
        self.root.bind("<Control-l>", lambda _e: self.clear_search())
        self.root.bind("<Control-L>", lambda _e: self.clear_search())
        self.root.bind("<Control-f>", lambda _e: self._focus_search())
        self.root.bind("<Control-F>", lambda _e: self._focus_search())
        self.root.bind("<Escape>", self._on_escape)
        self.root.bind("<Control-c>", self._on_ctrl_c, add="+")

        # 页面内的 1×1 焦点接收器，用于临时改键，不弹窗。
        self._key_sink = tk.Entry(
            self.root,
            width=1,
            bd=0,
            highlightthickness=0,
            bg=WHITE,
            takefocus=1,
        )
        self._key_sink.place(x=0, y=0, width=1, height=1)
        self._key_sink.bind("<KeyPress>", self._capture_binding_key)
        self._key_sink.bind("<FocusOut>", self._on_capture_focus_out)

        self._bind_screenshot_shortcut()

    def _bind_screenshot_shortcut(self) -> None:
        for seq in self._active_screenshot_sequences:
            try:
                self.search_box.unbind(seq)
                self.root.unbind(seq)
            except tk.TclError:
                pass
        self._active_screenshot_sequences = list(self.screenshot_sequences)
        for seq in self.screenshot_sequences:
            self.search_box.bind(seq, self._shortcut_screenshot)
            self.root.bind(seq, self._shortcut_screenshot)
        if hasattr(self, "snip_button") and not self._binding_capture:
            self.snip_button.configure(text=f"截屏搜题  {self.screenshot_display}")

    def start_key_capture(self) -> None:
        if self._binding_capture or self._snip_active or self._ocr_in_progress:
            return
        self._binding_capture = True
        self.snip_button.configure(text="按下新快捷键…")
        self.status_var.set("按下新的截屏快捷键，Esc 取消")
        self._key_sink.focus_set()

    def _on_capture_focus_out(self, _event=None) -> None:
        if self._binding_capture:
            self._cancel_key_capture()

    def _cancel_key_capture(self) -> None:
        self._binding_capture = False
        self.snip_button.configure(text=f"截屏搜题  {self.screenshot_display}")
        self.status_var.set("已取消改键")

    def _capture_binding_key(self, event: tk.Event) -> str | None:
        if not self._binding_capture:
            return None
        keysym = event.keysym
        if keysym == "Escape":
            self._cancel_key_capture()
            self._focus_search()
            return "break"
        if keysym in {
            "Control_L",
            "Control_R",
            "Shift_L",
            "Shift_R",
            "Alt_L",
            "Alt_R",
            "Win_L",
            "Win_R",
            "Super_L",
            "Super_R",
            "Caps_Lock",
        }:
            return "break"

        state = event.state
        ctrl = bool(state & 0x0004)
        alt = bool(state & 0x0008) or bool(state & 0x20000)
        shift = bool(state & 0x0001)

        key = keysym
        if len(key) == 1:
            key_display = key.upper()
        elif key == "Return":
            key_display = "Enter"
        elif key == "space":
            key_display = "Space"
        else:
            key_display = key

        if not (ctrl or alt or shift) and not re.match(r"^F([1-9]|1[0-2])$", key_display):
            self.status_var.set("请同时按 Ctrl / Alt / Shift，或使用 F1–F12")
            return "break"

        mods_display: list[str] = []
        mods_seq: list[str] = []
        if ctrl:
            mods_display.append("Ctrl")
            mods_seq.append("Control")
        if alt:
            mods_display.append("Alt")
            mods_seq.append("Alt")
        if shift:
            mods_display.append("Shift")
            mods_seq.append("Shift")

        display = "+".join(mods_display + [key_display])
        if display in {
            "F1",
            "Ctrl+L",
            "Ctrl+F",
            "Ctrl+C",
            "Ctrl+A",
            "Ctrl+Enter",
            "Alt+F4",
            "Ctrl+Alt+Delete",
        }:
            self.status_var.set("该快捷键已被占用")
            return "break"

        base = "<" + "-".join(mods_seq + [key]) + ">"
        sequences = [base]
        if len(key) == 1 and key.isalpha() and key.lower() != key:
            sequences.append("<" + "-".join(mods_seq + [key.lower()]) + ">")
        elif len(key) == 1 and key.isalpha():
            sequences.append("<" + "-".join(mods_seq + [key.upper()]) + ">")

        self.screenshot_display = display
        self.screenshot_sequences = sequences
        self._binding_capture = False
        self._bind_screenshot_shortcut()
        self.status_var.set(f"截屏快捷键：{display}（仅本次运行）")
        self._focus_search()
        return "break"

    def _shortcut_screenshot(self, _event=None) -> str:
        self.start_screenshot()
        return "break"

    def _select_all_search(self, _event=None) -> str:
        self.search_box.tag_add("sel", "1.0", "end-1c")
        self.search_box.mark_set("insert", "end-1c")
        return "break"

    def _on_escape(self, _event=None) -> None:
        if self.help_window is not None and self.help_window.winfo_exists():
            self._close_help()
        else:
            self.root.focus_set()

    def _on_ctrl_c(self, event: tk.Event) -> None:
        # 若焦点在详情/弹窗文字里，交给 Text 的默认复制行为。
        widget = event.widget
        if isinstance(widget, tk.Text):
            return
        if self.current_result:
            self._copy_result(self.current_result)

    def _log_exception(self, exc: BaseException) -> None:
        try:
            with ERROR_LOG.open("a", encoding="utf-8") as fh:
                fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
                fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        except Exception:
            pass

def _enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

def _silence_standard_streams() -> None:
    """windowed exe 没有控制台，避免第三方库写 stdout 时出错。
    """
    try:
        stream = open(os.devnull, "w", encoding="utf-8", buffering=1)
        sys.stdout = stream
        sys.stderr = stream
    except Exception:
        pass

def _run_self_test() -> int:
    """打包后自检：题库加载、检索、OCR 模型与 CPU 推理。

    :returns: 进程退出码
    """
    report: list[str] = []
    log_path = APP_DIR / "selftest.log"
    try:
        bank = QuestionBank.load_txt(TXT_PATH)
        report.append(f"题库：{bank.count} 题")
        if bank.count < 1000:
            raise RuntimeError("题库数量异常")

        results, elapsed = bank.search("诉讼时效 三年")
        if not results:
            raise RuntimeError("关键词检索没有返回结果")
        report.append(f"检索：{results[0].question.qid} {results[0].percent}% {elapsed:.1f}ms")

        from PIL import Image, ImageDraw, ImageFont

        q = bank.questions[0]
        font_path = r"C:\Windows\Fonts\msyh.ttc"
        font = ImageFont.truetype(font_path, 34)
        image = Image.new("RGB", (1680, 460), "white")
        draw = ImageDraw.Draw(image)
        text_lines: list[str] = []
        current = ""
        for ch in q.stem:
            if draw.textlength(current + ch, font=font) > 1600:
                text_lines.append(current)
                current = ch
            else:
                current += ch
        if current:
            text_lines.append(current)
        y = 24
        for line in text_lines:
            draw.text((24, y), line, font=font, fill="black")
            y += 48

        from . import ocr_engine

        ocr_result = ocr_engine.recognize(image)
        report.append(f"OCR：{ocr_result.engine}，{len(ocr_result.lines)} 行")
        stem, full = ocr_engine.build_queries(ocr_result)
        if not stem:
            raise RuntimeError("OCR 没有产生查询文本")
        ocr_hits, _ = bank.search(stem, limit=5)
        if not ocr_hits:
            raise RuntimeError("OCR 查询没有返回结果")
        report.append(
            f"OCR 检索：{ocr_hits[0].question.qid} {ocr_hits[0].percent}%"
        )
        if ocr_hits[0].score < 0.55:
            raise RuntimeError(f"OCR 匹配度过低：{ocr_hits[0].score:.2f}")

        report.append("SELFTEST OK")
        code = 0
    except BaseException as exc:
        report.append("SELFTEST FAILED")
        report.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        code = 1
    try:
        log_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    except Exception:
        pass
    return code

def main() -> int:
    if FROZEN:
        _silence_standard_streams()
    if "--self-test" in sys.argv:
        return _run_self_test()

    _enable_windows_dpi_awareness()
    root = tk.Tk()
    root.withdraw()
    try:
        bank = _load_bank()
    except Exception as exc:
        try:
            with ERROR_LOG.open("a", encoding="utf-8") as fh:
                fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
                fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        except Exception:
            pass
        messagebox.showerror(
            "启动失败",
            "题库加载失败。\n\n"
            f"{exc}\n\n"
            f"请确认以下文件存在：\n{TXT_PATH.name}\n"
            "或运行 python -m lawsjector.build_bank 重新生成题库。\n\n"
            f"错误日志：{ERROR_LOG.name}",
            parent=root,
        )
        root.destroy()
        return 1

    try:
        app = QuizApp(root, bank)
        root.deiconify()
        root.mainloop()
    except Exception as exc:
        try:
            with ERROR_LOG.open("a", encoding="utf-8") as fh:
                fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
                fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        except Exception:
            pass
        messagebox.showerror("程序异常", f"{exc}\n\n详情见 {ERROR_LOG.name}", parent=root)
        try:
            root.destroy()
        except Exception:
            pass
        return 1
    return 0
