from __future__ import annotations

import tkinter as tk
from typing import Callable

from PIL import Image, ImageEnhance, ImageTk

ACCENT = "#D4A72C"
DARK = "#0B2239"
TEXT = "#F4F7FB"

class RegionSnipper:
    def __init__(
        self,
        master: tk.Misc,
        image: Image.Image,
        on_select: Callable[[Image.Image], None],
        on_cancel: Callable[[], None],
    ) -> None:
        self.master = master
        self.image = image.convert("RGB")
        self.on_select = on_select
        self.on_cancel = on_cancel

        self.screen_w = master.winfo_screenwidth()
        self.screen_h = master.winfo_screenheight()
        self.scale_x = self.image.width / max(1, self.screen_w)
        self.scale_y = self.image.height / max(1, self.screen_h)

        self.start_x = 0
        self.start_y = 0
        self.dragging = False
        self.closed = False

        self.win = tk.Toplevel(master)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.win.configure(bg="#000000")
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)

        self.canvas = tk.Canvas(
            self.win,
            width=self.screen_w,
            height=self.screen_h,
            bg="#000000",
            highlightthickness=0,
            bd=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)

        display = self.image
        if (self.image.width, self.image.height) != (self.screen_w, self.screen_h):
            display = self.image.resize((self.screen_w, self.screen_h), Image.LANCZOS)
        display = ImageEnhance.Brightness(display).enhance(0.62)
        self._display_photo = ImageTk.PhotoImage(display)
        self.canvas.create_image(0, 0, image=self._display_photo, anchor="nw")

        self._draw_hint()
        self.rect = self.canvas.create_rectangle(
            0,
            0,
            0,
            0,
            outline=ACCENT,
            width=3,
            dash=(7, 4),
        )
        self.size_bg = self.canvas.create_rectangle(
            0, 0, 0, 0, fill=DARK, outline="", state="hidden"
        )
        self.size_text = self.canvas.create_text(
            0,
            0,
            text="",
            fill=TEXT,
            font=("Microsoft YaHei UI", 9, "bold"),
            anchor="se",
            state="hidden",
        )

        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<ButtonPress-3>", lambda _e: self._cancel())
        self.canvas.bind("<Escape>", lambda _e: self._cancel())
        self.win.bind("<Escape>", lambda _e: self._cancel())
        self.win.bind("<ButtonPress-3>", lambda _e: self._cancel())

        self.win.update_idletasks()
        self.win.lift()
        self.win.focus_force()
        try:
            self.win.grab_set()
        except tk.TclError:
            pass

    def _draw_hint(self) -> None:
        hint = "拖动鼠标框选题目区域    ·    右键 / Esc 取消"
        width = 440
        x = max(16, (self.screen_w - width) // 2)
        self.canvas.create_rectangle(
            x,
            18,
            x + width,
            58,
            fill=DARK,
            outline="",
        )
        self.canvas.create_text(
            x + width // 2,
            38,
            text=hint,
            fill=TEXT,
            font=("Microsoft YaHei UI", 10, "bold"),
        )

    def _press(self, event: tk.Event) -> None:
        self.start_x = max(0, min(self.screen_w, event.x))
        self.start_y = max(0, min(self.screen_h, event.y))
        self.dragging = True
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.start_x, self.start_y)
        self.canvas.itemconfigure(self.size_bg, state="hidden")
        self.canvas.itemconfigure(self.size_text, state="hidden")

    def _drag(self, event: tk.Event) -> None:
        if not self.dragging:
            return
        x = max(0, min(self.screen_w, event.x))
        y = max(0, min(self.screen_h, event.y))
        x0, x1 = sorted((self.start_x, x))
        y0, y1 = sorted((self.start_y, y))
        self.canvas.coords(self.rect, x0, y0, x1, y1)

        w, h = x1 - x0, y1 - y0
        if w >= 12 and h >= 12:
            label = f"{int(w * self.scale_x)} × {int(h * self.scale_y)}"
            tx = x1 - 6
            ty = y1 - 6
            if ty - 24 < 0:
                ty = y1 + 24
            self.canvas.coords(self.size_text, tx, ty)
            self.canvas.itemconfigure(self.size_text, text=label, state="normal")
            bbox = self.canvas.bbox(self.size_text)
            if bbox:
                self.canvas.coords(
                    self.size_bg,
                    bbox[0] - 7,
                    bbox[1] - 3,
                    bbox[2] + 7,
                    bbox[3] + 3,
                )
                self.canvas.itemconfigure(self.size_bg, state="normal")
                self.canvas.tag_raise(self.size_bg)
                self.canvas.tag_raise(self.size_text)

    def _release(self, event: tk.Event) -> None:
        if not self.dragging:
            return
        self.dragging = False
        x = max(0, min(self.screen_w, event.x))
        y = max(0, min(self.screen_h, event.y))
        x0, x1 = sorted((self.start_x, x))
        y0, y1 = sorted((self.start_y, y))

        if x1 - x0 < 14 or y1 - y0 < 12:
            self.canvas.coords(self.rect, 0, 0, 0, 0)
            self.canvas.itemconfigure(self.size_bg, state="hidden")
            self.canvas.itemconfigure(self.size_text, state="hidden")
            return

        left = max(0, min(self.image.width, int(x0 * self.scale_x)))
        top = max(0, min(self.image.height, int(y0 * self.scale_y)))
        right = max(0, min(self.image.width, int(x1 * self.scale_x)))
        bottom = max(0, min(self.image.height, int(y1 * self.scale_y)))
        if right - left < 8 or bottom - top < 8:
            return

        cropped = self.image.crop((left, top, right, bottom))
        self._close()
        self.on_select(cropped)

    def _cancel(self) -> None:
        if self.closed:
            return
        self._close()
        self.on_cancel()

    def _close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.win.grab_release()
        except tk.TclError:
            pass
        try:
            self.win.destroy()
        except tk.TclError:
            pass
