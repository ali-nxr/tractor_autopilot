"""
ui/debug_window.py — the second window opened when config.DEBUG is True.
Shows debug_view.py's panel grid (raw depth, SegFormer output, fusion,
height above plane) and, under the mouse, the exact per-pixel values with
a crosshair at the same scene point in every panel.

Refreshed from TractorVisionApp's own tick, on the Tk thread — the vision
thread only ever hands over a finished image + probe via SharedState.
"""

import tkinter as tk

import cv2
from PIL import Image, ImageTk

import config
import debug_view
from ui import theme

_HINT = "Hover any panel to read depth / driveable probability / class / height at that pixel"


class DebugWindow:
    def __init__(self, root, shared_state):
        self.state = shared_state
        self.closed = False
        self._mouse = None          # (x, y) in displayed-image coords, or None
        self._scale = 1.0
        self._tk_image = None

        self.top = tk.Toplevel(root)
        self.top.title(f"{config.WINDOW_TITLE} — Debug")
        self.top.configure(bg=theme.BG)
        pw = int(config.DEBUG_PANEL_WIDTH)
        ph = int(round(pw * config.FRAME_HEIGHT / config.FRAME_WIDTH))
        self.top.geometry(f"{3 * pw + 8}x{2 * ph + 40}")
        self.top.protocol("WM_DELETE_WINDOW", self._close)

        self.readout = tk.Label(self.top, text=_HINT, bg=theme.PANEL_BG, fg=theme.TEXT_DIM,
                                font=theme.FONT_MONO_SM, anchor="w", padx=theme.PAD_SM, pady=4)
        self.readout.pack(side=tk.BOTTOM, fill=tk.X)

        # Same fixed-container trick as the HUD's video panel: the image can
        # never resize its own container, so the window can be resized freely.
        self.frame = tk.Frame(self.top, bg="#000000")
        self.frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.frame.pack_propagate(False)
        self.image_label = tk.Label(self.frame, bg="#000000", bd=0, highlightthickness=0,
                                    text="waiting for the first frame...", fg=theme.TEXT_DIM)
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")
        self.image_label.bind("<Motion>", self._on_motion)
        self.image_label.bind("<Leave>", self._on_leave)

    def _close(self):
        # Closing only hides debugging; the HUD keeps running.
        self.closed = True
        self.top.destroy()

    def _on_motion(self, event):
        self._mouse = (event.x, event.y)

    def _on_leave(self, _event):
        self._mouse = None

    def refresh(self):
        if self.closed:
            return
        mosaic, probe = self.state.get_debug()
        if mosaic is None:
            return

        readout = _HINT
        if self._mouse is not None and self._scale > 0:
            pixel = debug_view.panel_pixel(probe, self._mouse[0] / self._scale,
                                           self._mouse[1] / self._scale)
            if pixel is not None:
                mosaic = debug_view.draw_crosshair(mosaic, probe, *pixel)
                readout = debug_view.describe(probe, *pixel)
        self.readout.configure(text=readout, fg=theme.TEXT if readout is not _HINT else theme.TEXT_DIM)

        avail_w = max(self.frame.winfo_width(), 160)
        avail_h = max(self.frame.winfo_height(), 90)
        h, w = mosaic.shape[:2]
        self._scale = min(avail_w / w, avail_h / h)
        disp_w, disp_h = max(1, int(w * self._scale)), max(1, int(h * self._scale))
        disp = cv2.resize(mosaic, (disp_w, disp_h), interpolation=cv2.INTER_AREA) \
            if (disp_w, disp_h) != (w, h) else mosaic
        self._tk_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        self.image_label.configure(image=self._tk_image, text="")
