"""
playback/ui_controls.py — a Tkinter transport bar for a PlaybackController:
play/pause, step, restart, skip, a scrub timeline, a speed selector and a
loop toggle.

Standalone on purpose. It imports only tkinter and the controller it is
given, takes its colours as a plain dict, and packs into any parent
frame — so it drops into an existing Tk app without that app knowing
anything about playback, and into a different project without edits.

    bar = PlaybackBar(root, controller, palette=PlaybackBar.palette_from(theme))
    bar.pack(side=tk.BOTTOM, fill=tk.X)
    bar.start_polling()        # refreshes itself from controller.status()
    bar.bind_keys(root)        # space / arrows / home

Scrubbing note: the slider only commits a seek on RELEASE, not while
being dragged. A seek is a perception discontinuity (trackers and
temporal filters get reset), so firing one per pixel of mouse travel
would be both wasteful and visibly unstable.
"""

import tkinter as tk

DEFAULT_PALETTE = {
    "bg": "#12161c",
    "card": "#1b222c",
    "text": "#e6edf3",
    "dim": "#9aa7b4",
    "faint": "#6b7785",
    "accent": "#ffc14d",
    "accent_text": "#1a1000",
    "border": "#2b3542",
    "good": "#4ade80",
}

SPEED_CHOICES = [("max", 0.0), ("0.25x", 0.25), ("0.5x", 0.5),
                 ("1x", 1.0), ("2x", 2.0), ("4x", 4.0)]


def _format_time(seconds):
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


class PlaybackBar(tk.Frame):
    """Transport controls for one PlaybackController."""

    def __init__(self, parent, controller, palette=None, refresh_ms=100,
                 skip_seconds=5.0, on_status=None):
        self.palette = dict(DEFAULT_PALETTE)
        if palette:
            self.palette.update({k: v for k, v in palette.items() if v})
        p = self.palette

        super().__init__(parent, bg=p["bg"], highlightbackground=p["border"],
                         highlightthickness=1)
        self.controller = controller
        self.refresh_ms = refresh_ms
        self.skip_seconds = skip_seconds
        self._on_status = on_status
        self._polling = False
        self._scrubbing = False
        self._last_status = {}

        self._build()

    # --------------------------------------------------------------- layout

    def _build(self):
        """
        ONE row, not three.

        Vertical space here is not free: this bar is added to a host
        window that is usually already as tall as the screen allows, so
        every row it takes comes straight out of the host's own panels.
        Measured on the tractor HUD at 1080p/125% scaling — a 1500x900
        request is clamped by Windows to 845 — a three-row bar (controls,
        timeline, sequence name) cut the bottom off the side panel and
        reserve_space() could not win the height back because there was
        none left to take. So the timeline shares the control row, and
        the sequence description goes in the window title instead, where
        it costs nothing. See describe_sequence().
        """
        p = self.palette

        row = tk.Frame(self, bg=p["bg"])
        row.pack(fill=tk.X, padx=10, pady=5)

        self.play_button = self._button(row, "▶  Play", self._toggle, primary=True, width=9)
        self.play_button.pack(side=tk.LEFT)
        self._button(row, "⏮", self.controller.restart, width=3).pack(side=tk.LEFT, padx=(5, 0))
        self._button(row, f"-{self.skip_seconds:g}s", self._skip_back, width=4).pack(side=tk.LEFT, padx=(5, 0))
        self._button(row, f"+{self.skip_seconds:g}s", self._skip_forward, width=4).pack(side=tk.LEFT, padx=(3, 0))
        self._button(row, "Step →", lambda: self.controller.step(1), width=6).pack(side=tk.LEFT, padx=(5, 0))

        # Right-hand items are packed before the slider so the slider is
        # the only thing that absorbs leftover width.
        self.counter_label = tk.Label(row, text="frame --", bg=p["bg"], fg=p["dim"],
                                      font=("Consolas", 9), width=20, anchor="e")
        self.counter_label.pack(side=tk.RIGHT, padx=(10, 0))

        self.loop_var = tk.BooleanVar(value=self.controller.loop)
        tk.Checkbutton(row, text="Loop", variable=self.loop_var, command=self._set_loop,
                       bg=p["bg"], fg=p["dim"], selectcolor=p["card"],
                       activebackground=p["bg"], activeforeground=p["text"],
                       highlightthickness=0, bd=0).pack(side=tk.RIGHT, padx=(10, 0))

        self.speed_var = tk.StringVar(value=self._speed_label(self.controller.speed))
        speed_menu = tk.OptionMenu(row, self.speed_var, *[label for label, _ in SPEED_CHOICES],
                                   command=self._set_speed)
        speed_menu.configure(bg=p["card"], fg=p["text"], activebackground=p["border"],
                             activeforeground=p["text"], relief=tk.FLAT, highlightthickness=0,
                             width=4, anchor="w", pady=1)
        speed_menu["menu"].configure(bg=p["card"], fg=p["text"], relief=tk.FLAT)
        speed_menu.pack(side=tk.RIGHT)
        tk.Label(row, text="Speed", bg=p["bg"], fg=p["faint"]).pack(side=tk.RIGHT, padx=(14, 4))

        self.duration_label = tk.Label(row, text=_format_time(self.controller.duration_s),
                                       bg=p["bg"], fg=p["faint"], font=("Consolas", 9),
                                       width=7, anchor="w")
        self.duration_label.pack(side=tk.RIGHT)

        self.position_label = tk.Label(row, text="0:00.0", bg=p["bg"], fg=p["text"],
                                       font=("Consolas", 9), width=7, anchor="e")
        self.position_label.pack(side=tk.LEFT, padx=(14, 0))

        self.scrub = tk.Scale(
            row, from_=0.0, to=max(0.001, self.controller.duration_s),
            resolution=0.01, orient=tk.HORIZONTAL, showvalue=False,
            bg=p["bg"], fg=p["text"], troughcolor=p["card"], activebackground=p["accent"],
            highlightthickness=0, bd=0, sliderrelief=tk.FLAT, length=200, width=10,
        )
        self.scrub.bind("<Button-1>", self._scrub_begin)
        self.scrub.bind("<ButtonRelease-1>", self._scrub_commit)
        self.scrub.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

    def describe_sequence(self):
        """What is being played, as one line — for the host's window
        title, since this bar has no row to spare for it."""
        return self.controller.source.info.summary()

    def _button(self, parent, text, command, primary=False, width=8):
        p = self.palette
        return tk.Button(
            parent, text=text, command=command, width=width,
            bg=p["accent"] if primary else p["card"],
            fg=p["accent_text"] if primary else p["text"],
            activebackground=p["accent"] if primary else p["border"],
            activeforeground=p["accent_text"] if primary else p["text"],
            relief=tk.FLAT, bd=0, padx=6, pady=3,
        )

    # ------------------------------------------------------------- commands

    def _toggle(self):
        self.controller.toggle_pause()
        self.refresh()

    def _skip_back(self):
        self.controller.seek(self.controller.position_s - self.skip_seconds)

    def _skip_forward(self):
        self.controller.seek(self.controller.position_s + self.skip_seconds)

    def _speed_label(self, speed):
        for label, value in SPEED_CHOICES:
            if abs(value - speed) < 1e-6:
                return label
        return f"{speed:g}x"

    def _set_speed(self, label):
        for name, value in SPEED_CHOICES:
            if name == label:
                self.controller.set_speed(value)
                return

    def _set_loop(self):
        self.controller.set_loop(self.loop_var.get())

    def _scrub_begin(self, _event=None):
        self._scrubbing = True

    def _scrub_commit(self, _event=None):
        self._scrubbing = False
        self.controller.seek(float(self.scrub.get()))

    # ------------------------------------------------------------- refresh

    def reserve_space(self, toplevel):
        """
        Grow `toplevel` by this bar's height, so adding the transport
        takes space from nothing else.

        Worth doing rather than leaving to the window manager: a host
        laid out with pack() gives leftover space to whatever was packed
        last, so a bar added afterwards silently squeezes a fixed-size
        panel until its lower rows are cut off — which is exactly what
        happened the first time this was wired into the tractor HUD (the
        vehicle-size card lost its bottom line). Also raises minsize by
        the same amount, so shrinking the window cannot recreate it.

        Best-effort, not a guarantee: the window manager clamps to the
        usable screen area, so a host window that is already full-height
        simply stays put (measured: a 1500x900 request becomes 845 at
        1080p/125% scaling). That is precisely why _build() is one row —
        the bar has to be affordable even when this can win nothing back.
        """
        self.update_idletasks()
        extra = self.winfo_reqheight()
        if extra <= 1:
            return

        try:
            toplevel.update_idletasks()
            geometry = toplevel.geometry()                  # "WxH+X+Y"
            size, _, _ = geometry.partition("+")
            width, height = (int(v) for v in size.split("x"))
            # An unmapped Tk window reports "1x1+0+0" no matter what
            # geometry() was set to; resizing from that would shrink the
            # host to nothing. Only act on a size that is really there.
            if width > 1 and height > 1:
                toplevel.geometry(f"{width}x{height + extra}")
        except Exception:
            pass

        try:
            min_w, min_h = toplevel.minsize()
            if min_w and min_h:
                toplevel.minsize(min_w, min_h + extra)
        except Exception:
            pass

    def bind_keys(self, widget):
        """Space = play/pause, Right/Left = step/skip, Home = restart.
        Bound on `widget` (usually the root window) rather than on this
        frame, so the shortcuts work wherever focus happens to be."""
        widget.bind("<space>", lambda _e: self._toggle())
        widget.bind("<Right>", lambda _e: self.controller.step(1))
        widget.bind("<Left>", lambda _e: self._skip_back())
        widget.bind("<Home>", lambda _e: self.controller.restart())

    def start_polling(self):
        if not self._polling:
            self._polling = True
            self._poll()

    def stop_polling(self):
        self._polling = False

    def _poll(self):
        if not self._polling:
            return
        self.refresh()
        self.after(self.refresh_ms, self._poll)

    def refresh(self):
        status = self.controller.status()
        self._last_status = status
        p = self.palette

        if status["at_end"]:
            label, primary = "↻  Replay", True
        elif status["paused"]:
            label, primary = "▶  Play", True
        else:
            label, primary = "⏸  Pause", False
        self.play_button.configure(
            text=label,
            bg=p["accent"] if primary else p["card"],
            fg=p["accent_text"] if primary else p["text"],
            activebackground=p["accent"] if primary else p["border"],
        )

        duration = status["duration_s"]
        if duration > 0 and abs(float(self.scrub.cget("to")) - duration) > 0.01:
            self.scrub.configure(to=duration)
            self.duration_label.configure(text=_format_time(duration))
        if not self._scrubbing:
            self.scrub.set(status["position_s"])
        self.position_label.configure(text=_format_time(status["position_s"]))

        total = status["frames_per_pass"]
        counter = f"frame {status['frame_index']}"
        if total:
            counter += f"/{total}"
        if status["pass_index"]:
            counter += f"  pass {status['pass_index'] + 1}"
        self.counter_label.configure(text=counter)

        if self._on_status:
            self._on_status(status)

    @staticmethod
    def palette_from(theme_module):
        """Borrow a host project's colours. Any name the host does not
        define is just left at this module's own default."""
        def pick(*names):
            for name in names:
                value = getattr(theme_module, name, None)
                if value:
                    return value
            return None

        return {
            "bg": pick("PANEL_BG", "BG"),
            "card": pick("CARD_BG"),
            "text": pick("TEXT"),
            "dim": pick("TEXT_DIM"),
            "faint": pick("TEXT_FAINT"),
            "accent": pick("ACCENT"),
            "border": pick("BORDER"),
            "good": pick("GOOD"),
        }
