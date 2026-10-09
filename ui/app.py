"""
ui/app.py — Tractor Vision cockpit HUD: a single always-visible screen, not
a sidebar with separate pages. The live camera feed (with segmentation,
obstacles, and the real drivable path already drawn onto it by overlay.py)
is the dominant center element; an instrument cluster sits either side of
it (IMU attitude on the left, speed/brake/steer on the right), with a
status strip along the bottom for the nearest-obstacle readout and system
health. Everything relevant is visible at once, all the time — there is
nothing to navigate to.
"""

import time
import tkinter as tk

import cv2
from PIL import Image, ImageTk

import config
import vehicle_profile
from ui import theme
from ui.widgets import AttitudeIndicator, ArcGauge, SteerBar, Sparkline, StatusDot, ObstacleReadout


class TractorVisionApp:
    def __init__(self, shared_state, on_close=None, on_imu_zero=None):
        self.state = shared_state
        self.on_close = on_close

        self.root = tk.Tk()
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry("1500x900")
        self.root.configure(bg=theme.BG)
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close)

        self._video_tk_image = None

        self.root.minsize(1200, 720)

        # Layout order matters in Tk's pack manager: whatever is packed
        # LAST only gets the space left over. Fixed-size parts (top bar,
        # bottom bar, then both side panels) are packed FIRST so they always
        # keep their size; the video is packed last and just fills the rest.
        self._build_top_bar()
        self._build_bottom_bar()
        self._build_middle(on_imu_zero=on_imu_zero)

        # Second window with raw depth / SegFormer output (config.DEBUG).
        self._debug_window = None
        if config.DEBUG:
            from ui.debug_window import DebugWindow
            self._debug_window = DebugWindow(self.root, shared_state)

        # Video of the HUD (+ debug window) into config.RECORD_VIDEO_DIR.
        self._recorder = None
        if config.RECORD_VIDEO:
            from ui.screen_recorder import AppRecorder
            sources = [("hud", lambda: self.root)]
            if self._debug_window is not None:
                sources.append(("debug", lambda: (None if self._debug_window.closed
                                                  else self._debug_window.top)))
            self._recorder = AppRecorder(self.root, sources)
            # Started from _tick() once the first processed frame is on
            # screen — not on a timer, or every video would open with the
            # empty startup UI while the camera/models are still loading.

        self._tick()

    def run(self):
        self.root.mainloop()

    def _handle_close(self):
        # Finalize the video files first, while the windows still exist.
        if self._recorder is not None:
            self._recorder.stop()
        if self.on_close:
            self.on_close()
        self.root.destroy()

    # ------------------------------------------------------------- top bar

    def _build_top_bar(self):
        top = tk.Frame(self.root, bg=theme.PANEL_BG, height=44)
        self._top_bar = top
        top.pack(side=tk.TOP, fill=tk.X)
        top.pack_propagate(False)

        left = tk.Frame(top, bg=theme.PANEL_BG)
        left.pack(side=tk.LEFT, padx=theme.PAD_LG)
        tk.Label(left, text="TRACTOR VISION", fg=theme.ACCENT, bg=theme.PANEL_BG,
                 font=theme.FONT_TITLE).pack(side=tk.LEFT, pady=8)
        tk.Label(left, text="  PERCEPTION + CONTROL", fg=theme.TEXT_FAINT, bg=theme.PANEL_BG,
                 font=theme.FONT_LABEL).pack(side=tk.LEFT, pady=8)

        right = tk.Frame(top, bg=theme.PANEL_BG)
        right.pack(side=tk.RIGHT, padx=theme.PAD_LG)
        self.fps_label = tk.Label(right, text="-- FPS", fg=theme.TEXT_DIM, bg=theme.PANEL_BG,
                                   font=theme.FONT_MONO_SM)
        self.fps_label.pack(side=tk.RIGHT, padx=(18, 0), pady=8)
        self.dot_yolo = StatusDot(right, "DETECT", bg=theme.PANEL_BG)
        self.dot_yolo.pack(side=tk.RIGHT, padx=(18, 0), pady=8)
        self.dot_imu = StatusDot(right, "IMU", bg=theme.PANEL_BG)
        self.dot_imu.pack(side=tk.RIGHT, padx=(18, 0), pady=8)
        self.dot_camera = StatusDot(right, "CAMERA", bg=theme.PANEL_BG)
        self.dot_camera.pack(side=tk.RIGHT, padx=(18, 0), pady=8)

        self.degraded_banner = tk.Label(
            self.root, text="", fg="#1a1400", bg=theme.DANGER,
            font=theme.FONT_STATUS, pady=6,
        )
        self._degraded_visible = False

    # --------------------------------------------------------------- middle

    def _build_middle(self, on_imu_zero=None):
        middle = tk.Frame(self.root, bg=theme.BG)
        middle.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._build_left_panel(middle, on_imu_zero=on_imu_zero)
        self._build_right_panel(middle)
        self._build_video_panel(middle)

    def _build_left_panel(self, parent, on_imu_zero=None):
        panel = tk.Frame(parent, bg=theme.PANEL_BG, width=280)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        tk.Label(panel, text="ATTITUDE (IMU)", fg=theme.IMU_ACCENT, bg=theme.PANEL_BG,
                 font=theme.FONT_LABEL_BOLD).pack(anchor="w", padx=theme.PAD_LG, pady=(theme.PAD_LG, 8))

        card = tk.Frame(panel, bg=theme.CARD_BG, highlightbackground=theme.BORDER, highlightthickness=1)
        card.pack(padx=theme.PAD_LG, fill=tk.X)
        self.attitude = AttitudeIndicator(card, size=180, bg=theme.CARD_BG)
        self.attitude.pack(pady=(theme.PAD_MD, 4))

        # Fixed-width grid (NOT pack(side=LEFT), whose column width silently
        # depends on each cell's current content) — three EQUAL columns
        # wide enough for the longest realistic value ("-999.9°", worst
        # case) so a longer number appearing in the LAST column can never
        # run past the card's edge the way it could with content-sized
        # packing (confirmed by testing: "-14.0°" in the first column
        # rendered fine while "26.0°" in the last column clipped, because
        # the last column had no leftover margin to grow into).
        readout = tk.Frame(card, bg=theme.CARD_BG)
        readout.pack(pady=(0, theme.PAD_MD), fill=tk.X, padx=4)
        for i in range(3):
            readout.columnconfigure(i, weight=1, uniform="imu_readout")
        self.pitch_label = self._mini_readout(readout, "PITCH", col=0)
        self.roll_label = self._mini_readout(readout, "ROLL", col=1)
        self.tilt_label = self._mini_readout(readout, "TILT", col=2)

        self.rollover_label = tk.Label(panel, text="ROLLOVER RISK: --", fg=theme.TEXT_DIM,
                                        bg=theme.PANEL_BG, font=theme.FONT_LABEL_BOLD)
        self.rollover_label.pack(anchor="w", padx=theme.PAD_LG, pady=(theme.PAD_MD, 4))

        self.imu_status_label = tk.Label(panel, text="IMU: not started", fg=theme.TEXT_FAINT,
                                          bg=theme.PANEL_BG, font=theme.FONT_LABEL, wraplength=245,
                                          justify="left")
        self.imu_status_label.pack(anchor="w", padx=theme.PAD_LG)

        if on_imu_zero:
            tk.Button(panel, text="Zero IMU Orientation", command=on_imu_zero,
                      bg=theme.CARD_BG, fg=theme.TEXT, relief=tk.FLAT,
                      font=theme.FONT_LABEL, padx=8, pady=6).pack(
                anchor="w", padx=theme.PAD_LG, pady=(theme.PAD_MD, 0))

        tk.Label(panel, text="TILT TREND", fg=theme.TEXT_FAINT, bg=theme.PANEL_BG,
                 font=theme.FONT_LABEL_BOLD).pack(anchor="w", padx=theme.PAD_LG, pady=(theme.PAD_LG, 4))
        self.tilt_spark = Sparkline(panel, width=245, height=36, color=theme.IMU_ACCENT, bg=theme.PANEL_BG)
        self.tilt_spark.pack(padx=theme.PAD_LG)

    def _mini_readout(self, parent, title, col):
        colframe = tk.Frame(parent, bg=theme.CARD_BG)
        colframe.grid(row=0, column=col, sticky="nsew")
        tk.Label(colframe, text=title, fg=theme.TEXT_FAINT, bg=theme.CARD_BG,
                 font=theme.FONT_LABEL).pack()
        value = tk.Label(colframe, text="0.0°", fg=theme.TEXT, bg=theme.CARD_BG,
                          font=theme.FONT_READOUT_SM)
        value.pack()
        return value

    def _build_video_panel(self, parent):
        # pack_propagate(False) + place(): the image inside can NEVER change
        # this frame's requested size. Before, the label was sized by its own
        # image, so it could only grow — when the red warning banner took
        # height away, the video kept its size and squeezed the right panel
        # out of the window.
        self.video_frame = tk.Frame(parent, bg="#000000", width=320, height=240)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.video_frame.pack_propagate(False)
        self.video_label = tk.Label(self.video_frame, bg="#000000", bd=0, highlightthickness=0)
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

    def _build_right_panel(self, parent):
        panel = tk.Frame(parent, bg=theme.PANEL_BG, width=250)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        gauges = tk.Frame(panel, bg=theme.PANEL_BG)
        gauges.pack(pady=(theme.PAD_LG, 0))
        self.speed_gauge = ArcGauge(gauges, size=112, max_value=30.0, color=theme.ACCENT,
                                     bg=theme.PANEL_BG, unit="km/h", title="SPEED")
        self.speed_gauge.pack(side=tk.LEFT, padx=8)
        self.brake_gauge = ArcGauge(gauges, size=112, max_value=100.0, color=theme.DANGER,
                                     bg=theme.PANEL_BG, unit="%", title="BRAKE")
        self.brake_gauge.pack(side=tk.LEFT, padx=8)

        tk.Label(panel, text="STEER SUGGESTION", fg=theme.TEXT_DIM, bg=theme.PANEL_BG,
                 font=theme.FONT_LABEL_BOLD).pack(anchor="w", padx=theme.PAD_LG, pady=(theme.PAD_LG, 4))
        steer_card = tk.Frame(panel, bg=theme.CARD_BG, highlightbackground=theme.BORDER,
                               highlightthickness=1)
        steer_card.pack(padx=theme.PAD_LG, fill=tk.X)
        self.steer_bar = SteerBar(steer_card, width=196, height=56,
                                   max_deg=config.MAX_STEER_SUGGESTION_DEG, bg=theme.CARD_BG)
        self.steer_bar.pack(padx=8, pady=8)

        tk.Label(panel, text="SPEED TREND", fg=theme.TEXT_FAINT, bg=theme.PANEL_BG,
                 font=theme.FONT_LABEL_BOLD).pack(anchor="w", padx=theme.PAD_LG, pady=(theme.PAD_LG, 4))
        self.speed_spark = Sparkline(panel, width=215, height=36, color=theme.ACCENT, bg=theme.PANEL_BG)
        self.speed_spark.pack(padx=theme.PAD_LG)

        tk.Label(panel, text="BRAKE TREND", fg=theme.TEXT_FAINT, bg=theme.PANEL_BG,
                 font=theme.FONT_LABEL_BOLD).pack(anchor="w", padx=theme.PAD_LG, pady=(theme.PAD_LG, 4))
        self.brake_spark = Sparkline(panel, width=215, height=36, color=theme.DANGER, bg=theme.PANEL_BG)
        self.brake_spark.pack(padx=theme.PAD_LG)

        self._build_vehicle_size_panel(panel)

    # ---------------------------------------------------------- vehicle size

    def _build_vehicle_size_panel(self, panel):
        """
        Runtime vehicle size. Apply updates config live (next frame uses
        it — corridor, path gap-fitting and the AR drivable surface are all
        width-driven) and saves to disk so it survives a restart. See
        vehicle_profile.py. Invalid input is rejected with a message and
        changes nothing.
        """
        tk.Label(panel, text="VEHICLE SIZE", fg=theme.TEXT_DIM, bg=theme.PANEL_BG,
                 font=theme.FONT_LABEL_BOLD).pack(anchor="w", padx=theme.PAD_LG, pady=(theme.PAD_LG, 4))
        card = tk.Frame(panel, bg=theme.CARD_BG, highlightbackground=theme.BORDER, highlightthickness=1)
        card.pack(padx=theme.PAD_LG, fill=tk.X)

        self.vehicle_unit = tk.StringVar(value="ft")
        unit_row = tk.Frame(card, bg=theme.CARD_BG)
        unit_row.pack(fill=tk.X, padx=8, pady=(8, 4))
        tk.Label(unit_row, text="Units", fg=theme.TEXT_FAINT, bg=theme.CARD_BG,
                 font=theme.FONT_LABEL).pack(side=tk.LEFT)
        for u in ("m", "ft"):
            tk.Radiobutton(unit_row, text=u, value=u, variable=self.vehicle_unit,
                           command=self._fill_vehicle_entries,
                           bg=theme.CARD_BG, fg=theme.TEXT, selectcolor=theme.PANEL_BG,
                           activebackground=theme.CARD_BG, activeforeground=theme.ACCENT,
                           font=theme.FONT_LABEL, highlightthickness=0, bd=0).pack(side=tk.RIGHT, padx=(6, 0))

        self.vehicle_width_entry = self._vehicle_entry_row(card, "Width")
        self.vehicle_length_entry = self._vehicle_entry_row(card, "Length")

        tk.Button(card, text="Apply", command=self._apply_vehicle_size,
                  bg=theme.ACCENT, fg="#1a1000", activebackground=theme.ACCENT_DIM,
                  relief=tk.FLAT, font=theme.FONT_LABEL_BOLD, padx=10, pady=4).pack(
            fill=tk.X, padx=8, pady=(6, 4))

        self.vehicle_status = tk.Label(card, text="", fg=theme.TEXT_FAINT, bg=theme.CARD_BG,
                                       font=theme.FONT_LABEL, wraplength=196, justify="left")
        self.vehicle_status.pack(anchor="w", padx=8, pady=(0, 8))

        self._fill_vehicle_entries()
        self._show_active_vehicle_size()

    def _vehicle_entry_row(self, parent, label):
        row = tk.Frame(parent, bg=theme.CARD_BG)
        row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(row, text=label, fg=theme.TEXT_FAINT, bg=theme.CARD_BG,
                 font=theme.FONT_LABEL, width=7, anchor="w").pack(side=tk.LEFT)
        entry = tk.Entry(row, width=9, bg=theme.PANEL_BG, fg=theme.TEXT, insertbackground=theme.TEXT,
                         relief=tk.FLAT, font=theme.FONT_READOUT_SM, justify="right",
                         highlightthickness=1, highlightbackground=theme.BORDER,
                         highlightcolor=theme.ACCENT)
        entry.pack(side=tk.RIGHT)
        entry.bind("<Return>", lambda _e: self._apply_vehicle_size())
        return entry

    def _fill_vehicle_entries(self):
        """Shows the ACTIVE size in the selected unit (also used when the
        unit toggle changes, so the numbers always mean what they say)."""
        unit = self.vehicle_unit.get()
        for entry, val_m in ((self.vehicle_width_entry, config.TRACTOR_WIDTH_M),
                             (self.vehicle_length_entry, config.TRACTOR_LENGTH_M)):
            entry.delete(0, tk.END)
            entry.insert(0, f"{vehicle_profile.from_meters(val_m, unit):.2f}")

    def _show_active_vehicle_size(self, prefix="Active"):
        w, l = config.TRACTOR_WIDTH_M, config.TRACTOR_LENGTH_M
        ft = vehicle_profile.FT_PER_M
        self.vehicle_status.configure(
            text=f"{prefix}: {w * ft:.1f} × {l * ft:.1f} ft\n({w:.2f} × {l:.2f} m)",
            fg=theme.GOOD if prefix != "Active" else theme.TEXT_FAINT)

    def _apply_vehicle_size(self):
        unit = self.vehicle_unit.get()
        try:
            width_m = vehicle_profile.to_meters(self.vehicle_width_entry.get().strip(), unit)
            length_m = vehicle_profile.to_meters(self.vehicle_length_entry.get().strip(), unit)
        except ValueError:
            self.vehicle_status.configure(text="Enter numbers only", fg=theme.DANGER)
            return
        try:
            vehicle_profile.set_vehicle_size(width_m, length_m)
        except ValueError as e:
            self.vehicle_status.configure(text=str(e), fg=theme.DANGER)
            return
        self._fill_vehicle_entries()
        self._show_active_vehicle_size(prefix="Applied")

    # ---------------------------------------------------------------- bottom

    def _build_bottom_bar(self):
        bottom = tk.Frame(self.root, bg=theme.PANEL_BG, height=90)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        bottom.pack_propagate(False)

        left = tk.Frame(bottom, bg=theme.PANEL_BG)
        left.pack(side=tk.LEFT, padx=theme.PAD_LG, pady=8)
        self.obstacle_readout = ObstacleReadout(left, bg=theme.PANEL_BG)
        self.obstacle_readout.pack()

        mid = tk.Frame(bottom, bg=theme.PANEL_BG)
        mid.pack(side=tk.LEFT, padx=theme.PAD_LG * 2, pady=8)
        tk.Label(mid, text="GROUND SEGMENTATION", fg=theme.TEXT_DIM, bg=theme.PANEL_BG,
                 font=theme.FONT_LABEL_BOLD).pack(anchor="w")
        row = tk.Frame(mid, bg=theme.PANEL_BG)
        row.pack(anchor="w", pady=(2, 0))
        self.land_pct_label = tk.Label(row, text="--", fg=theme.TEXT, bg=theme.PANEL_BG,
                                        font=theme.FONT_READOUT_MD)
        self.land_pct_label.pack(side=tk.LEFT)
        tk.Label(row, text=" % clear", fg=theme.TEXT_DIM, bg=theme.PANEL_BG,
                 font=theme.FONT_UNIT).pack(side=tk.LEFT, anchor="s", pady=(0, 4))
        self.path_status_label = tk.Label(mid, text="", fg=theme.TEXT_DIM, bg=theme.PANEL_BG,
                                           font=theme.FONT_LABEL)
        self.path_status_label.pack(anchor="w")

        right = tk.Frame(bottom, bg=theme.PANEL_BG)
        right.pack(side=tk.RIGHT, padx=theme.PAD_LG, pady=8)
        self.status_text_label = tk.Label(right, text="starting...", fg=theme.TEXT_DIM,
                                           bg=theme.PANEL_BG, font=theme.FONT_LABEL_BOLD,
                                           wraplength=320, justify="right")
        self.status_text_label.pack(anchor="e")

    # ------------------------------------------------------------------ tick

    def _tick(self):
        snap = self.state.snapshot()
        history = self.state.snapshot_history()
        imu = snap["imu"]

        # --- video ---
        frame = snap["annotated_frame"]
        if frame is not None:
            label_w = max(self.video_frame.winfo_width() - 2, 160)
            label_h = max(self.video_frame.winfo_height() - 2, 120)
            h, w = frame.shape[:2]
            scale = min(label_w / w, label_h / h)
            disp_w, disp_h = max(1, int(w * scale)), max(1, int(h * scale))
            disp = cv2.resize(frame, (disp_w, disp_h)) if scale != 1.0 else frame
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            self._video_tk_image = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.video_label.configure(image=self._video_tk_image)
            if self._recorder is not None and not self._recorder.started:
                self._recorder.start()

        # --- top bar ---
        self.fps_label.configure(text=f"{snap['fps']:.1f} FPS")
        camera_ok = "Camera error" not in snap["status_text"] and "Frame error" not in snap["status_text"]
        self.dot_camera.set("ok" if camera_ok else "danger")
        self.dot_imu.set("ok" if imu.get("available") else "off")
        yolo_status = snap.get("yolo_status", "not started")
        if yolo_status.startswith("running"):
            self.dot_yolo.set("ok")
        elif yolo_status.startswith("unavailable"):
            self.dot_yolo.set("off")
        else:
            self.dot_yolo.set("warning")

        # --- IMU / attitude ---
        pitch = imu.get("pitch_deg", 0.0)
        roll = imu.get("roll_deg", 0.0)
        tilt = imu.get("tilt_deg", 0.0)
        risk = imu.get("rollover_risk", "ok") if imu.get("available") else "off"
        self.attitude.set(pitch, roll, risk if risk in ("ok", "warning", "danger") else "off")
        self.pitch_label.configure(text=f"{pitch:+.1f}°")
        self.roll_label.configure(text=f"{roll:+.1f}°")
        self.tilt_label.configure(text=f"{tilt:.1f}°")
        risk_text = {"ok": "OK", "warning": "WARNING", "danger": "DANGER"}.get(risk, "--")
        self.rollover_label.configure(text=f"ROLLOVER RISK: {risk_text}", fg=theme.status_color(risk))
        if imu.get("available"):
            self.imu_status_label.configure(text="Streaming pitch/roll/yaw", fg=theme.TEXT_FAINT)
        else:
            self.imu_status_label.configure(
                text=imu.get("status", "unavailable"), fg=theme.TEXT_FAINT
            )
        self.tilt_spark.set(history.get("tilt_deg", []))

        # --- speed / brake / steer ---
        speed_kmh = snap["speed_mps"] * 3.6
        self.speed_gauge.set(speed_kmh, text_override=f"{speed_kmh:.0f}")
        brake_pct = snap["brake_percent"]
        brake_color = theme.GOOD if brake_pct < 20 else (theme.WARN if brake_pct < 60 else theme.DANGER)
        self.brake_gauge.set(brake_pct, text_override=f"{brake_pct:.0f}", color_override=brake_color)
        self.steer_bar.set(snap["steer_suggestion_deg"])
        self.speed_spark.set(history.get("speed_kmh", []))
        self.brake_spark.set(history.get("brake_pct", []))

        # --- obstacles ---
        self.obstacle_readout.set(
            snap["nearest_obstacle_m"], snap["closing_speed_mps"], len(snap["obstacles"])
        )

        # --- segmentation / path / status ---
        self.land_pct_label.configure(text=f"{snap['land_coverage_pct']:.0f}")
        blocked = snap["path_blocked_at_m"]
        if blocked is not None:
            self.path_status_label.configure(text=f"Path blocked at {blocked:.1f} m", fg=theme.DANGER)
        else:
            self.path_status_label.configure(text="Path clear ahead", fg=theme.GOOD)
        self.status_text_label.configure(text=snap["status_text"])

        # --- degraded banner ---
        if snap["watchdog_degraded"]:
            self.degraded_banner.configure(
                text=f'⚠ PERCEPTION DEGRADED {snap["degraded_duration_s"]:.1f}s — SAFETY BRAKE ENGAGED'
            )
            if not self._degraded_visible:
                self.degraded_banner.pack(side=tk.TOP, fill=tk.X, after=self._top_bar)
                self._degraded_visible = True
        else:
            if self._degraded_visible:
                self.degraded_banner.pack_forget()
                self._degraded_visible = False

        if self._debug_window is not None:
            self._debug_window.refresh()

        self.root.after(config.UPDATE_INTERVAL_MS, self._tick)
