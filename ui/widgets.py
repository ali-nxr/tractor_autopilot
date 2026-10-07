"""
ui/widgets.py — instrument-cluster widgets for the cockpit HUD layout (see
ui/app.py). Deliberately not a "cards with numbers" dashboard: an artificial-
horizon attitude indicator for IMU (the same visual language a real
aircraft/heavy-equipment attitude gauge uses — legible at a glance without
reading two separate numbers), circular arc gauges for speed/brake, a
horizontal steer bar, and small trend sparklines.
"""

import math
import tkinter as tk

from PIL import Image, ImageDraw, ImageTk

from ui import theme


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


class StatusDot(tk.Frame):
    """Small colored dot + label — connection/health status (Camera, IMU)."""

    def __init__(self, parent, label, bg=None):
        self._bg = bg or theme.PANEL_BG
        super().__init__(parent, bg=self._bg)
        self.canvas = tk.Canvas(self, width=10, height=10, bg=self._bg, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=(0, 6))
        self._dot = self.canvas.create_oval(1, 1, 9, 9, fill=theme.TEXT_FAINT, outline="")
        self.label = tk.Label(self, text=label, fg=theme.TEXT_DIM, bg=self._bg, font=theme.FONT_LABEL_BOLD)
        self.label.pack(side=tk.LEFT)

    def set(self, status):
        self.canvas.itemconfig(self._dot, fill=theme.status_color(status))


class AttitudeIndicator(tk.Label):
    """
    Artificial-horizon attitude indicator: sky/ground split by a horizon
    line that TILTS with roll and SHIFTS with pitch — exactly the visual
    language a real attitude gauge uses. Immediately legible ("which way,
    and how much, is it actually leaning") without reading two separate
    numbers and doing the mental geometry yourself. The bezel color itself
    carries the rollover-risk verdict, so a dangerous tilt is visible
    peripherally, not just in a number that has to be read.
    """

    def __init__(self, parent, size=190, bg=None):
        self.size = size
        self._bg = bg or theme.CARD_BG
        super().__init__(parent, bg=self._bg, highlightthickness=0, bd=0)
        self._img = None
        self.set(0.0, 0.0, "ok")

    def set(self, pitch_deg, roll_deg, risk="ok"):
        size = self.size
        big = size * 3  # oversized working canvas so rotation never reveals empty corners

        sky_color = (22, 48, 74)
        ground_color = (61, 42, 20)
        base = Image.new("RGB", (big, big), sky_color)
        draw = ImageDraw.Draw(base)

        pixels_per_deg = big / 70.0
        pitch_deg_clamped = max(-45.0, min(45.0, pitch_deg))
        horizon_y = big / 2.0 - pitch_deg_clamped * pixels_per_deg
        # Belt-and-suspenders: clamp the computed pixel position directly,
        # not just the input degrees — the pitch clamp (+-45) and the pixel
        # scale (big/70) weren't actually consistent with each other (at
        # the clamp boundary, horizon_y worked out to 1.14x big, past the
        # bottom of the canvas), which crashed PIL's rectangle() outright
        # at extreme pitch values. Clamping the pixel value itself is
        # correct regardless of how those two constants relate.
        horizon_y = max(0.0, min(float(big), horizon_y))
        draw.rectangle([0, horizon_y, big, big], fill=ground_color)
        draw.line([0, horizon_y, big, horizon_y], fill=(235, 233, 227), width=4)

        rotated = base.rotate(roll_deg, resample=Image.BICUBIC, center=(big / 2.0, big / 2.0))
        half = size / 2.0
        cropped = rotated.crop((big / 2.0 - half, big / 2.0 - half, big / 2.0 + half, big / 2.0 + half))

        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse([2, 2, size - 2, size - 2], fill=255)

        canvas = Image.new("RGB", (size, size), _hex_to_rgb(self._bg))
        canvas.paste(cropped, (0, 0), mask)
        cdraw = ImageDraw.Draw(canvas)

        cx = cy = size / 2.0

        # pitch ladder — small fixed ticks at +-10/+-20 deg, unrotated (they
        # belong to the vehicle's own frame of reference, not the horizon)
        small_ppd = (size / big) * pixels_per_deg
        for tick_deg, length in [(10, 16), (-10, 16), (20, 24), (-20, 24)]:
            ty = cy - tick_deg * small_ppd
            if 4 < ty < size - 4:
                cdraw.line([cx - length / 2, ty, cx + length / 2, ty], fill=(210, 208, 200), width=1)

        # fixed vehicle-reference symbol (wings + center dot), always level
        wing = size * 0.24
        gap = size * 0.05
        ref_color = _hex_to_rgb(theme.ACCENT)
        cdraw.line([cx - wing, cy, cx - gap, cy], fill=ref_color, width=3)
        cdraw.line([cx + gap, cy, cx + wing, cy], fill=ref_color, width=3)
        cdraw.ellipse([cx - 3.5, cy - 3.5, cx + 3.5, cy + 3.5], outline=ref_color, width=2)

        # bezel ring — carries the rollover-risk verdict as a color, visible
        # peripherally without reading a number
        bezel_color = _hex_to_rgb(theme.status_color(risk))
        bezel_w = 4 if risk == "ok" else 5
        cdraw.ellipse([1, 1, size - 1, size - 1], outline=bezel_color, width=bezel_w)

        self._img = ImageTk.PhotoImage(canvas)
        self.configure(image=self._img)


class ArcGauge(tk.Frame):
    """
    Circular arc gauge (speed, brake %) — a 270-degree sweep track with the
    live value as a colored arc, and the number itself rendered as a real
    Tk label overlaid via place() (crisper than baking text into the PIL
    image, and immune to any PIL font-availability differences).
    """

    def __init__(self, parent, size=150, max_value=100.0, color=None, bg=None,
                 unit="", title=""):
        self._bg = bg or theme.CARD_BG
        super().__init__(parent, bg=self._bg, width=size, height=size + 22)
        self.pack_propagate(False)
        self.size = size
        self.max_value = max_value
        self.color = color or theme.ACCENT

        self.ring_label = tk.Label(self, bg=self._bg, highlightthickness=0, bd=0)
        self.ring_label.place(x=0, y=0, width=size, height=size)

        self.value_label = tk.Label(self, text="--", fg=theme.TEXT, bg=self._bg,
                                     font=theme.FONT_READOUT_MD)
        self.value_label.place(relx=0.5, rely=0.44, anchor="center")
        self.unit_label = tk.Label(self, text=unit, fg=theme.TEXT_DIM, bg=self._bg,
                                    font=theme.FONT_UNIT)
        self.unit_label.place(relx=0.5, rely=0.62, anchor="center")

        tk.Label(self, text=title, fg=theme.TEXT_DIM, bg=self._bg,
                 font=theme.FONT_LABEL_BOLD).place(relx=0.5, y=size + 4, anchor="n")

        self._img = None
        self.set(0.0)

    def set(self, value, text_override=None, color_override=None):
        size = self.size
        img = Image.new("RGB", (size, size), _hex_to_rgb(self._bg))
        draw = ImageDraw.Draw(img)
        margin = 9
        bbox = [margin, margin, size - margin, size - margin]
        draw.arc(bbox, start=135, end=405, fill=_hex_to_rgb(theme.BORDER_BRIGHT), width=9)

        frac = 0.0
        if self.max_value:
            frac = max(0.0, min(1.0, value / self.max_value))
        color = color_override or self.color
        if frac > 0:
            draw.arc(bbox, start=135, end=135 + 270 * frac, fill=_hex_to_rgb(color), width=9)

        self._img = ImageTk.PhotoImage(img)
        self.ring_label.configure(image=self._img)
        self.value_label.configure(text=text_override if text_override is not None else f"{value:.0f}")


class SteerBar(tk.Frame):
    """
    Horizontal steer indicator: a track from -max to +max degrees with a
    center (straight-ahead) tick and a moving marker for the current
    suggestion — reads left/right at a glance the way a real analog
    instrument does, rather than as a signed number you have to interpret.
    """

    def __init__(self, parent, width=280, height=54, max_deg=25.0, bg=None):
        self._bg = bg or theme.CARD_BG
        super().__init__(parent, bg=self._bg)
        self.width = width
        self.height = height
        self.max_deg = max_deg
        self.canvas = tk.Canvas(self, width=width, height=height, bg=self._bg, highlightthickness=0)
        self.canvas.pack()
        self.set(0.0)

    def set(self, steer_deg):
        c = self.canvas
        c.delete("all")
        w, h = self.width, self.height
        track_y = h * 0.55
        track_x0, track_x1 = 14, w - 14
        c.create_line(track_x0, track_y, track_x1, track_y, fill=theme.BORDER_BRIGHT, width=4,
                       capstyle=tk.ROUND)
        # center (straight-ahead) tick
        cx = (track_x0 + track_x1) / 2.0
        c.create_line(cx, track_y - 10, cx, track_y + 10, fill=theme.TEXT_FAINT, width=2)

        clamped = max(-self.max_deg, min(self.max_deg, steer_deg))
        frac = clamped / self.max_deg if self.max_deg else 0.0
        marker_x = cx + frac * (track_x1 - track_x0) / 2.0

        color = theme.ACCENT if abs(steer_deg) > 1.0 else theme.TEXT_DIM
        c.create_line(cx, track_y, marker_x, track_y, fill=color, width=4, capstyle=tk.ROUND)
        c.create_oval(marker_x - 8, track_y - 8, marker_x + 8, track_y + 8, fill=color, outline="")

        label = "STRAIGHT" if abs(steer_deg) < 0.5 else (f"◄ LEFT {abs(steer_deg):.0f}°" if steer_deg < 0
                                                            else f"RIGHT {steer_deg:.0f}° ►")
        c.create_text(w / 2.0, h - 8, text=label, fill=theme.TEXT, font=theme.FONT_LABEL_BOLD)


class Sparkline(tk.Label):
    """Small trend line for a rolling history of values — no axes/labels,
    just the shape of the last N samples."""

    def __init__(self, parent, width=140, height=30, color=None, bg=None):
        self._bg = bg or theme.CARD_BG
        self.width = width
        self.height = height
        self.color = color or theme.ACCENT
        super().__init__(parent, bg=self._bg, highlightthickness=0, bd=0)
        self._img = None
        self.set([])

    def set(self, values):
        w, h = self.width, self.height
        img = Image.new("RGB", (w, h), _hex_to_rgb(self._bg))
        draw = ImageDraw.Draw(img)
        clean = [v for v in values if v == v]  # drop NaN
        if len(clean) >= 2:
            lo, hi = min(clean), max(clean)
            span = (hi - lo) or 1.0
            n = len(values)
            pts = []
            for i, v in enumerate(values):
                x = (i / max(n - 1, 1)) * (w - 4) + 2
                if v != v:  # NaN — skip, leave a gap
                    pts.append(None)
                    continue
                y = h - 2 - ((v - lo) / span) * (h - 4)
                pts.append((x, y))
            segment = []
            for p in pts:
                if p is None:
                    if len(segment) >= 2:
                        draw.line(segment, fill=_hex_to_rgb(self.color), width=2)
                    segment = []
                else:
                    segment.append(p)
            if len(segment) >= 2:
                draw.line(segment, fill=_hex_to_rgb(self.color), width=2)
        self._img = ImageTk.PhotoImage(img)
        self.configure(image=self._img)


class ObstacleReadout(tk.Frame):
    """Nearest-obstacle distance + closing speed + count — the numbers that
    actually drive braking, shown together as one glanceable block."""

    def __init__(self, parent, bg=None):
        self._bg = bg or theme.CARD_BG
        super().__init__(parent, bg=self._bg)

        tk.Label(self, text="NEAREST OBSTACLE", fg=theme.TEXT_DIM, bg=self._bg,
                 font=theme.FONT_LABEL_BOLD).pack(anchor="w")

        row = tk.Frame(self, bg=self._bg)
        row.pack(anchor="w", pady=(2, 0))
        self.distance_label = tk.Label(row, text="--", fg=theme.TEXT, bg=self._bg,
                                        font=theme.FONT_READOUT_MD)
        self.distance_label.pack(side=tk.LEFT)
        tk.Label(row, text=" m", fg=theme.TEXT_DIM, bg=self._bg, font=theme.FONT_UNIT).pack(
            side=tk.LEFT, anchor="s", pady=(0, 4))

        self.closing_label = tk.Label(self, text="", fg=theme.TEXT_DIM, bg=self._bg,
                                       font=theme.FONT_LABEL)
        self.closing_label.pack(anchor="w")
        self.count_label = tk.Label(self, text="", fg=theme.TEXT_FAINT, bg=self._bg,
                                     font=theme.FONT_LABEL)
        self.count_label.pack(anchor="w")

    def set(self, distance_m, closing_speed_mps, count):
        if distance_m is None:
            self.distance_label.configure(text="--", fg=theme.TEXT_FAINT)
            self.closing_label.configure(text="")
        else:
            urgent = distance_m < 5.0
            self.distance_label.configure(
                text=f"{distance_m:.1f}", fg=theme.DANGER if urgent else theme.TEXT
            )
            if closing_speed_mps > 0.3:
                self.closing_label.configure(text=f"closing at {closing_speed_mps:.1f} m/s",
                                              fg=theme.WARN)
            elif closing_speed_mps < -0.3:
                self.closing_label.configure(text=f"moving away {abs(closing_speed_mps):.1f} m/s",
                                              fg=theme.GOOD)
            else:
                self.closing_label.configure(text="stationary", fg=theme.TEXT_DIM)
        self.count_label.configure(text=f"{count} tracked" if count else "none in corridor")
