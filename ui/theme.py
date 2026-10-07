"""
ui/theme.py — a deliberately different visual identity from the old
sidebar-dashboard look: a dark industrial/cockpit HUD, not a web-app-style
admin panel. Near-black surfaces, amber as the primary instrument color
(the same family used on real heavy-equipment dashboards), cyan reserved
for IMU/attitude readouts specifically so it reads as a distinct
instrument cluster, monospace for every live numeric readout so values
don't visually jitter as digits change width.
"""

# --- surfaces, darkest to lightest ---
BG = "#08090a"             # window background — near black
PANEL_BG = "#111214"       # HUD panel background
CARD_BG = "#17181b"        # instrument cluster background
BORDER = "#26282c"
BORDER_BRIGHT = "#3a3d42"

# --- text ---
TEXT = "#f4f3ef"
TEXT_DIM = "#8c8d90"
TEXT_FAINT = "#4a4b4e"

# --- primary instrument color (amber, industrial-panel convention) ---
ACCENT = "#ff9f1c"
ACCENT_DIM = "#7a4d10"

# --- secondary instrument color (cyan, reserved for IMU/attitude) ---
IMU_ACCENT = "#3ee6e0"
IMU_ACCENT_DIM = "#1c6b68"

# --- status ---
GOOD = "#3ddc84"
WARN = "#ffd60a"
DANGER = "#ff3b30"

# --- fonts ---
# Numeric HUD readouts are monospace on purpose — digits don't visually
# jitter/reflow as the value changes, which matters a lot for something
# glanced at while driving.
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_LABEL = ("Segoe UI", 9)
FONT_LABEL_BOLD = ("Segoe UI", 9, "bold")
FONT_READOUT_LG = ("Consolas", 34, "bold")
FONT_READOUT_MD = ("Consolas", 20, "bold")
FONT_READOUT_SM = ("Consolas", 13, "bold")
FONT_UNIT = ("Segoe UI", 10)
FONT_MONO_SM = ("Consolas", 9)
FONT_STATUS = ("Segoe UI", 10, "bold")

# --- spacing ---
PAD_SM = 6
PAD_MD = 10
PAD_LG = 16

STATUS_COLORS = {"ok": GOOD, "warning": WARN, "danger": DANGER, "off": TEXT_FAINT}


def status_color(risk_or_level):
    return STATUS_COLORS.get(risk_or_level, TEXT_DIM)
