"""Generate PROJECT 08 diagrams: overview, architecture, perf_timeline.

Same style as _make_p01/p04 — white bg, warm-neutral panels, single accent per role.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

KOREAN_FONT = "/home1/sota/.fonts/NanumGothic-Regular.ttf"
font_manager.fontManager.addfont(KOREAN_FONT)
_kor_name = font_manager.FontProperties(fname=KOREAN_FONT).get_name()

OUT = Path(__file__).parent / "p08"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#FFFFFF"
INK = "#111827"
SUB = "#6B7280"
LINE = "#E5E7EB"
ACCENT = "#3182F6"
ACCENT_SOFT = "#EAF2FF"
WARM = "#F9FAFB"
OK = "#10B981"
WARN_BG = "#FFF7ED"
WARN_EDGE = "#F59E0B"
WARN_INK = "#B45309"

plt.rcParams.update({
    "font.family": [_kor_name, "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "axes.edgecolor": LINE,
    "axes.linewidth": 0.8,
    "axes.unicode_minus": False,
})


def box(ax, x, y, w, h, text, sub=None, fill=WARM, edge=LINE, ink=INK, subink=SUB,
        radius=0.12, fontsize=13, subsize=10):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.2, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2 + (0.08 if sub else 0), text,
            ha="center", va="center", color=ink, fontsize=fontsize, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.16, sub,
                ha="center", va="center", color=subink, fontsize=subsize)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.5, label=None):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color,
        shrinkA=6, shrinkB=6,
    )
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label,
                ha="center", va="center", color=SUB, fontsize=9)


# =========================================================
# 1) Overview — MP4 → Adaptive ffmpeg → Qwen3-VL → Structured JSON → Timestamped segments
# =========================================================
fig, ax = plt.subplots(figsize=(13.2, 4.2), dpi=180)
ax.set_xlim(0, 13.2)
ax.set_ylim(0, 4.2)
ax.set_axis_off()
fig.patch.set_facecolor(BG)

ax.text(0.4, 3.85, "System Overview", color=INK, fontsize=15, weight="bold")
ax.text(0.4, 3.55,
        "Data Flow: MP4 → Adaptive ffmpeg → Qwen3-VL (structured) → Timestamped Action Segments",
        color=SUB, fontsize=11)

y0 = 1.5
h0 = 1.4
w0 = 2.2
gap = 0.35
x_start = 0.4
stages = [
    ("MP4 Clip",         "원본 CCTV · ~수분 길이",         WARM,        LINE,   INK),
    ("Adaptive ffmpeg",  "fps · width · metadata strip",  ACCENT_SOFT, ACCENT, ACCENT),
    ("Qwen3-VL 32B",     "vLLM API · structured prompt",  ACCENT_SOFT, ACCENT, ACCENT),
    ("Post-processing",  "repair · filter · merge",       WARN_BG,     WARN_EDGE, WARN_INK),
    ("Action Segments",  "MM:SS – MM:SS · 15 labels",     "#F0FDF4",   OK,     OK),
]
positions = [(x_start + i * (w0 + gap), *s) for i, s in enumerate(stages)]
for x, title, sub, fill, edge, ink in positions:
    box(ax, x, y0, w0, h0, title, sub=sub, fill=fill, edge=edge, ink=ink,
        subink=SUB, fontsize=13, subsize=9.5)

for i in range(len(positions) - 1):
    x1 = positions[i][0] + w0
    x2 = positions[i + 1][0]
    y = y0 + h0 / 2
    arrow(ax, x1 + 0.05, y, x2 - 0.05, y)

# Feedback: hallucination detection → prompt refine
ax.annotate("", xy=(3.0 + w0 / 2, y0), xytext=(8.4 + w0 / 2, y0 - 0.7),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2,
                            connectionstyle="arc3,rad=-0.28"))
ax.text(6.0, y0 - 0.55, "Hallucination 검출 · 프롬프트 · fps 재조정 루프",
        ha="center", color=ACCENT, fontsize=10, weight="bold")

fig.tight_layout()
fig.savefig(OUT / "overview.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", OUT / "overview.png")


# =========================================================
# 2) Architecture — Detailed pipeline
#    Left: input + adaptive ffmpeg
#    Middle: Qwen3-VL structured prompt + call
#    Right: post-processing chain → segments
#    Bottom: SLURM batch layer
# =========================================================
FIG_W, FIG_H = 15.4, 8.4
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=180)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_axis_off()
fig.patch.set_facecolor(BG)

ax.text(0.4, 8.0, "Architecture", color=INK, fontsize=16, weight="bold")
ax.text(0.4, 7.65,
        "MP4 → adaptive ffmpeg → Qwen3-VL structured JSON → validators & repair → timestamped segments",
        color=SUB, fontsize=11)


def container(ax, x, y, w, h, title, fill=WARM, edge=LINE, ink=INK,
              fontsize=13, radius=0.12, title_pad=0.32):
    """Draw a labeled container with title anchored at top-center.
    Returns the y-coordinate below the title reserved for inner content.
    """
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.2, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(p)
    title_y = y + h - title_pad
    ax.text(x + w / 2, title_y, title,
            ha="center", va="top", color=ink, fontsize=fontsize, weight="bold")
    return title_y - 0.20  # top of usable inner area


# Row anchors
ROW_TOP_Y = 3.35        # bottom of the middle row (containers)
ROW_TOP_H = 3.55        # container height (was 2.85 → +0.7 for headroom)

# --- Left column: MP4 + Adaptive ffmpeg ---
box(ax, 0.4, ROW_TOP_Y + 2.05, 2.7, 1.15,
    "MP4 Clip", sub="≤10min / >10min",
    fill=WARM, edge=LINE, ink=INK, subink=SUB, fontsize=12, subsize=9.5)

box(ax, 0.4, ROW_TOP_Y + 0.05, 2.7, 1.6,
    "Adaptive ffmpeg",
    sub="fps ∈ {0.33, 0.1}\nwidth=512 · -map_metadata -1",
    fill=ACCENT_SOFT, edge=ACCENT, ink=ACCENT, subink=SUB, fontsize=12, subsize=9.5)
arrow(ax, 1.75, ROW_TOP_Y + 2.05, 1.75, ROW_TOP_Y + 1.65, lw=1.4)

# --- Middle: Qwen3-VL container (title at top; 3 sub-boxes stacked below) ---
qv_x, qv_y, qv_w, qv_h = 3.9, ROW_TOP_Y, 4.6, ROW_TOP_H
qv_inner_top = container(ax, qv_x, qv_y, qv_w, qv_h,
                         "Qwen3-VL 32B  ·  vLLM API",
                         fill=WARM, edge=LINE, ink=INK, fontsize=13)

sub_x = qv_x + 0.30
sub_w = qv_w - 0.60
sub_h = 0.68
sub_gap = 0.16
sub_items = [
    ("Constraint-aware Prompt", "action pool · priority · visibility · ≥5s"),
    ("Structured Output  (dict)", '{"MM:SS – MM:SS":  <int index>}'),
    ("Repetition penalty · max_tokens", "1.05  ·  2048  (dynamic)"),
]
for i, (t, s) in enumerate(sub_items):
    y = qv_inner_top - sub_h - i * (sub_h + sub_gap)
    box(ax, sub_x, y, sub_w, sub_h, t, sub=s,
        fill=ACCENT_SOFT, edge=ACCENT, ink=ACCENT, subink=SUB,
        radius=0.08, fontsize=10.5, subsize=9)

arrow(ax, 3.1, ROW_TOP_Y + 0.85, qv_x - 0.02, qv_y + qv_h / 2, color=ACCENT, lw=1.4)

# --- Right: Post-processing container ---
pp_x, pp_y, pp_w, pp_h = 9.3, ROW_TOP_Y, 3.2, ROW_TOP_H
pp_inner_top = container(ax, pp_x, pp_y, pp_w, pp_h,
                         "Post-processing",
                         fill=WARN_BG, edge=WARN_EDGE, ink=WARN_INK, fontsize=13)
pp_items = [
    "parse_response",
    "_repair_alternating",
    "_is_hallucination",
    "_filter_invalid_actions",
    "_filter_short_segments  (<3s)",
]
pp_line = 0.42
for i, t in enumerate(pp_items):
    y = pp_inner_top - 0.20 - i * pp_line
    ax.text(pp_x + 0.30, y, "· " + t,
            color=WARN_INK, fontsize=10.5, ha="left", va="center")

arrow(ax, qv_x + qv_w + 0.02, qv_y + qv_h / 2, pp_x - 0.02, pp_y + pp_h / 2,
      color=WARN_INK, lw=1.4)

# --- Right-most: Action Segments output ---
os_x, os_y, os_w, os_h = 13.0, ROW_TOP_Y + 0.6, 2.20, ROW_TOP_H - 1.2
os_inner_top = container(ax, os_x, os_y, os_w, os_h,
                         "Action Segments",
                         fill="#F0FDF4", edge=OK, ink=OK, fontsize=12)
ax.text(os_x + os_w / 2, os_inner_top - 0.35, "15 labels",
        ha="center", va="center", color=SUB, fontsize=10)
ax.text(os_x + os_w / 2, os_inner_top - 0.70, "JSONL 저장",
        ha="center", va="center", color=SUB, fontsize=10)
arrow(ax, pp_x + pp_w + 0.02, pp_y + pp_h / 2, os_x - 0.02, os_y + os_h / 2,
      color=OK, lw=1.4)

# --- Bottom: SLURM batch layer ---
sl_y = 1.15
sl_h = 1.35
sl_w = 12.8
sl_x = 1.3
sl_inner_top = container(ax, sl_x, sl_y, sl_w, sl_h,
                         "SLURM Batch  ·  gpu-106  ·  전 데이터셋 처리",
                         fill=WARM, edge=LINE, ink=INK, fontsize=12)
ax.text(sl_x + sl_w / 2, sl_inner_top - 0.35,
        "OOM 방지 (동적 fps)  ·  JSONL append  ·  logs/<jobid>.err  ·  precision / summary 스크립트 자동 후속",
        ha="center", va="center", color=SUB, fontsize=10)

# arrows: SLURM wraps the whole pipeline (dashed lines to top blocks)
for x_top in [1.75, qv_x + qv_w / 2, pp_x + pp_w / 2]:
    ax.annotate("", xy=(x_top, ROW_TOP_Y - 0.02),
                xytext=(x_top, sl_y + sl_h + 0.02),
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.8, linestyle="--"))

# Legend
lx = 0.4
ly = 0.35
ax.add_patch(FancyBboxPatch((lx, ly), 0.35, 0.25,
             boxstyle="round,pad=0.02,rounding_size=0.06",
             linewidth=1, edgecolor=ACCENT, facecolor=ACCENT_SOFT))
ax.text(lx + 0.45, ly + 0.12, "Model / prompt component",
        color=INK, fontsize=9, va="center")

ax.add_patch(FancyBboxPatch((lx + 3.4, ly), 0.35, 0.25,
             boxstyle="round,pad=0.02,rounding_size=0.06",
             linewidth=1, edgecolor=WARN_EDGE, facecolor=WARN_BG))
ax.text(lx + 3.85, ly + 0.12, "Validation / repair component",
        color=INK, fontsize=9, va="center")

ax.add_patch(FancyBboxPatch((lx + 7.2, ly), 0.35, 0.25,
             boxstyle="round,pad=0.02,rounding_size=0.06",
             linewidth=1, edgecolor=OK, facecolor="#F0FDF4"))
ax.text(lx + 7.65, ly + 0.12, "Output / storage",
        color=INK, fontsize=9, va="center")

fig.tight_layout()
fig.savefig(OUT / "architecture.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", OUT / "architecture.png")


# =========================================================
# 3) Qualitative Output — VLM-generated action timeline vs raw MP4
#    Gantt-style timeline over ~6 min sample, with priority-colored segments
# =========================================================
fig = plt.figure(figsize=(14, 6.6), dpi=180)
fig.patch.set_facecolor(BG)

fig.text(0.05, 0.955,
         "Qualitative Output — VLM-generated Action Timeline (structured, ≥5s segments)",
         color=INK, fontsize=15, weight="bold", ha="left", va="top")
fig.text(0.05, 0.918,
         "Qwen3-VL structured output을 초 단위 타임라인으로 시각화 · 15 카테고리 · Priority-aware 색상",
         color=SUB, fontsize=10.5, ha="left", va="top")

# --- Top strip: raw MP4 frames (illustrative filmstrip, no faces) ---
ax_top = fig.add_axes([0.05, 0.66, 0.90, 0.16])
ax_top.set_xlim(0, 360)
ax_top.set_ylim(0, 1)
ax_top.set_axis_off()
# Filmstrip cells
n = 24
w_cell = 360 / n
for i in range(n):
    x = i * w_cell
    # Warm neutral cells with subtle gradient (fake frame content)
    fill = "#E5E7EB" if i % 2 == 0 else "#F3F4F6"
    ax_top.add_patch(Rectangle((x + 1, 0.2), w_cell - 2, 0.6,
                                facecolor=fill, edgecolor=LINE, linewidth=0.8))
ax_top.text(0, 0.92, "Raw MP4  ·  duration 06:00  ·  20fps · 2560×1440",
            fontsize=10.5, color=INK, weight="bold", va="bottom")
ax_top.text(360, 0.92, "24 frames shown (illustrative)",
            fontsize=9.5, color=SUB, ha="right", va="bottom")

# --- Middle: sampled frames after adaptive ffmpeg ---
ax_mid = fig.add_axes([0.05, 0.44, 0.90, 0.16])
ax_mid.set_xlim(0, 360)
ax_mid.set_ylim(0, 1)
ax_mid.set_axis_off()
# Show sampled indices as highlighted cells
sampled = [1, 4, 7, 10, 13, 16, 19, 22]  # ~0.33fps effective
for i in range(n):
    x = i * w_cell
    if i in sampled:
        ax_mid.add_patch(Rectangle((x + 1, 0.2), w_cell - 2, 0.6,
                                    facecolor=ACCENT_SOFT, edgecolor=ACCENT, linewidth=1.2))
    else:
        ax_mid.add_patch(Rectangle((x + 1, 0.35), w_cell - 2, 0.3,
                                    facecolor="#F9FAFB", edgecolor=LINE, linewidth=0.6))
ax_mid.text(0, 0.92,
            "Adaptive ffmpeg  ·  fps=0.33  ·  width=512  ·  metadata stripped",
            fontsize=10.5, color=ACCENT, weight="bold", va="bottom")
ax_mid.text(360, 0.92, "8 keyframes into VLM",
            fontsize=9.5, color=SUB, ha="right", va="bottom")

# --- Bottom: VLM action timeline (Gantt) ---
ax_bot = fig.add_axes([0.05, 0.11, 0.90, 0.24])
ax_bot.set_xlim(0, 360)   # 6 minutes in seconds
ax_bot.set_ylim(-0.5, 1.5)
ax_bot.set_yticks([])
# Priority palette
P1 = "#3182F6"   # using phone / laptop / drinking
P2 = "#8B5CF6"   # reading / dressing
P3 = "#F59E0B"   # walking
P4 = "#10B981"   # sitting / watching

segments = [
    (0,   45,  "walking",         P3),
    (45,  62,  "sitting on chair", P4),
    (62,  148, "using phone",     P1),
    (148, 205, "watching tv",     P4),
    (205, 218, "walking",         P3),
    (218, 292, "using laptop",    P1),
    (292, 336, "drinking",        P1),
    (336, 360, "sitting on chair", P4),
]
for s, e, lab, col in segments:
    ax_bot.add_patch(Rectangle((s, 0.15), e - s, 0.75,
                                facecolor=col, edgecolor="none", alpha=0.92))
    if e - s >= 30:
        ax_bot.text((s + e) / 2, 0.53, lab,
                    ha="center", va="center", color="white",
                    fontsize=9.5, weight="bold")
# Axis: minute ticks
for m in range(0, 7):
    x = m * 60
    ax_bot.axvline(x, ymin=0, ymax=0.08, color=SUB, linewidth=0.6)
    ax_bot.text(x, -0.15, f"{m:02d}:00", ha="center", va="top",
                fontsize=9, color=SUB)
ax_bot.spines["top"].set_visible(False)
ax_bot.spines["right"].set_visible(False)
ax_bot.spines["left"].set_visible(False)
ax_bot.spines["bottom"].set_color(LINE)
ax_bot.tick_params(axis="both", length=0, colors=SUB)
ax_bot.set_xticks([])

# Timeline title
ax_bot.text(0, 1.1,
            "Qwen3-VL Output  ·  8 segments  ·  min. 5s enforced  ·  hallucination filters passed",
            fontsize=10.5, color=INK, weight="bold")

# Priority legend
leg_y = -0.42
leg_x = 0
for lab, col in [("P1 · using phone / laptop / drinking", P1),
                 ("P2 · reading / dressing", P2),
                 ("P3 · walking", P3),
                 ("P4 · sitting / watching", P4)]:
    ax_bot.add_patch(Rectangle((leg_x, leg_y - 0.08), 12, 0.16,
                                facecolor=col, edgecolor="none"))
    ax_bot.text(leg_x + 16, leg_y, lab, va="center", fontsize=8.5, color=INK)
    leg_x += 90

fig.savefig(OUT / "perf_timeline.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", OUT / "perf_timeline.png")
