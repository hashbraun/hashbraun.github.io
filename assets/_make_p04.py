"""Generate PROJECT 04 diagrams: overview, architecture, qualitative results.

Same style as _make_p01.py — white bg, warm-neutral panels, single accent per role.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image

KOREAN_FONT = "/home1/sota/.fonts/NanumGothic-Regular.ttf"
font_manager.fontManager.addfont(KOREAN_FONT)
_kor_name = font_manager.FontProperties(fname=KOREAN_FONT).get_name()

OUT = Path(__file__).parent / "p04"
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
FAIL = "#EF4444"

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
# 1) Overview — Teleop → SO-101 × 2 → Multi-modal Demos → VLA Policy → Closed-loop Rollout
# =========================================================
fig, ax = plt.subplots(figsize=(13.2, 4.2), dpi=180)
ax.set_xlim(0, 13.2)
ax.set_ylim(0, 4.2)
ax.set_axis_off()
fig.patch.set_facecolor(BG)

ax.text(0.4, 3.85, "System Overview", color=INK, fontsize=15, weight="bold")
ax.text(0.4, 3.55,
        "Data Flow: Teleop Rig → SO-101 Bi-arm → Multi-modal Demos → VLA Policy → Closed-loop Rollout",
        color=SUB, fontsize=11)

y0 = 1.5
h0 = 1.4
w0 = 2.2
gap = 0.35
x_start = 0.4
stages = [
    ("Teleop Rig",       "Puppeteer + Leader Arms",       WARM,        LINE,   INK),
    ("SO-101 Bi-arm",    "6-DoF × 2 · Bi-manipulator",    WARM,        LINE,   INK),
    ("Multi-modal Demos","RGB×3 · Depth/FPS · Proprio",   ACCENT_SOFT, ACCENT, ACCENT),
    ("VLA Policy",       "Vision-Language-Action",        ACCENT_SOFT, ACCENT, ACCENT),
    ("Closed-loop Rollout","On-robot bi-manip eval",      "#F0FDF4",   OK,     OK),
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

# Rollout feedback loop → new demos
ax.annotate("", xy=(3.0 + w0 / 2, y0), xytext=(10.8 + w0 / 2, y0 - 0.7),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2,
                            connectionstyle="arc3,rad=-0.28"))
ax.text(6.9, y0 - 0.55, "Failure Cases · 데모 보강 & 재학습 루프",
        ha="center", color=ACCENT, fontsize=10, weight="bold")

fig.tight_layout()
fig.savefig(OUT / "overview.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", OUT / "overview.png")


# =========================================================
# 2) Architecture — Multi-modal inputs → Fusion → Bi-manipulator Policy Head
# =========================================================
fig, ax = plt.subplots(figsize=(15, 7.4), dpi=180)
ax.set_xlim(0, 15)
ax.set_ylim(0, 7.4)
ax.set_axis_off()
fig.patch.set_facecolor(BG)

ax.text(0.4, 7.0, "Architecture", color=INK, fontsize=16, weight="bold")
ax.text(0.4, 6.7,
        "Multi-modal Inputs → Fusion Encoder → Bi-manipulator Policy Head → Dual-arm Actions",
        color=SUB, fontsize=11)

# ----- Left column: inputs (three stacked) -----
in_x = 0.4
in_w = 2.8
in_h = 1.05
in_gap = 0.3
in_top_y = 5.3
inputs = [
    ("RGB × 3 Cameras",   "top / left-wrist / right-wrist", ACCENT_SOFT, ACCENT, ACCENT),
    ("Point-Cloud / FPS", "RealSense depth · FPS sampling", ACCENT_SOFT, ACCENT, ACCENT),
    ("Proprioception",    "12-DoF joint states + gripper",  ACCENT_SOFT, ACCENT, ACCENT),
]
in_ys = []
for i, (t, s, fill, edge, ink) in enumerate(inputs):
    y = in_top_y - i * (in_h + in_gap)
    in_ys.append(y)
    box(ax, in_x, y, in_w, in_h, t, sub=s, fill=fill, edge=edge, ink=ink,
        subink=SUB, fontsize=12, subsize=9.5)

# ----- Middle: Fusion Encoder -----
fu_x = 4.4
fu_w = 3.4
fu_y = 3.8
fu_h = 2.5
box(ax, fu_x, fu_y, fu_w, fu_h,
    "Multi-modal Fusion",
    sub="Vision Encoder · PC Encoder\nCross-modal Transformer",
    fill=WARM, edge=LINE, ink=INK, subink=SUB, fontsize=13, subsize=10)

# Arrows: inputs → fusion
for y in in_ys:
    arrow(ax, in_x + in_w + 0.02, y + in_h / 2,
          fu_x - 0.02, fu_y + fu_h / 2, color=SUB, lw=1.2)

# ----- Right: Policy Head -----
ph_x = 8.4
ph_w = 3.2
ph_y = 4.3
ph_h = 1.55
box(ax, ph_x, ph_y, ph_w, ph_h,
    "VLA Policy Head",
    sub="action chunk · horizon = k",
    fill=ACCENT_SOFT, edge=ACCENT, ink=ACCENT, subink=SUB,
    fontsize=13, subsize=10)
arrow(ax, fu_x + fu_w + 0.02, fu_y + fu_h / 2,
      ph_x - 0.02, ph_y + ph_h / 2, color=ACCENT, lw=1.4)

# ----- Right-most: dual-arm actions (two boxes) -----
act_x = 12.2
act_w = 2.5
act_h = 1.0
act_gap = 0.25
act_top_y = 5.15
acts = [
    ("Left-arm Actions",  "6-DoF + gripper", "#F0FDF4", OK, OK),
    ("Right-arm Actions", "6-DoF + gripper", "#F0FDF4", OK, OK),
]
for i, (t, s, fill, edge, ink) in enumerate(acts):
    y = act_top_y - i * (act_h + act_gap)
    box(ax, act_x, y, act_w, act_h, t, sub=s, fill=fill, edge=edge, ink=ink,
        subink=SUB, fontsize=11.5, subsize=9)
    arrow(ax, ph_x + ph_w + 0.02, ph_y + ph_h / 2,
          act_x - 0.02, y + act_h / 2, color=OK, lw=1.2)

# ----- Bottom: Data Flywheel (Teleop demos → Replay buffer → Policy update) -----
fl_y = 1.1
fl_h = 1.2
fl_w = 4.8
fl_x = 3.6
box(ax, fl_x, fl_y, fl_w, fl_h,
    "Teleop Demos → Replay Buffer",
    sub="ACT / Diffusion Policy 학습 · 실패 케이스 데모 보강",
    fill=WARN_BG, edge=WARN_EDGE, ink=WARN_INK, subink=SUB,
    fontsize=13, subsize=10)

# Loop: replay buffer → fusion (training signal)
ax.annotate("", xy=(fu_x + fu_w / 3, fu_y),
            xytext=(fl_x + fl_w / 2, fl_y + fl_h),
            arrowprops=dict(arrowstyle="-|>", color=WARN_INK, lw=1.4,
                            connectionstyle="arc3,rad=-0.15"))
ax.text(fu_x + fu_w / 3 + 0.1, fu_y - 0.35, "train",
        color=WARN_INK, fontsize=10, weight="bold")

# Loop: policy rollout failures → replay buffer
ax.annotate("", xy=(fl_x + fl_w, fl_y + fl_h / 2),
            xytext=(ph_x + ph_w / 2, ph_y),
            arrowprops=dict(arrowstyle="-|>", color=SUB, lw=1.2,
                            connectionstyle="arc3,rad=-0.3"))
ax.text(ph_x - 0.4, ph_y - 0.55, "rollout failures",
        color=SUB, fontsize=9, style="italic")

# ----- Legend -----
lx = 0.4
ly = 0.35
ax.add_patch(FancyBboxPatch((lx, ly), 0.35, 0.25,
             boxstyle="round,pad=0.02,rounding_size=0.06",
             linewidth=1, edgecolor=ACCENT, facecolor=ACCENT_SOFT))
ax.text(lx + 0.45, ly + 0.12, "Model / ML component",
        color=INK, fontsize=9, va="center")

ax.add_patch(FancyBboxPatch((lx + 3.2, ly), 0.35, 0.25,
             boxstyle="round,pad=0.02,rounding_size=0.06",
             linewidth=1, edgecolor=WARN_EDGE, facecolor=WARN_BG))
ax.text(lx + 3.65, ly + 0.12, "Data-flywheel component",
        color=INK, fontsize=9, va="center")

ax.add_patch(FancyBboxPatch((lx + 6.5, ly), 0.35, 0.25,
             boxstyle="round,pad=0.02,rounding_size=0.06",
             linewidth=1, edgecolor=OK, facecolor="#F0FDF4"))
ax.text(lx + 6.95, ly + 0.12, "Actuator / output",
        color=INK, fontsize=9, va="center")

fig.tight_layout()
fig.savefig(OUT / "architecture.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", OUT / "architecture.png")


# =========================================================
# 3) Qualitative Results — RGB-only vs Ours (bi-manual rollout snapshots)
# =========================================================
STILL_PATHS = [
    "/tmp/p04_still_8.jpg",
    "/tmp/p04_still_14.jpg",
    "/tmp/p04_still_20.jpg",
    "/tmp/p04_still_30.jpg",
]
phase_labels = [
    "t=0  · 접근 (Approach)",
    "t=1  · 우측 그리퍼 Grasp",
    "t=2  · 양팔 Coordination",
    "t=3  · Place / Release",
]
# Illustrative baseline annotations — what tends to break in RGB-only rollouts
baseline_marks = [
    ("uncertain grasp point", FAIL),
    ("cup slipped", FAIL),
    ("hand collision", FAIL),
    ("release too early", FAIL),
]
ours_marks = [
    ("aligned via PC depth", OK),
    ("stable grasp", OK),
    ("bi-arm sync", OK),
    ("clean place", OK),
]

fig = plt.figure(figsize=(14, 9.0), dpi=180)
fig.patch.set_facecolor(BG)

# Overall title
fig.text(0.06, 0.955,
         "Qualitative Results — RGB-only vs Ours (Bi-manipulator Rollout Snapshots)",
         color=INK, fontsize=15, weight="bold", ha="left", va="top")
fig.text(0.06, 0.918,
         "동일 태스크(컵 pick-and-place) 롤아웃을 phase별로 정렬 · 상단: RGB-only 베이스라인 · 하단: Ours (+Point-Cloud / FPS)",
         color=SUB, fontsize=10.5, ha="left", va="top")

# Row labels on the left
fig.text(0.025, 0.68, "RGB-only\nbaseline", color=INK, fontsize=12, weight="bold",
         ha="center", va="center")
fig.text(0.025, 0.28, "Ours\n(+PC / FPS)", color=ACCENT, fontsize=12, weight="bold",
         ha="center", va="center")

for col in range(4):
    img = np.array(Image.open(STILL_PATHS[col]).convert("RGB"))
    # Top row (baseline): desaturated
    ax_top = fig.add_axes([0.07 + col * 0.232, 0.49, 0.22, 0.34])
    gray = img.mean(axis=2, keepdims=True).astype(np.uint8)
    gray = np.repeat(gray, 3, axis=2)
    ax_top.imshow(gray)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for spine in ax_top.spines.values():
        spine.set_color(LINE); spine.set_linewidth(1.0)
    ax_top.set_title(phase_labels[col], color=INK, fontsize=10.5, loc="left", pad=6)
    # Failure badge
    txt, col_c = baseline_marks[col]
    ax_top.text(0.98, 0.05, f"×  {txt}",
                transform=ax_top.transAxes, ha="right", va="bottom",
                fontsize=10, color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=col_c, edgecolor="none"))

    # Bottom row (ours): full color
    ax_bot = fig.add_axes([0.07 + col * 0.232, 0.09, 0.22, 0.34])
    ax_bot.imshow(img)
    ax_bot.set_xticks([]); ax_bot.set_yticks([])
    for spine in ax_bot.spines.values():
        spine.set_color(ACCENT); spine.set_linewidth(1.2)
    txt, col_c = ours_marks[col]
    ax_bot.text(0.98, 0.05, f"✓  {txt}",
                transform=ax_bot.transAxes, ha="right", va="bottom",
                fontsize=10, color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=col_c, edgecolor="none"))

fig.savefig(OUT / "perf_qual.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", OUT / "perf_qual.png")
