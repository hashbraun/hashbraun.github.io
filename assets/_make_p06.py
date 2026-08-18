"""Generate PROJECT 06 (UR5e Dynamic Obstacle-Aware Replanning) diagrams.

Assets:
    - overview.png     : Full pipeline (RGB-D → PointCloud → OctoMap → Planning Scene
                         → MoveIt2/OMPL → JointTrajectory → UR5e)
    - architecture.png : 3-tier system (Perception ↔ Planning ↔ Control) with
                         bytes-diff trigger and stop → back-off → replan FSM

Palette / helpers mirror _make_p05.py (uniform title-sub gap + linespacing).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

KOREAN_FONT = "/home1/sota/.fonts/NanumGothic-Regular.ttf"
font_manager.fontManager.addfont(KOREAN_FONT)
_kor_name = font_manager.FontProperties(fname=KOREAN_FONT).get_name()

OUT = Path(__file__).parent / "p06"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#FFFFFF"
INK = "#111827"
SUB = "#6B7280"
LINE = "#E5E7EB"
ACCENT = "#3182F6"
ACCENT_SOFT = "#EAF2FF"
WARM = "#F9FAFB"
OK = "#10B981"
OK_SOFT = "#F0FDF4"
WARN_BG = "#FFF7ED"
WARN_EDGE = "#F59E0B"
WARN_INK = "#B45309"
PURPLE = "#8B5CF6"
PURPLE_SOFT = "#F5F3FF"

plt.rcParams.update({
    "font.family": [_kor_name, "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "axes.edgecolor": LINE,
    "axes.linewidth": 0.8,
    "axes.unicode_minus": False,
})

# --- text layout constants (same as _make_p05.py) ---
TITLE_ANCHOR_DY = 0.14
SUB_TOP_DY      = -0.02
LINESPACING     = 1.35


def wrap_text(text, max_chars):
    """Wrap text at ' · ' or whitespace boundaries under max_chars per line."""
    if len(text) <= max_chars:
        return text
    parts = []
    for chunk in text.split(" · "):
        if parts:
            parts[-1] = parts[-1] + " ·"
        parts.append(chunk)
    tokens = []
    for p in parts:
        if len(p) <= max_chars:
            tokens.append(p)
        else:
            tokens.extend(p.split())
    lines, cur = [], ""
    for tok in tokens:
        candidate = (cur + " " + tok).strip() if cur else tok
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = tok
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def box(ax, x, y, w, h, text, sub=None, fill=WARM, edge=LINE, ink=INK, subink=SUB,
        radius=0.12, fontsize=13, subsize=10, sub_wrap=None):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.2, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(p)
    if sub:
        rendered = wrap_text(sub, sub_wrap) if sub_wrap else sub
        ax.text(x + w / 2, y + h / 2 + TITLE_ANCHOR_DY, text,
                ha="center", va="center", color=ink, fontsize=fontsize, weight="bold")
        ax.text(x + w / 2, y + h / 2 + SUB_TOP_DY, rendered,
                ha="center", va="top", color=subink, fontsize=subsize,
                linespacing=LINESPACING)
    else:
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", color=ink, fontsize=fontsize, weight="bold")


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.5, rad=0.0):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color,
        shrinkA=6, shrinkB=6,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)


def container(ax, x, y, w, h, title, fill=WARM, edge=LINE, ink=INK,
              fontsize=13, radius=0.12, title_pad=0.32):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.2, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(p)
    title_y = y + h - title_pad
    ax.text(x + w / 2, title_y, title,
            ha="center", va="top", color=ink, fontsize=fontsize, weight="bold")
    return title_y - 0.22


# =========================================================
# 1) Overview — RGB-D → OctoMap → Planning Scene → MoveIt2 → Controller → UR5e
# =========================================================
FIG_W, FIG_H = 15.4, 4.8
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=180)
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H); ax.set_axis_off()
fig.patch.set_facecolor(BG)

ax.text(0.4, 4.45, "System Overview",
        color=INK, fontsize=15, weight="bold")
ax.text(0.4, 4.15,
        "동적 장애물이 궤적에 들어오면 지도 데이터 변화량으로 감지 → 정지 → back-off → replan (Gazebo · ROS 2 Humble · MoveIt2)",
        color=SUB, fontsize=10.5)

# 7 stages (add controller + UR5e for clarity)
y0 = 1.55
h0 = 1.55
w0 = 1.98
gap = 0.15
x_start = 0.4

stages = [
    ("Depth Camera",     "Gazebo 시뮬 · RGB-D",         WARM,        LINE,   INK),
    ("PointCloud",       "3D 포인트 스트림",             WARM,        LINE,   INK),
    ("OctoMap",          "확률적 3D 점유 격자 · Octree", ACCENT_SOFT, ACCENT, ACCENT),
    ("Planning Scene",   "C-Obstacle · Voxel 업데이트",  ACCENT_SOFT, ACCENT, ACCENT),
    ("MoveIt2 / OMPL",   "RRT-Connect + CCC",            PURPLE_SOFT, PURPLE, PURPLE),
    ("JointTraj Ctrl",   "follow_joint_trajectory",      PURPLE_SOFT, PURPLE, PURPLE),
    ("UR5e (6-DOF)",     "실행 · 정지 · replan",         OK_SOFT,     OK,     OK),
]
positions = [(x_start + i * (w0 + gap), *s) for i, s in enumerate(stages)]
for x, title, sub, fill, edge, ink in positions:
    box(ax, x, y0, w0, h0, title, sub=sub, fill=fill, edge=edge, ink=ink,
        subink=SUB, fontsize=11.5, subsize=9, sub_wrap=18)

for i in range(len(positions) - 1):
    x1 = positions[i][0] + w0
    x2 = positions[i + 1][0]
    y = y0 + h0 / 2
    arrow(ax, x1 + 0.02, y, x2 - 0.02, y)

# Replan feedback loop (bytes-diff trigger)
loop_from_x = positions[3][0] + w0 / 2  # Planning Scene
loop_to_x   = positions[4][0] + w0 / 2  # MoveIt2
# arc under boxes: from UR5e bottom → back to MoveIt2 (via Planning Scene)
ax.annotate("", xy=(positions[4][0] + w0 / 2, y0),
            xytext=(positions[6][0] + w0 / 2, y0 - 0.7),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.3,
                            connectionstyle="arc3,rad=-0.30"))
ax.text((positions[4][0] + w0 / 2 + positions[6][0] + w0 / 2) / 2, y0 - 0.55,
        "지도 데이터량 변화 감지 → stop → back-off → replan",
        ha="center", color=ACCENT, fontsize=10, weight="bold")

fig.savefig(OUT / "overview.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", OUT / "overview.png")


# =========================================================
# 2) Architecture — 3-tier (Perception / Planning / Control) + Recovery FSM
# =========================================================
FIG_W, FIG_H = 15.4, 9.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=180)
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H); ax.set_axis_off()
fig.patch.set_facecolor(BG)

ax.text(0.4, 8.62, "Architecture — Perception · Planning · Control",
        color=INK, fontsize=16, weight="bold")
ax.text(0.4, 8.26,
        "UR5e + Gazebo + ROS 2 Humble + MoveIt2 · Continuous Collision Checking + Reactive Recovery FSM",
        color=SUB, fontsize=11)

# --- Legend (top-right) — numbered flow ---
legend_items = [
    ("1", "Depth → OctoMap → Planning Scene  (환경 갱신)",  ACCENT),
    ("2", "Planning Scene → MoveIt2  (C-Obstacle 회피)",     PURPLE),
    ("3", "MoveIt2 → JointTrajectory Controller  (실행)",    PURPLE),
    ("4", "Trigger → Recovery FSM → replan  (반응)",         WARN_EDGE),
]
lg_x0 = 9.7
lg_y0 = 8.62
for i, (num, txt, col) in enumerate(legend_items):
    ly = lg_y0 - i * 0.30
    circle = plt.Circle((lg_x0, ly), 0.13, color=col, zorder=5)
    ax.add_patch(circle)
    ax.text(lg_x0, ly, num, ha="center", va="center",
            color="white", fontsize=10, weight="bold", zorder=6)
    ax.text(lg_x0 + 0.28, ly, txt, ha="left", va="center",
            color=INK, fontsize=10)

# ---- Perception container (left) ----
per_x, per_y, per_w, per_h = 0.4, 4.55, 5.6, 3.0
inner_top = container(ax, per_x, per_y, per_w, per_h,
                      "Perception  (환경 인식)",
                      fill=ACCENT_SOFT, edge=ACCENT, ink=ACCENT, fontsize=13)
per_box_w = per_w - 0.8
per_box_x = per_x + 0.4
box(ax, per_box_x, inner_top - 1.05, per_box_w, 0.95,
    "Depth Camera → PointCloud",
    sub="Gazebo 시뮬 depth · rclpy 노드",
    fill=BG, edge=ACCENT, ink=ACCENT, fontsize=12, subsize=9.5, sub_wrap=26)
box(ax, per_box_x, inner_top - 2.15, per_box_w, 0.95,
    "OctoMap (OccupancyMapUpdater)",
    sub="Octree · 베이즈 갱신 · Planning Scene 반영",
    fill=BG, edge=ACCENT, ink=ACCENT, fontsize=12, subsize=9.5, sub_wrap=26)

per_out_center = (per_x + per_w, per_y + per_h / 2 - 0.4)  # right edge midpoint

# ---- Planning container (center) ----
pln_x, pln_y, pln_w, pln_h = 6.3, 4.55, 4.5, 3.0
inner_top = container(ax, pln_x, pln_y, pln_w, pln_h,
                      "Planning  (경로 계획)",
                      fill=PURPLE_SOFT, edge=PURPLE, ink=PURPLE, fontsize=13)
pln_box_w = pln_w - 0.8
pln_box_x = pln_x + 0.4
box(ax, pln_box_x, inner_top - 1.05, pln_box_w, 0.95,
    "MoveIt2 + OMPL",
    sub="RRT-Connect · C-Space 샘플링",
    fill=BG, edge=PURPLE, ink=PURPLE, fontsize=12, subsize=9.5, sub_wrap=24)
box(ax, pln_box_x, inner_top - 2.15, pln_box_w, 0.95,
    "Continuous Collision Check",
    sub="관절 상태 사이 전 궤적 · 미래 충돌 예측",
    fill=BG, edge=PURPLE, ink=PURPLE, fontsize=12, subsize=9.5, sub_wrap=24)

pln_left  = (pln_x, pln_y + pln_h / 2 - 0.4)
pln_right = (pln_x + pln_w, pln_y + pln_h / 2 - 0.4)

# ---- Control container (right) ----
ctl_x, ctl_y, ctl_w, ctl_h = 11.1, 4.55, 3.9, 3.0
inner_top = container(ax, ctl_x, ctl_y, ctl_w, ctl_h,
                      "Control  (실행)",
                      fill=OK_SOFT, edge=OK, ink=OK, fontsize=13)
ctl_box_w = ctl_w - 0.8
ctl_box_x = ctl_x + 0.4
box(ax, ctl_box_x, inner_top - 1.05, ctl_box_w, 0.95,
    "JointTrajectory Ctrl",
    sub="follow_joint_trajectory Action",
    fill=BG, edge=OK, ink=OK, fontsize=12, subsize=9.5, sub_wrap=22)
box(ax, ctl_box_x, inner_top - 2.15, ctl_box_w, 0.95,
    "UR5e Driver (Gazebo)",
    sub="joint_states 피드백",
    fill=BG, edge=OK, ink=OK, fontsize=12, subsize=9.5, sub_wrap=22)

ctl_left = (ctl_x, ctl_y + ctl_h / 2 - 0.4)
ctl_bottom = (ctl_x + ctl_w / 2, ctl_y)

# ---- Numbered cross-container arrows ----
def numbered_arrow(x1, y1, x2, y2, color, num, rad=0.0, lw=1.6):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=16,
        linewidth=lw, color=color,
        shrinkA=4, shrinkB=6,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    mx = x1 + (x2 - x1) * 0.5
    my = y1 + (y2 - y1) * 0.5
    if rad != 0.0:
        dx, dy = (x2 - x1), (y2 - y1)
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        mx += -dy / norm * rad * 1.4
        my += dx / norm * rad * 1.4
    circle = plt.Circle((mx, my), 0.17, color=color, zorder=7,
                        ec="white", lw=1.4)
    ax.add_patch(circle)
    ax.text(mx, my, num, ha="center", va="center",
            color="white", fontsize=10, weight="bold", zorder=8)

# ① Perception → Planning
numbered_arrow(per_out_center[0] + 0.02, per_out_center[1],
               pln_left[0] - 0.02,      pln_left[1],
               color=ACCENT, num="1")
# ② Planning internal (implicit — omitted, both boxes inside container)
# ② Planning → Control (trajectory)
numbered_arrow(pln_right[0] + 0.02, pln_right[1],
               ctl_left[0] - 0.02,  ctl_left[1],
               color=PURPLE, num="2")
# ③ Control → UR5e (joint execution) — inside container, so annotate with tiny arrow
# skip — implicit inside Control container

# ---- Bottom: Recovery FSM (bytes-diff trigger) ----
fsm_y = 2.3
fsm_h = 1.7
fsm_x = 0.4
fsm_w = 14.6
p = FancyBboxPatch((fsm_x, fsm_y), fsm_w, fsm_h,
                   boxstyle="round,pad=0.02,rounding_size=0.10",
                   linewidth=1.4, edgecolor=WARN_EDGE, facecolor=WARN_BG)
ax.add_patch(p)
ax.text(fsm_x + fsm_w / 2, fsm_y + fsm_h - 0.28,
        "Reactive Recovery FSM  (지도 데이터량 변화 → replan)",
        ha="center", va="top", color=WARN_INK, fontsize=12.5, weight="bold")

fsm_states = [
    ("Monitor",       "|Δ voxel bytes| < τ"),
    ("Detect",        "|Δ bytes| ≥ τ  (예: 160)"),
    ("Stop",          "궤적 중단 · 현재 자세 고정"),
    ("Back-off",      "안전 자세로 후퇴"),
    ("Replan",        "RRT-Connect 재탐색"),
    ("Resume",        "새 궤적 실행"),
]
n = len(fsm_states)
fsm_gap = 0.18
fsm_sw = (fsm_w - 0.4 - fsm_gap * (n - 1)) / n
fsm_sh = 0.9
fsm_sy = fsm_y + 0.18
for i, (t, sub) in enumerate(fsm_states):
    sx = fsm_x + 0.2 + i * (fsm_sw + fsm_gap)
    box(ax, sx, fsm_sy, fsm_sw, fsm_sh, t, sub=sub,
        fill=BG, edge=WARN_EDGE, ink=WARN_INK, subink=SUB,
        fontsize=11, subsize=8.5, sub_wrap=18)
for i in range(n - 1):
    x1 = fsm_x + 0.2 + i * (fsm_sw + fsm_gap) + fsm_sw
    x2 = fsm_x + 0.2 + (i + 1) * (fsm_sw + fsm_gap)
    y = fsm_sy + fsm_sh / 2
    arrow(ax, x1, y, x2, y, color=WARN_EDGE, lw=1.0)

# ④ Control → Recovery FSM trigger (down)
numbered_arrow(ctl_bottom[0], ctl_bottom[1] - 0.05,
               fsm_x + fsm_w - 1.6, fsm_y + fsm_h + 0.02,
               color=WARN_EDGE, num="4", rad=0.18, lw=1.6)
# Recovery Replan → back into Planning (up)
replan_center_x = fsm_x + 0.2 + 4 * (fsm_sw + fsm_gap) + fsm_sw / 2  # 5th state = Replan
ax.annotate("", xy=(pln_x + pln_w / 2, pln_y),
            xytext=(replan_center_x, fsm_y + fsm_h + 0.02),
            arrowprops=dict(arrowstyle="-|>", color=WARN_EDGE, lw=1.4,
                            connectionstyle="arc3,rad=-0.18"))

# ---- HW footer ----
ax.plot([0.4, FIG_W - 0.4], [0.35, 0.35], color=LINE, lw=0.8)
ax.text(0.4, 0.75,
        "실험 환경: Ubuntu 22.04 · ROS 2 Humble · Gazebo Classic 11 · RViz 2 · Python 3.10 (rclpy)",
        ha="left", color=SUB, fontsize=10)
ax.text(0.4, 0.45,
        "Motion Planning: MoveIt 2 + OMPL (RRTConnect)   |   Mapping: OctoMap (MoveIt OccupancyMapUpdater)   |   Robot: UR5e (6-DOF)",
        ha="left", color=SUB, fontsize=10)

fig.savefig(OUT / "architecture.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", OUT / "architecture.png")

print("all p06 assets written to", OUT)
