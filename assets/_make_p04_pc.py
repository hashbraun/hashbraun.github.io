"""Compose pointcloud.png for project-04 — RealSense colored depth + FPS-sampled cloud."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
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

plt.rcParams.update({
    "font.family": [_kor_name, "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "axes.unicode_minus": False,
})

# Sources
COLORED = "/tmp/rils_extract/realsense_bulid/image.png"   # RealSense Viewer colored depth cloud
FPS_SAMPLED = "/tmp/rils_extract/realsense_bulid/image 4.png"  # sparser / cleaner bi-arm cloud

# --- Left: colored depth cloud, crop out the sidebar/settings pane ---
img_color = np.array(Image.open(COLORED).convert("RGB"))
# image.png is 1350x801. Sidebar ~x=0..250, top toolbar ~y=0..40.
# Crop tightly to the 3D viewport content (skip top toolbar row too).
img_color = img_color[75:790, 275:1345]  # (H, W, C)

# --- Right: FPS-sampled cloud, tight crop around the actual point cloud ---
img_fps = np.array(Image.open(FPS_SAMPLED).convert("RGB"))
# image 4.png is 1167x424 — actual points sit in a small central region.
# Crop tightly around the points and scale up.
img_fps = img_fps[130:400, 380:900]

# Match heights by rendering into two axes with same figure height ratio.
fig = plt.figure(figsize=(14, 6.4), dpi=180)
fig.patch.set_facecolor(BG)

# Title
fig.text(0.05, 0.955,
         "RealSense Depth Sensing → FPS-sampled Point Cloud",
         color=INK, fontsize=15, weight="bold", ha="left", va="top")
fig.text(0.05, 0.915,
         "Intel RealSense D435i로 획득한 depth를 point cloud로 재구성 · Farthest Point Sampling으로 균일한 기하학적 커버리지 확보",
         color=SUB, fontsize=10.5, ha="left", va="top")

# Left panel — colored RealSense depth cloud
ax_l = fig.add_axes([0.05, 0.11, 0.44, 0.72])
ax_l.imshow(img_color)
ax_l.set_xticks([]); ax_l.set_yticks([])
for spine in ax_l.spines.values():
    spine.set_color(LINE); spine.set_linewidth(1.0)
ax_l.set_title("RealSense Viewer · Colored Depth Cloud",
               color=INK, fontsize=11.5, loc="left", pad=8, weight="bold")
# Caption pill (bottom-left of image)
ax_l.text(0.02, 0.05, "raw depth · 헤어핀 컬러맵",
          transform=ax_l.transAxes, ha="left", va="bottom",
          fontsize=10, color="white", weight="bold",
          bbox=dict(boxstyle="round,pad=0.35", facecolor=INK, edgecolor="none"))

# Right panel — FPS-sampled bi-arm cloud
ax_r = fig.add_axes([0.53, 0.11, 0.44, 0.72])
ax_r.imshow(img_fps)
ax_r.set_xticks([]); ax_r.set_yticks([])
for spine in ax_r.spines.values():
    spine.set_color(ACCENT); spine.set_linewidth(1.2)
ax_r.set_title("Ours · FPS-sampled Point Cloud (bi-arm scene)",
               color=ACCENT, fontsize=11.5, loc="left", pad=8, weight="bold")
ax_r.text(0.02, 0.05, "policy 입력 · 균일 커버리지",
          transform=ax_r.transAxes, ha="left", va="bottom",
          fontsize=10, color="white", weight="bold",
          bbox=dict(boxstyle="round,pad=0.35", facecolor=ACCENT, edgecolor="none"))

fig.savefig(OUT / "pointcloud.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", OUT / "pointcloud.png")
