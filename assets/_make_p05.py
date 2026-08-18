"""Generate PROJECT 05 (SMIM · On-Device Personal Memory Agent) diagrams.

Assets:
    - overview.png      : Ambient Recall UX flow (작성 중 → 트리거 → 근거 카드)
    - architecture.png  : Electron App + Native AI Runtime split · with Yunjae's
                          contribution boxes (Hybrid Retrieval + Self-Correction) highlighted
    - perf_ablation.png : Real ablation bar chart (semantic_only → +BM25/RRF → full)
    - flow_card.png     : Trigger → Query Planning → Hybrid Retrieval →
                          Re-ranking → sLLM Card → Grounding Validator flow

Palette / helpers mirror _make_p08.py.
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

OUT = Path(__file__).parent / "p05"
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


def wrap_text(text, max_chars):
    """Wrap text at ' · ' or whitespace boundaries under max_chars per line.

    Handles Korean/English mixed content: split on ' · ' first (keeps separator),
    then on whitespace as fallback, packing tokens greedily.
    """
    if len(text) <= max_chars:
        return text
    # split on ' · ' keeping the separator attached to the previous token
    parts = []
    for chunk in text.split(" · "):
        if parts:
            parts[-1] = parts[-1] + " ·"
        parts.append(chunk)
    # further split any oversized part on whitespace
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


# --- text layout constants (동일 행간 · 동일 제목-서브 간격 통일) ---
TITLE_ANCHOR_DY = 0.14   # title baseline offset from box center (upward)
SUB_TOP_DY      = -0.02  # sub text top edge offset from box center (downward)
LINESPACING     = 1.35   # uniform inter-line spacing for all sub text


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
        # Fixed anchors: title-to-sub gap and linespacing are the same for every box
        ax.text(x + w / 2, y + h / 2 + TITLE_ANCHOR_DY, text,
                ha="center", va="center", color=ink, fontsize=fontsize, weight="bold")
        ax.text(x + w / 2, y + h / 2 + SUB_TOP_DY, rendered,
                ha="center", va="top", color=subink, fontsize=subsize,
                linespacing=LINESPACING)
    else:
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", color=ink, fontsize=fontsize, weight="bold")


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.5, label=None, rad=0.0):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color,
        shrinkA=6, shrinkB=6,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.14, label,
                ha="center", va="center", color=SUB, fontsize=9)


def container(ax, x, y, w, h, title, fill=WARM, edge=LINE, ink=INK,
              fontsize=13, radius=0.12, title_pad=0.32):
    """Container with title anchored top-center. Returns top of inner content area."""
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
# 1) Overview — Ambient Recall UX flow
# =========================================================
fig, ax = plt.subplots(figsize=(13.4, 4.4), dpi=180)
ax.set_xlim(0, 13.4); ax.set_ylim(0, 4.4); ax.set_axis_off()
fig.patch.set_facecolor(BG)

ax.text(0.4, 4.05, "System Overview", color=INK, fontsize=15, weight="bold")
ax.text(0.4, 3.75,
        "Ambient Recall: 작성 중 맥락 감지 → 개인 문서에서 검증된 근거 카드 (100% on-device)",
        color=SUB, fontsize=11)

y0 = 1.55
h0 = 1.35
w0 = 2.28
gap = 0.32
x_start = 0.4
stages = [
    ("작성 중 텍스트",   "노트 · 문서 · 자기소개서",           WARM,        LINE,      INK),
    ("Trigger 감지",     "맥락 변화 · idle · 누적 입력",       ACCENT_SOFT, ACCENT,    ACCENT),
    ("Hybrid Retrieval", "BM25 + Vector · RRF · Rerank",      ACCENT_SOFT, ACCENT,    ACCENT),
    ("sLLM 카드 생성",   "추천 이유 · 요약 · 활용 제안",       PURPLE_SOFT, PURPLE,    PURPLE),
    ("근거 카드",        "Grounding-validated · 원문 링크",    OK_SOFT,     OK,        OK),
]
positions = [(x_start + i * (w0 + gap), *s) for i, s in enumerate(stages)]
for x, title, sub, fill, edge, ink in positions:
    box(ax, x, y0, w0, h0, title, sub=sub, fill=fill, edge=edge, ink=ink,
        subink=SUB, fontsize=12.5, subsize=9.5)

for i in range(len(positions) - 1):
    x1 = positions[i][0] + w0
    x2 = positions[i + 1][0]
    y = y0 + h0 / 2
    arrow(ax, x1 + 0.05, y, x2 - 0.05, y)

# Learning feedback loop (bottom curve)
ax.annotate("", xy=(x_start + 2.5 * (w0 + gap) + w0 / 2, y0),
            xytext=(x_start + 4 * (w0 + gap) + w0 / 2, y0 - 0.7),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2,
                            connectionstyle="arc3,rad=-0.32"))
ax.text((x_start + 3.25 * (w0 + gap)) + w0 / 2, y0 - 0.55,
        "열기 · 저장 · 무시 → 관련도 재학습 (로컬만)",
        ha="center", color=ACCENT, fontsize=10, weight="bold")

fig.tight_layout()
fig.savefig(OUT / "overview.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", OUT / "overview.png")


# =========================================================
# 2) Architecture — Electron App + Native AI Runtime split
#    Yunjae's contribution: Hybrid Retrieval + Self-Correction (highlighted ACCENT)
# =========================================================
FIG_W, FIG_H = 15.4, 9.4
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=180)
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H); ax.set_axis_off()
fig.patch.set_facecolor(BG)

ax.text(0.4, 9.02, "Architecture", color=INK, fontsize=16, weight="bold")
ax.text(0.4, 8.66,
        "Electron App + Native AI Runtime · CPU/GPU/Neural Engine · 100% on-device",
        color=SUB, fontsize=11)
ax.text(0.4, 8.36,
        "본인 담당: Hybrid Retrieval (RRF+Rerank), Corrective/Reflection loops, 자체 벤치·pytest 인프라",
        color=ACCENT, fontsize=10.5, weight="bold")

# ---- Flow legend (top-right) — 번호로 화살표 흐름 설명 ----
legend_items = [
    ("1", "Orchestrator → Hybrid Retrieval  (질의 전달)",   ACCENT),
    ("2", "Retrieval → sLLM Card  (근거 스니펫)",           PURPLE),
    ("3", "sLLM Card → Grounding Validator  (초안 카드)",   PURPLE),
    ("4", "Validator → Orchestrator  (검증된 카드)",         OK),
]
lg_x0 = 8.6
lg_y0 = 9.02
for i, (num, txt, col) in enumerate(legend_items):
    ly = lg_y0 - i * 0.30
    circle = plt.Circle((lg_x0, ly), 0.13, color=col, zorder=5)
    ax.add_patch(circle)
    ax.text(lg_x0, ly, num, ha="center", va="center",
            color="white", fontsize=10, weight="bold", zorder=6)
    ax.text(lg_x0 + 0.28, ly, txt, ha="left", va="center",
            color=INK, fontsize=10)

# ---- Container 1: Electron Application (left) ----
electron_x, electron_y, electron_w, electron_h = 0.4, 0.6, 5.6, 7.4
inner_top = container(ax, electron_x, electron_y, electron_w, electron_h,
                      "Electron Application (Node · Vite · TypeScript)",
                      fill=WARM, edge=LINE, ink=INK, fontsize=13)

# Compute box centers up-front so arrows use exact coordinates
el_box_w = electron_w - 0.8
el_box_h = 1.15
el_box_x = electron_x + 0.4
el_ui_y     = inner_top - 1.25
el_orch_y   = inner_top - 2.60
el_db_y     = inner_top - 3.95
el_test_y   = inner_top - 5.30

box(ax, el_box_x, el_ui_y, el_box_w, el_box_h,
    "사용자 인터페이스",
    sub="카드 목록 · 앱 화이트리스트 · 인덱싱 관리 · 통계",
    fill=BG, edge=LINE, ink=INK, fontsize=12, subsize=9.5, sub_wrap=28)
box(ax, el_box_x, el_orch_y, el_box_w, el_box_h,
    "Agent Orchestrator",
    sub="Trigger 감지 · 맥락 파싱 · 카드 라이프사이클",
    fill=BG, edge=LINE, ink=INK, fontsize=12, subsize=9.5, sub_wrap=28)
box(ax, el_box_x, el_db_y, el_box_w, el_box_h,
    "Local Vector DB (LanceDB)",
    sub="벡터 + 원문 + 메타 (사용자 홈에만 저장)",
    fill=BG, edge=LINE, ink=INK, fontsize=12, subsize=9.5, sub_wrap=28)
box(ax, el_box_x, el_test_y, el_box_w, el_box_h,
    "Model-free pytest (150 cases · 5 modules)",
    sub="RRF · 캐시 · 재랭킹 계약 · Ollama/LanceDB 없이 검증",
    fill=ACCENT_SOFT, edge=ACCENT, ink=ACCENT, fontsize=12, subsize=9.5, sub_wrap=28)

# Orchestrator center (used by arrows)
orch_right_x = el_box_x + el_box_w
orch_center_y = el_orch_y + el_box_h / 2

# ---- Container 2: Native AI Runtime (right) ----
rt_x, rt_y, rt_w, rt_h = 6.3, 0.6, 8.7, 7.4
inner_top_rt = container(ax, rt_x, rt_y, rt_w, rt_h,
                         "Native AI Runtime (Ollama · Swift · Neural Engine)",
                         fill=WARM, edge=LINE, ink=INK, fontsize=13)

# --- Row A: Hybrid Retrieval band (Yunjae's contribution, ACCENT) ---
band_y = inner_top_rt - 2.15
band_h = 1.85
band_x = rt_x + 0.35
band_w = rt_w - 0.7
p = FancyBboxPatch((band_x, band_y), band_w, band_h,
                   boxstyle="round,pad=0.02,rounding_size=0.10",
                   linewidth=1.4, edgecolor=ACCENT, facecolor=ACCENT_SOFT)
ax.add_patch(p)
ax.text(band_x + band_w / 2, band_y + band_h - 0.28,
        "Hybrid Retrieval  (본인 담당)",
        ha="center", va="top", color=ACCENT, fontsize=12.5, weight="bold")

hr_labels = [
    ("BM25", "고유명사 · 수치 · 전문 용어"),
    ("Vector Search", "granite-embedding-311m"),
    ("RRF Fusion", "Cormack et al. 2009"),
    ("Cross-encoder", "Re-ranker · Top-K"),
]
n = len(hr_labels)
inner_gap = 0.18
sub_w = (band_w - 0.4 - inner_gap * (n - 1)) / n
sub_y = band_y + 0.15
sub_h = 0.95
# per-inner-box char cap (Nanum Korean ~9pt fits ~18 chars in sub_w ≈ 1.83)
inner_wrap = 16
for i, (t, sub) in enumerate(hr_labels):
    sx = band_x + 0.2 + i * (sub_w + inner_gap)
    box(ax, sx, sub_y, sub_w, sub_h, t, sub=sub,
        fill=BG, edge=ACCENT, ink=ACCENT, subink=SUB,
        fontsize=11.5, subsize=9, sub_wrap=inner_wrap)

band_left_center = (band_x, band_y + band_h / 2)
band_bottom_center = (band_x + band_w / 2, band_y)
# a point on the band's bottom edge directly above the sLLM box (for arrow 2)
band_bottom_above_sllm = None  # filled after sllm_x is known

# --- Row B: sLLM Card + Grounding Validator ---
row_b_y = inner_top_rt - 4.35
row_b_h = 1.65
left_w = 4.15
sllm_x = rt_x + 0.35
sllm_center_top = (sllm_x + left_w / 2, row_b_y + row_b_h)
sllm_center = (sllm_x + left_w / 2, row_b_y + row_b_h / 2)
sllm_right_center = (sllm_x + left_w, row_b_y + row_b_h / 2)
box(ax, sllm_x, row_b_y, left_w, row_b_h,
    "sLLM 카드 생성  (Qwen 3B/4B/7B)",
    sub="추천 이유 · 핵심 요약 · 활용 제안 · Structured Output",
    fill=PURPLE_SOFT, edge=PURPLE, ink=PURPLE, subink=SUB,
    fontsize=12, subsize=9.5, sub_wrap=30)

right_x = rt_x + 0.35 + left_w + 0.35
right_w = rt_w - 0.7 - left_w - 0.35
val_left_center = (right_x, row_b_y + row_b_h / 2)
val_top_center = (right_x + right_w / 2, row_b_y + row_b_h)
box(ax, right_x, row_b_y, right_w, row_b_h,
    "Grounding Validator  (본인 담당)",
    sub="원문 근거 검증 · low_confidence 시 폴백 · 재-refine 루프",
    fill=ACCENT_SOFT, edge=ACCENT, ink=ACCENT, subink=SUB,
    fontsize=12, subsize=9.5, sub_wrap=24)

# --- Row C: Ingest pipeline (bottom) ---
row_c_y = inner_top_rt - 6.55
row_c_h = 1.75
ing_x = rt_x + 0.35
ing_w = rt_w - 0.7
p = FancyBboxPatch((ing_x, row_c_y), ing_w, row_c_h,
                   boxstyle="round,pad=0.02,rounding_size=0.10",
                   linewidth=1.2, edgecolor=LINE, facecolor=WARM)
ax.add_patch(p)
ax.text(ing_x + ing_w / 2, row_c_y + row_c_h - 0.28,
        "Ingest Pipeline (오프라인 인덱싱)",
        ha="center", va="top", color=INK, fontsize=12, weight="bold")

ing_labels = [
    ("개인 문서",   "노트·회의록·보고서"),
    ("문서 파싱",   "본문·제목·출처"),
    ("Semantic Chunk", "문맥 단위 분할"),
    ("Embedding",   "granite-embed 311m"),
    ("LanceDB",     "벡터 + 원문 + 메타"),
]
n2 = len(ing_labels)
ing_gap = 0.22
ing_sub_w = (ing_w - 0.4 - ing_gap * (n2 - 1)) / n2
ing_sub_y = row_c_y + 0.18
ing_sub_h = 0.95
for i, (t, sub) in enumerate(ing_labels):
    sx = ing_x + 0.2 + i * (ing_sub_w + ing_gap)
    box(ax, sx, ing_sub_y, ing_sub_w, ing_sub_h, t, sub=sub,
        fill=BG, edge=LINE, ink=INK, subink=SUB,
        fontsize=11, subsize=8.5, sub_wrap=14)
# arrows between ingest steps
for i in range(n2 - 1):
    x1 = ing_x + 0.2 + i * (ing_sub_w + ing_gap) + ing_sub_w
    x2 = ing_x + 0.2 + (i + 1) * (ing_sub_w + ing_gap)
    y = ing_sub_y + ing_sub_h / 2
    arrow(ax, x1, y, x2, y, color=SUB, lw=1.0)

# ---- Cross-container arrows with numbered badges ----
def numbered_arrow(x1, y1, x2, y2, color, num, badge_pos=0.5, rad=0.0, lw=1.6):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=16,
        linewidth=lw, color=color,
        shrinkA=4, shrinkB=6,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    # badge on the arrow midpoint (approximate — offset perpendicular by rad)
    mx = x1 + (x2 - x1) * badge_pos
    my = y1 + (y2 - y1) * badge_pos
    # perpendicular offset when curved
    if rad != 0.0:
        dx, dy = (x2 - x1), (y2 - y1)
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        # shift toward the curve bulge
        mx += -dy / norm * rad * 1.4
        my += dx / norm * rad * 1.4
    circle = plt.Circle((mx, my), 0.17, color=color, zorder=7,
                        ec="white", lw=1.4)
    ax.add_patch(circle)
    ax.text(mx, my, num, ha="center", va="center",
            color="white", fontsize=10, weight="bold", zorder=8)

# ① Orchestrator right-edge → Hybrid Retrieval band left-edge (query)
numbered_arrow(orch_right_x + 0.05, orch_center_y,
               band_left_center[0] - 0.05, band_left_center[1],
               color=ACCENT, num="1", rad=-0.05)

# ② Hybrid Retrieval band bottom → sLLM top (retrieved snippets)
#    Straight vertical drop from band's bottom edge directly above sLLM box
#    so the badge does not overlap the Vector Search / RRF Fusion columns.
numbered_arrow(sllm_center_top[0], band_y - 0.02,
               sllm_center_top[0], sllm_center_top[1] + 0.05,
               color=PURPLE, num="2", rad=0.0)

# ③ sLLM right → Validator left (draft card)
numbered_arrow(sllm_right_center[0] + 0.02, sllm_right_center[1],
               val_left_center[0] - 0.02, val_left_center[1],
               color=PURPLE, num="3", rad=0.0)

# ④ Validator top → Orchestrator right (validated card back)
#    Route up and left across the top to avoid crossing arrow ①
numbered_arrow(val_top_center[0], val_top_center[1] + 0.05,
               orch_right_x + 0.05, orch_center_y - 0.4,
               color=OK, num="4", rad=0.35, lw=1.6)

# HW footer line
ax.plot([0.4, FIG_W - 0.4], [0.35, 0.35], color=LINE, lw=0.8)
ax.text(FIG_W - 0.4, 0.18,
        "HW: CPU · GPU · Neural Engine (macOS/Windows on-device NPU)",
        ha="right", color=SUB, fontsize=9.5)

fig.savefig(OUT / "architecture.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", OUT / "architecture.png")


# =========================================================
# 3) Ablation — real numbers (Phase 3 ablation_24288.out)
# =========================================================
fig, ax = plt.subplots(figsize=(12.4, 5.2), dpi=180)
fig.patch.set_facecolor(BG)

conditions = [
    ("semantic_only\n(vec-only baseline)", 0.700, 0.60, 0.80),
    ("full_no_bm25\n(vec + rerank)",       0.683, 0.60, 0.80),
    ("rrf_no_rerank\n(BM25 + vec)",        1.000, 1.00, 1.00),
    ("full_current\n(BM25+vec+rerank\n+구조신호)", 0.950, 0.90, 1.00),
]
labels = [c[0] for c in conditions]
mrr = [c[1] for c in conditions]
h1 = [c[2] for c in conditions]
h3 = [c[3] for c in conditions]

x = np.arange(len(labels))
w = 0.26

b1 = ax.bar(x - w, mrr, w, label="MRR@3", color=ACCENT)
b2 = ax.bar(x,     h1,  w, label="Hit@1", color=PURPLE)
b3 = ax.bar(x + w, h3,  w, label="Hit@3", color=OK)

for bars in (b1, b2, b3):
    for r in bars:
        v = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", color=INK, fontsize=9.5, weight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10, color=INK)
ax.set_ylim(0, 1.18)
ax.set_ylabel("Score (0 – 1)", color=SUB, fontsize=10.5)
ax.set_title("Phase 3 Ablation — 10-query 자체 벤치 · 40 chunks / 19 docs",
             loc="left", color=INK, fontsize=13.5, weight="bold", pad=14)
ax.legend(loc="upper right", frameon=False, fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=SUB, length=0)
ax.yaxis.grid(True, color=LINE, lw=0.7)
ax.set_axisbelow(True)

# Insight annotation
ax.annotate("BM25 추가 → MRR@3 +0.300",
            xy=(2, 1.00), xytext=(1.2, 1.13),
            fontsize=10, color=ACCENT, weight="bold",
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))

fig.tight_layout()
fig.savefig(OUT / "perf_ablation.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", OUT / "perf_ablation.png")


# =========================================================
# 4) flow_card — Trigger → Query Planning → Hybrid → Rerank → sLLM → Grounding Validator
#    (본인 담당 파트 강조)
# =========================================================
FIG_W, FIG_H = 14.6, 4.8
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=180)
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H); ax.set_axis_off()
fig.patch.set_facecolor(BG)

ax.text(0.4, 4.45, "Card Generation Flow — Trigger → 근거 검증 카드",
        color=INK, fontsize=15, weight="bold")
ax.text(0.4, 4.15,
        "자체 교정 루프 (Corrective) + 카드 단위 재생성 (Reflection) — 본인 담당 파트 (파란색 강조)",
        color=ACCENT, fontsize=10.5, weight="bold")

y0 = 1.85
h0 = 1.55
w0 = 1.85
gap = 0.20
x_start = 0.4

stages = [
    ("Trigger",       "맥락 변화 감지",           WARM,        LINE,   INK,    False),
    ("Query Plan",    "의도 분해 · 라우팅",       ACCENT_SOFT, ACCENT, ACCENT, True),
    ("Hybrid Retr.",  "BM25 + Vec + RRF",         ACCENT_SOFT, ACCENT, ACCENT, True),
    ("Re-ranker",     "Cross-encoder Top-K",      ACCENT_SOFT, ACCENT, ACCENT, True),
    ("sLLM Card",     "이유 · 요약 · 제안",       PURPLE_SOFT, PURPLE, PURPLE, False),
    ("Grounding Val", "원문 근거 검증",           ACCENT_SOFT, ACCENT, ACCENT, True),
    ("Card ✓",        "사용자 알림 카드",         OK_SOFT,     OK,     OK,     False),
]
positions = [(x_start + i * (w0 + gap), *s) for i, s in enumerate(stages)]
for x, title, sub, fill, edge, ink, _mine in positions:
    box(ax, x, y0, w0, h0, title, sub=sub, fill=fill, edge=edge, ink=ink,
        subink=SUB, fontsize=11.5, subsize=9)

for i in range(len(positions) - 1):
    x1 = positions[i][0] + w0
    x2 = positions[i + 1][0]
    y = y0 + h0 / 2
    arrow(ax, x1 + 0.02, y, x2 - 0.02, y)

# Corrective-loop back-arrow (Grounding Val → Query Plan when low_confidence)
gv_x = positions[5][0] + w0 / 2
qp_x = positions[1][0] + w0 / 2
ax.annotate("", xy=(qp_x, y0), xytext=(gv_x, y0 - 0.7),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.3,
                            connectionstyle="arc3,rad=-0.28"))
ax.text((qp_x + gv_x) / 2, y0 - 0.55,
        "low_confidence → Query Refinement (Corrective 외부 루프)",
        ha="center", color=ACCENT, fontsize=10, weight="bold")

# Reflection loop (sLLM Card self-critique, small arc above sLLM)
slm_x = positions[4][0] + w0 / 2
ax.annotate("", xy=(slm_x - 0.35, y0 + h0),
            xytext=(slm_x + 0.35, y0 + h0),
            arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.2,
                            connectionstyle="arc3,rad=-0.55"))
ax.text(slm_x, y0 + h0 + 0.35,
        "카드별 critique → 재생성 (Reflection 내부 루프)",
        ha="center", color=PURPLE, fontsize=9.5, weight="bold")

fig.savefig(OUT / "flow_card.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", OUT / "flow_card.png")

print("all p05 assets written to", OUT)
