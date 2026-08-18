"""Generate PROJECT 08 GIFs with pure PIL.

- demo_ffmpeg.gif : adaptive ffmpeg sampling animation (dense frames → sparse keyframes)
- demo_struct.gif : Qwen3-VL structured JSON typewriter with timeline filling in
"""

from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
import random

OUT = Path(__file__).parent / "p08"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- palette ----------
BG          = (255, 255, 255)
BG_SOFT     = (249, 250, 251)
INK         = (17, 24, 39)
SUB         = (75, 85, 99)
MUTE        = (107, 114, 128)
LINE        = (229, 231, 235)
ACCENT      = (49, 130, 246)
ACCENT_SOFT = (234, 242, 255)
OK          = (16, 185, 129)
OK_SOFT     = (209, 250, 229)
BAD         = (239, 68, 68)
WARN        = (245, 158, 11)
WARN_SOFT   = (255, 247, 237)
VIOLET      = (139, 92, 246)

# Priority-aware palette (matches perf_timeline.png)
P1 = (49, 130, 246)    # blue    — using phone / laptop / drinking
P2 = (139, 92, 246)    # violet  — reading / dressing
P3 = (245, 158, 11)    # orange  — walking
P4 = (16, 185, 129)    # green   — sitting / watching

FONT_PATH_KR = "/home1/sota/.fonts/NanumGothic-Regular.ttf"


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH_KR, size)
    except OSError:
        return ImageFont.load_default()


def rounded_rect(draw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_right(draw, xy, text, fnt, fill):
    tw, th = draw.textbbox((0, 0), text, font=fnt)[2:]
    draw.text((xy[0] - tw, xy[1] - th / 2), text, font=fnt, fill=fill)


def save_gif(frames, path, duration_ms=100, loop=0):
    p_frames = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    p_frames[0].save(
        path,
        save_all=True,
        append_images=p_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
        disposal=1,
    )
    print("wrote", path, f"({len(frames)} frames)")


# =========================================================
# 1) demo_ffmpeg.gif — adaptive sampling: dense → sparse
# =========================================================
def make_ffmpeg_gif():
    W, H = 720, 360
    N = 40  # ~4s at 100ms
    frames = []
    random.seed(7)

    # Two modes to demonstrate: ≤10min → fps=0.33 keeps 8 frames
    #                          >10min → fps=0.1  keeps 4 frames
    # We alternate: first half short-clip mode, second half long-clip mode.
    n_cells = 40  # dense raw frames per row

    for i in range(N):
        t = i / (N - 1)
        long_mode = i >= N // 2       # switch modes at halfway
        # Progress of the "sampling" sweep within the current mode
        sub_t = ((i % (N // 2)) + 1) / (N // 2)

        if long_mode:
            keep_idx = [4, 14, 24, 34]  # fps=0.1 → ~4 keyframes
            fps_txt = "fps = 0.1"
            duration_txt = "duration  12:30  (long clip mode)"
            keep_count = 4
        else:
            keep_idx = [2, 7, 12, 17, 22, 27, 32, 37]  # fps=0.33 → 8 keyframes
            fps_txt = "fps = 0.33"
            duration_txt = "duration  06:00  (standard mode)"
            keep_count = 8

        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img, "RGBA")

        # Outer frame
        rounded_rect(d, (0, 0, W - 1, H - 1), 12, outline=LINE, width=1)

        # Title
        d.text((18, 16), "Adaptive ffmpeg  ·  long-clip preprocessing",
               font=font(15), fill=INK)
        d.text((18, 38), duration_txt, font=font(11), fill=SUB)

        # Mode pill (top-right)
        pill = "> 10 min" if long_mode else "≤ 10 min"
        pill_col = WARN if long_mode else ACCENT
        pw = int(d.textlength(pill, font=font(11))) + 22
        rounded_rect(d, (W - 16 - pw, 20, W - 16, 40), 10,
                     fill=(pill_col[0], pill_col[1], pill_col[2], 40),
                     outline=pill_col, width=1)
        d.text((W - 16 - pw + 11, 22), pill, font=font(11), fill=pill_col)

        # ==== Top strip: raw frames (dense filmstrip) ====
        row_x0, row_x1 = 24, W - 24
        row_y = 84
        row_h = 60
        d.text((24, row_y - 20), "Raw MP4  ·  every frame",
               font=font(11), fill=INK)
        text_right(d, (W - 24, row_y - 14),
                   f"{n_cells} frames shown (illustrative)",
                   font(10), MUTE)

        cell_w = (row_x1 - row_x0) / n_cells
        for k in range(n_cells):
            x = row_x0 + k * cell_w
            fill = (245, 246, 248) if k % 2 == 0 else (238, 240, 244)
            rounded_rect(d, (x + 1, row_y, x + cell_w - 1, row_y + row_h),
                         2, fill=fill, outline=LINE, width=1)

        # ==== Sweep line ====
        sweep_x = row_x0 + (row_x1 - row_x0) * sub_t
        d.line([(sweep_x, row_y - 4), (sweep_x, row_y + row_h + 4)],
               fill=(ACCENT[0], ACCENT[1], ACCENT[2], 220), width=2)

        # ==== Highlight keyframes that have been "swept over" so far ====
        keyframes_selected = [k for k in keep_idx if k * cell_w + row_x0 <= sweep_x]
        for k in keyframes_selected:
            x = row_x0 + k * cell_w
            rounded_rect(d, (x + 1, row_y - 3, x + cell_w - 1, row_y + row_h + 3),
                         3, fill=(ACCENT_SOFT[0], ACCENT_SOFT[1], ACCENT_SOFT[2], 220),
                         outline=ACCENT, width=2)

        # ==== Bottom strip: sampled keyframes with drop lines ====
        out_y = 224
        out_h = 60
        d.text((24, out_y - 20), "Adaptive keyframes  ·  width=512  ·  metadata stripped",
               font=font(11), fill=ACCENT)
        text_right(d, (W - 24, out_y - 14),
                   f"{len(keyframes_selected)} / {keep_count} kept",
                   font(10), SUB)

        # Baseline "output track" as light rounded row
        rounded_rect(d, (row_x0, out_y, row_x1, out_y + out_h),
                     6, fill=BG_SOFT, outline=LINE, width=1)

        # Even-spaced output slots
        slot_w = (row_x1 - row_x0) / keep_count
        for j, k in enumerate(keep_idx):
            src_x = row_x0 + k * cell_w + cell_w / 2
            slot_x = row_x0 + j * slot_w
            if k * cell_w + row_x0 <= sweep_x:
                # draw drop line
                d.line([(src_x, row_y + row_h + 2), (slot_x + slot_w / 2, out_y - 2)],
                       fill=(ACCENT[0], ACCENT[1], ACCENT[2], 160), width=1)
                # filled sampled cell
                rounded_rect(d, (slot_x + 6, out_y + 6, slot_x + slot_w - 6, out_y + out_h - 6),
                             4, fill=ACCENT_SOFT, outline=ACCENT, width=2)
                d.text((slot_x + slot_w / 2 - 6, out_y + out_h / 2 - 8),
                       f"{j + 1}", font=font(12), fill=ACCENT)
            else:
                # empty slot placeholder
                rounded_rect(d, (slot_x + 6, out_y + 6, slot_x + slot_w - 6, out_y + out_h - 6),
                             4, outline=LINE, width=1)

        # ==== Bottom status line: fps chip + size chip ====
        chip_y = 314
        # fps chip
        cx = 24
        chip_w = int(d.textlength(fps_txt, font=font(11))) + 22
        rounded_rect(d, (cx, chip_y, cx + chip_w, chip_y + 22), 8,
                     fill=(ACCENT[0], ACCENT[1], ACCENT[2], 30),
                     outline=ACCENT, width=1)
        d.text((cx + 11, chip_y + 4), fps_txt, font=font(11), fill=ACCENT)

        # width chip
        cx += chip_w + 8
        w_txt = "width = 512"
        chip_w = int(d.textlength(w_txt, font=font(11))) + 22
        rounded_rect(d, (cx, chip_y, cx + chip_w, chip_y + 22), 8,
                     fill=(ACCENT[0], ACCENT[1], ACCENT[2], 30),
                     outline=ACCENT, width=1)
        d.text((cx + 11, chip_y + 4), w_txt, font=font(11), fill=ACCENT)

        # metadata chip
        cx += chip_w + 8
        m_txt = "-map_metadata -1"
        chip_w = int(d.textlength(m_txt, font=font(11))) + 22
        rounded_rect(d, (cx, chip_y, cx + chip_w, chip_y + 22), 8,
                     fill=(ACCENT[0], ACCENT[1], ACCENT[2], 30),
                     outline=ACCENT, width=1)
        d.text((cx + 11, chip_y + 4), m_txt, font=font(11), fill=ACCENT)

        # OOM-guard right-side chip
        oom = "OOM-guard: dynamic fps"
        chip_w = int(d.textlength(oom, font=font(11))) + 22
        cx = W - 24 - chip_w
        rounded_rect(d, (cx, chip_y, cx + chip_w, chip_y + 22), 8,
                     fill=(OK[0], OK[1], OK[2], 30),
                     outline=OK, width=1)
        d.text((cx + 11, chip_y + 4), oom, font=font(11), fill=OK)

        frames.append(img)

    # Hold the last frame of each mode a bit longer for readability
    # Duplicate mid-frame and end-frame
    frames = frames[:N // 2] + [frames[N // 2 - 1]] * 4 + frames[N // 2:] + [frames[-1]] * 6
    save_gif(frames, OUT / "demo_ffmpeg.gif", duration_ms=110)


# =========================================================
# 2) demo_struct.gif — structured JSON typewriter + timeline fill
# =========================================================
def make_struct_gif():
    W, H = 720, 400
    frames = []

    # Action pool with priority — subset shown, matches project code
    actions = [
        ("walking",           P3),
        ("sitting on chair",  P4),
        ("using phone",       P1),
        ("watching tv",       P4),
        ("using laptop",      P1),
        ("drinking",          P1),
    ]
    # Segment list produced by Qwen3-VL structured output
    segs = [
        ("00:00", "00:45", 0, "walking",          P3),
        ("00:45", "01:02", 1, "sitting on chair", P4),
        ("01:02", "02:28", 2, "using phone",      P1),
        ("02:28", "03:25", 3, "watching tv",      P4),
        ("03:25", "04:52", 4, "using laptop",     P1),
        ("04:52", "06:00", 5, "drinking",         P1),
    ]
    # Convert MM:SS → seconds
    def to_sec(s):
        m, ss = s.split(":")
        return int(m) * 60 + int(ss)

    total = 360  # 06:00

    # Timeline area
    tl_x0, tl_x1 = 380, W - 24
    tl_y = 320
    tl_h = 32

    # JSON area
    json_x = 24
    json_y_start = 100
    line_h = 26

    # Frame plan: one segment reveals over ~7 frames (type + fill), then hold.
    frames_per_seg = 7
    N = frames_per_seg * len(segs) + 8  # trailing hold

    for i in range(N):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img, "RGBA")

        # Outer frame + title
        rounded_rect(d, (0, 0, W - 1, H - 1), 12, outline=LINE, width=1)
        d.text((18, 16),
               "Qwen3-VL Structured Output  ·  MM:SS – MM:SS → action index",
               font=font(14), fill=INK)
        d.text((18, 38),
               "constraint-aware prompt + JSON dict + priority-aware post-processing",
               font=font(11), fill=SUB)

        # ==== Left column: JSON typewriter ====
        # Header bar
        rounded_rect(d, (json_x, 70, 355, 92), 6,
                     fill=BG_SOFT, outline=LINE, width=1)
        d.text((json_x + 10, 72), "response.json", font=font(11), fill=SUB)
        # Little OK badge if hallucination filters passed
        if i >= frames_per_seg * len(segs) - 3:
            pill = "✓ filters passed"
            pw = int(d.textlength(pill, font=font(11))) + 20
            rounded_rect(d, (355 - pw + 355 - 355, 70, 355, 92), 6, fill=None)  # noop
            # place OK pill top-right of column
            rounded_rect(d, (355 - pw, 70, 355, 92), 6,
                         fill=(OK[0], OK[1], OK[2], 30), outline=OK, width=1)
            d.text((355 - pw + 8, 72), pill, font=font(11), fill=OK)

        # Opening brace
        d.text((json_x, json_y_start - 6), "{", font=font(14), fill=INK)

        # Reveal per-segment
        revealed = min(len(segs), i // frames_per_seg + 1)
        for j in range(revealed):
            start_ss, end_ss, idx, lab, col = segs[j]
            phase = i - j * frames_per_seg  # 0..frames_per_seg-1 while typing this line
            # Full typed line
            line = f'  "{start_ss} - {end_ss}"  :  {idx},'
            # Progressive type effect for the freshly revealing line
            if j == revealed - 1 and phase < 5:
                chars_total = len(line)
                shown_chars = int(chars_total * (phase + 1) / 5)
                shown = line[:shown_chars]
            else:
                shown = line
            ly = json_y_start + 18 + j * line_h
            d.text((json_x + 6, ly), shown, font=font(12), fill=INK)
            # Trailing action label as a soft comment
            if shown == line:
                cmt = f"# {lab}"
                cmt_x = json_x + 6 + d.textlength(line, font=font(12)) + 10
                d.text((cmt_x, ly), cmt, font=font(12), fill=MUTE)

        # Closing brace after all revealed
        if revealed >= len(segs):
            ly = json_y_start + 18 + len(segs) * line_h
            d.text((json_x, ly), "}", font=font(14), fill=INK)

        # ==== Right column top: action pool ====
        d.text((tl_x0, 70), "Action pool  ·  15 labels (P1–P4)",
               font=font(11), fill=INK)
        pool_x = tl_x0
        pool_y = 92
        for k, (name, col) in enumerate(actions):
            tw = int(d.textlength(name, font=font(11))) + 22
            if pool_x + tw > tl_x1:
                pool_x = tl_x0
                pool_y += 26
            rounded_rect(d, (pool_x, pool_y, pool_x + tw, pool_y + 20), 8,
                         fill=(col[0], col[1], col[2], 30),
                         outline=col, width=1)
            d.text((pool_x + 10, pool_y + 2), name, font=font(11), fill=col)
            pool_x += tw + 6

        # ==== Right column middle: parsed segments list (compact) ====
        d.text((tl_x0, 178),
               "Parsed segments  ·  min. 5s enforced",
               font=font(11), fill=INK)
        for j in range(min(len(segs), revealed)):
            start_ss, end_ss, idx, lab, col = segs[j]
            sy = 200 + j * 18
            # colored dot
            d.ellipse((tl_x0, sy + 3, tl_x0 + 10, sy + 13), fill=col)
            d.text((tl_x0 + 16, sy),
                   f"{start_ss}–{end_ss}   {lab}",
                   font=font(11), fill=INK)

        # ==== Right column bottom: Gantt timeline being filled ====
        d.text((tl_x0, tl_y - 18),
               "Timeline (00:00 – 06:00)",
               font=font(11), fill=INK)
        # Baseline track
        rounded_rect(d, (tl_x0, tl_y, tl_x1, tl_y + tl_h),
                     4, fill=BG_SOFT, outline=LINE, width=1)
        # Draw segments completed so far
        for j in range(min(len(segs), revealed)):
            start_ss, end_ss, idx, lab, col = segs[j]
            s = to_sec(start_ss); e = to_sec(end_ss)
            phase = i - j * frames_per_seg
            # Fill grows during this segment's frames
            if j == revealed - 1 and phase < 5:
                grow = min(1.0, (phase + 1) / 5)
            else:
                grow = 1.0
            xs = tl_x0 + (tl_x1 - tl_x0) * s / total
            xe = tl_x0 + (tl_x1 - tl_x0) * (s + (e - s) * grow) / total
            rounded_rect(d, (xs, tl_y + 4, xe, tl_y + tl_h - 4),
                         3, fill=col, outline=None, width=0)

        # Minute ticks
        for m in range(0, 7):
            x = tl_x0 + (tl_x1 - tl_x0) * m / 6
            d.line([(x, tl_y + tl_h), (x, tl_y + tl_h + 4)], fill=SUB, width=1)
            lbl = f"{m:02d}:00"
            tw = int(d.textlength(lbl, font=font(9)))
            d.text((x - tw / 2, tl_y + tl_h + 6), lbl, font=font(9), fill=SUB)

        frames.append(img)

    save_gif(frames, OUT / "demo_struct.gif", duration_ms=120)


if __name__ == "__main__":
    make_ffmpeg_gif()
    make_struct_gif()
