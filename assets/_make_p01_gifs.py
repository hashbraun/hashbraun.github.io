"""Generate PROJECT 01 GIF demos with pure PIL (no external deps).

- demo_centroid.gif : baseline drift over 60 frames (IoU decays)
- demo_polygon.gif  : polygon-mask propagation stays stable
- tool_ux.gif       : baseline 7 clicks/frame vs ours 2 clicks/frame

Style: clean white background, blue accent, mono-ish sans font,
matches the portfolio tone. Sizes kept modest to keep GIF small.
"""

from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
import random

OUT = Path(__file__).parent / "p01"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- palette (must match portfolio tone) ----------
BG          = (255, 255, 255)
BG_SOFT     = (249, 250, 251)
INK         = (17, 24, 39)
SUB         = (75, 85, 99)
MUTE        = (107, 114, 128)
LINE        = (229, 231, 235)
ACCENT      = (49, 130, 246)
ACCENT_SOFT = (234, 242, 255)
OK          = (16, 185, 129)
BAD         = (239, 68, 68)
WARN        = (245, 158, 11)

# ---------- fonts ----------
FONT_PATH_KR   = "/home1/sota/.fonts/NanumGothic-Regular.ttf"
FONT_PATH_MONO = None  # fall back to default bitmap font if unavailable


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH_KR, size)
    except OSError:
        return ImageFont.load_default()


# ---------- helpers ----------
def rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_center(draw, xy, text, fnt, fill):
    tw, th = draw.textbbox((0, 0), text, font=fnt)[2:]
    draw.text((xy[0] - tw / 2, xy[1] - th / 2), text, font=fnt, fill=fill)


def text_right(draw, xy, text, fnt, fill):
    tw, th = draw.textbbox((0, 0), text, font=fnt)[2:]
    draw.text((xy[0] - tw, xy[1] - th / 2), text, font=fnt, fill=fill)


def make_scene_frame(size, t_norm, mask_offset, mask_size, color_mask,
                     dot_pos=None, polygon_pts=None,
                     title="", iou_text="", subtitle="",
                     legend=("baseline",), badge_text=None, badge_color=ACCENT):
    """Draw a single frame simulating a top-down CCTV scene with a person mask.

    Args:
      size        : (W, H) frame size
      t_norm      : 0..1 progress across the clip
      mask_offset : (dx, dy) shift applied to mask position (drift simulation)
      mask_size   : (mw, mh) mask ellipse dimensions
      color_mask  : RGBA fill of the mask overlay
      dot_pos     : (x, y) center dot (for centroid prompt display) or None
      polygon_pts : list of (x, y) polygon points (for polygon prompt display) or None
      title       : text at top-left
      iou_text    : IoU string at top-right
      subtitle    : caption at bottom-left
      badge_text  : optional pill in top-right (e.g. "drift")
    """
    W, H = size
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")

    # --- frame border ---
    rounded_rect(d, (0, 0, W - 1, H - 1), 12, outline=LINE, width=1)

    # --- CCTV area (rounded panel) ---
    panel = (24, 60, W - 24, H - 60)
    rounded_rect(d, panel, 10, fill=BG_SOFT, outline=LINE, width=1)

    # subtle grid inside the CCTV panel
    for gx in range(panel[0] + 24, panel[2], 24):
        d.line([(gx, panel[1] + 4), (gx, panel[3] - 4)], fill=(238, 240, 244), width=1)
    for gy in range(panel[1] + 24, panel[3], 24):
        d.line([(panel[0] + 4, gy), (panel[2] - 4, gy)], fill=(238, 240, 244), width=1)

    # --- ground-truth "person" ellipse (fixed anchor moves slowly across the scene) ---
    # Person drifts slightly across the frame (simulating movement).
    gt_cx = panel[0] + 100 + int(80 * t_norm)
    gt_cy = panel[1] + 90  + int(30 * math.sin(t_norm * math.pi * 2))
    mw, mh = mask_size
    gt_bbox = (gt_cx - mw / 2, gt_cy - mh / 2, gt_cx + mw / 2, gt_cy + mh / 2)
    d.ellipse(gt_bbox, fill=(209, 213, 219, 200), outline=(107, 114, 128, 255), width=2)
    # tiny "head"
    d.ellipse((gt_cx - 12, gt_cy - mh / 2 - 22, gt_cx + 12, gt_cy - mh / 2 + 2),
              fill=(156, 163, 175, 220), outline=(107, 114, 128, 255), width=2)

    # --- Predicted mask overlay (with drift or stable, depending on caller) ---
    pred_cx = gt_cx + mask_offset[0]
    pred_cy = gt_cy + mask_offset[1]
    pred_bbox = (pred_cx - mw / 2, pred_cy - mh / 2, pred_cx + mw / 2, pred_cy + mh / 2)
    d.ellipse(pred_bbox, fill=color_mask, outline=(ACCENT[0], ACCENT[1], ACCENT[2], 220), width=2)

    # --- Centroid dot ---
    if dot_pos is not None:
        r = 5
        cx, cy = dot_pos
        d.ellipse((cx - r, cy - r, cx + r, cy + r),
                  fill=ACCENT, outline=(255, 255, 255, 255), width=2)

    # --- Polygon prompt outline ---
    if polygon_pts is not None and len(polygon_pts) >= 3:
        d.line(polygon_pts + [polygon_pts[0]], fill=ACCENT, width=2)
        for px, py in polygon_pts:
            r = 3
            d.ellipse((px - r, py - r, px + r, py + r),
                      fill=(255, 255, 255), outline=ACCENT, width=2)

    # --- Title bar ---
    d.text((16, 20), title, font=font(15), fill=INK)
    # IoU
    text_right(d, (W - 18, 32), iou_text, font(14), SUB)
    # subtitle at bottom
    d.text((28, H - 40), subtitle, font=font(12), fill=MUTE)

    # --- optional badge (e.g. "drift") ---
    if badge_text:
        pad_x, pad_y = 10, 4
        bw = d.textlength(badge_text, font=font(12))
        bx1 = W - 16 - bw - 2 * pad_x
        bx2 = W - 16
        by1 = 52
        by2 = by1 + 22
        rounded_rect(d, (bx1, by1, bx2, by2), 11,
                     fill=(badge_color[0], badge_color[1], badge_color[2], 40),
                     outline=badge_color, width=1)
        d.text((bx1 + pad_x, by1 + pad_y - 2), badge_text,
               font=font(12), fill=badge_color)

    return img


def save_gif(frames, path, duration_ms=100, loop=0):
    # GIF is a palette format. Converting each RGB frame to 'P' mode with an
    # adaptive palette makes playback reliable across browsers, and disabling
    # optimize preserves every animation frame (otherwise PIL may drop frames
    # it considers visually similar).
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
# 1) demo_centroid.gif  — baseline drifts as clip progresses
# =========================================================
def make_centroid_gif():
    W, H = 640, 360
    N = 30  # 30 frames * 100ms = 3s loop
    frames = []
    random.seed(1)
    for i in range(N):
        t = i / (N - 1)
        # Drift grows non-linearly, plus a bit of noise
        drift_x = 40 * t + random.uniform(-1.5, 1.5)
        drift_y = 25 * t + random.uniform(-1.5, 1.5)

        # Simulated IoU that decays with drift
        iou = max(0.10, 0.86 - 0.60 * t + random.uniform(-0.02, 0.02))
        iou_color = OK if iou > 0.7 else (WARN if iou > 0.4 else BAD)

        # For drawing, we need the person's GT position to place the centroid dot on it.
        # We recompute using the same formula as make_scene_frame internally, then pass
        # the *drifted* prediction ellipse (via mask_offset) but keep the centroid dot
        # on the GT to show that the prompt was "correct at t=0" but the tracker drifts.
        gt_cx = 24 + 100 + int(80 * t)
        gt_cy = 60 + 90  + int(30 * math.sin(t * math.pi * 2))

        img = make_scene_frame(
            (W, H), t,
            mask_offset=(int(drift_x), int(drift_y)),
            mask_size=(72, 108),
            color_mask=(ACCENT[0], ACCENT[1], ACCENT[2], 110),
            dot_pos=(gt_cx, gt_cy),
            polygon_pts=None,
            title="Centroid Prompt · SAM2 propagation",
            iou_text=f"IoU  {iou:.2f}",
            subtitle=f"frame {i*2+1:02d} / 60   ·   initial dot prompt drifts as person moves",
        )
        # After ~half the clip, add a small "drift" badge
        if i > N * 0.35:
            d = ImageDraw.Draw(img, "RGBA")
            pad_x, pad_y = 10, 4
            txt = "drift"
            bw = d.textlength(txt, font=font(12))
            bx1 = W - 16 - bw - 2 * pad_x
            bx2 = W - 16
            by1 = 52
            by2 = by1 + 22
            rounded_rect(d, (bx1, by1, bx2, by2), 11,
                         fill=(BAD[0], BAD[1], BAD[2], 40),
                         outline=BAD, width=1)
            d.text((bx1 + pad_x, by1 + pad_y - 2), txt, font=font(12), fill=BAD)
        # IoU color chip
        chip = ImageDraw.Draw(img, "RGBA")
        chip.ellipse((W - 200, 25, W - 190, 35), fill=iou_color)

        frames.append(img)
    save_gif(frames, OUT / "demo_centroid.gif", duration_ms=110)


# =========================================================
# 2) demo_polygon.gif — polygon prompt stays stable
# =========================================================
def make_polygon_gif():
    W, H = 640, 360
    N = 30
    frames = []
    random.seed(2)
    for i in range(N):
        t = i / (N - 1)

        # Almost no drift; polygon prompt keeps prediction aligned.
        drift_x = random.uniform(-1.5, 1.5)
        drift_y = random.uniform(-1.5, 1.5)

        iou = 0.86 + random.uniform(-0.02, 0.02) - 0.01 * t  # tiny natural jitter
        iou_color = OK

        gt_cx = 24 + 100 + int(80 * t)
        gt_cy = 60 + 90  + int(30 * math.sin(t * math.pi * 2))

        # Polygon vertices roughly outline the person ellipse
        mw, mh = 72, 108
        poly = []
        n_pts = 10
        for k in range(n_pts):
            ang = 2 * math.pi * k / n_pts
            px = gt_cx + (mw / 2 + random.uniform(-2, 2)) * math.cos(ang)
            py = gt_cy + (mh / 2 + random.uniform(-2, 2)) * math.sin(ang)
            poly.append((px, py))

        img = make_scene_frame(
            (W, H), t,
            mask_offset=(int(drift_x), int(drift_y)),
            mask_size=(mw, mh),
            color_mask=(ACCENT[0], ACCENT[1], ACCENT[2], 130),
            dot_pos=None,
            polygon_pts=poly,
            title="Polygon-Mask Prompt (ours) · SAM2 propagation",
            iou_text=f"IoU  {iou:.2f}",
            subtitle=f"frame {i*2+1:02d} / 60   ·   confirmed initial polygon anchors propagation",
        )
        # Green "stable" badge throughout
        d = ImageDraw.Draw(img, "RGBA")
        pad_x, pad_y = 10, 4
        txt = "stable"
        bw = d.textlength(txt, font=font(12))
        bx1 = W - 16 - bw - 2 * pad_x
        bx2 = W - 16
        by1 = 52
        by2 = by1 + 22
        rounded_rect(d, (bx1, by1, bx2, by2), 11,
                     fill=(OK[0], OK[1], OK[2], 40),
                     outline=OK, width=1)
        d.text((bx1 + pad_x, by1 + pad_y - 2), txt, font=font(12), fill=OK)
        chip = ImageDraw.Draw(img, "RGBA")
        chip.ellipse((W - 200, 25, W - 190, 35), fill=iou_color)
        frames.append(img)
    save_gif(frames, OUT / "demo_polygon.gif", duration_ms=110)


# =========================================================
# 3) tool_ux.gif — baseline 7 clicks/frame vs ours 2 clicks/frame
# =========================================================
def make_tool_ux_gif():
    W, H = 720, 360
    N = 34  # a bit longer so counters have time to settle
    frames = []
    random.seed(3)

    # Click positions (predetermined, so animation is deterministic)
    # Panel positions are relative to each panel's rect.
    def click_seq(count, seed):
        rng = random.Random(seed)
        pts = []
        while len(pts) < count:
            x = rng.randint(40, 260)
            y = rng.randint(60, 200)
            pts.append((x, y))
        return pts

    left_clicks  = click_seq(7, 11)   # baseline
    right_clicks = click_seq(2, 13)   # ours

    for i in range(N):
        t = i / (N - 1)
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img, "RGBA")

        # Outer border
        rounded_rect(d, (0, 0, W - 1, H - 1), 12, outline=LINE, width=1)

        # Two panels
        pad = 24
        gap = 24
        pw = (W - 2 * pad - gap) // 2
        ph = H - 90
        left_panel  = (pad, 68, pad + pw, 68 + ph)
        right_panel = (pad + pw + gap, 68, W - pad, 68 + ph)

        rounded_rect(d, left_panel,  10, fill=BG_SOFT, outline=LINE, width=1)
        rounded_rect(d, right_panel, 10, fill=ACCENT_SOFT, outline=ACCENT, width=1)

        # Titles
        d.text((left_panel[0]  + 12, 24), "Baseline",     font=font(15), fill=INK)
        d.text((left_panel[0]  + 12, 44), "7 clicks / frame",  font=font(12), fill=MUTE)
        d.text((right_panel[0] + 12, 24), "Ours (Polygon Prompt)",     font=font(15), fill=ACCENT)
        d.text((right_panel[0] + 12, 44), "2 clicks / frame",           font=font(12), fill=SUB)

        # ==== Left panel: animate 7 clicks appearing sequentially ====
        # Show clicks 0..min(7, floor(t*10)) using a sweep timer
        left_visible = min(7, int(t * 10))
        for k in range(left_visible):
            cx = left_panel[0] + left_clicks[k][0]
            cy = left_panel[1] + left_clicks[k][1]
            # Pulse for the freshest click
            r = 6 + (2 if k == left_visible - 1 else 0)
            d.ellipse((cx - r, cy - r, cx + r, cy + r),
                      fill=(255, 255, 255), outline=BAD, width=2)
            d.text((cx + 8, cy - 8), str(k + 1), font=font(11), fill=BAD)
        # Click counter (top-right of panel)
        text_right(d, (left_panel[2] - 12, 34),
                   f"clicks: {left_visible}/7", font(13), BAD)

        # ==== Right panel: animate 2 clicks ====
        right_visible = min(2, int(t * 4))
        for k in range(right_visible):
            cx = right_panel[0] + right_clicks[k][0]
            cy = right_panel[1] + right_clicks[k][1]
            r = 6 + (2 if k == right_visible - 1 else 0)
            d.ellipse((cx - r, cy - r, cx + r, cy + r),
                      fill=(255, 255, 255), outline=ACCENT, width=2)
            d.text((cx + 8, cy - 8), str(k + 1), font=font(11), fill=ACCENT)
        text_right(d, (right_panel[2] - 12, 34),
                   f"clicks: {right_visible}/2", font(13), ACCENT)

        # ==== Bottom summary bar ====
        d.text((pad, H - 32),
               "60-frame clip labeling time:",
               font=font(12), fill=MUTE)
        # baseline time
        d.text((pad + 200, H - 32), "30 min", font=font(13), fill=BAD)
        d.text((pad + 260, H - 32), "→", font=font(13), fill=MUTE)
        d.text((pad + 280, H - 32), "4 min", font=font(13), fill=ACCENT)

        frames.append(img)

    save_gif(frames, OUT / "tool_ux.gif", duration_ms=120)


if __name__ == "__main__":
    make_centroid_gif()
    make_polygon_gif()
    make_tool_ux_gif()
