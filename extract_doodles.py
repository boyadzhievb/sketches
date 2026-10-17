#!/usr/bin/env python3
"""
Extract individual doodles from scanned sketch sheets.

Processes JPEG/PNG photos of hand-drawn blue pen doodles on white paper.
Detects each doodle as a connected ink region, crops it with padding,
removes the white background, and saves as a transparent PNG.

Three-stage detection:
  1. Progressive erosion with post-merge evaluation
  2. Watershed segmentation via distance transform (scipy)
  3. Dilation-based spatial clustering for disconnected fragments

Dependencies: opencv-python, numpy, Pillow, scipy

Usage:
    python3 extract_doodles.py [input_dir] [output_dir] [options]

    --debug                    saves ink mask, bounding boxes, and watershed debug
    --watermark "Your Name"    semi-transparent text tiled diagonally across each doodle
    --watermark-logo logo.png  small logo stamp in the bottom-right corner
    --copyright "Your Name"    embeds copyright info in PNG metadata

Defaults:
    input_dir  = current directory (looks for *.jpeg, *.jpg, *.png)
    output_dir = ./extracted_doodles/
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from scipy import ndimage

A4_RATIO = 297 / 210  # ~1.4143


def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def crop_to_paper(img_bgr, debug_dir=None, stem=None):
    """Detect the A4 paper in the photo and perspective-correct to it."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < h * w * 0.2:
        return img_bgr

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        pts = order_points(approx.reshape(4, 2))

        width_top = np.linalg.norm(pts[1] - pts[0])
        width_bot = np.linalg.norm(pts[2] - pts[3])
        height_left = np.linalg.norm(pts[3] - pts[0])
        height_right = np.linalg.norm(pts[2] - pts[1])

        out_w = int(max(width_top, width_bot))
        out_h = int(max(height_left, height_right))

        if out_w > out_h:
            out_h = int(out_w / A4_RATIO)
        else:
            out_w = int(out_h / A4_RATIO)

        dst = np.array([
            [0, 0], [out_w - 1, 0],
            [out_w - 1, out_h - 1], [0, out_h - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
        warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h))

        if debug_dir and stem:
            debug_img = img_bgr.copy()
            cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 8)
            for p in pts:
                cv2.circle(debug_img, (int(p[0]), int(p[1])), 20, (0, 0, 255), -1)
            cv2.imwrite(str(debug_dir / f"{stem}_paper.jpg"), debug_img)

        return warped
    else:
        x, y, rw, rh = cv2.boundingRect(largest)
        margin = 10
        x1 = max(0, x + margin)
        y1 = max(0, y + margin)
        x2 = min(w, x + rw - margin)
        y2 = min(h, y + rh - margin)
        cropped = img_bgr[y1:y2, x1:x2]

        if debug_dir and stem:
            debug_img = img_bgr.copy()
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 8)
            cv2.imwrite(str(debug_dir / f"{stem}_paper.jpg"), debug_img)

        return cropped


def detect_ink_mask(img_bgr):
    """Blue-ink color detection via HSV + LAB color spaces."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    blue_hsv = cv2.inRange(hsv, (85, 30, 20), (145, 255, 240))
    blue_lab = cv2.inRange(lab, (0, 105, 0), (140, 155, 118))

    combined = cv2.bitwise_or(blue_hsv, blue_lab)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=2)

    return combined


def merge_nearby_boxes(boxes, distance):
    """Merge bounding boxes that overlap or are within distance pixels."""
    if not boxes:
        return []

    rects = [(b[0], b[1], b[2], b[3]) for b in boxes]
    merged = True
    while merged:
        merged = False
        new_rects = []
        used = [False] * len(rects)

        for i in range(len(rects)):
            if used[i]:
                continue
            x1, y1, w1, h1 = rects[i]
            rx1, ry1 = x1, y1
            rx2, ry2 = x1 + w1, y1 + h1

            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = rects[j]
                sx1, sy1 = x2, y2
                sx2, sy2 = x2 + w2, y2 + h2

                if (rx1 - distance <= sx2 and rx2 + distance >= sx1 and
                        ry1 - distance <= sy2 and ry2 + distance >= sy1):
                    rx1 = min(rx1, sx1)
                    ry1 = min(ry1, sy1)
                    rx2 = max(rx2, sx2)
                    ry2 = max(ry2, sy2)
                    used[j] = True
                    merged = True

            new_rects.append((rx1, ry1, rx2 - rx1, ry2 - ry1))
            used[i] = True

        rects = new_rects

    return rects


def _full_pipeline(raw_boxes, img_h, img_w):
    """Merge nearby boxes + filter page-spanning blobs and tiny specks."""
    max_dim = max(img_h, img_w)
    merge_dist = int(max_dim * 0.008)
    merged = merge_nearby_boxes(raw_boxes, distance=merge_dist)

    filtered = []
    for b in merged:
        x, y, bw, bh = b[:4]
        if bw > img_w * 0.6 and bh > img_h * 0.6:
            continue
        if bw < int(max_dim * 0.025) or bh < int(max_dim * 0.025):
            continue
        filtered.append((x, y, bw, bh))
    return filtered


# ---------------------------------------------------------------------------
# Stage 1: Progressive erosion (evaluates AFTER merge+filter)
# ---------------------------------------------------------------------------

def find_boxes_erosion(mask, img_h, img_w, verbose=False):
    """Try increasingly aggressive erosion. Evaluate each strategy on the
    FINAL merged+filtered result, not raw component count."""
    max_dim = max(img_h, img_w)
    total_px = img_h * img_w

    # (erosion_iters, kernel_size, min_area_frac, max_area_frac)
    strategies = [
        (1,  3, 0.0002, 0.15),
        (3,  3, 0.0002, 0.15),
        (6,  3, 0.0001, 0.12),
        (8,  3, 0.0001, 0.10),
        (12, 3, 0.00008, 0.10),
        (16, 3, 0.00005, 0.08),
        (6,  5, 0.00008, 0.10),
        (10, 5, 0.00005, 0.08),
        (15, 5, 0.00003, 0.06),
        (20, 5, 0.00002, 0.05),
    ]

    best_final = []

    for erode_iter, kern_sz, min_a_frac, max_a_frac in strategies:
        min_area = int(total_px * min_a_frac)
        max_area = int(total_px * max_a_frac)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_sz, kern_sz))
        eroded = cv2.erode(mask, kernel, iterations=erode_iter)
        dilated = cv2.dilate(eroded, kernel, iterations=max(1, erode_iter // 3))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dilated, connectivity=8
        )

        boxes = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area < min_area or area > max_area:
                continue
            if bw > img_w * 0.55 and bh > img_h * 0.55:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 8:
                continue
            boxes.append((x, y, bw, bh))

        final = _full_pipeline(boxes, img_h, img_w)

        if verbose:
            print(f"    erosion e={erode_iter} k={kern_sz}: "
                  f"{num_labels-1} comps, {len(boxes)} raw, {len(final)} final")

        if len(final) > len(best_final):
            best_final = final

        if len(final) >= 5:
            break

    return best_final


# ---------------------------------------------------------------------------
# Stage 2: Watershed segmentation (scipy)
# ---------------------------------------------------------------------------

def find_boxes_watershed(mask, img_h, img_w, verbose=False,
                         debug_dir=None, stem=None):
    """Distance-transform + watershed for dense sheets where erosion fails."""
    total_px = img_h * img_w

    dist = ndimage.distance_transform_edt(mask)
    if dist.max() == 0:
        return []

    num_peaks = 0
    peak_labels = None

    for peak_frac in (0.35, 0.25, 0.18, 0.12, 0.08, 0.05):
        peak_thresh = dist.max() * peak_frac
        peaks = (dist > peak_thresh).astype(np.uint8)
        peak_labels, num_peaks = ndimage.label(peaks)

        if verbose:
            print(f"    watershed peak_frac={peak_frac}: {num_peaks} peaks")
        if num_peaks >= 3:
            break

    if num_peaks < 2 or peak_labels is None:
        return []

    cost = (255 - dist * 255 / dist.max()).astype(np.uint8)
    labels = ndimage.watershed_ift(cost, peak_labels)
    labels[mask == 0] = 0

    if debug_dir and stem:
        vis = (labels * 17 % 256).astype(np.uint8)
        cv2.imwrite(str(debug_dir / f"{stem}_watershed.jpg"), vis)

    min_area = int(total_px * 0.00005)
    max_area = int(total_px * 0.12)

    boxes = []
    for label_id in range(1, num_peaks + 1):
        region = (labels == label_id).astype(np.uint8)
        area = int(region.sum())
        if area < min_area or area > max_area:
            continue

        coords = cv2.findNonZero(region)
        if coords is None:
            continue
        x, y, bw, bh = cv2.boundingRect(coords)

        if bw > img_w * 0.55 and bh > img_h * 0.55:
            continue
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 8:
            continue
        boxes.append((x, y, bw, bh))

    final = _full_pipeline(boxes, img_h, img_w)

    if verbose:
        print(f"    watershed: {len(boxes)} raw, {len(final)} final")

    return final


# ---------------------------------------------------------------------------
# Stage 3: Dilation-based spatial clustering
# ---------------------------------------------------------------------------

def find_boxes_dilate_cluster(mask, img_h, img_w, verbose=False):
    """Dilate the mask to merge nearby fragments, then find components.

    This is the image-space equivalent of DBSCAN: nearby ink blobs
    merge into one connected component when dilated enough.
    """
    max_dim = max(img_h, img_w)
    total_px = img_h * img_w

    best_final = []

    for dilate_px in (15, 25, 40, 60):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (dilate_px, dilate_px))
        dilated = cv2.dilate(mask, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dilated, connectivity=8
        )

        min_area = int(total_px * 0.001)
        max_area = int(total_px * 0.15)

        boxes = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if bw > img_w * 0.55 and bh > img_h * 0.55:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 8:
                continue
            boxes.append((x, y, bw, bh))

        final = _full_pipeline(boxes, img_h, img_w)

        if verbose:
            print(f"    dilate_cluster d={dilate_px}: "
                  f"{num_labels-1} comps, {len(final)} final")

        if len(final) > len(best_final):
            best_final = final

        if len(final) >= 5:
            break

    return best_final


# ---------------------------------------------------------------------------
# Main detection dispatcher
# ---------------------------------------------------------------------------

def find_doodle_boxes(mask, img_h, img_w, verbose=False,
                      debug_dir=None, stem=None):
    """Three-stage detection: erosion → watershed → dilation clustering."""

    # Stage 1
    final1 = find_boxes_erosion(mask, img_h, img_w, verbose=verbose)
    if len(final1) >= 3:
        if verbose:
            print(f"    -> erosion: {len(final1)} doodles ✓")
        return final1

    if verbose:
        print(f"    -> erosion: only {len(final1)}, trying watershed...")

    # Stage 2
    final2 = find_boxes_watershed(
        mask, img_h, img_w, verbose=verbose,
        debug_dir=debug_dir, stem=stem
    )
    if len(final2) >= 3:
        if verbose:
            print(f"    -> watershed: {len(final2)} doodles ✓")
        return final2

    if verbose:
        print(f"    -> watershed: only {len(final2)}, trying dilation clustering...")

    # Stage 3
    final3 = find_boxes_dilate_cluster(mask, img_h, img_w, verbose=verbose)
    if verbose:
        print(f"    -> dilate_cluster: {len(final3)} doodles")

    # Return whichever found the most
    best = max([final1, final2, final3], key=len)
    return best


# ---------------------------------------------------------------------------
# Alpha channel: color-based transparency (no white halo)
# ---------------------------------------------------------------------------

def make_alpha_mask(img_bgr, ink_mask, crop_region):
    """Build a smooth alpha channel based on ink color distance.

    Instead of just dilating the binary mask (which creates white halos),
    compute how "blue" each pixel is and use that as soft alpha.
    """
    x1, y1, x2, y2 = crop_region
    crop = img_bgr[y1:y2, x1:x2]
    crop_ink = ink_mask[y1:y2, x1:x2]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    base = cv2.dilate(crop_ink, kernel, iterations=2)

    # Soft edges: measure distance from white paper in LAB space
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)
    b_ch = lab[:, :, 2].astype(np.float32)

    # Paper: high L (~230+), neutral B (~128). Blue ink: lower L, lower B.
    paper_dist = np.sqrt(((230 - l_ch) / 230) ** 2 + ((128 - b_ch) / 128) ** 2)
    paper_dist = np.clip(paper_dist * 2.5, 0, 1)

    soft = (paper_dist * 255).astype(np.uint8)
    alpha = np.maximum(base, soft)

    # Clamp to neighborhood of detected ink
    far_from_ink = cv2.dilate(crop_ink, kernel, iterations=6)
    alpha[far_from_ink == 0] = 0

    return alpha


# ---------------------------------------------------------------------------
# Watermarking
# ---------------------------------------------------------------------------

def apply_watermark(pil_img, text=None, logo_path=None, opacity=0.25):
    """Stamp a semi-transparent watermark on a RGBA PIL image.

    text:       string to overlay diagonally across the image
    logo_path:  path to a PNG logo to place in the bottom-right corner
    opacity:    0.0 (invisible) to 1.0 (fully opaque)
    """
    w, h = pil_img.size

    if text:
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Scale font to ~5% of the image diagonal
        diag = int((w ** 2 + h ** 2) ** 0.5)
        font_size = max(12, diag // 20)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",
                                      font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        alpha_val = int(255 * opacity)
        color = (41, 75, 170, alpha_val)  # #294baa — matches ink color

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Tile diagonally across the image
        spacing_x = tw + font_size * 3
        spacing_y = th + font_size * 4

        # Create a larger canvas to rotate
        canvas_size = max(w, h) * 2
        text_layer = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)

        for ty in range(-canvas_size // 2, canvas_size, spacing_y):
            for tx in range(-canvas_size // 2, canvas_size, spacing_x):
                text_draw.text((tx, ty), text, fill=color, font=font)

        # Rotate 30 degrees and crop to original size
        text_layer = text_layer.rotate(30, resample=Image.BICUBIC, expand=False)
        cx = (canvas_size - w) // 2
        cy = (canvas_size - h) // 2
        text_layer = text_layer.crop((cx, cy, cx + w, cy + h))

        pil_img = Image.alpha_composite(pil_img, text_layer)

    if logo_path and Path(logo_path).exists():
        logo = Image.open(logo_path).convert("RGBA")

        # Scale logo to ~15% of the smaller dimension
        logo_size = max(20, min(w, h) // 7)
        logo.thumbnail((logo_size, logo_size), Image.LANCZOS)

        # Adjust opacity
        r, g, b, a = logo.split()
        a = a.point(lambda x: int(x * opacity))
        logo = Image.merge("RGBA", (r, g, b, a))

        margin = max(4, min(w, h) // 30)
        pos = (w - logo.width - margin, h - logo.height - margin)
        pil_img.paste(logo, pos, logo)

    return pil_img


def add_png_metadata(pil_img, copyright_text):
    """Embed copyright info in PNG metadata."""
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Copyright", copyright_text)
    meta.add_text("Author", copyright_text)
    meta.add_text("Comment", f"© {copyright_text}")
    return meta


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_doodles(input_path, output_dir, debug=False,
                    watermark_text=None, watermark_logo=None,
                    copyright_text=None):
    """Extract individual doodles from a sketch sheet photo."""
    raw_img = cv2.imread(str(input_path))
    if raw_img is None:
        print(f"  Could not read: {input_path}")
        return 0

    stem = Path(input_path).stem
    debug_dir = None
    if debug:
        debug_dir = Path(output_dir) / "_debug"
        debug_dir.mkdir(exist_ok=True)

    img = crop_to_paper(raw_img, debug_dir=debug_dir, stem=stem)

    h, w = img.shape[:2]
    max_dim = max(h, w)

    mask = detect_ink_mask(img)

    if debug:
        cv2.imwrite(str(debug_dir / f"{stem}_mask.jpg"), mask)

    filtered = find_doodle_boxes(
        mask, h, w, verbose=debug, debug_dir=debug_dir, stem=stem
    )

    # Sort top-to-bottom, left-to-right
    row_height = h // 8
    filtered.sort(key=lambda b: (b[1] // row_height, b[0]))

    if debug:
        annotated = img.copy()
        for (x, y, bw, bh) in filtered:
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
        cv2.imwrite(str(debug_dir / f"{stem}_boxes.jpg"), annotated)

    padding = int(max_dim * 0.012)
    count = 0

    for i, (x, y, bw, bh) in enumerate(filtered):
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)

        crop_rgb = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        alpha = make_alpha_mask(img, mask, (x1, y1, x2, y2))

        pil_img = Image.fromarray(crop_rgb).convert("RGBA")
        pil_img.putalpha(Image.fromarray(alpha))

        if watermark_text or watermark_logo:
            pil_img = apply_watermark(pil_img, text=watermark_text,
                                      logo_path=watermark_logo)

        out_name = f"{stem}_doodle_{i + 1:03d}.png"

        if copyright_text:
            png_meta = add_png_metadata(pil_img, copyright_text)
            pil_img.save(Path(output_dir) / out_name, "PNG", pnginfo=png_meta)
        else:
            pil_img.save(Path(output_dir) / out_name, "PNG")
        count += 1

    return count


def _get_flag(flag_name):
    """Extract --flag=value or --flag value from sys.argv."""
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == flag_name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith(flag_name + "="):
            return arg.split("=", 1)[1]
    return None


def main():
    positional = []
    skip_next = False
    for i, a in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if a.startswith("--"):
            if "=" not in a and a in ("--watermark", "--watermark-logo", "--copyright"):
                skip_next = True
            continue
        positional.append(a)

    debug = "--debug" in sys.argv
    watermark_text = _get_flag("--watermark")
    watermark_logo = _get_flag("--watermark-logo")
    copyright_text = _get_flag("--copyright")

    input_dir = Path(positional[0]) if len(positional) > 0 else Path(".")
    output_dir = Path(positional[1]) if len(positional) > 1 else Path("extracted_doodles")

    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpeg", ".jpg", ".png"}
    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in extensions and p.is_file()
    )

    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Processing {len(images)} images from {input_dir}")
    print(f"Output: {output_dir}/")
    if debug:
        print(f"Debug: {output_dir}/_debug/")
    if watermark_text:
        print(f"Watermark: \"{watermark_text}\"")
    if watermark_logo:
        print(f"Logo: {watermark_logo}")
    if copyright_text:
        print(f"Copyright: {copyright_text}")
    print()

    total = 0
    for img_path in images:
        count = extract_doodles(
            img_path, output_dir, debug=debug,
            watermark_text=watermark_text,
            watermark_logo=watermark_logo,
            copyright_text=copyright_text,
        )
        print(f"  {img_path.name}: {count} doodles extracted")
        total += count

    print(f"\nDone. {total} doodles extracted to {output_dir}/")


if __name__ == "__main__":
    main()
