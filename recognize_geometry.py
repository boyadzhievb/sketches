#!/usr/bin/env python3
"""
Recognize geometric shapes in hand-drawn doodles and reconstruct clean versions.

Approach: trace the outline of the ink at uniform thickness — that outline IS
the shape. Find outer + inner contours, simplify with polygon approximation,
classify each (circle, triangle, rectangle, polygon, arc, freeform), then
redraw with clean uniform-width strokes.

Dependencies: opencv-python, numpy

Usage:
    python3 recognize_geometry.py <input.png> [--output-dir DIR] [--debug]
    python3 recognize_geometry.py --batch <dir> [--output-dir DIR] [--debug]
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np


CLEAN_COLOR = (170, 75, 41)
STROKE_WIDTH = 2
MIN_CONTOUR_AREA = 30
SIMPLIFY_EPSILON_RATIO = 0.015


@dataclass
class DetectedShape:
    shape_type: str
    center_x: int
    center_y: int
    confidence: float = 0.0
    is_hole: bool = False
    params: dict = field(default_factory=dict)


@dataclass
class RecognitionResult:
    source_file: str
    image_width: int
    image_height: int
    shapes: list = field(default_factory=list)
    rotation_degrees: float = 0.0
    similarity_score: float = 0.0
    ink_coverage_original: float = 0.0
    ink_coverage_clean: float = 0.0


def extract_ink_mask(image_bgra):
    if image_bgra.shape[2] == 4:
        alpha = image_bgra[:, :, 3]
        bgr = image_bgra[:, :, :3]
    else:
        alpha = np.full(image_bgra.shape[:2], 255, dtype=np.uint8)
        bgr = image_bgra

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    has_alpha = image_bgra.shape[2] == 4 and np.any(alpha < 255)

    if has_alpha:
        _, ink_by_alpha = cv2.threshold(alpha, 100, 255, cv2.THRESH_BINARY)
        _, ink_by_dark = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(ink_by_alpha, ink_by_dark)
    else:
        _, mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def classify_contour(contour):
    """Classify a contour by its geometry: circle, triangle, rectangle, polygon, or freeform."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "freeform", {}, 0.0

    circularity = 4 * math.pi * area / (perimeter * perimeter)
    epsilon = SIMPLIFY_EPSILON_RATIO * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertex_count = len(approx)

    (fitted_x, fitted_y), fitted_radius = cv2.minEnclosingCircle(contour)
    enclosing_area = math.pi * fitted_radius * fitted_radius
    fill_ratio = area / enclosing_area if enclosing_area > 0 else 0

    if circularity > 0.75 and fill_ratio > 0.7:
        return "circle", {
            "radius": int(fitted_radius),
            "fitted_center_x": int(fitted_x),
            "fitted_center_y": int(fitted_y),
        }, min(1.0, circularity)

    if vertex_count == 3:
        vertices = approx.reshape(-1, 2).tolist()
        side_lengths = []
        for i in range(3):
            next_vertex = (i + 1) % 3
            side_length = math.hypot(
                vertices[next_vertex][0] - vertices[i][0],
                vertices[next_vertex][1] - vertices[i][1],
            )
            side_lengths.append(side_length)
        return "triangle", {
            "vertices": vertices,
            "side_lengths": [round(side, 1) for side in side_lengths],
        }, 0.85

    if vertex_count == 4:
        rect = cv2.minAreaRect(contour)
        box_points = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box_points)
        rectangularity = area / box_area if box_area > 0 else 0
        vertices = approx.reshape(-1, 2).tolist()

        if rectangularity > 0.85:
            return "rectangle", {
                "vertices": vertices,
                "width": int(max(rect[1])),
                "height": int(min(rect[1])),
                "angle": round(rect[2], 1),
            }, rectangularity

        return "quadrilateral", {"vertices": vertices}, 0.65

    if circularity > 0.55 and vertex_count >= 6:
        return "circle", {
            "radius": int(fitted_radius),
            "fitted_center_x": int(fitted_x),
            "fitted_center_y": int(fitted_y),
        }, circularity

    if 5 <= vertex_count <= 12:
        vertices = approx.reshape(-1, 2).tolist()
        return "polygon", {
            "vertices": vertices,
            "sides": vertex_count,
        }, 0.5 + 0.3 * circularity

    vertices = approx.reshape(-1, 2).tolist()
    return "freeform", {
        "vertices": vertices,
        "vertex_count": vertex_count,
        "circularity": round(circularity, 3),
    }, max(0.2, circularity)


def detect_shapes(mask, image_shape):
    """Find all contours (outer + holes), classify each, return shape list."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    shapes = []

    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        parent_index = hierarchy[index][3]
        is_hole = parent_index >= 0

        shape_type, params, confidence = classify_contour(contour)

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        params["contour_index"] = index
        params["area"] = int(area)

        shapes.append(DetectedShape(
            shape_type=shape_type,
            center_x=center_x,
            center_y=center_y,
            confidence=confidence,
            is_hole=is_hole,
            params=params,
        ))

    return shapes, contours


def compute_rotation_angle(shapes):
    """Find the dominant angle to rotate the doodle for alignment."""
    points = []
    for shape in shapes:
        if not shape.is_hole:
            points.append((shape.center_x, shape.center_y))

    if len(points) < 3:
        return 0.0

    points_array = np.array(points, dtype=np.float64)
    mean = np.mean(points_array, axis=0)
    centered = points_array - mean

    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    angle_radians = math.atan2(principal_axis[1], principal_axis[0])
    angle_degrees = math.degrees(angle_radians)

    if abs(angle_degrees) < 5:
        return 0.0
    if abs(angle_degrees - 90) < 5 or abs(angle_degrees + 90) < 5:
        return 0.0

    snap_targets = [0, 45, 90, -45, -90]
    nearest = min(snap_targets, key=lambda t: abs(angle_degrees - t))
    correction = nearest - angle_degrees

    if abs(correction) < 2:
        return 0.0

    return round(correction, 1)


def normalize_similar_shapes(shapes):
    """Snap similar-sized circles to the same radius; align close coordinates."""
    circles = [shape for shape in shapes if shape.shape_type == "circle"]
    if len(circles) < 2:
        return shapes

    radii = [shape.params.get("radius", 0) for shape in circles]
    if not radii:
        return shapes

    sorted_radii = sorted(set(radii))
    clusters = []
    current_cluster = [sorted_radii[0]]

    for radius in sorted_radii[1:]:
        if radius <= current_cluster[-1] * 1.4:
            current_cluster.append(radius)
        else:
            clusters.append(current_cluster)
            current_cluster = [radius]
    clusters.append(current_cluster)

    radius_map = {}
    for cluster in clusters:
        canonical = int(np.median(cluster))
        for radius in cluster:
            radius_map[radius] = canonical

    for shape in circles:
        original_radius = shape.params.get("radius", 0)
        if original_radius in radius_map:
            shape.params["radius"] = radius_map[original_radius]

    return shapes


def draw_clean_shape(canvas, shape, contour, rotation_matrix, image_center):
    """Draw a single shape with clean strokes, applying rotation."""

    if shape.shape_type == "circle":
        radius = shape.params.get("radius", 10)
        fitted_x = shape.params.get("fitted_center_x", shape.center_x)
        fitted_y = shape.params.get("fitted_center_y", shape.center_y)
        rotated_center = apply_rotation(fitted_x, fitted_y, rotation_matrix)
        cv2.circle(canvas, rotated_center, radius, CLEAN_COLOR, STROKE_WIDTH, cv2.LINE_AA)

    elif shape.shape_type in ("triangle", "rectangle", "quadrilateral", "polygon"):
        vertices = shape.params.get("vertices", [])
        if vertices:
            rotated_vertices = np.array(
                [apply_rotation(vertex[0], vertex[1], rotation_matrix) for vertex in vertices],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [rotated_vertices], True, CLEAN_COLOR, STROKE_WIDTH, cv2.LINE_AA)

    elif shape.shape_type == "freeform":
        vertices = shape.params.get("vertices", [])
        if vertices:
            rotated_vertices = np.array(
                [apply_rotation(vertex[0], vertex[1], rotation_matrix) for vertex in vertices],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [rotated_vertices], True, CLEAN_COLOR, STROKE_WIDTH, cv2.LINE_AA)


def apply_rotation(point_x, point_y, rotation_matrix):
    if rotation_matrix is None:
        return (int(point_x), int(point_y))
    point = np.array([point_x, point_y, 1.0])
    rotated = rotation_matrix @ point
    return (int(rotated[0]), int(rotated[1]))


def draw_clean_image(image_shape, shapes, contours, rotation_angle):
    height, width = image_shape[:2]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    center = (width / 2, height / 2)
    rotation_matrix = None
    if abs(rotation_angle) > 0.5:
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    for shape in sorted(shapes, key=lambda s: (s.is_hole, -s.params.get("area", 0))):
        contour_index = shape.params.get("contour_index")
        if contour_index is not None and contour_index < len(contours):
            draw_clean_shape(canvas, shape, contours[contour_index], rotation_matrix, center)

    return canvas


def compute_similarity(original_mask, clean_image):
    clean_gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
    _, clean_mask = cv2.threshold(clean_gray, 200, 255, cv2.THRESH_BINARY_INV)

    if original_mask.shape != clean_mask.shape:
        clean_mask = cv2.resize(clean_mask, (original_mask.shape[1], original_mask.shape[0]))

    original_dilated = cv2.dilate(original_mask, np.ones((5, 5), np.uint8))
    clean_dilated = cv2.dilate(clean_mask, np.ones((5, 5), np.uint8))

    overlap_forward = cv2.bitwise_and(clean_mask, original_dilated)
    overlap_backward = cv2.bitwise_and(original_mask, clean_dilated)

    forward_score = cv2.countNonZero(overlap_forward) / max(1, cv2.countNonZero(clean_mask))
    backward_score = cv2.countNonZero(overlap_backward) / max(1, cv2.countNonZero(original_mask))
    similarity = (forward_score + backward_score) / 2

    original_coverage = cv2.countNonZero(original_mask) / original_mask.size
    clean_coverage = cv2.countNonZero(clean_mask) / clean_mask.size

    return similarity, original_coverage, clean_coverage


def create_comparison(original_bgra, clean_image, shapes, rotation_angle):
    if original_bgra.shape[2] == 4:
        original_bgr = np.full((*original_bgra.shape[:2], 3), 255, dtype=np.uint8)
        alpha_channel = original_bgra[:, :, 3:4] / 255.0
        original_bgr = (original_bgra[:, :, :3] * alpha_channel + original_bgr * (1 - alpha_channel)).astype(np.uint8)
    else:
        original_bgr = original_bgra.copy()

    height, width = original_bgr.shape[:2]
    gap = 16
    info_width = max(200, int(width * 0.6))
    total_width = width + gap + width + gap + info_width

    comparison = np.full((height, total_width, 3), 255, dtype=np.uint8)
    comparison[:height, :width] = original_bgr

    clean_resized = cv2.resize(clean_image, (width, height))
    comparison[:height, width + gap:width + gap + width] = clean_resized

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (50, 50, 50)
    dim_color = (130, 130, 130)
    info_x = 2 * width + 2 * gap + 8
    text_y = 22

    cv2.putText(comparison, "Detected shapes:", (info_x, text_y), font, 0.38, text_color, 1, cv2.LINE_AA)
    text_y += 20

    shape_counts = {}
    for shape in shapes:
        label = f"{'hole ' if shape.is_hole else ''}{shape.shape_type}"
        shape_counts[label] = shape_counts.get(label, 0) + 1

    for label, count in sorted(shape_counts.items()):
        cv2.putText(comparison, f"  {count}x {label}", (info_x, text_y), font, 0.33, text_color, 1, cv2.LINE_AA)
        text_y += 16

    text_y += 8
    total = len(shapes)
    cv2.putText(comparison, f"Total: {total} elements", (info_x, text_y), font, 0.38, (30, 30, 30), 1, cv2.LINE_AA)
    text_y += 20

    if abs(rotation_angle) > 0.5:
        cv2.putText(comparison, f"Rotated: {rotation_angle:.1f} deg", (info_x, text_y), font, 0.33, dim_color, 1, cv2.LINE_AA)

    cv2.putText(comparison, "Original", (4, height - 6), font, 0.33, dim_color, 1, cv2.LINE_AA)
    cv2.putText(comparison, "Clean", (width + gap + 4, height - 6), font, 0.33, dim_color, 1, cv2.LINE_AA)

    return comparison


def recognize_doodle(input_path, output_directory=None, enable_debug=False):
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        return None

    if output_directory is None:
        output_directory = input_path.parent / "recognized"
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: could not read {input_path}", file=sys.stderr)
        return None

    print(f"Processing: {input_path.name} ({image.shape[1]}x{image.shape[0]})")

    ink_mask = extract_ink_mask(image)

    if enable_debug:
        cv2.imwrite(str(output_directory / f"{input_path.stem}_mask.png"), ink_mask)

    shapes, contours = detect_shapes(ink_mask, image.shape)
    shapes = normalize_similar_shapes(shapes)

    rotation_angle = compute_rotation_angle(shapes)

    shape_summary = {}
    for shape in shapes:
        shape_summary[shape.shape_type] = shape_summary.get(shape.shape_type, 0) + 1
    print(f"  Shapes: {len(shapes)} — {shape_summary}")
    if abs(rotation_angle) > 0.5:
        print(f"  Rotation correction: {rotation_angle:.1f} deg")

    clean_image = draw_clean_image(image.shape, shapes, contours, rotation_angle)
    similarity, original_coverage, clean_coverage = compute_similarity(ink_mask, clean_image)

    stem = input_path.stem
    cv2.imwrite(str(output_directory / f"{stem}_clean.png"), clean_image)

    comparison = create_comparison(image, clean_image, shapes, rotation_angle)
    cv2.imwrite(str(output_directory / f"{stem}_comparison.png"), comparison)

    result = RecognitionResult(
        source_file=input_path.name,
        image_width=image.shape[1],
        image_height=image.shape[0],
        shapes=[asdict(s) for s in shapes],
        rotation_degrees=rotation_angle,
        similarity_score=round(similarity, 4),
        ink_coverage_original=round(original_coverage, 4),
        ink_coverage_clean=round(clean_coverage, 4),
    )

    metadata_path = output_directory / f"{stem}_metadata.json"
    with open(metadata_path, "w") as metadata_file:
        result_dict = asdict(result)
        json.dump(result_dict, metadata_file, indent=2, default=str)

    print(f"  Similarity: {similarity:.1%}")
    print(f"  Output: {stem}_clean.png, {stem}_comparison.png, {stem}_metadata.json")

    return result


def main():
    parser = argparse.ArgumentParser(description="Recognize geometry in hand-drawn doodles")
    parser.add_argument("input", help="Input PNG file or directory (with --batch)")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: <input_dir>/recognized)")
    parser.add_argument("--batch", action="store_true", help="Process all PNGs in the input directory")
    parser.add_argument("--debug", action="store_true", help="Save intermediate debug images")
    parsed_args = parser.parse_args()

    input_path = Path(parsed_args.input)

    if parsed_args.batch:
        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory", file=sys.stderr)
            sys.exit(1)

        png_files = sorted(input_path.glob("*.png"))
        if not png_files:
            print(f"No PNG files found in {input_path}", file=sys.stderr)
            sys.exit(1)

        output_directory = Path(parsed_args.output_dir) if parsed_args.output_dir else input_path / "recognized"
        all_results = []

        for png_file in png_files:
            result = recognize_doodle(png_file, output_directory, parsed_args.debug)
            if result:
                all_results.append(asdict(result))

        summary_path = output_directory / "batch_summary.json"
        with open(summary_path, "w") as summary_file:
            json.dump(all_results, summary_file, indent=2)
        print(f"\nBatch complete: {len(all_results)} doodles -> {summary_path}")

    else:
        if not input_path.is_file():
            print(f"Error: {input_path} is not a file", file=sys.stderr)
            sys.exit(1)

        output_directory = Path(parsed_args.output_dir) if parsed_args.output_dir else None
        recognize_doodle(input_path, output_directory, parsed_args.debug)


if __name__ == "__main__":
    main()
