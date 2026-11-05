from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class DewarpError(RuntimeError):
    """Raised when an image/mask pair cannot be dewarped safely."""


@dataclass(frozen=True)
class DewarpResult:
    image: np.ndarray
    rotation_degrees: float
    output_size: tuple[int, int]
    grid2d: np.ndarray


def normalize_mask(mask: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    """Convert a 0/1, 0/255, grayscale, or color mask to one filled component."""
    if mask is None:
        raise DewarpError("mask is empty or unreadable")
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2).astype(np.uint8)
    elif mask.ndim == 2:
        mask = (mask > 0).astype(np.uint8)
    else:
        raise DewarpError(f"unsupported mask shape: {mask.shape}")
    if mask.shape != expected_shape:
        raise DewarpError(
            f"image and mask sizes differ: image={expected_shape}, mask={mask.shape}"
        )
    if not np.any(mask):
        raise DewarpError("mask contains no foreground pixels")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise DewarpError("mask contains no foreground component")
    largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    largest = (labels == largest_label).astype(np.uint8)
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise DewarpError("cannot extract the document contour")
    filled = np.zeros_like(largest)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return filled


def _rotate_and_crop(
    image: np.ndarray, mask: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    tuple[int, int],
]:
    height, width = mask.shape
    kernel_size = max(3, int(round(min(height, width) * 0.008)))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(
        expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise DewarpError("cannot find a document contour for rotation")
    points = np.vstack(contours)
    rectangle = cv2.minAreaRect(points)
    base_angle = float(rectangle[2] if rectangle[2] < 60 else rectangle[2] - 90)
    rect_w, rect_h = rectangle[1]

    # 让文档的短边对齐输入图像短边、长边对齐输入图像长边：
    # minAreaRect 对竖版文档会给出约 -90° 的角（把长边当作 width 边），直接旋转会把
    # 竖页整页翻成横版。这里比较"按 base_angle 旋转后文档长边是否水平"与"输入图像长边
    # 是否水平"，不一致则额外补 90°，使文档朝向与输入图像一致，避免无谓的 90° 翻转。
    # 该判断对任意输入都适用（横/竖/方），不依赖单页文档特判。
    doc_long_horizontal = rect_w >= rect_h          # base_angle 旋转后文档长边是否水平
    img_long_horizontal = width >= height           # 输入图像长边是否水平
    angle = base_angle + 90.0 if doc_long_horizontal != img_long_horizontal else base_angle

    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    rotated_width = int(width * cosine + height * sine)
    rotated_height = int(height * cosine + width * sine)
    matrix[0, 2] += rotated_width / 2.0 - width / 2.0
    matrix[1, 2] += rotated_height / 2.0 - height / 2.0

    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    box = cv2.boxPoints(rectangle)
    homogeneous_box = np.column_stack((box, np.ones(4, dtype=np.float32)))
    rotated_box = homogeneous_box @ matrix.T
    x1 = max(0, int(np.floor(rotated_box[:, 0].min())))
    y1 = max(0, int(np.floor(rotated_box[:, 1].min())))
    x2 = min(rotated_width, int(np.ceil(rotated_box[:, 0].max())))
    y2 = min(rotated_height, int(np.ceil(rotated_box[:, 1].max())))
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise DewarpError("document region is too small after rotation")

    return (
        rotated_image[y1:y2, x1:x2],
        rotated_mask[y1:y2, x1:x2],
        angle,
        matrix,
        (x1, y1),
    )


def _odd_window(length: int, fraction: float, maximum: int) -> int:
    """Return a scale-aware odd smoothing window that fits ``length``."""
    window = max(3, int(round(length * fraction)))
    window = min(window, maximum, length if length % 2 else length - 1)
    if window % 2 == 0:
        window -= 1
    return max(1, window)


def _smooth_envelope(values: np.ndarray) -> np.ndarray:
    """Suppress isolated mask defects without flattening the page curvature."""
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 5:
        return values.copy()
    median_window = _odd_window(len(values), 0.006, 15)
    gaussian_window = _odd_window(len(values), 0.012, 31)
    padded = np.pad(values, median_window // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, median_window)
    filtered = np.median(windows, axis=1).astype(np.float32)
    return cv2.GaussianBlur(
        filtered.reshape(-1, 1), (1, gaussian_window), 0
    ).reshape(-1)


def _mask_envelopes(
    mask: np.ndarray, *, smooth: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract upper/lower/left/right foreground envelopes from a filled mask."""
    foreground = mask > 0
    valid_x = np.any(foreground, axis=0)
    valid_y = np.any(foreground, axis=1)
    if not np.any(valid_x):
        raise DewarpError("cannot extract boundaries from an empty mask")
    if np.count_nonzero(valid_x) < 8 or np.count_nonzero(valid_y) < 8:
        raise DewarpError("document mask is too small for boundary estimation")
    x_axis = np.flatnonzero(valid_x).astype(np.int32)
    y_axis = np.flatnonzero(valid_y).astype(np.int32)
    top_values = np.argmax(foreground, axis=0)[valid_x].astype(np.float32)
    bottom_values = (
        mask.shape[0] - 1 - np.argmax(foreground[::-1], axis=0)[valid_x]
    ).astype(np.float32)
    left_values = np.argmax(foreground, axis=1)[valid_y].astype(np.float32)
    right_values = (
        mask.shape[1] - 1 - np.argmax(foreground[:, ::-1], axis=1)[valid_y]
    ).astype(np.float32)
    if smooth:
        top_values = _smooth_envelope(top_values)
        bottom_values = _smooth_envelope(bottom_values)
        left_values = _smooth_envelope(left_values)
        right_values = _smooth_envelope(right_values)

    top = np.column_stack((x_axis, top_values)).astype(np.float32)
    bottom = np.column_stack((x_axis, bottom_values)).astype(np.float32)
    left = np.column_stack((left_values, y_axis)).astype(np.float32)
    right = np.column_stack((right_values, y_axis)).astype(np.float32)
    return top, bottom, left, right


def _envelope_junction(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    *,
    horizontal_side: str,
    vertical_side: str,
) -> tuple[np.ndarray, float]:
    """Find a shared corner where one horizontal and one vertical envelope meet."""
    # Keep the search out of the central gutter: its upper/lower cusp is also an
    # envelope intersection, but it is not an outer book corner.
    corner_fraction = 0.45
    x_cut = float(
        horizontal[:, 0].min() + corner_fraction * np.ptp(horizontal[:, 0])
    )
    y_cut = float(vertical[:, 1].min() + corner_fraction * np.ptp(vertical[:, 1]))
    if horizontal_side == "left":
        candidates = horizontal[horizontal[:, 0] <= x_cut]
    else:
        x_cut = float(
            horizontal[:, 0].max() - corner_fraction * np.ptp(horizontal[:, 0])
        )
        candidates = horizontal[horizontal[:, 0] >= x_cut]
    if vertical_side == "top":
        candidates = candidates[candidates[:, 1] <= y_cut]
    else:
        y_cut = float(
            vertical[:, 1].max() - corner_fraction * np.ptp(vertical[:, 1])
        )
        candidates = candidates[candidates[:, 1] >= y_cut]
    if len(candidates) == 0:
        raise DewarpError("cannot locate a corner in the expected document quadrant")

    vertical_x = np.interp(
        candidates[:, 1], vertical[:, 1], vertical[:, 0], left=np.nan, right=np.nan
    )
    residuals = np.abs(candidates[:, 0] - vertical_x)
    valid = np.isfinite(residuals)
    if not np.any(valid):
        raise DewarpError("adjacent document envelopes do not overlap")
    candidates = candidates[valid]
    vertical_x = vertical_x[valid]
    residuals = residuals[valid]
    # There can be another exact intersection at the central gutter.  Retain all
    # geometrically plausible intersections, then choose the one furthest into
    # the requested outer corner instead of taking the first array occurrence.
    tolerance = max(
        2.0,
        0.008
        * min(float(np.ptp(horizontal[:, 0])), float(np.ptp(vertical[:, 1]))),
    )
    plausible = residuals <= float(residuals.min()) + tolerance
    plausible_indices = np.flatnonzero(plausible)
    x_span = max(1.0, float(np.ptp(horizontal[:, 0])))
    y_span = max(1.0, float(np.ptp(vertical[:, 1])))
    x_normalized = (candidates[plausible, 0] - horizontal[:, 0].min()) / x_span
    y_normalized = (candidates[plausible, 1] - vertical[:, 1].min()) / y_span
    x_outward = 1.0 - x_normalized if horizontal_side == "left" else x_normalized
    y_outward = 1.0 - y_normalized if vertical_side == "top" else y_normalized
    best = int(plausible_indices[int(np.argmax(x_outward + y_outward))])
    corner = np.array(
        [(candidates[best, 0] + vertical_x[best]) * 0.5, candidates[best, 1]],
        dtype=np.float32,
    )
    return corner, float(residuals[best])


def _snap_to_contour(point: np.ndarray, contour: np.ndarray) -> np.ndarray:
    contour_points = contour.reshape(-1, 2).astype(np.float32)
    index = int(np.argmin(np.sum((contour_points - point) ** 2, axis=1)))
    return contour_points[index]


def _trim_envelope(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    horizontal: bool,
) -> np.ndarray:
    axis = 0 if horizontal else 1
    low, high = sorted((float(start[axis]), float(end[axis])))
    selected = points[(points[:, axis] >= low) & (points[:, axis] <= high)]
    if len(selected) < 4:
        raise DewarpError("a fitted document boundary is too short")
    selected = selected[np.argsort(selected[:, axis], kind="stable")].copy()
    selected[0] = start
    selected[-1] = end
    return selected.astype(np.float32)


def _validate_boundary_edges(
    mask: np.ndarray,
    edges: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    corners: np.ndarray,
    junction_residuals: np.ndarray,
) -> None:
    top, bottom, left, right = edges
    height, width = mask.shape
    scale = float(min(height, width))
    if not np.isfinite(corners).all() or not np.isfinite(junction_residuals).all():
        raise DewarpError("boundary estimation produced non-finite coordinates")
    if float(junction_residuals.max()) > max(4.0, 0.035 * scale):
        raise DewarpError("adjacent boundary curves do not meet reliably")

    tl, tr, br, bl = corners
    foreground = mask > 0
    foreground_x = np.flatnonzero(np.any(foreground, axis=0))
    foreground_y = np.flatnonzero(np.any(foreground, axis=1))
    document_width = float(foreground_x[-1] - foreground_x[0] + 1)
    document_height = float(foreground_y[-1] - foreground_y[0] + 1)
    # Strong perspective can make the far edge much shorter than the near edge.
    # Forty-five percent still rejects a gutter mistaken for both outer corners.
    if tr[0] - tl[0] < max(8.0, 0.45 * document_width):
        raise DewarpError("estimated upper corners are too close")
    if br[0] - bl[0] < max(8.0, 0.45 * document_width):
        raise DewarpError("estimated lower corners are too close")
    if bl[1] - tl[1] < max(8.0, 0.55 * document_height):
        raise DewarpError("estimated left corners are too close")
    if br[1] - tr[1] < max(8.0, 0.55 * document_height):
        raise DewarpError("estimated right corners are too close")
    polygon_area = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2))))
    foreground_area = float(np.count_nonzero(mask))
    if polygon_area < 0.35 * foreground_area:
        raise DewarpError("estimated corners cover too little of the document")

    common_x_min = max(float(top[:, 0].min()), float(bottom[:, 0].min()))
    common_x_max = min(float(top[:, 0].max()), float(bottom[:, 0].max()))
    check_x = np.linspace(common_x_min, common_x_max, 128)
    top_y = np.interp(check_x, top[:, 0], top[:, 1])
    bottom_y = np.interp(check_x, bottom[:, 0], bottom[:, 1])
    if np.mean(bottom_y - top_y > 1.0) < 0.98:
        raise DewarpError("upper and lower document boundaries cross")

    common_y_min = max(float(left[:, 1].min()), float(right[:, 1].min()))
    common_y_max = min(float(left[:, 1].max()), float(right[:, 1].max()))
    check_y = np.linspace(common_y_min, common_y_max, 128)
    left_x = np.interp(check_y, left[:, 1], left[:, 0])
    right_x = np.interp(check_y, right[:, 1], right[:, 0])
    if np.mean(right_x - left_x > 1.0) < 0.98:
        raise DewarpError("left and right document boundaries cross")


def _boundary_edges(
    mask: np.ndarray, contour: np.ndarray, *, smooth: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    top, bottom, left, right = _mask_envelopes(mask, smooth=smooth)
    tl, tl_error = _envelope_junction(
        top, left, horizontal_side="left", vertical_side="top"
    )
    tr, tr_error = _envelope_junction(
        top, right, horizontal_side="right", vertical_side="top"
    )
    bl, bl_error = _envelope_junction(
        bottom, left, horizontal_side="left", vertical_side="bottom"
    )
    br, br_error = _envelope_junction(
        bottom, right, horizontal_side="right", vertical_side="bottom"
    )
    tl = _snap_to_contour(tl, contour)
    tr = _snap_to_contour(tr, contour)
    br = _snap_to_contour(br, contour)
    bl = _snap_to_contour(bl, contour)

    top = _trim_envelope(top, tl, tr, horizontal=True)
    bottom = _trim_envelope(bottom, bl, br, horizontal=True)
    left = _trim_envelope(left, tl, bl, horizontal=False)
    right = _trim_envelope(right, tr, br, horizontal=False)
    # Coons interpolation assumes that adjacent curves have identical corners.
    top[0] = left[0] = tl
    top[-1] = right[0] = tr
    bottom[-1] = right[-1] = br
    bottom[0] = left[-1] = bl
    edges = top, bottom, left, right
    corners = np.stack((tl, tr, br, bl)).astype(np.float32)
    residuals = np.array((tl_error, tr_error, br_error, bl_error), np.float32)
    _validate_boundary_edges(mask, edges, corners, residuals)
    return edges


def _edge_map(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    edge_map = np.zeros(shape, dtype=np.uint8)
    rounded = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(edge_map, [rounded], False, 255, thickness=1)
    return edge_map


def _edge_maps(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = min(mask.shape)
    kernel_size = _odd_window(scale, 0.003, 9)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise DewarpError("cannot extract document boundaries from mask")
    contour = max(contours, key=cv2.contourArea)
    component = np.zeros_like(cleaned)
    cv2.drawContours(component, [contour], -1, 255, cv2.FILLED)

    try:
        edges = _boundary_edges(component, contour, smooth=True)
    except DewarpError:
        # A raw-envelope retry is safer than inventing corners from a bounding box.
        edges = _boundary_edges(component, contour, smooth=False)
    return tuple(_edge_map(mask.shape, edge) for edge in edges)


def _continuous_edge(
    axis_coordinates: np.ndarray,
    value_coordinates: np.ndarray,
    *,
    value_on_x: bool,
) -> np.ndarray:
    if len(axis_coordinates) == 0:
        raise DewarpError("a document boundary is empty")
    order = np.argsort(axis_coordinates, kind="stable")
    axis_coordinates = axis_coordinates[order]
    value_coordinates = value_coordinates[order]
    unique_axis, starts, counts = np.unique(
        axis_coordinates, return_index=True, return_counts=True
    )
    unique_values = np.array(
        [
            np.median(value_coordinates[start : start + count])
            for start, count in zip(starts, counts)
        ],
        dtype=np.float32,
    )
    full_axis = np.arange(unique_axis.min(), unique_axis.max() + 1)
    values = np.interp(full_axis, unique_axis, unique_values)
    if value_on_x:
        points = np.column_stack((values, full_axis))
    else:
        points = np.column_stack((full_axis, values))
    return points.astype(np.float32)


def _sample_edge(points: np.ndarray, count: int) -> np.ndarray:
    indices = np.linspace(0, len(points) - 1, count).astype(np.int32)
    return points[indices]


def _coons_grid(
    top: np.ndarray,
    bottom: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    rows = len(left)
    columns = len(top)
    grid = np.empty((rows, columns, 2), dtype=np.float32)
    for row in range(rows):
        v = row / float(rows - 1)
        for column in range(columns):
            u = column / float(columns - 1)
            grid[row, column] = (
                (1.0 - u) * left[row]
                + u * right[row]
                + (1.0 - v) * top[column]
                + v * bottom[column]
                - (
                    (1.0 - u) * (1.0 - v) * top[0]
                    + u * (1.0 - v) * top[-1]
                    + (1.0 - u) * v * bottom[0]
                    + u * v * bottom[-1]
                )
            )
    return grid


def _harmonic_grid(
    top: np.ndarray,
    bottom: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    initial: np.ndarray,
    *,
    iterations: int = 1200,
) -> np.ndarray:
    """Interpolate a bounded interior while preserving all four fitted curves."""
    grid = initial.astype(np.float32, copy=True)
    grid[0, :] = top
    grid[-1, :] = bottom
    grid[:, 0] = left
    grid[:, -1] = right
    for _ in range(iterations):
        previous = grid.copy()
        grid[1:-1, 1:-1] = 0.25 * (
            previous[:-2, 1:-1]
            + previous[2:, 1:-1]
            + previous[1:-1, :-2]
            + previous[1:-1, 2:]
        )
        if float(np.max(np.abs(grid - previous))) < 1e-3:
            break
    return grid


def _validate_source_grid(source_grid: np.ndarray, image_shape: tuple[int, int]) -> None:
    """Reject non-finite, out-of-frame, or substantially folded sampling meshes."""
    if not np.isfinite(source_grid).all():
        raise DewarpError("dewarp grid contains non-finite coordinates")
    height, width = image_shape
    # A Coons patch can overshoot a strongly curved boundary slightly even when
    # every boundary point is valid.  Keep that small interpolation margin, but
    # reject excursions large enough to sample unrelated reflected content.
    tolerance = max(2.0, 0.025 * min(height, width))
    if (
        source_grid[:, :, 0].min() < -tolerance
        or source_grid[:, :, 0].max() > width - 1 + tolerance
        or source_grid[:, :, 1].min() < -tolerance
        or source_grid[:, :, 1].max() > height - 1 + tolerance
    ):
        raise DewarpError("dewarp grid extends outside the cropped image")

    horizontal = source_grid[:-1, 1:] - source_grid[:-1, :-1]
    vertical = source_grid[1:, :-1] - source_grid[:-1, :-1]
    signed_area = (
        horizontal[:, :, 0] * vertical[:, :, 1]
        - horizontal[:, :, 1] * vertical[:, :, 0]
    )
    if float(np.mean(signed_area > 1e-4)) < 0.995:
        raise DewarpError("dewarp grid folds over itself")


def _valid_output_bounds(valid_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Find an inner rectangle whose rows and columns stay inside the mask."""
    valid = valid_mask > 0
    if not np.any(valid):
        raise DewarpError("dewarped mask contains no valid document pixels")
    row_coverage = np.mean(valid, axis=1)
    usable_rows = np.flatnonzero(row_coverage >= 0.85)
    if len(usable_rows) == 0:
        usable_rows = np.flatnonzero(np.any(valid, axis=1))
    y1 = int(usable_rows[0])
    y2 = int(usable_rows[-1]) + 1

    x1, x2 = 0, valid.shape[1]
    # The warped silhouette can be jagged by a few pixels.  Intersect the valid
    # interval of every retained row/column so those protrusions are cropped,
    # instead of appearing as barcode-like strips around the final rectangle.
    for _ in range(2):
        row_slice = valid[y1:y2, x1:x2]
        if not np.all(np.any(row_slice, axis=1)):
            raise DewarpError("dewarped mask contains an invalid interior row")
        row_starts = np.argmax(row_slice, axis=1)
        row_ends = row_slice.shape[1] - np.argmax(row_slice[:, ::-1], axis=1)
        x1 += int(row_starts.max())
        x2 = x1 + int((row_ends - row_starts.max()).min())

        column_slice = valid[y1:y2, x1:x2]
        if not np.all(np.any(column_slice, axis=0)):
            raise DewarpError("dewarped mask contains an invalid interior column")
        column_starts = np.argmax(column_slice, axis=0)
        column_ends = column_slice.shape[0] - np.argmax(
            column_slice[::-1], axis=0
        )
        y1 += int(column_starts.max())
        y2 = y1 + int((column_ends - column_starts.max()).min())
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise DewarpError("valid dewarped document region is too small")
    return x1, y1, x2, y2


def _dewarp_from_edges(
    image: np.ndarray,
    mask: np.ndarray,
    top_map: np.ndarray,
    bottom_map: np.ndarray,
    left_map: np.ndarray,
    right_map: np.ndarray,
    grid_columns: int,
    grid_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    top_y, top_x = np.where(top_map.T == 255)
    bottom_y, bottom_x = np.where(bottom_map.T == 255)
    left_y, left_x = np.where(left_map == 255)
    right_y, right_x = np.where(right_map == 255)
    top = _continuous_edge(top_y, top_x, value_on_x=False)
    bottom = _continuous_edge(bottom_y, bottom_x, value_on_x=False)
    left = _continuous_edge(left_y, left_x, value_on_x=True)
    right = _continuous_edge(right_y, right_x, value_on_x=True)

    top_sample = _sample_edge(top, grid_columns)
    bottom_sample = _sample_edge(bottom, grid_columns)
    left_sample = _sample_edge(left, grid_rows)
    right_sample = _sample_edge(right, grid_rows)
    source_grid = _coons_grid(top_sample, bottom_sample, left_sample, right_sample)

    height, width = image.shape[:2]
    try:
        _validate_source_grid(source_grid, (height, width))
    except DewarpError:
        source_grid = _harmonic_grid(
            top_sample,
            bottom_sample,
            left_sample,
            right_sample,
            source_grid,
        )
        _validate_source_grid(source_grid, (height, width))
    map_x = cv2.resize(
        source_grid[:, :, 0], (width, height), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    map_y = cv2.resize(
        source_grid[:, :, 1], (width, height), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

    # Boundary curves define the geometry, but the mask must also define which
    # source pixels are allowed to enter the result.  Clearing the source first
    # prevents in-frame background from leaking through a curved Coons interior.
    masked_image = image.copy()
    masked_image[mask == 0] = 255
    border_value: int | tuple[int, ...]
    if image.ndim == 2:
        border_value = 255
    else:
        border_value = tuple(255 for _ in range(image.shape[2]))
    output = cv2.remap(
        masked_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    output_mask = cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    x1, y1, x2, y2 = _valid_output_bounds(output_mask)
    output = output[y1:y2, x1:x2]
    output_mask = output_mask[y1:y2, x1:x2]
    map_x = map_x[y1:y2, x1:x2]
    map_y = map_y[y1:y2, x1:x2]
    output[output_mask < 128] = 255
    output = cv2.GaussianBlur(output, (3, 3), 0.5)
    output[output_mask < 128] = 255

    # Cropping changes the output-domain origin.  Re-sample the dense cropped
    # maps so the exported grid2d describes the exact image returned here.
    source_grid = np.stack(
        (
            cv2.resize(map_x, (grid_columns, grid_rows), interpolation=cv2.INTER_AREA),
            cv2.resize(map_y, (grid_columns, grid_rows), interpolation=cv2.INTER_AREA),
        ),
        axis=-1,
    ).astype(np.float32)

    flat_width = 0.5 * (
        np.linalg.norm(np.diff(top_sample, axis=0), axis=1).sum()
        + np.linalg.norm(np.diff(bottom_sample, axis=0), axis=1).sum()
    )
    flat_height = 0.5 * (
        np.linalg.norm(np.diff(left_sample, axis=0), axis=1).sum()
        + np.linalg.norm(np.diff(right_sample, axis=0), axis=1).sum()
    )
    if flat_width <= 0 or flat_height <= 0:
        raise DewarpError("cannot estimate the flattened document aspect ratio")
    aspect = float(flat_width / flat_height)
    height, width = output.shape[:2]
    original_aspect = width / float(height)
    if aspect > original_aspect:
        output_width = width
        output_height = max(8, int(width / aspect))
    else:
        output_height = height
        output_width = max(8, int(height * aspect))
    output = cv2.resize(
        output, (output_width, output_height), interpolation=cv2.INTER_LINEAR
    )
    resized_mask = cv2.resize(
        output_mask, (output_width, output_height), interpolation=cv2.INTER_LINEAR
    )
    output[resized_mask < 128] = 255
    return output, source_grid


def _grid_in_original_coordinates(
    source_grid: np.ndarray,
    rotation_matrix: np.ndarray,
    crop_origin: tuple[int, int],
) -> np.ndarray:
    """Map crop-local remap coordinates back to the untouched input image."""
    grid = source_grid.astype(np.float32, copy=True)
    grid[:, :, 0] += float(crop_origin[0])
    grid[:, :, 1] += float(crop_origin[1])
    inverse = cv2.invertAffineTransform(rotation_matrix)
    points = grid.reshape(-1, 1, 2)
    return cv2.transform(points, inverse).reshape(grid.shape).astype(np.float32)


def _build_uvdoc_grid2d(
    original_grid: np.ndarray,
    output_size: tuple[int, int],
    upsample: float,
) -> np.ndarray:
    """Build UVDoc's (control_height, control_width, xy) reverse sampling grid."""
    output_width, output_height = output_size
    control_width = max(2, int(round(output_width / upsample)) + 1)
    control_height = max(2, int(round(output_height / upsample)) + 1)
    grid_x = cv2.resize(
        original_grid[:, :, 0],
        (control_width, control_height),
        interpolation=cv2.INTER_CUBIC,
    )
    grid_y = cv2.resize(
        original_grid[:, :, 1],
        (control_width, control_height),
        interpolation=cv2.INTER_CUBIC,
    )
    return np.stack((grid_x, grid_y), axis=-1).astype(np.float64)


def dewarp_document(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    grid_columns: int = 80,
    grid_rows: int = 60,
    grid_upsample: float = 14.0,
) -> DewarpResult:
    """Flatten one document using the four outer boundaries of its mask."""
    if image is None or image.ndim not in (2, 3):
        raise DewarpError("image is empty or has an unsupported shape")
    if grid_columns < 4 or grid_rows < 4:
        raise DewarpError("grid dimensions must both be at least 4")
    if not np.isfinite(grid_upsample) or grid_upsample <= 0:
        raise DewarpError("grid upsample must be a positive finite number")

    normalized_mask = normalize_mask(mask, image.shape[:2])
    rotated_image, rotated_mask, angle, rotation_matrix, crop_origin = (
        _rotate_and_crop(image, normalized_mask)
    )
    edges = _edge_maps(rotated_mask)
    output, source_grid = _dewarp_from_edges(
        rotated_image,
        rotated_mask,
        *edges,
        grid_columns=grid_columns,
        grid_rows=grid_rows,
    )
    output_size = (output.shape[1], output.shape[0])
    original_grid = _grid_in_original_coordinates(
        source_grid, rotation_matrix, crop_origin
    )
    grid2d = _build_uvdoc_grid2d(original_grid, output_size, grid_upsample)
    return DewarpResult(output, angle, output_size, grid2d)
