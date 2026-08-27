from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from docflow.vendor.table_rec.mixed_rules import Rect


@dataclass
class Axis:
    rows: list[Rect] = field(default_factory=list)
    cols: list[Rect] = field(default_factory=list)


@dataclass
class BandCandidate:
    bbox: Rect
    objects: list[object]
    side: str
    insert_before_row: int | None = None


@dataclass
class TopBandBuildResult:
    cells_added: int = 0
    text_splits: int = 0
    image_assignments: int = 0
    inferred_columns: int = 0


@dataclass
class EvidenceRowCandidate:
    objects: list[object]
    bbox: Rect
    covered_cols: set[int]
    score: float


class StructurePostProcessor:
    """Canonicalize table structure using model, layout, OCR, and geometry evidence.

    This layer intentionally avoids sample-specific semantics. It treats missing
    header bands, ghost rows/columns, and spans as generic structural evidence
    problems after the table-recognition model has produced an initial grid.
    """

    def process(
        self,
        cells: list[object],
        layout_objects: list[object],
        table_bbox: Rect,
        image_size: dict,
        table_meta: dict | None = None,
    ) -> dict:
        if not cells:
            return {"structure_postprocess": "skipped_empty"}
        table_meta = table_meta or {}
        ocr_objects = [obj for obj in table_meta.get("_ocr_objects", []) if getattr(obj, "text", "")]
        diagnostics: dict = {
            "ghost_columns_removed": 0,
            "ghost_rows_removed": 0,
            "top_rows_added": 0,
            "cell_text_reassigned": 0,
        }

        self._remove_ghost_columns(cells, diagnostics)
        self._remove_ghost_rows(cells, diagnostics)
        self._repair_roles(cells)
        diagnostics["wide_image_header_splits"] = self._split_wide_multi_image_header_cells(
            cells,
            layout_objects,
            table_bbox,
        )

        top_band = self._detect_missing_top_band(cells, layout_objects, ocr_objects, table_bbox)
        if top_band is not None:
            result = MissingTopBandStructureBuilder(self).build(cells, top_band, table_bbox)
            diagnostics["top_rows_added"] += int(result.cells_added > 0)
            diagnostics["top_band_cells_added"] = result.cells_added
            diagnostics["top_band_text_splits"] = result.text_splits
            diagnostics["top_band_image_assignments"] = result.image_assignments
            diagnostics["top_band_inferred_columns"] = result.inferred_columns

        internal_results = self._insert_internal_missing_bands(cells, layout_objects, ocr_objects, table_bbox)
        diagnostics["internal_bands_added"] = len(internal_results)
        diagnostics["internal_band_cells_added"] = sum(result.cells_added for result in internal_results)

        reassigned = self._reassign_text_from_ocr(cells, ocr_objects)
        diagnostics["cell_text_reassigned"] = reassigned
        diagnostics["row_header_splits"] = self._split_row_header_value_cells(cells, ocr_objects)
        diagnostics["wide_ocr_row_splits"] = self._split_wide_ocr_into_row_cells(cells, ocr_objects)
        self._fill_grid_holes(cells)
        self._repair_roles(cells)
        diagnostics["row_count"], diagnostics["col_count"] = self._shape(cells)
        return diagnostics

    def _split_wide_multi_image_header_cells(
        self,
        cells: list[object],
        layout_objects: list[object],
        table_bbox: Rect,
    ) -> int:
        """按正文列边界拆开被模型合并的多图片表头。"""
        if not cells:
            return 0
        row_count, col_count = self._shape(cells)
        if row_count < 2 or col_count < 3:
            return 0

        min_row = min(getattr(cell, "row") for cell in cells)
        split_count = 0
        for candidate in list(cells):
            images = [
                obj
                for obj in getattr(candidate, "layout_objects", [])
                if getattr(obj, "label", "") == "image"
            ]
            if (
                getattr(candidate, "row") > min_row + 1
                or len(images) < 2
                or getattr(candidate, "bbox").w < table_bbox.w * 0.70
            ):
                continue

            start_row = int(getattr(candidate, "row"))
            row_span = max(1, int(getattr(candidate, "rowspan", 1)))
            end_row = min(start_row + row_span, row_count)
            col_bands = self._stable_column_bands(cells, end_row, col_count, table_bbox)
            if len(col_bands) < 3:
                continue
            image_columns = {
                min(col_bands, key=lambda col: abs(col_bands[col].cx - getattr(image, "bbox").cx))
                for image in images
            }
            if len(image_columns) < 2:
                continue

            row_ranges = self._replacement_row_ranges(cells, candidate, start_row, end_row)
            if len(row_ranges) != end_row - start_row:
                continue

            cells[:] = [
                cell
                for cell in cells
                if not (start_row <= int(getattr(cell, "row")) < end_row)
            ]
            replacements: list[object] = []
            for row, (y0, y1) in zip(range(start_row, end_row), row_ranges):
                for col, col_rect in sorted(col_bands.items()):
                    replacements.append(
                        self._new_cell(
                            row=row,
                            col=col,
                            bbox=Rect(col_rect.x0, y0, col_rect.x1, y1),
                            text="",
                            role=(
                                "corner"
                                if row == min_row and col == min(col_bands)
                                else "column_header"
                                if row == min_row
                                else "row_header"
                                if col == min(col_bands)
                                else "body"
                            ),
                            confidence=min(0.75, float(getattr(candidate, "confidence", 1.0))),
                        )
                    )

            for obj in layout_objects:
                bbox = getattr(obj, "bbox")
                if getattr(obj, "label", "") != "image" or not getattr(candidate, "bbox").contains_point(
                    bbox.cx, bbox.cy, tol=4
                ):
                    continue
                owner = min(
                    replacements,
                    key=lambda cell: abs(getattr(cell, "bbox").cx - bbox.cx)
                    + abs(getattr(cell, "bbox").cy - bbox.cy),
                )
                owner.layout_objects.append(obj)

            cells.extend(replacements)
            split_count += 1
        return split_count

    def _stable_column_bands(
        self,
        cells: list[object],
        first_body_row: int,
        col_count: int,
        table_bbox: Rect,
    ) -> dict[int, Rect]:
        bands: dict[int, Rect] = {}
        for col in range(col_count):
            boxes = [
                getattr(cell, "bbox")
                for cell in cells
                if int(getattr(cell, "row")) >= first_body_row
                and int(getattr(cell, "col")) == col
                and max(1, int(getattr(cell, "colspan", 1))) == 1
                and 1 < getattr(cell, "bbox").w < table_bbox.w * 0.60
            ]
            if not boxes:
                continue
            bands[col] = Rect(
                float(np.median([box.x0 for box in boxes])),
                table_bbox.y0,
                float(np.median([box.x1 for box in boxes])),
                table_bbox.y1,
            )
        return bands

    def _replacement_row_ranges(
        self,
        cells: list[object],
        candidate: object,
        start_row: int,
        end_row: int,
    ) -> list[tuple[float, float]]:
        bbox = getattr(candidate, "bbox")
        boundaries = [bbox.y0]
        for row in range(start_row + 1, end_row):
            starts = [
                getattr(cell, "bbox").y0
                for cell in cells
                if int(getattr(cell, "row")) == row
                and cell is not candidate
                and getattr(cell, "bbox").area > 0
            ]
            if not starts:
                return []
            boundaries.append(float(np.median(starts)))
        boundaries.append(bbox.y1)
        if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
            return []
        return list(zip(boundaries, boundaries[1:]))

    def _insert_internal_missing_bands(
        self,
        cells: list[object],
        layout_objects: list[object],
        ocr_objects: list[object],
        table_bbox: Rect,
    ) -> list[TopBandBuildResult]:
        results: list[TopBandBuildResult] = []
        while True:
            band = self._detect_next_internal_missing_band(cells, layout_objects, ocr_objects, table_bbox)
            if band is None:
                break
            results.append(MissingTopBandStructureBuilder(self).build(cells, band, table_bbox))
            if len(results) >= 4:
                break
        return results

    def _detect_next_internal_missing_band(
        self,
        cells: list[object],
        layout_objects: list[object],
        ocr_objects: list[object],
        table_bbox: Rect,
    ) -> BandCandidate | None:
        row_bands = self._row_bands(cells)
        if len(row_bands) < 3:
            return None
        heights = [rect.h for rect in row_bands.values() if rect.h > 1]
        median_h = self._median(heights, default=48.0)
        rows = sorted(row_bands)
        for prev_row, next_row in zip(rows, rows[1:]):
            prev_rect = row_bands[prev_row]
            next_rect = row_bands[next_row]
            gap = next_rect.y0 - prev_rect.y1
            if gap < max(45.0, median_h * 1.45):
                continue
            band_rect = Rect(table_bbox.x0, prev_rect.y1, table_bbox.x1, next_rect.y0)
            objects = [
                obj
                for obj in [*layout_objects, *ocr_objects]
                if self._is_table_content_object(obj)
                and band_rect.overlap_area(getattr(obj, "bbox")) / max(1.0, getattr(obj, "bbox").area) > 0.35
            ]
            if len(objects) < 3:
                continue
            candidate_groups = self._internal_row_candidate_groups(objects, self._col_bands(cells), table_bbox)
            if candidate_groups:
                group = min(candidate_groups, key=lambda values: min(getattr(obj, "bbox").y0 for obj in values))
                return BandCandidate(
                    bbox=self._union([getattr(obj, "bbox") for obj in group]),
                    objects=group,
                    side="internal",
                    insert_before_row=next_row,
                )
        return None

    def _internal_row_candidate_groups(
        self,
        objects: list[object],
        col_bands: dict[int, Rect],
        table_bbox: Rect,
    ) -> list[list[object]]:
        """Find missing logical rows from evidence distribution, not field names.

        A candidate row is accepted when objects in the gap align with existing
        table columns and provide enough cross-column support. Adjacent y-clusters
        can be merged when they describe the same row, such as image objects above
        their OCR captions or two OCR lines aligned to the same column set.
        """

        if len(col_bands) < 2:
            return []
        clusters = [group for group in self._cluster_objects_by_y(objects) if self._row_evidence_candidate(group, col_bands, table_bbox)]
        if not clusters:
            return []

        candidates: list[EvidenceRowCandidate] = []
        idx = 0
        while idx < len(clusters):
            current = list(clusters[idx])
            idx += 1
            while idx < len(clusters) and self._should_merge_evidence_rows(current, clusters[idx], col_bands, table_bbox):
                current.extend(clusters[idx])
                idx += 1
            candidate = self._score_evidence_row(current, col_bands, table_bbox)
            if candidate.score >= 2.35:
                candidates.append(candidate)

        if not candidates:
            return []
        return [candidate.objects for candidate in sorted(candidates, key=lambda item: item.bbox.y0)]

    def _row_evidence_candidate(
        self,
        objects: list[object],
        col_bands: dict[int, Rect],
        table_bbox: Rect,
    ) -> bool:
        return self._score_evidence_row(objects, col_bands, table_bbox).score >= 1.65

    def _score_evidence_row(
        self,
        objects: list[object],
        col_bands: dict[int, Rect],
        table_bbox: Rect,
    ) -> EvidenceRowCandidate:
        objects = [obj for obj in objects if self._is_table_content_object(obj)]
        if not objects:
            return EvidenceRowCandidate([], Rect(0, 0, 0, 0), set(), 0.0)
        bbox = self._union([getattr(obj, "bbox") for obj in objects])
        covered_cols = self._covered_columns(objects, col_bands)
        min_col = min(col_bands)
        data_cols = {col for col in covered_cols if col != min_col}
        images = [obj for obj in objects if getattr(obj, "label", "") == "image"]
        text_objs = [obj for obj in objects if str(getattr(obj, "text", "")).strip()]
        structured_text = [
            obj
            for obj in text_objs
            if self._looks_like_compact_structured_value(str(getattr(obj, "text", "")))
        ]
        left_support = sum(1 for obj in text_objs if getattr(obj, "bbox").cx <= col_bands[min_col].x1 + 10)
        x_span = bbox.w / max(1.0, table_bbox.w)
        y_compactness = 1.0 - min(1.0, bbox.h / max(1.0, table_bbox.h * 0.18))

        score = 0.0
        score += min(2.0, len(data_cols) * 0.42)
        score += min(1.2, len(images) * 0.22)
        score += min(1.0, len(text_objs) * 0.12)
        score += min(0.8, len(structured_text) * 0.22)
        score += 0.45 if left_support else 0.0
        score += 0.45 if x_span >= 0.35 else 0.0
        score += max(0.0, y_compactness) * 0.35
        if len(data_cols) < 2 and len(images) < 2:
            score *= 0.55
        return EvidenceRowCandidate(objects, bbox, covered_cols, score)

    def _should_merge_evidence_rows(
        self,
        upper: list[object],
        lower: list[object],
        col_bands: dict[int, Rect],
        table_bbox: Rect,
    ) -> bool:
        upper_score = self._score_evidence_row(upper, col_bands, table_bbox)
        lower_score = self._score_evidence_row(lower, col_bands, table_bbox)
        if not upper_score.objects or not lower_score.objects:
            return False
        vertical_gap = lower_score.bbox.y0 - upper_score.bbox.y1
        median_h = self._median(
            [getattr(obj, "bbox").h for obj in [*upper_score.objects, *lower_score.objects]],
            default=18.0,
        )
        if vertical_gap > max(28.0, median_h * 1.25):
            return False
        shared = upper_score.covered_cols & lower_score.covered_cols
        union = upper_score.covered_cols | lower_score.covered_cols
        column_iou = len(shared) / max(1, len(union))
        has_visual_text_pair = (
            any(getattr(obj, "label", "") == "image" for obj in upper_score.objects)
            and any(str(getattr(obj, "text", "")).strip() for obj in lower_score.objects)
            and column_iou >= 0.25
        )
        has_continuation_text = (
            any(str(getattr(obj, "text", "")).strip() for obj in upper_score.objects)
            and any(str(getattr(obj, "text", "")).strip() for obj in lower_score.objects)
            and column_iou >= 0.35
        )
        combined = self._score_evidence_row([*upper_score.objects, *lower_score.objects], col_bands, table_bbox)
        improves_support = combined.score >= max(upper_score.score, lower_score.score) + 0.35
        return has_visual_text_pair or has_continuation_text or improves_support

    def _covered_columns(self, objects: list[object], col_bands: dict[int, Rect]) -> set[int]:
        out: set[int] = set()
        for obj in objects:
            bbox = getattr(obj, "bbox")
            best_col = None
            best_score = 0.0
            for col, col_rect in col_bands.items():
                overlap = self._x_overlap_ratio(col_rect, bbox)
                center_inside = col_rect.x0 - 6 <= bbox.cx <= col_rect.x1 + 6
                score = overlap + (0.55 if center_inside else 0.0)
                if score > best_score:
                    best_col, best_score = col, score
            if best_col is not None and best_score >= 0.20:
                out.add(best_col)
        return out

    def _detect_missing_top_band(
        self,
        cells: list[object],
        layout_objects: list[object],
        ocr_objects: list[object],
        table_bbox: Rect,
    ) -> BandCandidate | None:
        valid_cells = [cell for cell in cells if getattr(cell, "bbox").area > 0]
        if not valid_cells:
            return None
        first_y = min(getattr(cell, "bbox").y0 for cell in valid_cells)
        gap_h = first_y - table_bbox.y0
        median_h = self._median([getattr(cell, "bbox").h for cell in cells], default=40.0)
        if gap_h < max(28.0, median_h * 0.55):
            return None

        band = Rect(table_bbox.x0, table_bbox.y0, table_bbox.x1, first_y)
        objects = [
            obj
            for obj in [*layout_objects, *ocr_objects]
            if self._is_table_content_object(obj)
            and band.overlap_area(getattr(obj, "bbox")) / max(1.0, getattr(obj, "bbox").area) > 0.35
        ]
        if len(objects) < 3:
            return None

        row_groups = self._cluster_objects_by_y(objects)
        best = max(row_groups, key=lambda group: len(group), default=[])
        if len(best) < 3:
            return None

        x_centers = sorted(getattr(obj, "bbox").cx for obj in best)
        if len(x_centers) < 3 or (x_centers[-1] - x_centers[0]) < table_bbox.w * 0.35:
            return None
        return BandCandidate(bbox=self._union([getattr(obj, "bbox") for obj in objects]), objects=objects, side="top")

    def _insert_top_band_row(
        self,
        cells: list[object],
        band: BandCandidate,
        table_bbox: Rect,
    ) -> int:
        col_bands = self._col_bands(cells)
        if len(col_bands) < 2:
            return 0
        min_row = min(getattr(cell, "row") for cell in cells)
        for cell in cells:
            cell.row += 1

        y0 = max(table_bbox.y0, band.bbox.y0 - 2)
        valid_cells = [cell for cell in cells if getattr(cell, "bbox").area > 0]
        y1 = min(min(getattr(cell, "bbox").y0 for cell in valid_cells), band.bbox.y1 + 4)
        if y1 <= y0:
            y0, y1 = band.bbox.y0, band.bbox.y1

        label_objects = [obj for obj in band.objects if getattr(obj, "bbox").cx <= col_bands[0].x1 + 8]
        header_text = self._join_text(label_objects) or self._join_text(
            [obj for obj in band.objects if getattr(obj, "bbox").cx < table_bbox.x0 + table_bbox.w * 0.18]
        )

        cells.append(
            self._new_cell(
                row=min_row,
                col=0,
                bbox=Rect(col_bands[0].x0, y0, col_bands[0].x1, y1),
                text=header_text,
                role="corner",
                confidence=0.55,
            )
        )

        content_objects = [
            obj
            for obj in band.objects
            if not self._is_title_like_band_object(obj, band.bbox)
            and not self._is_wide_text_object(obj, band.bbox)
            and not (getattr(obj, "bbox").cx <= col_bands[0].x1 + 8 and getattr(obj, "label", "") != "image")
        ]
        for col, col_rect in sorted(col_bands.items()):
            if col == 0:
                continue
            objs = [
                obj
                for obj in content_objects
                if self._x_overlap_ratio(col_rect, getattr(obj, "bbox")) > 0.25
                or (col_rect.x0 - 10 <= getattr(obj, "bbox").cx <= col_rect.x1 + 10)
            ]
            if not objs:
                continue
            cell = self._new_cell(
                row=min_row,
                col=col,
                bbox=Rect(col_rect.x0, y0, col_rect.x1, y1),
                text=self._join_text(objs),
                role="column_header",
                confidence=0.55,
            )
            cell.layout_objects = [obj for obj in objs if getattr(obj, "label", "") == "image"]
            cells.append(cell)
        return 1

    def _remove_ghost_columns(self, cells: list[object], diagnostics: dict) -> None:
        col_bands = self._col_bands(cells)
        if len(col_bands) <= 2:
            return
        widths = [rect.w for rect in col_bands.values() if rect.w > 1]
        median_w = self._median(widths, default=0.0)
        if median_w <= 0:
            return
        ghost_cols: list[int] = []
        row_count, _ = self._shape(cells)
        for col, rect in sorted(col_bands.items()):
            direct = [cell for cell in cells if getattr(cell, "col") == col and max(1, getattr(cell, "colspan")) == 1]
            text_count = sum(1 for cell in direct if str(getattr(cell, "text", "")).strip() or getattr(cell, "layout_objects", []))
            direct_widths = [getattr(cell, "bbox").w for cell in direct if getattr(cell, "bbox").area > 0]
            direct_med_w = self._median(direct_widths, default=rect.w)
            coverage = len(direct) / max(1, row_count)
            is_thin = rect.w < median_w * 0.32 or direct_med_w < median_w * 0.32
            is_empty = text_count <= max(1, int(len(direct) * 0.18))
            is_orphan_edge = (
                text_count == 0
                and len(direct) <= max(1, int(row_count * 0.20))
                and (col == min(col_bands) or col == max(col_bands))
            )
            if (is_thin and is_empty and coverage >= 0.25) or is_orphan_edge:
                ghost_cols.append(col)
        if not ghost_cols:
            return
        for col in sorted(ghost_cols, reverse=True):
            self._merge_column_into_neighbor(cells, col)
        diagnostics["ghost_columns_removed"] += len(ghost_cols)
        self._compact_columns(cells)

    def _remove_ghost_rows(self, cells: list[object], diagnostics: dict) -> None:
        row_bands = self._row_bands(cells)
        if len(row_bands) <= 2:
            return
        heights = [rect.h for rect in row_bands.values() if rect.h > 1]
        median_h = self._median(heights, default=0.0)
        if median_h <= 0:
            return
        ghost_rows = []
        _, col_count = self._shape(cells)
        for row, rect in sorted(row_bands.items()):
            direct = [cell for cell in cells if getattr(cell, "row") == row and max(1, getattr(cell, "rowspan")) == 1]
            text_count = sum(1 for cell in direct if str(getattr(cell, "text", "")).strip() or getattr(cell, "layout_objects", []))
            coverage = len(direct) / max(1, col_count)
            if rect.h < median_h * 0.30 and text_count == 0 and coverage >= 0.35:
                ghost_rows.append(row)
        for row in sorted(ghost_rows, reverse=True):
            self._merge_row_into_neighbor(cells, row)
        diagnostics["ghost_rows_removed"] += len(ghost_rows)
        self._compact_rows(cells)

    def _merge_column_into_neighbor(self, cells: list[object], col: int) -> None:
        _, col_count = self._shape(cells)
        target = col + 1 if col + 1 < col_count else col - 1
        if target < 0:
            return
        for cell in list(cells):
            start = getattr(cell, "col")
            span = max(1, getattr(cell, "colspan"))
            end = start + span - 1
            if start == col and span == 1:
                self._move_cell_content(cells, cell, getattr(cell, "row"), target)
                cells.remove(cell)
            elif start <= col <= end:
                cell.colspan = max(1, span - 1)
            elif start > col:
                cell.col -= 1

    def _merge_row_into_neighbor(self, cells: list[object], row: int) -> None:
        row_count, _ = self._shape(cells)
        target = row + 1 if row + 1 < row_count else row - 1
        if target < 0:
            return
        for cell in list(cells):
            start = getattr(cell, "row")
            span = max(1, getattr(cell, "rowspan"))
            end = start + span - 1
            if start == row and span == 1:
                self._move_cell_content(cells, cell, target, getattr(cell, "col"))
                cells.remove(cell)
            elif start <= row <= end:
                cell.rowspan = max(1, span - 1)
            elif start > row:
                cell.row -= 1

    def _move_cell_content(self, cells: list[object], src: object, row: int, col: int) -> None:
        dst = next((cell for cell in cells if getattr(cell, "row") == row and getattr(cell, "col") == col), None)
        if dst is None:
            return
        text = str(getattr(src, "text", "")).strip()
        if text and not str(getattr(dst, "text", "")).strip():
            dst.text = text
        dst.layout_objects.extend(getattr(src, "layout_objects", []))

    def _reassign_text_from_ocr(self, cells: list[object], ocr_objects: list[object]) -> int:
        if not cells or not ocr_objects:
            return 0
        buckets: dict[int, list[object]] = {idx: [] for idx, _ in enumerate(cells)}
        for obj in ocr_objects:
            best_idx = -1
            best_score = 0.0
            bbox = getattr(obj, "bbox")
            for idx, cell in enumerate(cells):
                cb = getattr(cell, "bbox")
                if cb.area <= 0:
                    continue
                if self._is_ocr_too_wide_for_cell(obj, cell):
                    continue
                overlap = cb.overlap_area(bbox) / max(1.0, bbox.area)
                center = cb.contains_point(bbox.cx, bbox.cy, tol=3)
                score = overlap + (0.8 if center else 0.0)
                if score > best_score:
                    best_idx, best_score = idx, score
            if best_idx >= 0 and best_score >= 0.20:
                buckets[best_idx].append(obj)

        changed = 0
        for idx, objs in buckets.items():
            if not objs:
                continue
            text = self._join_text(objs)
            if text and text != str(getattr(cells[idx], "text", "")):
                cells[idx].text = text
                changed += 1
        return changed

    def _split_row_header_value_cells(self, cells: list[object], ocr_objects: list[object]) -> int:
        if not cells or not ocr_objects:
            return 0
        col_bands = self._col_bands(cells)
        if len(col_bands) < 3:
            return 0
        min_col = min(col_bands)
        split_count = 0
        for cell in list(cells):
            if getattr(cell, "col") != min_col or max(1, getattr(cell, "colspan")) < 2:
                continue
            bbox = getattr(cell, "bbox")
            tokens = [
                obj
                for obj in ocr_objects
                if bbox.overlap_area(getattr(obj, "bbox")) / max(1.0, getattr(obj, "bbox").area) > 0.35
                and not self._is_ocr_too_wide_for_cell(obj, cell)
            ]
            if len(tokens) < 2:
                continue
            left_boundary = col_bands[min_col].x1
            left_tokens = [obj for obj in tokens if getattr(obj, "bbox").cx <= left_boundary + 8]
            right_tokens = [obj for obj in tokens if getattr(obj, "bbox").cx > left_boundary + 8]
            if not left_tokens or not right_tokens:
                continue
            old_colspan = max(1, getattr(cell, "colspan"))
            cell.colspan = 1
            cell.bbox = Rect(col_bands[min_col].x0, bbox.y0, col_bands[min_col].x1, bbox.y1)
            cell.text = self._join_text(left_tokens)
            right_col = min_col + 1
            right_span = max(1, old_colspan - 1)
            right_rects = [col_bands[col] for col in range(right_col, right_col + right_span) if col in col_bands]
            right_union = self._union(right_rects) if right_rects else Rect(left_boundary, bbox.y0, bbox.x1, bbox.y1)
            right_bbox = Rect(right_union.x0, bbox.y0, right_union.x1, bbox.y1)
            cells.append(
                self._new_cell(
                    row=getattr(cell, "row"),
                    col=right_col,
                    bbox=right_bbox,
                    text=self._join_text(right_tokens),
                    rowspan=max(1, getattr(cell, "rowspan")),
                    colspan=right_span,
                    source_rowspan=max(1, getattr(cell, "source_rowspan", 1)),
                    source_colspan=right_span,
                    role="body",
                    confidence=min(0.75, getattr(cell, "confidence", 1.0)),
                )
            )
            split_count += 1
        return split_count

    def _split_wide_ocr_into_row_cells(self, cells: list[object], ocr_objects: list[object]) -> int:
        if not cells or not ocr_objects:
            return 0
        row_groups: dict[int, list[object]] = {}
        for cell in cells:
            if getattr(cell, "bbox").area <= 0:
                continue
            row_groups.setdefault(getattr(cell, "row"), []).append(cell)
        splitter = MissingTopBandStructureBuilder(self)
        split_count = 0
        for obj in ocr_objects:
            bbox = getattr(obj, "bbox")
            text = str(getattr(obj, "text", "")).strip()
            if not text:
                continue
            best_row = None
            best_score = 0.0
            for row, row_cells in row_groups.items():
                row_bbox = self._union([getattr(cell, "bbox") for cell in row_cells])
                y_overlap = max(0.0, min(row_bbox.y1, bbox.y1) - max(row_bbox.y0, bbox.y0))
                score = y_overlap / max(1.0, min(row_bbox.h, bbox.h))
                if row_bbox.y0 - 4 <= bbox.cy <= row_bbox.y1 + 4:
                    score += 0.5
                if score > best_score:
                    best_row = row
                    best_score = score
            if best_row is None or best_score < 0.35:
                continue
            row_cells = sorted(row_groups[best_row], key=lambda cell: getattr(cell, "col"))
            covered = [
                idx
                for idx, cell in enumerate(row_cells)
                if self._x_overlap_ratio(getattr(cell, "bbox"), bbox) > 0.18
            ]
            if len(covered) <= 1:
                continue
            assignments = splitter._split_text_by_cell_x(text, bbox, row_cells)
            used = 0
            for idx, value in assignments.items():
                if idx >= len(row_cells):
                    continue
                cell = row_cells[idx]
                if getattr(cell, "col") == min(getattr(c, "col") for c in row_cells):
                    continue
                value = value.strip()
                if not value:
                    continue
                current = str(getattr(cell, "text", "")).strip()
                if value in current:
                    continue
                if self._looks_like_compact_structured_value(value) and self._text_is_compatible(current, value):
                    cell.text = current + "\n" + value if current else value
                    used += 1
                elif not current:
                    cell.text = value
                    used += 1
            if used:
                split_count += 1
        return split_count

    @staticmethod
    def _looks_like_dimension_value(text: str) -> bool:
        value = str(text or "")
        return bool(re.search(r"\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\s*[x×]\s*\d+", value, flags=re.I))

    @staticmethod
    def _looks_like_model_value(text: str) -> bool:
        value = "".join(str(text or "").split())
        return bool(re.search(r"[A-Za-z]{1,}\d+[A-Za-z0-9]*", value))

    @staticmethod
    def _looks_like_compact_structured_value(text: str) -> bool:
        value = "".join(str(text or "").split())
        if not value:
            return False
        patterns = [
            r"\d+(?:\.\d+)?(?:[x×]\d+(?:\.\d+)?){1,3}(?:[a-zA-Z\u4e00-\u9fff]{0,4})?",
            r"[A-Za-z]+[A-Za-z0-9()（）._/-]*\d[A-Za-z0-9()（）._/-]*",
            r"\d+(?:\.\d+)?\s*(?:mm|cm|m|kg|g|w|kw|mah|l|ml|gb|tb|hz|寸|英寸|瓦|升|克|千克)",
        ]
        return any(re.fullmatch(pattern, value, flags=re.I) for pattern in patterns)

    @staticmethod
    def _text_is_compatible(current: str, value: str) -> bool:
        current = str(current or "").strip()
        value = str(value or "").strip()
        if not current or not value:
            return True
        if value in current:
            return False
        compact_current = "".join(current.split())
        compact_value = "".join(value.split())
        if compact_value in compact_current:
            return False
        return True

    def _fill_grid_holes(self, cells: list[object]) -> None:
        row_count, col_count = self._shape(cells)
        occupied = set()
        for cell in cells:
            for r in range(getattr(cell, "row"), min(row_count, getattr(cell, "row") + max(1, getattr(cell, "rowspan")))):
                for c in range(getattr(cell, "col"), min(col_count, getattr(cell, "col") + max(1, getattr(cell, "colspan")))):
                    occupied.add((r, c))
        row_bands = self._row_bands(cells)
        col_bands = self._col_bands(cells)
        for row in range(row_count):
            for col in range(col_count):
                if (row, col) in occupied:
                    continue
                rect = row_bands.get(row, Rect(0, 0, 0, 0)).intersect(col_bands.get(col, Rect(0, 0, 0, 0)))
                cells.append(self._new_cell(row=row, col=col, bbox=rect, confidence=0.1))

    def _repair_roles(self, cells: list[object]) -> None:
        if not cells:
            return
        min_row = min(getattr(cell, "row") for cell in cells)
        min_col = min(getattr(cell, "col") for cell in cells)
        for cell in cells:
            if getattr(cell, "row") == min_row and getattr(cell, "col") == min_col:
                cell.role = "corner"
            elif getattr(cell, "row") == min_row:
                cell.role = "column_header"
            elif getattr(cell, "col") == min_col:
                cell.role = "row_header"
            elif getattr(cell, "role", "") in {"corner", "column_header", "row_header"}:
                cell.role = "body"

    def _compact_columns(self, cells: list[object]) -> None:
        starts = sorted({getattr(cell, "col") for cell in cells})
        mapping = {old: new for new, old in enumerate(starts)}
        for cell in cells:
            old_start = getattr(cell, "col")
            old_end = old_start + max(1, getattr(cell, "colspan")) - 1
            covered = [mapping[col] for col in starts if old_start <= col <= old_end]
            if not covered:
                cell.col = mapping.get(old_start, old_start)
                cell.colspan = 1
            else:
                cell.col = min(covered)
                cell.colspan = max(covered) - min(covered) + 1

    def _compact_rows(self, cells: list[object]) -> None:
        starts = sorted({getattr(cell, "row") for cell in cells})
        mapping = {old: new for new, old in enumerate(starts)}
        for cell in cells:
            old_start = getattr(cell, "row")
            old_end = old_start + max(1, getattr(cell, "rowspan")) - 1
            covered = [mapping[row] for row in starts if old_start <= row <= old_end]
            if not covered:
                cell.row = mapping.get(old_start, old_start)
                cell.rowspan = 1
            else:
                cell.row = min(covered)
                cell.rowspan = max(covered) - min(covered) + 1

    def _row_bands(self, cells: list[object]) -> dict[int, Rect]:
        by_row: dict[int, list[Rect]] = {}
        for cell in cells:
            if getattr(cell, "bbox").area <= 0:
                continue
            by_row.setdefault(getattr(cell, "row"), []).append(getattr(cell, "bbox"))
        return {row: self._union(rects) for row, rects in by_row.items()}

    def _col_bands(self, cells: list[object]) -> dict[int, Rect]:
        by_col: dict[int, list[Rect]] = {}
        for cell in cells:
            if getattr(cell, "bbox").area <= 0:
                continue
            col0 = getattr(cell, "col")
            colspan = max(1, getattr(cell, "colspan"))
            bbox = getattr(cell, "bbox")
            if colspan == 1:
                by_col.setdefault(col0, []).append(bbox)
                continue
            step = bbox.w / colspan if colspan else bbox.w
            for offset in range(colspan):
                by_col.setdefault(col0 + offset, []).append(
                    Rect(bbox.x0 + step * offset, bbox.y0, bbox.x0 + step * (offset + 1), bbox.y1)
                )
        return {col: self._union(rects) for col, rects in by_col.items()}

    def _cluster_objects_by_y(self, objects: list[object]) -> list[list[object]]:
        if not objects:
            return []
        tol = max(10.0, self._median([getattr(obj, "bbox").h for obj in objects], default=16.0) * 0.75)
        groups: list[list[object]] = []
        for obj in sorted(objects, key=lambda item: getattr(item, "bbox").cy):
            if not groups:
                groups.append([obj])
                continue
            center = float(np.mean([getattr(item, "bbox").cy for item in groups[-1]]))
            if abs(getattr(obj, "bbox").cy - center) <= tol:
                groups[-1].append(obj)
            else:
                groups.append([obj])
        return groups

    def _join_text(self, objects: list[object]) -> str:
        text_objects = [obj for obj in objects if str(getattr(obj, "text", "")).strip()]
        if not text_objects:
            return ""
        lines = self._cluster_objects_by_y(text_objects)
        return "\n".join(
            " ".join(str(getattr(obj, "text", "")).strip() for obj in sorted(line, key=lambda item: getattr(item, "bbox").x0))
            for line in lines
        )

    @staticmethod
    def _is_table_content_object(obj: object) -> bool:
        return getattr(obj, "role", "") not in {"title", "noise", "table_region"} and getattr(obj, "bbox").area > 0

    @staticmethod
    def _is_title_like_band_object(obj: object, band: Rect) -> bool:
        bbox = getattr(obj, "bbox")
        role = getattr(obj, "role", "")
        if role == "title":
            return True
        if bbox.w > band.w * 0.45 and bbox.cy < band.y0 + band.h * 0.35:
            return True
        return False

    @staticmethod
    def _is_wide_text_object(obj: object, band: Rect) -> bool:
        if getattr(obj, "label", "") == "image":
            return False
        text = str(getattr(obj, "text", "")).strip()
        bbox = getattr(obj, "bbox")
        if not text:
            return False
        return bbox.w > band.w * 0.42

    @staticmethod
    def _is_ocr_too_wide_for_cell(obj: object, cell: object) -> bool:
        if getattr(obj, "label", "") == "image":
            return False
        bbox = getattr(obj, "bbox")
        cb = getattr(cell, "bbox")
        text = str(getattr(obj, "text", "")).strip()
        if not text or cb.w <= 1:
            return False
        if StructurePostProcessor._looks_like_compact_structured_value(text) and bbox.w <= cb.w * 1.08:
            return False
        if bbox.w > cb.w * 1.18:
            return True
        return bbox.w > cb.w * 0.72 and len(text) > 18 and max(1, getattr(cell, "colspan")) == 1

    @staticmethod
    def _x_overlap_ratio(a: Rect, b: Rect) -> float:
        overlap = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
        return overlap / max(1.0, min(a.w, b.w))

    @staticmethod
    def _shape(cells: list[object]) -> tuple[int, int]:
        rows = 0
        cols = 0
        for cell in cells:
            rows = max(rows, getattr(cell, "row") + max(1, getattr(cell, "rowspan")))
            cols = max(cols, getattr(cell, "col") + max(1, getattr(cell, "colspan")))
        return rows, cols

    @staticmethod
    def _union(rects: Iterable[Rect]) -> Rect:
        values = [rect for rect in rects if rect and rect.area >= 0]
        if not values:
            return Rect(0, 0, 0, 0)
        return Rect(
            min(rect.x0 for rect in values),
            min(rect.y0 for rect in values),
            max(rect.x1 for rect in values),
            max(rect.y1 for rect in values),
        )

    @staticmethod
    def _median(values: Iterable[float], default: float) -> float:
        vals = [float(value) for value in values if float(value) > 1]
        return float(np.median(vals)) if vals else default

    @staticmethod
    def _new_cell(**kwargs):
        # Avoid importing TableCell here; fusion.py imports this module.
        from docflow.vendor.table_rec.fusion import TableCell

        return TableCell(**kwargs)


class MissingTopBandStructureBuilder:
    """Infer a missing top row from body columns and top-band evidence.

    The body grid supplies the structural prior. Top-band images and OCR supply
    local evidence. Wide OCR objects that span multiple inferred columns are not
    assigned wholesale; they are split geometrically as a fallback.
    """

    def __init__(self, ops: StructurePostProcessor):
        self.ops = ops

    def build(
        self,
        cells: list[object],
        band: BandCandidate,
        table_bbox: Rect,
    ) -> TopBandBuildResult:
        col_bands = self.ops._col_bands(cells)
        if len(col_bands) < 2:
            return TopBandBuildResult()

        insert_row = (
            int(band.insert_before_row)
            if band.insert_before_row is not None
            else min(getattr(cell, "row") for cell in cells)
        )
        for cell in cells:
            if getattr(cell, "row") >= insert_row:
                cell.row += 1

        y0, y1 = self._top_row_y_range(cells, band, table_bbox)
        top_cells = self._create_virtual_cells(insert_row, col_bands, y0, y1, band)
        evidence = self._usable_evidence(band)
        regular_text, wide_text, images = self._partition_evidence(evidence, top_cells)

        text_assignments = self._assign_regular_text(top_cells, regular_text)
        image_assignments = self._assign_images(top_cells, images)
        text_splits = self._split_wide_text(top_cells, wide_text)

        cells.extend(top_cells)
        return TopBandBuildResult(
            cells_added=len(top_cells),
            text_splits=text_splits,
            image_assignments=image_assignments,
            inferred_columns=len(top_cells),
        )

    def _top_row_y_range(
        self,
        shifted_cells: list[object],
        band: BandCandidate,
        table_bbox: Rect,
    ) -> tuple[float, float]:
        if band.side == "internal":
            return band.bbox.y0 - 2, band.bbox.y1 + 4
        valid_cells = [cell for cell in shifted_cells if getattr(cell, "bbox").area > 0]
        next_y = min(getattr(cell, "bbox").y0 for cell in valid_cells) if valid_cells else band.bbox.y1
        y0 = max(table_bbox.y0, band.bbox.y0 - 2)
        y1 = min(next_y, band.bbox.y1 + 4)
        if y1 <= y0:
            return band.bbox.y0, band.bbox.y1
        return y0, y1

    def _create_virtual_cells(
        self,
        row: int,
        col_bands: dict[int, Rect],
        y0: float,
        y1: float,
        band: BandCandidate,
    ) -> list[object]:
        cells: list[object] = []
        for col, col_rect in sorted(col_bands.items()):
            role = "corner" if col == min(col_bands) else "column_header"
            cells.append(
                self.ops._new_cell(
                    row=row,
                    col=col,
                    bbox=Rect(col_rect.x0, y0, col_rect.x1, y1),
                    text="",
                    role=role,
                    confidence=0.55,
                )
            )
        return cells

    def _usable_evidence(self, band: BandCandidate) -> list[object]:
        return [
            obj
            for obj in band.objects
            if not self.ops._is_title_like_band_object(obj, band.bbox)
            and not (
                getattr(obj, "label", "") != "image"
                and not str(getattr(obj, "text", "")).strip()
            )
        ]

    def _partition_evidence(
        self,
        objects: list[object],
        top_cells: list[object],
    ) -> tuple[list[object], list[object], list[object]]:
        regular_text: list[object] = []
        wide_text: list[object] = []
        images: list[object] = []
        for obj in objects:
            if getattr(obj, "label", "") == "image":
                images.append(obj)
                continue
            if not str(getattr(obj, "text", "")).strip():
                continue
            covered = self._covered_cells(top_cells, getattr(obj, "bbox"), min_ratio=0.20)
            if len(covered) > 1 or self._is_wide_for_nearest_cell(obj, top_cells):
                wide_text.append(obj)
            else:
                regular_text.append(obj)
        return regular_text, wide_text, images

    def _assign_regular_text(self, top_cells: list[object], text_objects: list[object]) -> int:
        buckets: dict[int, list[object]] = {idx: [] for idx, _ in enumerate(top_cells)}
        for obj in text_objects:
            idx = self._nearest_cell_index(top_cells, getattr(obj, "bbox"))
            if idx is not None:
                buckets[idx].append(obj)

        changed = 0
        for idx, objs in buckets.items():
            if not objs:
                continue
            text = self.ops._join_text(objs)
            if not text:
                continue
            top_cells[idx].text = self._merge_text(top_cells[idx].text, text)
            changed += 1
        return changed

    def _assign_images(self, top_cells: list[object], images: list[object]) -> int:
        count = 0
        for obj in images:
            idx = self._nearest_cell_index(top_cells, getattr(obj, "bbox"))
            if idx is None:
                continue
            top_cells[idx].layout_objects.append(obj)
            count += 1
        return count

    def _split_wide_text(self, top_cells: list[object], text_objects: list[object]) -> int:
        split_count = 0
        for obj in text_objects:
            assignments = self._split_text_by_cell_x(str(getattr(obj, "text", "")), getattr(obj, "bbox"), top_cells)
            used = 0
            for idx, text in assignments.items():
                value = text.strip()
                if not value:
                    continue
                # Wide text is fallback evidence: do not overwrite a cell that
                # already has a local OCR token.
                if str(getattr(top_cells[idx], "text", "")).strip():
                    continue
                top_cells[idx].text = value
                used += 1
            if used:
                split_count += 1
        return split_count

    def _split_text_by_cell_x(
        self,
        text: str,
        bbox: Rect,
        top_cells: list[object],
    ) -> dict[int, str]:
        token_assignments = self._split_repeated_tokens_by_x(text, bbox, top_cells)
        if token_assignments:
            return token_assignments

        chars = [ch for ch in text if not ch.isspace()]
        if not chars or bbox.w <= 1:
            return {}
        assignments: dict[int, list[str]] = {}
        n = len(chars)
        for i, ch in enumerate(chars):
            cx = bbox.x0 + (i + 0.5) / n * bbox.w
            idx = self._cell_index_at_x(top_cells, cx)
            if idx is None:
                continue
            assignments.setdefault(idx, []).append(ch)
        return {idx: "".join(values) for idx, values in assignments.items()}

    def _split_repeated_tokens_by_x(
        self,
        text: str,
        bbox: Rect,
        top_cells: list[object],
    ) -> dict[int, str]:
        spans = self._repeated_token_spans(text)
        if bbox.w <= 1:
            return {}
        covered = self._covered_cells(top_cells, bbox, min_ratio=0.18)
        if len(spans) < 2:
            spans = self._split_compact_text_for_covered_cells(text, len(covered))
        elif len(covered) > len(spans):
            spans = self._expand_spans_for_covered_cells(text, spans, len(covered))
        if len(spans) < 2:
            return {}
        compact_len = len("".join(str(text or "").split()))
        assignments: dict[int, list[str]] = {}
        for start, end, token in spans:
            cx = bbox.x0 + ((start + end) / 2.0) / max(1, compact_len) * bbox.w
            idx = self._cell_index_at_x(top_cells, cx)
            if idx is None:
                continue
            assignments.setdefault(idx, []).append(token)
        return {idx: "\n".join(values) for idx, values in assignments.items()}

    @staticmethod
    def _split_compact_text_for_covered_cells(text: str, target_count: int) -> list[tuple[int, int, str]]:
        value = "".join(str(text or "").split())
        if target_count < 2 or len(value) < target_count * 2:
            return []
        return MissingTopBandStructureBuilder._even_spans(value, target_count)

    @staticmethod
    def _expand_spans_for_covered_cells(
        text: str,
        spans: list[tuple[int, int, str]],
        target_count: int,
    ) -> list[tuple[int, int, str]]:
        value = "".join(str(text or "").split())
        if target_count <= len(spans) or not spans:
            return spans
        expanded = list(spans)
        while len(expanded) < target_count:
            longest_idx = max(range(len(expanded)), key=lambda idx: expanded[idx][1] - expanded[idx][0])
            start, end, token = expanded[longest_idx]
            if end - start < 6:
                break
            pieces = MissingTopBandStructureBuilder._uppercase_boundary_spans(token)
            if len(pieces) <= 1:
                pieces = MissingTopBandStructureBuilder._even_spans(token, 2)
            replacement = [(start + p0, start + p1, piece) for p0, p1, piece in pieces]
            expanded = expanded[:longest_idx] + replacement + expanded[longest_idx + 1 :]
        return expanded[:target_count] if len(expanded) > target_count else expanded

    @staticmethod
    def _even_spans(value: str, count: int) -> list[tuple[int, int, str]]:
        if count < 2:
            return []
        spans = []
        for idx in range(count):
            start = round(idx * len(value) / count)
            end = round((idx + 1) * len(value) / count)
            token = value[start:end]
            if token:
                spans.append((start, end, token))
        return spans

    @staticmethod
    def _repeated_token_spans(text: str) -> list[tuple[int, int, str]]:
        value = "".join(str(text or "").split())
        if not value:
            return []

        measurement_pattern = re.compile(
            r"\d+(?:\.\d+)?(?:[x×]\d+(?:\.\d+)?){1,3}(?:mm|cm|m)?"
            r"|\d+(?:\.\d+)?(?:mm|cm|m|kg|g|w|kw|mah|l|ml|gb|tb|hz|寸|英寸|瓦|升|克|千克)",
            re.I,
        )
        measurement_spans = [(match.start(), match.end(), match.group(0)) for match in measurement_pattern.finditer(value)]
        if len(measurement_spans) >= 2:
            return measurement_spans

        alnum_spans = MissingTopBandStructureBuilder._compact_alnum_spans(value)
        if len(alnum_spans) >= 2:
            return alnum_spans

        return MissingTopBandStructureBuilder._uppercase_boundary_spans(value)

    @staticmethod
    def _compact_alnum_spans(value: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        idx = 0
        while idx < len(value):
            match = re.search(r"[A-Za-z]+", value[idx:])
            if not match:
                break
            start = idx + match.start()
            end = start
            has_digit = False
            while end < len(value):
                ch = value[end]
                if ch.isascii() and (ch.isalnum() or ch in {"-", "_", ".", "/"}):
                    has_digit = has_digit or ch.isdigit()
                    end += 1
                    continue
                if ch in {"(", "（"}:
                    close = ")" if ch == "(" else "）"
                    close_pos = value.find(close, end + 1)
                    ascii_close_pos = value.find(")", end + 1)
                    candidates = [pos for pos in [close_pos, ascii_close_pos] if pos >= 0]
                    if candidates:
                        end = min(candidates) + 1
                    break
                break
            if has_digit:
                for part_start, part_end, token in MissingTopBandStructureBuilder._uppercase_boundary_spans(value[start:end]):
                    spans.append((start + part_start, start + part_end, token))
            idx = max(end, start + 1)
        return spans

    @staticmethod
    def _uppercase_boundary_spans(value: str) -> list[tuple[int, int, str]]:
        starts = [0]
        for idx in range(1, len(value) - 1):
            prev = value[idx - 1]
            cur = value[idx]
            nxt = value[idx + 1]
            if not (cur.isascii() and cur.isupper() and nxt.isascii() and nxt.isupper()):
                continue
            if prev.isdigit() or prev in {")", "）"}:
                starts.append(idx)
        starts = sorted(set(starts))

        spans: list[tuple[int, int, str]] = []
        for pos, start in enumerate(starts):
            end = starts[pos + 1] if pos + 1 < len(starts) else len(value)
            token = value[start:end]
            if not token:
                continue
            if re.search(r"[A-Za-z]", token) and re.search(r"\d", token):
                spans.append((start, end, token))
        return spans

    def _covered_cells(
        self,
        top_cells: list[object],
        bbox: Rect,
        min_ratio: float,
    ) -> list[int]:
        out: list[int] = []
        for idx, cell in enumerate(top_cells):
            cb = getattr(cell, "bbox")
            if self.ops._x_overlap_ratio(cb, bbox) >= min_ratio:
                out.append(idx)
        return out

    def _nearest_cell_index(self, top_cells: list[object], bbox: Rect) -> int | None:
        best_idx = None
        best_score = -1.0
        for idx, cell in enumerate(top_cells):
            cb = getattr(cell, "bbox")
            x_overlap = self.ops._x_overlap_ratio(cb, bbox)
            center = cb.x0 - 8 <= bbox.cx <= cb.x1 + 8
            distance = abs(cb.cx - bbox.cx) / max(1.0, cb.w)
            score = x_overlap + (0.75 if center else 0.0) - distance * 0.05
            if score > best_score:
                best_idx, best_score = idx, score
        return best_idx if best_idx is not None and best_score > 0.05 else None

    def _cell_index_at_x(self, top_cells: list[object], x: float) -> int | None:
        for idx, cell in enumerate(top_cells):
            bbox = getattr(cell, "bbox")
            if bbox.x0 <= x <= bbox.x1:
                return idx
        if not top_cells:
            return None
        return min(range(len(top_cells)), key=lambda idx: abs(getattr(top_cells[idx], "bbox").cx - x))

    def _is_wide_for_nearest_cell(self, obj: object, top_cells: list[object]) -> bool:
        idx = self._nearest_cell_index(top_cells, getattr(obj, "bbox"))
        if idx is None:
            return False
        cell_bbox = getattr(top_cells[idx], "bbox")
        obj_bbox = getattr(obj, "bbox")
        text = str(getattr(obj, "text", ""))
        return obj_bbox.w > cell_bbox.w * 1.18 or (obj_bbox.w > cell_bbox.w * 0.78 and len(text) > 18)

    @staticmethod
    def _merge_text(existing: str, new_text: str) -> str:
        existing = str(existing or "").strip()
        new_text = str(new_text or "").strip()
        if not existing:
            return new_text
        if not new_text or new_text in existing:
            return existing
        if existing in new_text:
            return new_text
        return existing + "\n" + new_text
