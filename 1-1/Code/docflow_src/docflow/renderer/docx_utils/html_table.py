"""Small HTML table geometry helpers used by the reflow renderer."""

from __future__ import annotations

import math

from docflow.planning.text_metrics import estimate_text_units


def get_table_rows(table_soup) -> list:
    rows = []
    for section in table_soup.find_all(["thead", "tbody", "tfoot"], recursive=False):
        rows.extend(section.find_all("tr", recursive=False))
    rows.extend(row for row in table_soup.find_all("tr", recursive=False) if row not in rows)
    return rows


def get_table_columns(row_soup) -> list:
    return row_soup.find_all(["td", "th"], recursive=False)


def get_table_dimensions(table_soup) -> tuple[int, int]:
    rows = get_table_rows(table_soup)
    columns = max(
        (sum(int(cell.get("colspan", 1)) for cell in get_table_columns(row)) for row in rows),
        default=0,
    )
    return len(rows), columns


def get_table_cell_placements(table_soup) -> list[tuple[int, int, int, int, object]]:
    rows, columns = get_table_dimensions(table_soup)
    occupied = [[False] * columns for _ in range(rows)]
    placements = []
    for row_index, row in enumerate(get_table_rows(table_soup)):
        column_index = 0
        for cell in get_table_columns(row):
            while column_index < columns and occupied[row_index][column_index]:
                column_index += 1
            if column_index >= columns:
                break
            row_span = min(int(cell.get("rowspan", 1)), rows - row_index)
            column_span = min(int(cell.get("colspan", 1)), columns - column_index)
            placements.append((row_index, column_index, row_span, column_span, cell))
            for target_row in range(row_index, row_index + row_span):
                for target_column in range(column_index, column_index + column_span):
                    occupied[target_row][target_column] = True
            column_index += column_span
    return placements


def get_table_cell_text_lines(cell_soup) -> tuple[str, ...]:
    return tuple(line.strip() for line in cell_soup.get_text("\n", strip=True).splitlines() if line.strip())


def estimate_table_cell_lines(cell_soup, font_size: float, available_width: float) -> int:
    return sum(
        max(1, math.ceil(estimate_text_units(line) * font_size / max(available_width, 1.0)))
        for line in get_table_cell_text_lines(cell_soup)
    )


def get_table_column_weights(table_soup, source_weights=()) -> tuple[float, ...]:
    _rows, columns = get_table_dimensions(table_soup)
    if len(source_weights) == columns and all(float(weight) > 0 for weight in source_weights):
        return tuple(float(weight) for weight in source_weights)
    weights = [1.0] * columns
    for _row, column, _row_span, column_span, cell in get_table_cell_placements(table_soup):
        text = cell.get_text(" ", strip=True)
        units = estimate_text_units(text)
        demand = max(units / max(column_span, 1), 1.0)
        for target in range(column, column + column_span):
            weights[target] = max(weights[target], demand)
    return tuple(weights)
