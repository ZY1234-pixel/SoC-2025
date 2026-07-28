"""Small HTML table geometry helpers used by the reflow renderer."""

from __future__ import annotations


def estimate_text_units(text: str) -> float:
    return sum(1.0 if ord(char) >= 0x2E80 else 0.42 for char in text)


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


def collapse_sparse_header_spans(table_soup) -> None:
    for row_index, column, _row_span, column_span, header in tuple(get_table_cell_placements(table_soup)):
        if column_span <= 1:
            continue
        rows = get_table_rows(table_soup)
        region_rows = []
        for target_row in range(row_index + 1, len(rows)):
            cells = [
                (cell_column, span, cell)
                for source_row, cell_column, _span_rows, span, cell in get_table_cell_placements(table_soup)
                if source_row == target_row and column <= cell_column < column + column_span
            ]
            if (
                not cells
                or sum(span for _cell_column, span, _cell in cells) != column_span
                or any(cell_column + span > column + column_span for cell_column, span, _cell in cells)
                or sum(bool(cell.get_text(" ", strip=True)) for _cell_column, _span, cell in cells) > 1
            ):
                region_rows = []
                break
            region_rows.append(cells)
        if not region_rows or not any(len(cells) > 1 for cells in region_rows):
            continue
        header.attrs.pop("colspan", None)
        for cells in region_rows:
            tags = [cell for _column, _span, cell in cells]
            text = next((cell.get_text(" ", strip=True) for cell in tags if cell.get_text(" ", strip=True)), "")
            tags[0].string = text
            tags[0].attrs.pop("colspan", None)
            for cell in tags[1:]:
                cell.decompose()
        break


def get_table_column_weights(table_soup) -> tuple[float, ...]:
    _rows, columns = get_table_dimensions(table_soup)
    weights = [1.0] * columns
    for _row, column, _row_span, column_span, cell in get_table_cell_placements(table_soup):
        text = cell.get_text(" ", strip=True)
        units = estimate_text_units(text)
        demand = max(units / max(column_span, 1), 1.0)
        for target in range(column, column + column_span):
            weights[target] = max(weights[target], demand)
    return tuple(weights)
