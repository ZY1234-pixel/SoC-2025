"""Small HTML table geometry helpers used by the reflow renderer."""

from __future__ import annotations


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
