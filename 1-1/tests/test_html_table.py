from bs4 import BeautifulSoup

from docflow.renderer.docx_utils.html_table import get_table_dimensions, get_table_rows


def test_identical_table_rows_are_not_deduplicated() -> None:
    table = BeautifulSoup(
        "<table><tr><td>相同内容</td></tr><tr><td>相同内容</td></tr></table>",
        "html.parser",
    ).find("table")

    assert len(get_table_rows(table)) == 2
    assert get_table_dimensions(table) == (2, 1)
