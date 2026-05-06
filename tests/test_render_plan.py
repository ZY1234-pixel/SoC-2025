from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code" / "docflow_src"))

from docflow.model.base import BBox, BlockType
from docflow.model.blocks.text_block import TextBlock, TextLine
from docflow.model.page import Document, Page
from docflow.model.zone import Zone
from docflow.utils.render_plan import build_render_plan


def test_render_plan_contains_layout_profile_and_zone_strategy():
    block = TextBlock(
        bbox=BBox(0, 0, 100, 40),
        block_type=BlockType.TEXT,
        lines=[TextLine(text="hello world")],
        col_count=1,
        col_index=0,
        spanned_cols=[0],
    )
    page = Page(index=0, image_width=1000, image_height=1400)
    page.attributes = {"layout_profile": "single_column", "rule_stats": {"category_fix_count": 1}}
    page.zones = [Zone(col_count=1, blocks=[block], has_spanned=False)]
    document = Document(pages=[page], metadata={})

    plan = build_render_plan(document, output_format="docx")
    assert plan["summary"]["page_count"] == 1
    assert plan["pages"][0]["layout_profile"] == "single_column"
    assert plan["pages"][0]["zones"][0]["rendering_strategy"] == "single_col"


def test_render_plan_exposes_region_metadata():
    block = TextBlock(
        bbox=BBox(0, 0, 100, 40),
        block_type=BlockType.TEXT,
        lines=[TextLine(text="hello world")],
        col_count=3,
        col_index=1,
        spanned_cols=[1],
        attributes={"xycutpp_proto": {"region_id": "local_parallel_1", "region_kind": "local_parallel_text_band"}},
    )
    page = Page(index=0, image_width=1000, image_height=1400)
    page.attributes = {"layout_profile": "generic_complex", "rule_stats": {}}
    page.zones = [
        Zone(
            col_count=3,
            blocks=[block],
            has_spanned=False,
            region_id="local_parallel_1",
            region_kind="local_parallel_text_band",
        )
    ]
    document = Document(pages=[page], metadata={})

    plan = build_render_plan(document, output_format="docx")
    zone = plan["pages"][0]["zones"][0]

    assert zone["region_id"] == "local_parallel_1"
    assert zone["region_kind"] == "local_parallel_text_band"
