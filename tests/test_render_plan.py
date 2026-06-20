from pathlib import Path
import sys
from io import BytesIO
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code" / "docflow_src"))

from docflow.model.base import BBox, BlockType
from docflow.model.blocks.image_block import ImageBlock
from docflow.model.blocks.text_block import TextBlock, TextLine
from docflow.model.page import Document, Page
from docflow.model.zone import Zone
from docflow.layout.style_inferrer import infer_block_styles
from docflow.pipeline import RecoveryPipeline
from docflow.renderer.docx_renderer import DocxRenderer
from docflow.schema.models import BlockStyle
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


def test_font_size_estimation_uses_full_page_scale_not_margin_scale():
    page = Page(index=0, image_width=800, image_height=1132)
    page.detect_page_size()
    page.margin_top_pt = 120
    page.margin_bottom_pt = 120
    line = TextLine(
        text="字号估计",
        text_region=[[80, 100], [260, 100], [260, 124], [80, 124]],
    )
    block = TextBlock(
        bbox=BBox(80, 96, 260, 128),
        block_type=BlockType.TEXT,
        lines=[line],
    )

    block.estimate_font_size(page.full_coord_mapper)

    assert block.estimated_font_size_pt == 14.0


def test_weak_multicolumn_evidence_profiles_as_single_column():
    page = Page(index=0, image_width=800, image_height=1132)
    text_blocks = [
        TextBlock(
            bbox=BBox(90, 80, 710, 120),
            block_type=BlockType.TITLE,
            lines=[TextLine(text="通用标题文本")],
            col_count=2,
            col_index=0,
            spanned_cols=[0, 1],
        ),
        TextBlock(
            bbox=BBox(95, 170, 705, 360),
            block_type=BlockType.TEXT,
            lines=[TextLine(text="这是一个跨越页面主体宽度的正文段落，用来模拟普通单栏文档中的长段落。")],
            col_count=2,
            col_index=0,
            spanned_cols=[0, 1],
        ),
        TextBlock(
            bbox=BBox(96, 390, 704, 620),
            block_type=BlockType.TEXT,
            lines=[TextLine(text="第二个正文块同样占据大部分页面宽度，不形成左右并行的连续阅读流。")],
            col_count=2,
            col_index=0,
            spanned_cols=[0, 1],
        ),
        TextBlock(
            bbox=BBox(100, 660, 250, 690),
            block_type=BlockType.TEXT,
            lines=[TextLine(text="短小左侧块")],
            col_count=2,
            col_index=0,
            spanned_cols=[0],
        ),
        TextBlock(
            bbox=BBox(520, 710, 690, 740),
            block_type=BlockType.TEXT,
            lines=[TextLine(text="短小右侧块")],
            col_count=2,
            col_index=1,
            spanned_cols=[1],
        ),
    ]
    page.zones = [Zone(col_count=2, blocks=text_blocks, has_spanned=True)]

    assert RecoveryPipeline._infer_layout_profile(page, text_blocks) == "single_column"


def test_weak_multicolumn_collapse_clears_column_metadata():
    page = Page(index=0, image_width=800, image_height=1132)
    block = TextBlock(
        bbox=BBox(520, 320, 690, 350),
        block_type=BlockType.TEXT,
        lines=[TextLine(text="短小右侧块")],
        col_count=2,
        col_index=1,
        spanned_cols=[1],
    )
    page.zones = [Zone(col_count=2, blocks=[block], has_spanned=True)]

    RecoveryPipeline._collapse_to_single_column(page, [block])

    assert page.zones[0].col_count == 1
    assert block.col_count == 1
    assert block.col_index == 0
    assert block.spanned_cols == [0]


def test_reflow_text_block_merges_visual_ocr_lines():
    lines = [
        TextLine(text="以习近平新时代中国特色社会主义思想为指导，全面贯彻党", text_region=[[90, 100], [690, 100], [690, 124], [90, 124]]),
        TextLine(text="的二十大精神，坚持以人民为中心的发展思想，以切实保障农民", text_region=[[90, 130], [690, 130], [690, 154], [90, 154]]),
        TextLine(text="住宅财产权和宅基地使用权为目标。", text_region=[[90, 160], [500, 160], [500, 184], [90, 184]]),
    ]
    block = TextBlock(
        bbox=BBox(90, 100, 690, 184),
        block_type=BlockType.TEXT,
        lines=lines,
        style=BlockStyle(font_size_pt=18.0, alignment="justify"),
    )
    page = Page(index=0, image_width=800, image_height=1132)
    page.attributes = {"layout_profile": "single_column", "render_mode": "reflow"}
    page.zones = [Zone(col_count=1, blocks=[block], has_spanned=False)]
    doc = Document(pages=[page], metadata={})

    docx_bytes = DocxRenderer().render_bytes(doc, enforce_single_page=False)
    with ZipFile(BytesIO(docx_bytes)) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")

    assert "<w:br" not in xml
    assert "全面贯彻党" in xml
    assert "的二十大精神" in xml


def test_chinese_numbered_same_level_titles_share_font_size():
    page = Page(index=0, image_width=800, image_height=1132)
    first = TextBlock(
        bbox=BBox(120, 200, 280, 230),
        block_type=BlockType.TITLE,
        lines=[TextLine(text="一、总体要求")],
        estimated_font_size_pt=24.0,
    )
    second = TextBlock(
        bbox=BBox(120, 500, 280, 526),
        block_type=BlockType.TITLE,
        lines=[TextLine(text="二、适用范围")],
        estimated_font_size_pt=18.0,
    )
    zone = Zone(col_count=1, blocks=[first, second], has_spanned=False)

    infer_block_styles(
        [zone],
        page.coord_mapper,
        page_width_px=page.image_width,
        font_mapper=page.full_coord_mapper,
    )

    assert first.style.font_size_pt == second.style.font_size_pt
    assert first.style.font_size_pt == 21.0


def test_split_visual_rows_title_alignment_uses_merged_row_edges():
    page = Page(index=0, image_width=800, image_height=1132)
    masthead = TextBlock(
        bbox=BBox(122, 165, 723, 280),
        block_type=BlockType.TITLE,
        lines=[
            TextLine(text="广东省自然资源", text_region=[[122, 167], [561, 165], [561, 222], [123, 224]]),
            TextLine(text="厅", text_region=[[554, 170], [611, 170], [611, 216], [554, 216]]),
            TextLine(text="文件", text_region=[[623, 182], [723, 182], [723, 254], [623, 254]]),
            TextLine(text="广东省农业农村", text_region=[[122, 215], [564, 211], [564, 276], [123, 280]]),
            TextLine(text="厅", text_region=[[557, 217], [613, 217], [613, 271], [557, 271]]),
        ],
        estimated_font_size_pt=28.0,
    )
    zone = Zone(col_count=1, blocks=[masthead], has_spanned=False)

    infer_block_styles(
        [zone],
        page.coord_mapper,
        page_width_px=page.image_width,
        font_mapper=page.full_coord_mapper,
    )

    assert masthead.style.alignment == "center"


def test_multiline_title_uses_modest_font_scale():
    page = Page(index=0, image_width=800, image_height=1132)
    title = TextBlock(
        bbox=BBox(117, 421, 663, 536),
        block_type=BlockType.TITLE,
        lines=[
            TextLine(text="广东省自然资源厅广东省农业农村厅关于", text_region=[[117, 424], [663, 421], [663, 455], [117, 457]]),
            TextLine(text="妥善处理农村不动产登记历史", text_region=[[199, 462], [583, 460], [583, 492], [199, 494]]),
            TextLine(text="遗留问题的若干意见", text_region=[[259, 501], [526, 499], [526, 533], [259, 536]]),
        ],
        estimated_font_size_pt=21.0,
    )

    resolved = DocxRenderer()._resolve_title_font_size_pt(
        block=title,
        page=page,
        font_size_pt=21.0,
        alignment="center",
    )

    assert resolved == 22.26


def test_split_masthead_title_does_not_use_single_line_scale():
    page = Page(index=0, image_width=800, image_height=1132)
    masthead = TextBlock(
        bbox=BBox(122, 165, 723, 280),
        block_type=BlockType.TITLE,
        lines=[
            TextLine(text="广东省自然资源", text_region=[[122, 167], [561, 165], [561, 222], [123, 224]]),
            TextLine(text="厅", text_region=[[554, 170], [611, 170], [611, 216], [554, 216]]),
            TextLine(text="文件", text_region=[[623, 182], [723, 182], [723, 254], [623, 254]]),
            TextLine(text="广东省农业农村", text_region=[[122, 215], [564, 211], [564, 276], [123, 280]]),
            TextLine(text="厅", text_region=[[557, 217], [613, 217], [613, 271], [557, 271]]),
        ],
        estimated_font_size_pt=28.0,
    )

    resolved = DocxRenderer()._resolve_title_font_size_pt(
        block=masthead,
        page=page,
        font_size_pt=28.0,
        alignment="center",
    )

    assert resolved == 28.0


def test_title_block_does_not_force_bold_without_style_signal():
    title = TextBlock(
        bbox=BBox(120, 200, 320, 230),
        block_type=BlockType.TITLE,
        lines=[TextLine(text="一、总体要求")],
        style=BlockStyle(font_size_pt=16.0, alignment="left", bold=False),
    )
    page = Page(index=0, image_width=800, image_height=1132)
    page.zones = [Zone(col_count=1, blocks=[title], has_spanned=False)]
    doc = Document(pages=[page], metadata={})

    docx_bytes = DocxRenderer().render_bytes(doc, enforce_single_page=False)
    with ZipFile(BytesIO(docx_bytes)) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")

    assert "一、总体要求" in xml
    assert '<w:b w:val="0"/>' in xml
    assert '<w:b/>' not in xml
    assert '<w:b w:val="1"/>' not in xml


def test_title_block_preserves_explicit_bold_style():
    title = TextBlock(
        bbox=BBox(120, 200, 320, 230),
        block_type=BlockType.TITLE,
        lines=[TextLine(text="显式加粗标题")],
        style=BlockStyle(font_size_pt=16.0, alignment="left", bold=True),
    )
    page = Page(index=0, image_width=800, image_height=1132)
    page.zones = [Zone(col_count=1, blocks=[title], has_spanned=False)]
    doc = Document(pages=[page], metadata={})

    docx_bytes = DocxRenderer().render_bytes(doc, enforce_single_page=False)
    with ZipFile(BytesIO(docx_bytes)) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")

    assert "显式加粗标题" in xml
    assert "<w:b" in xml


def test_renderer_merges_adjacent_visual_text_band_for_docx_layout():
    page = Page(index=0, image_width=1200, image_height=1600)
    figure = ImageBlock(
        bbox=BBox(80, 1000, 500, 1300),
        block_type=BlockType.FIGURE,
        col_count=2,
        col_index=0,
        spanned_cols=[0],
    )
    caption = TextBlock(
        bbox=BBox(180, 940, 1020, 990),
        block_type=BlockType.TEXT,
        lines=[TextLine(text="图下说明文字应跟随左侧图片，而不是独立撑开一整行。")],
        col_count=1,
        col_index=0,
        spanned_cols=[0],
    )
    side_text = TextBlock(
        bbox=BBox(570, 1010, 1080, 1320),
        block_type=BlockType.TEXT,
        lines=[TextLine(text="右侧正文与图片垂直重叠，应合并到同一个局部双栏图文带。")],
        col_count=2,
        col_index=1,
        spanned_cols=[1],
    )
    zones = [
        Zone(col_count=2, blocks=[figure], has_spanned=False),
        Zone(col_count=1, blocks=[caption], has_spanned=False),
        Zone(col_count=2, blocks=[side_text], has_spanned=False),
    ]

    merged = DocxRenderer()._merge_adjacent_visual_text_zones(zones, page)

    assert len(merged) == 1
    assert merged[0].col_count == 2
    assert merged[0].blocks == [figure, caption, side_text]
    assert figure.spanned_cols == [0]
    assert caption.spanned_cols == [0]
    assert side_text.spanned_cols == [1]


def test_renderer_splits_embedded_visual_text_band_inside_single_zone():
    page = Page(index=0, image_width=1000, image_height=1500)
    intro = TextBlock(
        bbox=BBox(120, 620, 520, 660),
        block_type=BlockType.TEXT,
        block_id="intro",
        lines=[TextLine(text="前一小节收尾文字。")],
        col_count=1,
        col_index=0,
        spanned_cols=[0],
    )
    title = TextBlock(
        bbox=BBox(500, 720, 590, 760),
        block_type=BlockType.TITLE,
        block_id="title",
        lines=[TextLine(text="肉桂")],
        col_count=1,
        col_index=0,
        spanned_cols=[0],
    )
    source = TextBlock(
        bbox=BBox(130, 780, 430, 815),
        block_type=BlockType.TEXT,
        block_id="source",
        lines=[TextLine(text="【来源】樟科植物肉桂的树皮。")],
        col_count=1,
        col_index=0,
        spanned_cols=[0],
    )
    figure = ImageBlock(
        bbox=BBox(650, 780, 930, 1020),
        block_type=BlockType.FIGURE,
        block_id="figure",
        col_count=1,
        col_index=0,
        spanned_cols=[0],
    )
    harvest = TextBlock(
        bbox=BBox(130, 830, 430, 865),
        block_type=BlockType.TEXT,
        block_id="harvest",
        lines=[TextLine(text="【采收加工】一般在秋季剥取。")],
        col_count=1,
        col_index=0,
        spanned_cols=[0],
    )
    caption = TextBlock(
        bbox=BBox(700, 1035, 880, 1070),
        block_type=BlockType.FIGURE_CAPTION,
        block_id="caption",
        lines=[TextLine(text="图 2-169 肉桂")],
        col_count=1,
        col_index=0,
        spanned_cols=[0],
    )
    tail = TextBlock(
        bbox=BBox(130, 1120, 520, 1180),
        block_type=BlockType.TEXT,
        block_id="tail",
        lines=[TextLine(text="后续贮藏要求。")],
        col_count=1,
        col_index=0,
        spanned_cols=[0],
    )
    zone = Zone(col_count=1, blocks=[intro, title, source, figure, harvest, caption, tail], has_spanned=False)

    split = DocxRenderer()._split_embedded_visual_text_bands([zone], page)

    assert [len(item.blocks) for item in split] == [1, 5, 1]
    assert split[1].col_count == 2
    assert [block.block_id for block in split[1].blocks] == [title.block_id, source.block_id, figure.block_id, harvest.block_id, caption.block_id]
    assert figure.spanned_cols == [1]
    assert source.spanned_cols == [0]
    assert harvest.spanned_cols == [0]
    assert caption.spanned_cols == [1]


def test_renderer_narrow_strip_block_does_not_create_spanned_segment():
    header = TextBlock(
        bbox=BBox(820, 30, 1080, 70),
        block_type=BlockType.HEADER,
        lines=[TextLine(text="The Economist November 11th 2023")],
        col_count=3,
        col_index=2,
        spanned_cols=[0, 1, 2],
    )

    cols = DocxRenderer._layout_block_cols(header, num_cols=3, page_width_px=1200)

    assert cols == [2]


def test_long_narrow_title_uses_body_sized_cap():
    page = Page(index=0, image_width=1386, image_height=1859)
    title = TextBlock(
        bbox=BBox(67, 159, 108, 337),
        block_type=BlockType.TITLE,
        lines=[TextLine(text="GUSHIZHONGDEKEXUE 故事中的科学")],
        estimated_font_size_pt=30.0,
    )

    resolved = DocxRenderer()._resolve_title_font_size_pt(
        block=title,
        page=page,
        font_size_pt=30.0,
        alignment="center",
    )

    assert resolved == 12.075


def test_multicolumn_latin_body_escapes_font_floor():
    page = Page(index=0, image_width=5000, image_height=6567)
    block = TextBlock(
        bbox=BBox(300, 3000, 1720, 4260),
        block_type=BlockType.TEXT,
        lines=[TextLine(text="A long Latin magazine paragraph should not stay at the tiny fit floor.")],
        col_count=3,
        col_index=0,
        spanned_cols=[0],
    )

    resolved = DocxRenderer()._resolve_body_font_size_pt(
        block=block,
        page=page,
        font_size_pt=8.5,
    )

    assert resolved == 9.3
