from pathlib import Path
import sys
import json

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code" / "docflow_src"))

from docflow.layout.sorter import sort_layout
from docflow.layout.xycutpp import _sort_layout_xycutpp_core, postprocess_xycutpp_local_attachments
from docflow.adapters.paddle_adapter import PaddleAdapter
from docflow.model.base import BBox, BlockType
from docflow.model.blocks.factory import BlockFactory
from docflow.model.blocks.image_block import ImageBlock
from docflow.model.blocks.text_block import TextBlock, TextLine


def _text_block(block_id: str, bbox: tuple[float, float, float, float], block_type: BlockType = BlockType.TEXT) -> TextBlock:
    return TextBlock(
        bbox=BBox(*bbox),
        block_type=block_type,
        block_id=block_id,
        lines=[TextLine(text=block_id)],
    )


def _figure_block(block_id: str, bbox: tuple[float, float, float, float]) -> ImageBlock:
    return ImageBlock(
        bbox=BBox(*bbox),
        block_type=BlockType.FIGURE,
        block_id=block_id,
    )


def _text_block_with_text(
    block_id: str,
    bbox: tuple[float, float, float, float],
    text: str,
    block_type: BlockType = BlockType.TEXT,
) -> TextBlock:
    return TextBlock(
        bbox=BBox(*bbox),
        block_type=block_type,
        block_id=block_id,
        lines=[TextLine(text=text)],
    )


def _table_block(block_id: str, bbox: tuple[float, float, float, float]) -> ImageBlock:
    return ImageBlock(
        bbox=BBox(*bbox),
        block_type=BlockType.TABLE,
        block_id=block_id,
    )


def test_xycutpp_promotes_wide_title_before_multicolumn_body():
    blocks = [
        _text_block("title", (40, 20, 960, 92), BlockType.TITLE),
        _text_block("l1", (40, 130, 430, 220)),
        _text_block("l2", (40, 235, 430, 330)),
        _text_block("r1", (560, 130, 950, 220)),
        _text_block("r2", (560, 235, 950, 330)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1000,
        image_height=1400,
        strategy="xycutpp",
    )

    assert [blk.block_id for blk in ordered] == ["title", "l1", "l2", "r1", "r2"]
    assert ordered[0].col_count == 2
    assert ordered[0].spanned_cols == [0, 1]


def test_xycutpp_detects_cross_layout_intro_text():
    blocks = [
        _text_block("intro", (60, 20, 940, 105)),
        _text_block("l1", (40, 150, 430, 240)),
        _text_block("l2", (40, 255, 430, 345)),
        _text_block("r1", (560, 150, 950, 240)),
        _text_block("r2", (560, 255, 950, 345)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1000,
        image_height=1400,
        strategy="xycutpp",
    )

    assert [blk.block_id for blk in ordered] == ["intro", "l1", "l2", "r1", "r2"]
    assert ordered[0].col_count == 2
    assert ordered[0].spanned_cols == [0, 1]


def test_xycutpp_remaps_figure_and_caption_back_into_right_column_flow():
    blocks = [
        _text_block("l1", (40, 120, 430, 210)),
        _text_block("l2", (40, 225, 430, 315)),
        _text_block("r1", (560, 120, 950, 180)),
        _figure_block("fig", (560, 195, 950, 335)),
        _text_block("cap", (575, 340, 940, 372), BlockType.FIGURE_CAPTION),
        _text_block("r2", (560, 390, 950, 480)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1000,
        image_height=1400,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids[:2] == ["l1", "l2"]
    assert ids.index("r1") < ids.index("fig") < ids.index("cap") < ids.index("r2")
    fig = next(blk for blk in ordered if blk.block_id == "fig")
    assert fig.col_count == 2 and fig.col_index == 1


def test_xycutpp_restores_formula_before_following_left_column_body():
    blocks = [
        _text_block("l_formula", (40, 40, 300, 95), BlockType.FORMULA),
        _text_block("l_formula_no", (320, 48, 360, 88), BlockType.FIGURE_CAPTION),
        _text_block("l1", (40, 120, 430, 210)),
        _text_block("l2", (40, 230, 430, 320)),
        _text_block("r1", (560, 120, 950, 210)),
        _text_block("r_title", (560, 230, 950, 265), BlockType.TITLE),
        _text_block("r2", (560, 290, 950, 380)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1000,
        image_height=1400,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids[:4] == ["l_formula", "l_formula_no", "l1", "l2"]
    assert ids.index("r_title") < ids.index("r2")


def test_xycutpp_attaches_table_caption_back_to_the_table():
    blocks = [
        _text_block("l1", (40, 120, 430, 220)),
        _text_block("l2", (40, 240, 430, 340)),
        _text_block("r1", (560, 120, 950, 220)),
        _text_block("r2", (560, 240, 950, 340)),
        _text_block("tab_cap", (180, 430, 760, 462), BlockType.TABLE_CAPTION),
        _text_block("tab_src", (160, 655, 520, 690), BlockType.TABLE_FOOTNOTE),
        _figure_block("tab", (120, 480, 900, 650)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1000,
        image_height=1400,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids.index("r2") < ids.index("tab_cap") < ids.index("tab") < ids.index("tab_src")


def test_xycutpp_keeps_table_continuation_label_before_table_and_source_after_table():
    blocks = [
        _text_block("cont", (820, 40, 910, 92), BlockType.TABLE_CAPTION),
        _figure_block("tab", (120, 100, 920, 420)),
        _text_block("src", (120, 430, 520, 468), BlockType.TABLE_FOOTNOTE),
        _text_block("body_h", (140, 520, 420, 560)),
        _text_block("body", (120, 580, 940, 760)),
    ]
    blocks[0].lines = [TextLine(text="续表")]
    blocks[2].lines = [TextLine(text="资料来源：统计年鉴")]

    ordered = sort_layout(
        blocks,
        image_width=1000,
        image_height=1400,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids[:3] == ["cont", "tab", "src"]
    assert ids.index("src") < ids.index("body_h") < ids.index("body")


def test_xycutpp_keeps_inline_equation_numbers_adjacent_to_each_formula():
    blocks = [
        _text_block("eq1", (40, 60, 250, 110), BlockType.FORMULA),
        _text_block("n1", (300, 66, 350, 104), BlockType.FIGURE_CAPTION),
        _text_block("eq2", (40, 150, 320, 205), BlockType.FORMULA),
        _text_block("n2", (360, 156, 410, 194), BlockType.FIGURE_CAPTION),
        _text_block("body", (40, 240, 430, 330)),
        _text_block("r1", (560, 80, 950, 180)),
    ]
    blocks[1].lines = [TextLine(text="(18)")]
    blocks[3].lines = [TextLine(text="(19)")]

    ordered = sort_layout(
        blocks,
        image_width=1000,
        image_height=1400,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids[:5] == ["eq1", "n1", "eq2", "n2", "body"]


def test_xycutpp_places_wide_title_and_subtitle_before_following_body_band():
    blocks = [
        _text_block("prev_l", (40, 80, 420, 180)),
        _text_block("prev_r", (560, 80, 940, 180)),
        _text_block("section", (40, 260, 260, 300)),
        _text_block("title", (40, 340, 560, 410), BlockType.TITLE),
        _text_block("subtitle", (40, 430, 860, 470)),
        _text_block("body_l", (40, 520, 420, 760)),
        _text_block("body_r", (560, 520, 940, 760)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1000,
        image_height=1400,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids.index("title") < ids.index("subtitle") < ids.index("body_l")
    assert ids.index("section") < ids.index("subtitle")
    assert ids.index("subtitle") < ids.index("body_r")


def test_xycutpp_places_interview_title_before_question_body():
    blocks = [
        _text_block("sidebar_title", (40, 60, 320, 120), BlockType.TITLE),
        _text_block("sidebar_body", (40, 180, 320, 460)),
        _figure_block("hero", (520, 0, 920, 260)),
        _text_block("interview_title", (520, 300, 900, 340), BlockType.TITLE),
        _text_block("question", (520, 360, 820, 470)),
        _text_block("answer", (520, 490, 820, 780)),
        _text_block("right_body", (900, 360, 1180, 780)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1200,
        image_height=1600,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids.index("interview_title") < ids.index("question") < ids.index("answer")


def test_xycutpp_core_keeps_stable_double_column_page_column_major_despite_right_title():
    blocks = [
        _text_block("l_top", (40, 100, 430, 220)),
        _text_block("l_mid", (40, 260, 430, 420)),
        _text_block("l_low", (40, 460, 430, 620)),
        _text_block("r_top", (560, 100, 950, 220)),
        _text_block("r_mid", (560, 260, 950, 420)),
        _text_block("r_title", (560, 520, 860, 560), BlockType.TITLE),
        _text_block("r_low", (560, 600, 950, 760)),
        _figure_block("l_fig", (60, 820, 420, 1080)),
        _text_block("l_fig_cap", (60, 1090, 420, 1130), BlockType.FIGURE_CAPTION),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1000,
        image_height=1400,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids.index("l_low") < ids.index("r_top")
    assert ids.index("l_fig_cap") < ids.index("r_top")
    assert ids.index("r_title") < ids.index("r_low")


def test_xycutpp_later_column_title_does_not_anchor_to_adjacent_column_text():
    blocks = [
        _figure_block("wide_fig", (774, 199, 1504, 692)),
        _text_block("fig_cap", (812, 705, 1486, 752), BlockType.FIGURE_CAPTION),
        _text_block("left_top", (773, 776, 1124, 1064)),
        _text_block("left_mid", (775, 1065, 1125, 1184)),
        _text_block("left_title", (792, 1200, 1105, 1265), BlockType.TITLE),
        _text_block("left_low", (775, 1281, 1125, 1352)),
        _text_block("right_title", (1153, 776, 1334, 800), BlockType.TITLE),
        _text_block("right_top", (1154, 801, 1505, 990)),
        _text_block("right_mid", (1154, 993, 1504, 1112)),
        _text_block("right_low", (1154, 1112, 1505, 1352)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1524,
        image_height=1368,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids.index("left_top") < ids.index("right_title") < ids.index("right_top")
    right_title = next(blk for blk in ordered if blk.block_id == "right_title")
    proto = right_title.attributes.get("xycutpp_proto", {})
    assert proto.get("restore_anchor_id") == "right_top"


def test_xycutpp_keeps_single_figure_caption_with_figure_before_lower_flows():
    blocks = [
        _text_block("page_title", (130, 34, 1386, 135), BlockType.TITLE),
        _text_block("byline", (589, 159, 931, 184)),
        _text_block("left_intro1", (9, 201, 361, 345)),
        _text_block("left_intro2", (11, 346, 361, 442)),
        _text_block("mid_intro1", (392, 201, 742, 296)),
        _text_block("mid_intro2", (392, 298, 743, 442)),
        _figure_block("wide_fig", (774, 199, 1504, 692)),
        _text_block("fig_cap", (812, 705, 1486, 752), BlockType.FIGURE_CAPTION),
        _text_block("left_title", (28, 456, 342, 521), BlockType.TITLE),
        _text_block("left_section", (11, 538, 360, 657)),
        _text_block("middle_title", (410, 456, 721, 523), BlockType.TITLE),
        _text_block("middle_section", (392, 538, 742, 608)),
        _text_block("right_section_title", (1153, 776, 1334, 800), BlockType.TITLE),
        _text_block("right_section_body", (1154, 801, 1505, 990)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1524,
        image_height=1368,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids.index("wide_fig") + 1 == ids.index("fig_cap")
    assert ids.index("fig_cap") < ids.index("left_section")
    assert ids.index("fig_cap") < ids.index("middle_section")
    assert ids.index("right_section_title") < ids.index("right_section_body")


def test_xycutpp_postprocess_does_not_break_real_newspaper_four_column_skeleton():
    raw = json.loads((ROOT / "test-result" / "run_20260507_050647" / "newspaper_01" / "raw_result.json").read_text())
    image = cv2.imread(str(ROOT / "dataset" / "newspaper_01.png"))
    converted = PaddleAdapter().convert(raw["regions"], image)
    page = converted["pages"][0]
    blocks = [BlockFactory.create(block) for block in page["blocks"]]

    core = _sort_layout_xycutpp_core(
        blocks,
        image_width=page["width"],
        image_height=page["height"],
    )
    by_id_core = {blk.block_id: blk for blk in core}
    assert by_id_core["blk_26"].col_count == 4
    assert by_id_core["blk_25"].col_count == 4
    assert by_id_core["blk_27"].col_count == 4
    assert by_id_core["blk_30"].col_count == 4
    assert by_id_core["blk_26"].col_index == 0
    assert by_id_core["blk_25"].col_index == 1
    assert by_id_core["blk_27"].col_index == 2
    assert by_id_core["blk_30"].col_index == 3

    post = postprocess_xycutpp_local_attachments(
        core,
        image_width=page["width"],
        image_height=page["height"],
    )
    by_id_post = {blk.block_id: blk for blk in post}
    assert by_id_post["blk_26"].col_count == 4
    assert by_id_post["blk_25"].col_count == 4
    assert by_id_post["blk_27"].col_count == 4
    assert by_id_post["blk_30"].col_count == 4
    assert "region_kind" not in (by_id_post["blk_27"].attributes or {}).get("xycutpp_proto", {})


def test_xycutpp_hybrid_sorter_keeps_real_newspaper_four_column_skeleton():
    raw = json.loads((ROOT / "test-result" / "run_20260507_050647" / "newspaper_01" / "raw_result.json").read_text())
    image = cv2.imread(str(ROOT / "dataset" / "newspaper_01.png"))
    converted = PaddleAdapter().convert(raw["regions"], image)
    page = converted["pages"][0]
    blocks = [BlockFactory.create(block) for block in page["blocks"]]

    ordered = sort_layout(
        blocks,
        image_width=page["width"],
        image_height=page["height"],
        strategy="xycutpp_hybrid",
    )
    by_id = {blk.block_id: blk for blk in ordered}

    assert by_id["blk_26"].col_count == 4
    assert by_id["blk_25"].col_count == 4
    assert by_id["blk_27"].col_count == 4
    assert by_id["blk_30"].col_count == 4
    assert by_id["blk_26"].col_index == 0
    assert by_id["blk_25"].col_index == 1
    assert by_id["blk_27"].col_index == 2
    assert by_id["blk_30"].col_index == 3


def test_xycutpp_keeps_parallel_figure_group_before_following_section():
    blocks = [
        _figure_block("left_fig", (220, 225, 715, 594)),
        _figure_block("right_fig", (723, 226, 1217, 593)),
        _text_block("left_subcap", (404, 599, 536, 624), BlockType.FIGURE_CAPTION),
        _text_block("right_subcap", (884, 599, 1015, 624), BlockType.FIGURE_CAPTION),
        _text_block("main_cap", (579, 661, 863, 696), BlockType.FIGURE_CAPTION),
        _text_block("section", (149, 744, 605, 776), BlockType.TITLE),
        _text_block("body", (148, 785, 1288, 1086)),
        _text_block("table_cap", (580, 1111, 859, 1140), BlockType.TABLE_CAPTION),
        _table_block("table", (163, 1152, 1288, 1646)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1434,
        image_height=2070,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids[:6] == [
        "left_fig",
        "right_fig",
        "left_subcap",
        "right_subcap",
        "main_cap",
        "section",
    ]


def test_xycutpp_uses_column_figures_as_structural_cut_anchors():
    blocks = [
        _figure_block("l_fig_top", (206, 116, 653, 471)),
        _text_block("l_cap_top", (282, 493, 538, 521), BlockType.FIGURE_CAPTION),
        _text_block("l_body_top", (138, 575, 686, 696)),
        _figure_block("l_fig_mid", (250, 789, 646, 1210)),
        _text_block("l_title_mid", (192, 1251, 339, 1283), BlockType.TITLE),
        _text_block("l_body_mid", (138, 1332, 689, 1460)),
        _text_block("r_title_top", (788, 101, 938, 133), BlockType.TITLE),
        _text_block("r_body_top", (733, 182, 1282, 398)),
        _figure_block("r_fig_top", (742, 451, 1257, 736)),
        _text_block("r_cap_top", (914, 764, 1101, 792), BlockType.FIGURE_CAPTION),
        _text_block("r_title_mid", (788, 847, 1076, 880), BlockType.TITLE),
        _text_block("r_body_mid", (736, 929, 1280, 1050)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1417,
        image_height=2024,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids.index("l_fig_top") < ids.index("r_title_top")
    assert ids.index("l_body_top") < ids.index("r_title_top")
    assert ids.index("l_title_mid") < ids.index("r_title_top")


def test_xycutpp_column_metadata_uses_body_anchors_not_wide_titles():
    blocks = [
        _text_block("masthead", (62, 34, 479, 70), BlockType.TITLE),
        _text_block("c0_top", (58, 106, 272, 395)),
        _text_block("c0_mid", (58, 409, 271, 732)),
        _text_block("c0_low", (60, 745, 269, 1017)),
        _text_block("c1_top", (289, 106, 494, 304)),
        _text_block("c1_title", (287, 339, 482, 375), BlockType.TITLE),
        _text_block("c1_mid", (289, 357, 503, 696)),
        _text_block("c1_low", (287, 711, 493, 837)),
        _text_block("c2_top", (518, 109, 721, 180)),
        _text_block("c2_mid", (517, 410, 728, 626)),
        _text_block("c2_low", (517, 643, 733, 837)),
        _text_block("c3_top", (747, 106, 919, 144)),
        _text_block("c3_mid", (746, 161, 956, 376)),
        _text_block("c3_low", (746, 404, 958, 838)),
        _figure_block("bottom_fig", (290, 857, 956, 1302)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1022,
        image_height=1344,
        strategy="xycutpp",
    )
    by_id = {blk.block_id: blk for blk in ordered}

    assert max(blk.col_count for blk in ordered) == 4
    assert by_id["c0_top"].col_index == 0
    assert by_id["c1_top"].col_index == 1
    assert by_id["c2_top"].col_index == 2
    assert by_id["c3_top"].col_index == 3
    assert ordered.index(by_id["c1_title"]) < ordered.index(by_id["c1_mid"])
    assert by_id["masthead"].spanned_cols == [0, 1]
    assert by_id["bottom_fig"].spanned_cols == [1, 2, 3]


def test_xycutpp_delays_subset_spanning_visual_until_uncovered_left_column_tail():
    blocks = [
        _text_block("masthead", (62, 34, 479, 70), BlockType.TITLE),
        _text_block("c0_top", (58, 106, 272, 395)),
        _text_block("c0_mid", (58, 409, 271, 732)),
        _text_block("c0_low", (60, 745, 269, 1017)),
        _text_block("c0_tail", (60, 1034, 268, 1231)),
        _text_block("c0_end", (60, 1248, 269, 1301)),
        _text_block("c1_top", (289, 106, 494, 304)),
        _text_block("c1_title", (287, 339, 482, 375), BlockType.TITLE),
        _text_block("c1_mid", (289, 357, 503, 696)),
        _text_block("c1_low", (287, 711, 493, 837)),
        _text_block("c2_top", (518, 109, 721, 180)),
        _text_block("c2_mid", (517, 410, 728, 626)),
        _text_block("c2_low", (517, 643, 733, 837)),
        _text_block("c3_top", (747, 106, 919, 144)),
        _text_block("c3_mid", (746, 161, 956, 376)),
        _text_block("c3_low", (746, 404, 958, 838)),
        _figure_block("bottom_fig", (290, 857, 956, 1302)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1022,
        image_height=1344,
        strategy="xycutpp",
    )
    by_id = {blk.block_id: blk for blk in ordered}

    assert by_id["bottom_fig"].spanned_cols == [1, 2, 3]
    assert ordered.index(by_id["c0_low"]) < ordered.index(by_id["c0_tail"]) < ordered.index(by_id["c0_end"])
    assert ordered.index(by_id["c0_tail"]) < ordered.index(by_id["bottom_fig"])
    assert ordered.index(by_id["c0_end"]) < ordered.index(by_id["bottom_fig"])
    assert by_id["bottom_fig"].attributes["xycutpp_proto"]["spanning_visual_waits_for_uncovered_prefix"] is True
    assert by_id["c0_tail"].attributes["xycutpp_proto"]["spanning_visual_anchor_id"] == "bottom_fig"


def test_xycutpp_keeps_local_title_after_prior_same_column_context():
    blocks = [
        _text_block("top", (20, 200, 360, 340)),
        _text_block("intro", (20, 346, 360, 442)),
        _text_block("section_title", (30, 456, 342, 521), BlockType.TITLE),
        _text_block("body", (20, 538, 360, 656)),
        _text_block("right_top", (395, 201, 742, 295)),
        _text_block("right_body", (395, 298, 743, 440)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1524,
        image_height=1368,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids.index("intro") < ids.index("section_title") < ids.index("body")


def test_xycutpp_keeps_top_byline_after_main_title():
    blocks = [
        _text_block("main_title", (130, 34, 1386, 135), BlockType.TITLE),
        _text_block("left_top", (12, 201, 361, 343)),
        _text_block("right_top", (395, 201, 742, 295)),
        _text_block("byline", (589, 159, 931, 184)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1524,
        image_height=1368,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids[:2] == ["main_title", "byline"]
    assert ordered[1].col_count == 1


def test_xycutpp_keeps_exam_like_page_as_two_columns_under_wide_title():
    blocks = [
        _text_block("title", (416, 142, 1297, 219), BlockType.TITLE),
        _text_block("a1", (157, 298, 837, 555)),
        _text_block("a2", (157, 575, 834, 640)),
        _text_block("a3", (156, 661, 834, 726)),
        _text_block("a4", (155, 914, 834, 1014)),
        _text_block("a5", (240, 1121, 834, 1185)),
        _text_block("a6", (157, 1398, 834, 1432)),
        _text_block("a7", (211, 1675, 833, 1742)),
        _text_block("b1", (965, 296, 1562, 364)),
        _text_block("b2", (944, 406, 1564, 534)),
        _text_block("b3", (887, 552, 1565, 684)),
        _text_block("b4", (887, 704, 1564, 801)),
        _text_block("b5", (887, 981, 1564, 1114)),
        _text_block("b6", (889, 1309, 1563, 1375)),
        _text_block("b7", (887, 1469, 1562, 1535)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1700,
        image_height=2200,
        strategy="xycutpp",
    )
    by_id = {blk.block_id: blk for blk in ordered}

    assert max(blk.col_count for blk in ordered) == 2
    assert by_id["title"].spanned_cols == [0, 1]
    assert all(by_id[f"a{idx}"].col_index == 0 for idx in range(1, 8))
    assert all(by_id[f"b{idx}"].col_index == 1 for idx in range(1, 8))


def test_xycutpp_recovers_local_three_column_band_below_single_column_exam_text():
    blocks = [
        _text_block("intro_1", (145, 139, 574, 177)),
        _text_block("intro_2", (140, 202, 1487, 376)),
        _text_block("intro_3", (147, 402, 771, 436)),
        _text_block("intro_4", (143, 463, 1498, 566)),
        _text_block("question_21", (143, 663, 1053, 699)),
        _text_block("question_22", (144, 724, 947, 764)),
        _text_block("question_23", (145, 792, 800, 828)),
        _text_block("question_24", (143, 856, 1126, 894)),
        _text_block("question_25", (143, 919, 854, 957)),
        _text_block_with_text("section_b", (810, 984, 842, 1018), "B"),
        _text_block("left_head", (172, 1121, 544, 1196)),
        _text_block("left_body", (167, 1152, 549, 1769)),
        _text_block("left_sig", (300, 1808, 512, 1895)),
        _text_block("middle_body", (556, 1108, 940, 1847)),
        _text_block("middle_sig", (660, 1849, 898, 1939)),
        _text_block("right_title", (1115, 1118, 1223, 1156), BlockType.TITLE),
        _text_block("right_body", (955, 1206, 1329, 1506)),
        _text_block("right_sig", (952, 1761, 1187, 1897)),
        _text_block("tail_prompt", (145, 2046, 1134, 2085)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1654,
        image_height=2339,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]
    by_id = {blk.block_id: blk for blk in ordered}

    assert ids.index("question_25") < ids.index("section_b") < ids.index("left_head")
    assert ids.index("left_head") < ids.index("left_body") < ids.index("left_sig")
    assert ids.index("left_sig") < ids.index("middle_body") < ids.index("middle_sig")
    assert ids.index("middle_sig") < ids.index("right_title") < ids.index("right_body") < ids.index("right_sig")
    assert ids.index("right_sig") < ids.index("tail_prompt")
    assert by_id["left_body"].col_count == 3
    assert by_id["middle_body"].col_index == 1
    assert by_id["right_body"].col_index == 2
    assert by_id["intro_2"].col_count == 1


def test_xycutpp_local_multicol_region_allows_single_large_block_columns():
    blocks = [
        _text_block("top_prompt", (90, 80, 910, 130)),
        _text_block("left_notice", (80, 260, 330, 760)),
        _text_block("middle_notice", (380, 250, 640, 770)),
        _text_block("right_notice", (690, 270, 940, 735)),
        _text_block("tail_prompt", (90, 900, 900, 950)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1000,
        image_height=1200,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]
    by_id = {blk.block_id: blk for blk in ordered}

    assert ids == ["top_prompt", "left_notice", "middle_notice", "right_notice", "tail_prompt"]
    assert by_id["left_notice"].col_count == 3
    assert by_id["middle_notice"].col_index == 1
    assert by_id["right_notice"].col_index == 2


def test_xycutpp_local_multicol_region_ignores_scattered_narrow_question_lines():
    blocks = [
        _text_block("q1", (80, 100, 420, 132)),
        _text_block("q2", (500, 320, 900, 352)),
        _text_block("q3", (80, 560, 430, 592)),
        _text_block("q4", (500, 820, 900, 852)),
        _text_block("q5", (80, 1080, 430, 1112)),
        _text_block("q6", (500, 1340, 900, 1372)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1000,
        image_height=1500,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids == ["q1", "q3", "q5", "q2", "q4", "q6"]


def test_xycutpp_keeps_stable_academic_two_column_page_column_major_after_tables():
    blocks = [
        _text_block("table_cap_l", (138, 141, 785, 221), BlockType.TABLE_CAPTION),
        _table_block("table_l", (174, 243, 746, 405)),
        _text_block("stage_text", (108, 461, 815, 530)),
        _text_block("eq", (271, 546, 641, 588), BlockType.FORMULA),
        _text_block_with_text("eq_no", (756, 546, 817, 585), "(10)", BlockType.FIGURE_CAPTION),
        _text_block("left_intro", (103, 602, 812, 836)),
        _text_block("results", (258, 856, 662, 888), BlockType.TITLE),
        _text_block("dataset", (106, 897, 519, 932), BlockType.TITLE),
        _text_block("dataset_body", (106, 943, 815, 1208)),
        _text_block("impl", (108, 1228, 411, 1261), BlockType.TITLE),
        _text_block("impl_body", (103, 1272, 817, 1606)),
        _text_block("quant", (108, 1626, 660, 1658), BlockType.TITLE),
        _text_block("quant_body", (106, 1665, 820, 1802)),
        _text_block("downstream", (106, 1818, 649, 1855), BlockType.TITLE),
        _text_block("downstream_body", (108, 1865, 822, 2000)),
        _text_block("table_cap_r1", (840, 141, 1541, 246), BlockType.TABLE_CAPTION),
        _table_block("table_r1", (867, 268, 1517, 582)),
        _text_block("table_cap_r2", (873, 619, 1509, 700), BlockType.TABLE_CAPTION),
        _table_block("table_r2", (867, 719, 1519, 898)),
        _text_block("right_text", (836, 953, 1544, 1222)),
        _text_block("ablation", (837, 1238, 1287, 1270), BlockType.TITLE),
        _text_block("ablation_body", (834, 1280, 1544, 1513)),
        _text_block("qual", (837, 1531, 1309, 1563), BlockType.TITLE),
        _text_block("qual_body", (834, 1570, 1546, 1838)),
        _text_block("conclusion", (1083, 1855, 1300, 1888), BlockType.TITLE),
        _text_block("conclusion_body", (832, 1894, 1544, 2000)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1654,
        image_height=2340,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids.index("downstream_body") < ids.index("table_cap_r1")
    assert ids.index("downstream_body") < ids.index("ablation")
    assert ids.index("qual_body") < ids.index("conclusion")


def test_xycutpp_precut_splits_text_around_isolated_central_visual():
    blocks = [
        _text_block("top", (120, 60, 900, 150)),
        _text_block("left", (60, 220, 300, 520)),
        _text_block("right", (700, 220, 940, 520)),
        _figure_block("center_fig", (340, 200, 660, 540)),
        _text_block("bottom", (120, 620, 900, 760)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1000,
        image_height=1400,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids.index("top") < ids.index("left") < ids.index("right") < ids.index("bottom")
    assert ids.index("left") < ids.index("center_fig") < ids.index("bottom")


def test_xycutpp_demotes_peripheral_sidebar_after_main_flow():
    blocks = [
        _text_block("page_title", (40, 60, 520, 120), BlockType.TITLE),
        _text_block("sidebar1", (40, 220, 250, 420)),
        _text_block("sidebar2", (40, 430, 250, 620)),
        _text_block("main_title", (340, 180, 900, 220), BlockType.TITLE),
        _text_block("main_q", (340, 260, 700, 360)),
        _text_block("main_a", (340, 380, 700, 620)),
        _text_block("main_r", (760, 260, 1040, 620)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1100,
        image_height=1600,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids.index("page_title") < ids.index("main_title") < ids.index("main_q")
    assert ids.index("main_q") < ids.index("sidebar1")
    assert ids.index("main_r") < ids.index("sidebar2")


def test_xycutpp_keeps_spanning_article_title_and_subtitle_before_column_body():
    blocks = [
        _text_block("old1", (40, 60, 320, 160)),
        _text_block("old2", (360, 60, 640, 160)),
        _text_block("old3", (680, 60, 960, 160)),
        _text_block("section", (40, 260, 240, 300)),
        _text_block("title", (40, 340, 520, 410), BlockType.TITLE),
        _text_block("subtitle", (40, 440, 820, 480)),
        _text_block("c1a", (40, 520, 260, 760)),
        _text_block("c2a", (340, 520, 560, 700)),
        _text_block("c3a", (680, 520, 940, 900)),
        _text_block("c1b", (40, 770, 260, 980)),
        _text_block("c2b", (340, 710, 560, 980)),
        _figure_block("fig", (40, 1020, 620, 1320)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1000,
        image_height=1600,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]
    by_id = {blk.block_id: blk for blk in ordered}

    assert ids.index("section") < ids.index("title") < ids.index("subtitle") < ids.index("c1a")
    assert ids.index("c1a") < ids.index("c1b") < ids.index("c2a") < ids.index("c2b") < ids.index("c3a")
    proto = (by_id["title"].attributes or {}).get("xycutpp_proto", {})
    assert proto.get("region_kind") == "spanning_article_band"


def test_xycutpp_keeps_local_centered_title_before_right_side_figure():
    blocks = [
        _text_block("chapter", (120, 80, 320, 125), BlockType.TITLE),
        _text_block("intro", (160, 150, 540, 210)),
        _figure_block("fig", (640, 300, 930, 520)),
        _text_block("title", (470, 240, 610, 285), BlockType.TITLE),
        _text_block("body_short", (170, 320, 510, 380)),
        _text_block("body_long", (130, 400, 960, 640)),
        _text_block("cap", (680, 532, 910, 564), BlockType.FIGURE_CAPTION),
        _text_block("tail", (170, 660, 520, 710)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1100,
        image_height=1500,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]

    assert ids.index("title") < ids.index("fig")


def test_xycutpp_defers_right_side_figure_family_until_after_continuing_body():
    blocks = [
        _text_block("chapter", (120, 80, 320, 125), BlockType.TITLE),
        _text_block("section", (470, 240, 610, 285), BlockType.TITLE),
        _text_block("lead", (160, 305, 520, 365)),
        _figure_block("fig", (640, 300, 930, 520)),
        _text_block("cont1", (140, 530, 970, 590)),
        _text_block("cont2", (130, 595, 980, 770)),
        _text_block("cap", (680, 532, 910, 564), BlockType.FIGURE_CAPTION),
        _text_block("tail", (170, 790, 520, 840)),
    ]

    ordered = sort_layout(
        blocks,
        image_width=1100,
        image_height=1500,
        strategy="xycutpp",
    )
    ids = [blk.block_id for blk in ordered]
    by_id = {blk.block_id: blk for blk in ordered}

    assert ids.index("cont2") < ids.index("fig") < ids.index("cap")
    proto = (by_id["fig"].attributes or {}).get("xycutpp_proto", {})
    assert proto.get("body_continuation_deferred") is True
