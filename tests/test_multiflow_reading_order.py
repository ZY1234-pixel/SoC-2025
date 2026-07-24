from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code" / "docflow_src"))

from docflow.config import RecoveryConfig
from docflow.model.blocks.factory import BlockFactory
from docflow.model.base import BBox, BlockType
from docflow.model.blocks.text_block import TextBlock, TextLine
from docflow.pipeline import RecoveryPipeline
from docflow.renderer.markdown_renderer import MarkdownRenderer
from docflow.utils.render_plan import build_render_plan


def _text_block(block_id: str, bbox: list[float], text: str, category: str = "text") -> dict:
    lines = []
    y = bbox[1]
    for part in text.split("\n"):
        lines.append(
            {
                "text": part,
                "confidence": 1.0,
                "poly": [
                    [bbox[0], y],
                    [bbox[2], y],
                    [bbox[2], y + 18],
                    [bbox[0], y + 18],
                ],
            }
        )
        y += 24
    return {
        "id": block_id,
        "category": category,
        "bbox": bbox,
        "text_lines": lines,
        "text": text,
    }


def _model_order_text_block(
    block_id: str,
    bbox: list[float],
    text: str,
    model_order: int,
    category: str = "text",
) -> dict:
    block = _text_block(block_id, bbox, text, category=category)
    block["attributes"] = {
        "layout_model": "pp-doclayout-v3",
        "model_order": model_order,
        "raw_layout_label": category,
    }
    return block


def _figure_block(block_id: str, bbox: list[float]) -> dict:
    return {
        "id": block_id,
        "category": "figure",
        "bbox": bbox,
    }


def _formula_block(block_id: str, bbox: list[float]) -> dict:
    return {
        "id": block_id,
        "category": "formula",
        "bbox": bbox,
    }


def _model_order_figure_block(block_id: str, bbox: list[float], model_order: int) -> dict:
    block = _figure_block(block_id, bbox)
    block["attributes"] = {
        "layout_model": "pp-doclayout-v3",
        "model_order": model_order,
        "raw_layout_label": "figure",
    }
    return block


def test_pipeline_reclassifies_cjk_figure_caption_and_page_number():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1417,
                "height": 2024,
                "blocks": [
                    _text_block("cap_as_title", [318, 1169, 505, 1197], "图6-20绘制椭圆", category="title"),
                    _text_block("page_no", [1198, 1933, 1280, 1963], "-081-"),
                    _text_block("section", [188, 1248, 339, 1283], "3.绘制曲线", category="title"),
                    _text_block("body", [135, 1332, 689, 1460], "在左视图中捕捉点和椭圆的四分点。"),
                ],
            }
        ],
    }
    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    doc = pipeline.build_document(page)
    blocks = [blk for zone in doc.pages[0].zones for blk in zone.blocks]
    by_id = {blk.block_id: blk for blk in blocks}

    assert by_id["cap_as_title"].block_type == BlockType.FIGURE_CAPTION
    assert by_id["page_no"].block_type == BlockType.PAGE_NUMBER


def test_pipeline_model_order_strategy_preserves_upstream_reading_order():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1000,
                "height": 1000,
                "blocks": [
                    _model_order_text_block("left_second", [100, 100, 400, 180], "model says second", 1),
                    _model_order_text_block("bottom_third", [100, 500, 900, 580], "model says third", 2),
                    _model_order_text_block("right_first", [600, 100, 900, 180], "model says first", 0),
                ],
            }
        ],
    }

    pipeline = RecoveryPipeline(
        config=RecoveryConfig(
            reading_order_strategy="model_order",
            font_classification_enabled=False,
        )
    )
    doc = pipeline.build_document(page)
    blocks = [blk for zone in doc.pages[0].zones for blk in zone.blocks]

    assert [block.block_id for block in blocks] == ["right_first", "left_second", "bottom_third"]
    assert all((block.attributes or {}).get("reading_order_strategy") == "model_order" for block in blocks)


def test_model_order_four_column_magazine_keeps_regular_narrow_tracks():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1022,
                "height": 1344,
                "blocks": [
                    _model_order_text_block("header", [732, 39, 976, 57], "The Economist December 9th 2023", 0, category="header"),
                    _model_order_text_block("masthead", [61, 36, 478, 70], "The world this week Business", 1, category="title"),
                    _model_order_text_block("c0_a", [58, 107, 274, 396], "ByteDance has offered to buy back stock from investors.\nAnother line in the first column.", 2),
                    _model_order_text_block("c0_b", [58, 409, 271, 734], "Meanwhile, a federal judge imposed an injunction.\nThe first column continues here.", 3),
                    _model_order_text_block("c1_a", [287, 107, 496, 306], "The conglomerate is writing off assets acquired almost two decades ago.", 4),
                    _model_order_text_block("c1_b", [287, 339, 502, 697], "Return to never-ever land Disney was forced to defend its business strategy.", 5),
                    _model_order_text_block("c2_a", [516, 108, 723, 182], "Gold, another asset that does well when interest rates are lower, hit a record.", 6),
                    _model_order_text_block("c2_b", [516, 410, 729, 628], "The bullish mood on rate cuts spurred investors to push up stockmarkets.", 7),
                    _model_order_text_block("c3_a", [745, 107, 919, 145], "Scheme took their toll on personal finances.", 8),
                    _model_order_text_block("c3_b", [746, 447, 960, 836], "After being warned about greenwashing, companies are now being told not to engage in AI-washing.", 9),
                    _model_order_figure_block("bottom_fig", [288, 859, 958, 1298], 10),
                ],
            }
        ],
    }

    document = RecoveryPipeline(
        config=RecoveryConfig(
            reading_order_strategy="model_order",
            font_classification_enabled=False,
        )
    ).build_document(page)
    page_obj = document.pages[0]
    blocks = [blk for zone in page_obj.zones for blk in zone.blocks]
    by_id = {blk.block_id: blk for blk in blocks}
    multicol = next(zone for zone in page_obj.zones if any(block.block_id == "bottom_fig" for block in zone.blocks))

    assert multicol.col_count == 4
    assert by_id["c0_a"].col_index == 0
    assert by_id["c1_a"].col_index == 1
    assert by_id["c2_a"].col_index == 2
    assert by_id["c3_b"].col_index == 3
    assert by_id["bottom_fig"].spanned_cols == [1, 2, 3]
    assert by_id["header"].style.alignment == "right"
    assert by_id["masthead"].style.alignment == "left"


def test_pipeline_repairs_anomalous_model_order_for_two_column_magazine():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "blocks": [
                    _model_order_text_block("right_mid", [620, 720, 1120, 920], "right middle body", 0),
                    _model_order_text_block("left_bottom", [80, 980, 560, 1210], "left bottom body", 1),
                    _model_order_text_block("right_top", [620, 300, 1120, 520], "right top body", 2),
                    _model_order_text_block("left_top", [80, 300, 560, 520], "left top body", 3),
                    _model_order_text_block("title", [80, 90, 560, 150], "Magazine Title", 4, category="title"),
                    _model_order_text_block("right_bottom", [620, 960, 1120, 1210], "right bottom body", 5),
                ],
            }
        ],
    }

    pipeline = RecoveryPipeline(
        config=RecoveryConfig(
            reading_order_strategy="model_order",
            model_order_geometric_repair_enabled=True,
            font_classification_enabled=False,
        )
    )
    doc = pipeline.build_document(page)
    blocks = [blk for zone in doc.pages[0].zones for blk in zone.blocks]

    assert [block.block_id for block in blocks] == [
        "title",
        "left_top",
        "left_bottom",
        "right_top",
        "right_mid",
        "right_bottom",
    ]
    assert doc.pages[0].attributes["rule_stats"]["model_order_geometric_repair"] == 1
    assert all((block.attributes or {}).get("reading_order_strategy") == "model_order_geometric_repair" for block in blocks)


def test_repaired_model_order_short_titles_anchor_to_following_text_column():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1000,
                "height": 1500,
                "blocks": [
                    _model_order_figure_block("right_image_placeholder", [650, 260, 900, 520], 0),
                    _model_order_text_block("left_body", [120, 300, 480, 430], "left body paragraph\nleft body more", 1),
                    _model_order_text_block("short_title", [485, 230, 570, 260], "Section", 2, category="title"),
                    _model_order_text_block("left_tail", [120, 455, 480, 560], "left tail paragraph", 3),
                    _model_order_text_block("right_body", [620, 620, 900, 760], "right body paragraph", 4),
                    _model_order_text_block("top_heading", [120, 80, 260, 120], "Top", 5, category="title"),
                ],
            }
        ],
    }

    pipeline = RecoveryPipeline(
        config=RecoveryConfig(
            reading_order_strategy="model_order",
            model_order_geometric_repair_enabled=True,
            font_classification_enabled=False,
        )
    )
    doc = pipeline.build_document(page)
    blocks = {blk.block_id: blk for zone in doc.pages[0].zones for blk in zone.blocks}

    assert doc.pages[0].attributes["rule_stats"]["model_order_geometric_repair"] == 1
    assert blocks["short_title"].col_count == 2
    assert blocks["short_title"].col_index == blocks["left_body"].col_index


def test_model_order_geometric_repair_is_disabled_by_default_for_model_order_strategy():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "blocks": [
                    _model_order_text_block("right_mid", [620, 720, 1120, 920], "right middle body", 0),
                    _model_order_text_block("left_bottom", [80, 980, 560, 1210], "left bottom body", 1),
                    _model_order_text_block("right_top", [620, 300, 1120, 520], "right top body", 2),
                    _model_order_text_block("left_top", [80, 300, 560, 520], "left top body", 3),
                    _model_order_text_block("title", [80, 90, 560, 150], "Magazine Title", 4, category="title"),
                    _model_order_text_block("right_bottom", [620, 960, 1120, 1210], "right bottom body", 5),
                ],
            }
        ],
    }

    pipeline = RecoveryPipeline(
        config=RecoveryConfig(
            reading_order_strategy="model_order",
            font_classification_enabled=False,
        )
    )
    doc = pipeline.build_document(page)
    blocks = [blk for zone in doc.pages[0].zones for blk in zone.blocks]

    assert [block.block_id for block in blocks] == [
        "right_mid",
        "left_bottom",
        "right_top",
        "left_top",
        "title",
        "right_bottom",
    ]
    assert doc.pages[0].attributes["rule_stats"]["model_order_geometric_repair"] == 0
    assert all((block.attributes or {}).get("reading_order_strategy") == "model_order" for block in blocks)


def test_weak_multicolumn_collapse_preserves_model_reading_order():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1653,
                "height": 2339,
                "blocks": [
                    _model_order_text_block("header", [185, 274, 1198, 356], "HYDROLOGICAL PROCESSES", 0, category="header"),
                    _model_order_text_block("paper_title", [200, 418, 1442, 527], "Daily streamflow modelling", 1, category="title"),
                    _model_order_text_block("authors", [419, 574, 1223, 612], "Jin-Yong Choi Bernard Engel", 2),
                    _model_order_text_block("affiliations", [232, 616, 1412, 673], "Department of Agricultural Engineering", 3),
                    _model_order_text_block("abstract_title", [771, 784, 899, 817], "Abstract:", 4, category="title"),
                    _model_order_text_block("abstract", [216, 835, 1428, 1368], "A cell-based long-term hydrological model " * 8, 5, category="abstract"),
                    _model_order_text_block("keywords", [216, 1383, 1450, 1441], "KEY WORDS watershed modelling GIS", 6),
                    _model_order_text_block("intro_title", [710, 1506, 934, 1537], "INTRODUCTION", 7, category="title"),
                    _model_order_text_block("intro_1", [182, 1557, 1459, 1789], "Water resources development and watershed management " * 6, 8),
                    _model_order_text_block("intro_2", [183, 1791, 1462, 1891], "Continuous models are useful " * 6, 9),
                    _model_order_text_block("footnote", [187, 1937, 1457, 1991], "Correspondence to Jin-Yong Choi", 10, category="footnote"),
                ],
            }
        ],
    }

    pipeline = RecoveryPipeline(
        config=RecoveryConfig(
            reading_order_strategy="model_order",
            font_classification_enabled=False,
        )
    )
    doc = pipeline.build_document(page)
    blocks = [blk for zone in doc.pages[0].zones for blk in zone.blocks]

    assert [block.block_id for block in blocks] == [
        "header",
        "paper_title",
        "authors",
        "affiliations",
        "abstract_title",
        "abstract",
        "keywords",
        "intro_title",
        "intro_1",
        "intro_2",
        "footnote",
    ]


def test_pipeline_suppresses_visual_boxes_that_duplicate_text_regions():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1654,
                "height": 2340,
                "blocks": [
                    _text_block(
                        "right_top_text",
                        [836, 953, 1544, 1222],
                        "Segmentation and recognition are evaluated with two baselines.\n"
                        "The proposed pipeline keeps text regions as text.\n"
                        "False visual proposals should not cover this paragraph.\n"
                        "Additional evidence is reported for each component.",
                    ),
                    _text_block(
                        "right_bottom_text",
                        [834, 1280, 1544, 1513],
                        "F. Evaluation of noise impact\n"
                        "This paragraph continues the same right column discussion.\n"
                        "It contains enough lines to represent body text.\n"
                        "A large figure box spanning it would be a duplicate.",
                    ),
                    _figure_block("false_right_column_figure", [844, 961, 1541, 1511]),
                    _text_block("section_title", [261, 859, 664, 888], "III. EXPERIMENTS AND RESULTS", category="title"),
                    _formula_block("false_section_formula", [261, 859, 664, 888]),
                    _text_block(
                        "fig_reference_body",
                        [834, 1570, 1546, 1838],
                        "Fig. 3 visualizes the precision and recall curves for the evaluated methods.\n"
                        "Unlike a caption, this is a full paragraph inside the main text column.\n"
                        "It discusses the experimental trend and then compares different settings.\n"
                        "The text should remain part of the reading flow.",
                    ),
                    _formula_block("real_formula", [261, 536, 664, 585]),
                ],
            }
        ],
    }

    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(page)
    blocks = [blk for zone in document.pages[0].zones for blk in zone.blocks]
    by_id = {blk.block_id: blk for blk in blocks}
    stats = document.pages[0].attributes["rule_stats"]

    assert "false_right_column_figure" not in by_id
    assert "false_section_formula" not in by_id
    assert by_id["fig_reference_body"].block_type == BlockType.TEXT
    assert by_id["section_title"].block_type == BlockType.TITLE
    assert by_id["real_formula"].block_type == BlockType.EQUATION
    assert stats["spurious_visual_suppressed"] == 2


def test_pipeline_suppresses_footer_page_number_visual_container():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1102,
                "height": 1631,
                "blocks": [
                    _text_block("body", [130, 1200, 990, 1370], "正文内容应保留。"),
                    {
                        **_figure_block("footer_page_number_box", [0, 1483, 250, 1554]),
                        "attributes": {
                            "nested_children": [
                                {
                                    "type": "page_number",
                                    "category": "page_number",
                                    "bbox": [140, 1504, 180, 1528],
                                }
                            ]
                        },
                    },
                    _text_block("footer", [387, 1524, 675, 1550], "连锁药店店员中药基础训练手册", category="footer"),
                ],
            }
        ],
    }

    document = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid")).build_document(page)
    block_ids = {blk.block_id for zone in document.pages[0].zones for blk in zone.blocks}

    assert "footer_page_number_box" not in block_ids
    assert "body" in block_ids
    assert "footer" in block_ids


def _magazine_like_page() -> dict:
    return {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "blocks": [
                    _text_block("gaza_l1", [40, 80, 340, 220], "Gaza lead left one\nGaza left one body"),
                    _text_block("gaza_l2", [40, 240, 340, 410], "Gaza left two body\nGaza left two end"),
                    _text_block("gaza_m1", [420, 80, 720, 220], "Gaza middle one\nGaza middle one body"),
                    _text_block("gaza_m2", [420, 240, 720, 410], "Gaza middle two body\nGaza middle two end"),
                    _text_block("right_frag", [820, 70, 1140, 120], "continuation fragment"),
                    _text_block("right_top", [820, 130, 1140, 470], "Right flow top body\nRight flow top more"),
                    _text_block("right_mid", [820, 490, 1140, 860], "Right flow middle body\nRight flow middle more"),
                    _text_block("right_low", [820, 880, 1140, 1460], "Right flow lower body\nRight flow ending"),
                    _text_block("kicker", [40, 540, 220, 600], "Marine technology"),
                    _text_block("title", [40, 620, 700, 700], "The wind in their sails", category="title"),
                    _text_block("subtitle", [40, 790, 720, 850], "Two teams one French one Swiss plan to smash the sailing speed record"),
                    _text_block("sail_l1", [40, 880, 340, 1120], "Sailing left body one\nSailing left body two"),
                    _text_block("sail_m1", [420, 880, 720, 1120], "Sailing middle body one\nSailing middle body two"),
                    _text_block("dup_intro", [40, 1125, 340, 1180], "Repeated transition line"),
                    _text_block("dup_body", [40, 1185, 340, 1340], "Repeated transition line\nSailing left continuation"),
                    _figure_block("figure", [40, 1345, 720, 1540]),
                    _text_block("caption", [40, 1545, 420, 1580], "And all I ask is a fast ship", category="figure_caption"),
                ],
            }
        ],
    }


def _textbook_like_sidebar_page(side: str = "left") -> dict:
    assert side in {"left", "right"}

    if side == "left":
        blocks = [
            _figure_block("hero", [936, 0, 1680, 555]),
            _text_block("side_title", [61, 234, 820, 301], "Environmental Research Group", category="title"),
            _text_block(
                "side_body",
                [58, 643, 437, 1034],
                "In our advanced research\n"
                "laboratories, WRI conducts\n"
                "sophisticated and\n"
                "comprehensive\n"
                "environmental analyses and\n"
                "research. This information is\n"
                "the foundation that helps us\n"
                "make decisions about living\n"
                "responsibly within our\n"
                "environment",
            ),
            _text_block(
                "side_quote",
                [61, 1130, 423, 1362],
                "\"The data we are able to\n"
                "extract from thorough\n"
                "testing procedures provides\n"
                "vital input in decisions that\n"
                "affect our choices and our\n"
                "quality of life.\"",
            ),
            _text_block(
                "main_title",
                [587, 597, 1166, 628],
                "An Interview with Dr. Rick Rediske, Program Manager",
                category="title",
            ),
            _text_block(
                "q1",
                [584, 643, 1033, 749],
                "Why is it important for WRI to continue\n"
                "to investigate the presence of chemical\n"
                "contaminates in the environment?",
            ),
            _text_block(
                "a1",
                [584, 756, 1063, 1116],
                "The testing we do serves two purpos-\n"
                "es. First, it helps define the nature and the\n"
                "extent of a problem or a potential prob-\n"
                "lem. The best means of protecting our\n"
                "water quality is by preventing problems\n"
                "before they occur. That's why we need to\n"
                "continually assess the condition of our\n"
                "environment so we can detect changes\n"
                "that might signal a potential problem.\n"
                "Second, it provides important informa-",
            ),
            _text_block(
                "a1b",
                [584, 1082, 1063, 1301],
                "Second, it provides important informa-\n"
                "tion that will help decision-makers identi-\n"
                "fy what should happen next. Without ade-\n"
                "quate data, those who are working\n"
                "towards change will have a difficult time\n"
                "developing viable solutions.",
            ),
            _text_block(
                "q2",
                [1091, 790, 1570, 1007],
                "You completed an extensive investigation\n"
                "of sediment contamination in the Tannery\n"
                "Bay of White Lake in Muskegon County.\n"
                "Can you give us a summary of the results\n"
                "and tell us how this investigation will\n"
                "help clean-up efforts?",
            ),
            _text_block(
                "a2",
                [1091, 1011, 1566, 1410],
                "The results of the project show that\n"
                "there are adverse ecological effects asso-\n"
                "ciated with the sediment contamination.\n"
                "In addition, mixing and resuspension of\n"
                "the sediments continues to move the con-\n"
                "taminates from Tannery Bay into other\n"
                "parts of White Lake. The identification\n"
                "of these issues has resulted in the\n"
                "Michigan Department of Environmental\n"
                "Quality raising the priority of sediment\n"
                "remediation of Tannery Bay.",
            ),
            _figure_block("bottom_fig", [252, 1554, 1057, 2122]),
        ]
    else:
        blocks = [
            _figure_block("hero", [0, 0, 744, 555]),
            _text_block(
                "main_title",
                [520, 597, 1099, 628],
                "An Interview with Dr. Rick Rediske, Program Manager",
                category="title",
            ),
            _text_block(
                "q1",
                [520, 643, 969, 749],
                "Why is it important for WRI to continue\n"
                "to investigate the presence of chemical\n"
                "contaminates in the environment?",
            ),
            _text_block(
                "a1",
                [520, 756, 999, 1116],
                "The testing we do serves two purpos-\n"
                "es. First, it helps define the nature and the\n"
                "extent of a problem or a potential prob-\n"
                "lem. The best means of protecting our\n"
                "water quality is by preventing problems\n"
                "before they occur.\n"
                "Second, it provides important informa-",
            ),
            _text_block(
                "a1b",
                [520, 1082, 999, 1301],
                "Second, it provides important informa-\n"
                "tion that will help decision-makers identi-\n"
                "fy what should happen next. Without ade-\n"
                "quate data, those who are working\n"
                "towards change will have a difficult time\n"
                "developing viable solutions.",
            ),
            _text_block(
                "q2",
                [1020, 790, 1499, 1007],
                "You completed an extensive investigation\n"
                "of sediment contamination in the Tannery\n"
                "Bay of White Lake in Muskegon County.",
            ),
            _text_block(
                "a2",
                [1020, 1011, 1495, 1410],
                "The results of the project show that\n"
                "there are adverse ecological effects asso-\n"
                "ciated with the sediment contamination.",
            ),
            _text_block("side_title", [1160, 234, 1640, 301], "Sidebar Notes", category="title"),
            _text_block(
                "side_body",
                [1240, 643, 1610, 1034],
                "Sidebar fact one\nSidebar fact two\nSidebar fact three",
            ),
            _text_block(
                "side_quote",
                [1240, 1130, 1610, 1362],
                "\"Sidebar quote\nwith supporting\nmaterial.\"",
            ),
        ]

    return {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1680,
                "height": 2120,
                "blocks": blocks,
            }
        ],
    }


def _banded_mixed_page() -> dict:
    return {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "blocks": [
                    _text_block("section_a", [500, 120, 760, 170], "Section A", category="title"),
                    _text_block("intro", [120, 190, 980, 260], "Intro line one\nIntro line two"),
                    _text_block("body_a", [120, 270, 920, 520], "Body A one\nBody A two"),
                    _text_block("section_b", [520, 560, 780, 610], "Section B", category="title"),
                    _text_block("table_cap", [180, 630, 1000, 680], "Wide table caption"),
                    {
                        "id": "table1",
                        "category": "table",
                        "bbox": [130, 700, 1100, 860],
                        "cells": [],
                    },
                    _text_block("section_c", [510, 900, 760, 950], "Section C", category="title"),
                    _text_block("left_h", [120, 980, 360, 1030], "Left block title", category="title"),
                    _text_block("left_b", [120, 1040, 520, 1300], "Left content one\nLeft content two"),
                    _text_block("right_h", [720, 980, 980, 1030], "Right block title", category="title"),
                    _text_block("right_b", [720, 1040, 1080, 1290], "Right content one\nRight content two"),
                ],
            }
        ],
    }


def _decorative_icon_page() -> dict:
    return {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "blocks": [
                    _figure_block("icon", [500, 560, 545, 620]),
                    _text_block("section_title", [570, 552, 760, 620], "Word List", category="title"),
                    _text_block("body", [120, 660, 920, 860], "word one\nword two\nword three"),
                ],
            }
        ],
    }


def _overlapped_figure_text_page() -> dict:
    return {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "blocks": [
                    _text_block("title", [120, 100, 620, 150], "Practice", category="title"),
                    _text_block("glyph", [240, 220, 760, 520], "K"),
                    _figure_block("practice_grid", [240, 220, 760, 520]),
                ],
            }
        ],
    }


def test_magazine_like_page_under_unified_xycutpp_does_not_emit_project_flow_ids():
    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(_magazine_like_page())
    page = document.pages[0]

    flow_ids = [zone.flow_id for zone in page.zones if zone.flow_id]
    assert flow_ids == []
    assert page.attributes["quality_metrics"]["zone_count"] >= 1
    assert "phase_counts" in page.attributes.get("xycutpp_debug", {})


def test_old_reading_order_strategy_names_are_hybrid_aliases():
    expected_ids = None
    for strategy in ("legacy", "auto", "xycutpp", "xycutpp_hybrid", "xycutpp_paper", "newspaper_hybrid"):
        pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy=strategy))
        document = pipeline.build_document(_magazine_like_page())
        page = document.pages[0]
        ids = [getattr(block, "block_id", "") for zone in page.zones for block in zone.blocks]
        flow_ids = {zone.flow_id for zone in page.zones if zone.flow_id}

        assert flow_ids == set()
        assert "phase_counts" in page.attributes.get("xycutpp_debug", {})
        if expected_ids is None:
            expected_ids = ids
        else:
            assert ids == expected_ids


def test_magazine_like_markdown_still_contains_key_story_markers_and_deduped_transition():
    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(_magazine_like_page())
    renderer = MarkdownRenderer()
    markdown = renderer.render_bytes(document).decode("utf-8")

    assert "# The wind in their sails" in markdown
    assert "Right flow top body" in markdown
    assert markdown.count("Repeated transition line") == 1


def test_magazine_like_render_plan_reports_no_flows_under_unified_xycutpp():
    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(_magazine_like_page())
    plan = build_render_plan(document, output_format="markdown")
    page = plan["pages"][0]

    assert page["flow_count"] == 0
    assert {zone["flow_id"] for zone in page["zones"] if zone["flow_id"]} == set()


def test_short_tail_fragment_is_absorbed_into_previous_body_block():
    blocks = [
        BlockFactory.create(
            _text_block(
                "body",
                [40, 100, 340, 320],
                "Line one\nLine two\nLine three\nLine four",
            )
        ),
        BlockFactory.create(
            _text_block(
                "tail",
                [42, 320, 342, 370],
                "Short continuation\nTail line",
            )
        ),
    ]
    for block in blocks:
        block.col_count = 1
        block.col_index = 0
        block.spanned_cols = [0]
        block.attributes = {"flow_id": "flow_x", "flow_kind": "title"}

    merged = RecoveryPipeline._absorb_short_tail_fragments_within_flows(blocks)
    assert len(merged) == 1
    assert merged[0].block_id == "body"
    assert "Short continuation" in merged[0].full_text()


def test_straighten_same_flow_boundaries_ignores_blocks_without_flow_id():
    prev = TextBlock(
        bbox=BBox(40, 100, 340, 320),
        block_type=BlockType.TEXT,
        block_id="prev",
        lines=[TextLine(text="prev")],
        col_count=1,
        col_index=0,
        spanned_cols=[0],
    )
    curr = TextBlock(
        bbox=BBox(40, 260, 340, 420),
        block_type=BlockType.TEXT,
        block_id="curr",
        lines=[TextLine(text="curr")],
        col_count=1,
        col_index=0,
        spanned_cols=[0],
    )

    RecoveryPipeline._straighten_same_flow_boundaries([prev, curr])

    assert (prev.bbox.y1, prev.bbox.y2) == (100, 320)
    assert (curr.bbox.y1, curr.bbox.y2) == (260, 420)


def test_academic_section_titles_do_not_trigger_article_flow_segmentation():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1200,
                "height": 1800,
                "blocks": [
                    _text_block("l_top", [60, 80, 500, 240], "Left top body\nMore left body"),
                    _text_block("r_top", [680, 80, 1140, 240], "Right top body\nMore right body"),
                    _text_block("sec_21", [60, 320, 360, 360], "2.1 Experimental Setup", category="title"),
                    _text_block("l_mid", [60, 390, 500, 620], "Left mid body\nMore left mid"),
                    _text_block("sec_22", [680, 320, 980, 360], "2.2 Sewage Properties", category="title"),
                    _text_block("r_mid", [680, 390, 1140, 620], "Right mid body\nMore right mid"),
                    _text_block("sec_31", [60, 760, 540, 805], "3.1 Influence of Pulse Electric Field", category="title"),
                    _text_block("l_low", [60, 830, 500, 1120], "Left lower body\nMore left lower"),
                    _text_block("sec_a", [680, 760, 980, 805], "A. Dataset and evaluation protocol", category="title"),
                    _text_block("r_low", [680, 830, 1140, 1120], "Right lower body\nMore right lower"),
                ],
            }
        ],
    }

    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(page)
    flow_ids = {zone.flow_id for zone in document.pages[0].zones if zone.flow_id}

    assert flow_ids == set()


def test_academic_two_column_page_prefers_column_major_order_under_xycutpp():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1200,
                "height": 1800,
                "blocks": [
                    _text_block("left_top", [60, 120, 500, 260], "Left top body\nMore left"),
                    _text_block("right_top", [700, 120, 1140, 260], "Right top body\nMore right"),
                    _text_block("table_cap_l", [60, 320, 500, 360], "TABLE I Results", category="table_caption"),
                    _figure_block("table_l", [60, 380, 500, 520]),
                    _text_block("left_mid", [60, 560, 500, 860], "Left middle body\nMore left middle"),
                    _text_block("sec_l", [60, 900, 360, 940], "2.1 Experimental Setup", category="title"),
                    _text_block("left_low", [60, 970, 500, 1260], "Left lower body\nMore left lower"),
                    _text_block("table_cap_r", [700, 320, 1140, 360], "TABLE II More Results", category="table_caption"),
                    _figure_block("table_r", [700, 380, 1140, 520]),
                    _text_block("right_mid", [700, 560, 1140, 860], "Right middle body\nMore right middle"),
                    _text_block("sec_r", [700, 900, 980, 940], "3.1 Analysis", category="title"),
                    _text_block("sec_mid", [60, 1320, 420, 1360], "III. RESULTS", category="title"),
                    _text_block("right_low", [700, 970, 1140, 1260], "Right lower body\nMore right lower"),
                ],
            }
        ],
    }

    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(page)
    ids = [getattr(block, "block_id", "") for zone in document.pages[0].zones for block in zone.blocks]

    assert ids.index("left_low") < ids.index("table_cap_r")
    assert ids.index("left_low") < ids.index("right_mid")


def test_stable_multicol_alias_prefers_column_major_without_fragmenting_into_many_flows():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1600,
                "height": 2000,
                "blocks": [
                    _text_block("masthead", [120, 40, 1400, 130], "Big story title", category="title"),
                    _text_block("byline", [520, 160, 980, 200], "By Reporter Name"),
                    _figure_block("hero", [760, 220, 1490, 700]),
                    _text_block("hero_cap", [800, 710, 1460, 755], "Hero caption", category="figure_caption"),
                    _text_block("l1", [40, 220, 360, 420], "Left col top\nLeft col top 2"),
                    _text_block("l2", [40, 430, 360, 700], "Left col mid\nLeft col mid 2"),
                    _text_block("l_title", [40, 760, 360, 805], "Secondary left title", category="title"),
                    _text_block("l3", [40, 820, 360, 1120], "Left lower body\nLeft lower body 2"),
                    _text_block("m1", [400, 220, 720, 520], "Middle top body\nMiddle top 2"),
                    _text_block("m2", [400, 540, 720, 900], "Middle lower body\nMiddle lower 2"),
                    _text_block("r1", [1120, 780, 1540, 1080], "Right lower body\nRight lower 2"),
                    _text_block("r2", [1120, 1090, 1540, 1380], "Right tail body\nRight tail 2"),
                ],
            }
        ],
    }

    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(page)
    page_obj = document.pages[0]
    ids = [getattr(block, "block_id", "") for zone in page_obj.zones for block in zone.blocks]
    flow_ids = {zone.flow_id for zone in page_obj.zones if zone.flow_id}

    assert flow_ids == set()
    assert ids.index("l3") < ids.index("r1")
    assert ids.index("l3") < ids.index("r2")


def test_xycutpp_hybrid_treats_stable_multicol_spanning_page_as_column_major_instead_of_article_flows():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1600,
                "height": 2000,
                "blocks": [
                    _text_block("masthead", [120, 40, 1400, 130], "Big story title", category="title"),
                    _text_block("byline", [520, 160, 980, 200], "By Reporter Name"),
                    _figure_block("hero", [760, 220, 1490, 700]),
                    _text_block("hero_cap", [800, 710, 1460, 755], "Hero caption", category="figure_caption"),
                    _text_block("l1", [40, 220, 360, 420], "Left col top\nLeft col top 2"),
                    _text_block("l2", [40, 430, 360, 700], "Left col mid\nLeft col mid 2"),
                    _text_block("l_title", [40, 760, 360, 805], "Secondary left title", category="title"),
                    _text_block("l3", [40, 820, 360, 1120], "Left lower body\nLeft lower body 2"),
                    _text_block("m1", [400, 220, 720, 520], "Middle top body\nMiddle top 2"),
                    _text_block("m2", [400, 540, 720, 900], "Middle lower body\nMiddle lower 2"),
                    _text_block("r1", [1120, 780, 1540, 1080], "Right lower body\nRight lower 2"),
                    _text_block("r2", [1120, 1090, 1540, 1380], "Right tail body\nRight tail 2"),
                ],
            }
        ],
    }

    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(page)
    page_obj = document.pages[0]
    ids = [getattr(block, "block_id", "") for zone in page_obj.zones for block in zone.blocks]
    flow_ids = {zone.flow_id for zone in page_obj.zones if zone.flow_id}

    assert flow_ids == set()
    assert ids[0] == "masthead"
    assert ids.index("masthead") < ids.index("l1")
    assert ids.index("masthead") < ids.index("hero")


def test_xycutpp_hybrid_keeps_local_headings_before_their_following_content():
    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(_banded_mixed_page())
    ids = [getattr(block, "block_id", "") for zone in document.pages[0].zones for block in zone.blocks]

    assert ids.index("section_a") < ids.index("table1")
    assert ids.index("left_h") < ids.index("left_b")
    assert ids.index("section_c") < ids.index("right_h")
    assert ids.index("right_h") < ids.index("right_b")


def test_textbook_like_left_sidebar_keeps_heading_before_top_figure_and_trims_duplicate_prefix():
    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(_textbook_like_sidebar_page("left"))
    markdown = MarkdownRenderer().render_bytes(document).decode("utf-8")

    assert markdown.lstrip().startswith("# Environmental Research Group")
    assert markdown.count("Second, it provides important informa-") == 1


def test_textbook_like_right_sidebar_also_trims_duplicate_prefix_without_left_specific_assumptions():
    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(_textbook_like_sidebar_page("right"))
    markdown = MarkdownRenderer().render_bytes(document).decode("utf-8")

    assert "# An Interview with Dr. Rick Rediske, Program Manager" in markdown
    assert "# Sidebar Notes" in markdown
    assert markdown.count("Second, it provides important informa-") == 1


def test_decorative_title_icon_is_suppressed_from_main_reading_order():
    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(_decorative_icon_page())
    ids = [getattr(block, "block_id", "") for zone in document.pages[0].zones for block in zone.blocks]
    markdown = MarkdownRenderer().render_bytes(document).decode("utf-8")
    stats = document.pages[0].attributes["rule_stats"]

    assert "icon" not in ids
    assert markdown.lstrip().startswith("# Word List")
    assert stats["decorative_icon_suppressed"] == 1


def test_short_text_overlapping_figure_is_deduplicated():
    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(_overlapped_figure_text_page())
    ids = [getattr(block, "block_id", "") for zone in document.pages[0].zones for block in zone.blocks]
    markdown = MarkdownRenderer().render_bytes(document).decode("utf-8")
    stats = document.pages[0].attributes["rule_stats"]

    assert "glyph" not in ids
    assert "practice_grid" in ids
    assert "\nK\n" not in markdown
    assert stats["figure_text_dedup_suppressed"] == 1


def test_exam_like_two_column_page_refreshes_profile_and_render_mode():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1700,
                "height": 2200,
                "attributes": {"layout_profile": "single_column", "render_mode": "reflow"},
                "blocks": [
                    _text_block("title", [416, 142, 1297, 219], "Competition title", category="title"),
                    _text_block("a1", [157, 298, 837, 555], "A one\nA one body"),
                    _text_block("a2", [157, 575, 834, 640], "A two"),
                    _text_block("a3", [156, 661, 834, 726], "A three"),
                    _text_block("a4", [155, 914, 834, 1014], "A four"),
                    _text_block("a5", [240, 1121, 834, 1185], "A five"),
                    _text_block("a6", [157, 1398, 834, 1432], "A six"),
                    _text_block("b1", [965, 296, 1562, 364], "B one"),
                    _text_block("b2", [944, 406, 1564, 534], "B two"),
                    _text_block("b3", [887, 552, 1565, 684], "B three"),
                    _text_block("b4", [887, 704, 1564, 801], "B four"),
                    _text_block("b5", [887, 981, 1564, 1114], "B five"),
                    _text_block("b6", [889, 1309, 1563, 1375], "B six"),
                ],
            }
        ],
    }

    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(page)
    page_obj = document.pages[0]

    assert page_obj.attributes["layout_profile"] == "academic_two_col"
    assert page_obj.attributes["render_mode"] == "native_columns"
    assert max(zone.col_count for zone in page_obj.zones) == 2


def test_local_parallel_region_isolated_into_its_own_zone_group():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1654,
                "height": 2339,
                "blocks": [
                    _text_block("intro_1", [145, 139, 574, 177], "Intro line 1"),
                    _text_block("intro_2", [140, 202, 1487, 376], "Wide intro paragraph\nMore intro"),
                    _text_block("q25", [143, 919, 854, 957], "25. Question line"),
                    _text_block("mark_b", [810, 984, 842, 1018], "B"),
                    _text_block("left_head", [172, 1121, 544, 1196], "Dear students"),
                    _text_block("left_body", [167, 1204, 549, 1769], "Left notice\nLeft detail\nLeft detail"),
                    _text_block("left_sig", [300, 1808, 512, 1895], "English Club\nMarch 3"),
                    _text_block("mid_body", [556, 1108, 940, 1847], "Middle notice\nMiddle detail\nMiddle detail"),
                    _text_block("mid_sig", [660, 1849, 898, 1939], "Students Union\nMarch 11"),
                    _text_block("right_title", [1115, 1118, 1223, 1156], "Found", category="title"),
                    _text_block("right_body", [955, 1206, 1329, 1506], "Right notice\nRight detail\nRight detail"),
                    _text_block("right_sig", [952, 1761, 1187, 1897], "Chen Dong\nMarch 2"),
                    _text_block("tail", [145, 2046, 1134, 2085], "Tail prompt"),
                ],
            }
        ],
    }

    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(page)
    page_obj = document.pages[0]

    assert len(page_obj.zones) == 3
    assert page_obj.zones[0].col_count == 1
    assert page_obj.zones[1].col_count == 3
    assert page_obj.zones[1].region_kind == "local_parallel_text_band"
    assert page_obj.zones[1].region_id.startswith("local_parallel_")
    assert page_obj.zones[2].col_count == 1


def test_subset_spanning_visual_preserves_uncovered_left_column_flow():
    page = {
        "version": "2.0",
        "metadata": {},
        "pages": [
            {
                "page_index": 0,
                "width": 1022,
                "height": 1344,
                "blocks": [
                    _text_block("masthead", [62, 34, 479, 70], "The world this week Business", category="title"),
                    _text_block("c0_top", [58, 106, 272, 395], "c0 top"),
                    _text_block("c0_mid", [58, 409, 271, 732], "c0 mid"),
                    _text_block("c0_low", [60, 745, 269, 1017], "c0 low"),
                    _text_block("c1_top", [289, 106, 494, 304], "c1 top"),
                    _text_block("c1_title", [287, 339, 482, 375], "c1 title", category="title"),
                    _text_block("c1_mid", [289, 357, 503, 696], "c1 mid"),
                    _text_block("c1_low", [287, 711, 493, 837], "c1 low"),
                    _text_block("c2_top", [518, 109, 721, 180], "c2 top"),
                    _text_block("c2_mid", [517, 410, 728, 626], "c2 mid"),
                    _text_block("c2_low", [517, 643, 733, 837], "c2 low"),
                    _text_block("c3_top", [747, 106, 919, 144], "c3 top"),
                    _text_block("c3_mid", [746, 161, 956, 376], "c3 mid"),
                    _text_block("c3_low", [746, 404, 958, 838], "c3 low"),
                    _text_block("c0_tail", [60, 1034, 268, 1231], "left tail"),
                    _text_block("c0_end", [60, 1248, 269, 1301], "left end"),
                    _figure_block("bottom_fig", [290, 857, 956, 1302]),
                ],
            }
        ],
    }

    pipeline = RecoveryPipeline(config=RecoveryConfig(reading_order_strategy="xycutpp_hybrid"))
    document = pipeline.build_document(page)
    zones = document.pages[0].zones
    assert len(zones) == 1
    zone = zones[0]
    ids = [block.block_id for block in zone.blocks]

    assert ids.index("c0_low") < ids.index("c0_tail") < ids.index("c0_end")
    assert ids.index("c0_end") < ids.index("c1_top")
    assert ids.index("c3_low") < ids.index("bottom_fig")
    assert zone.col_count == 4
    bottom_fig = next(block for block in zone.blocks if block.block_id == "bottom_fig")
    assert bottom_fig.spanned_cols == [1, 2, 3]
