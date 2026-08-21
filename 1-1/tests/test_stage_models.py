from __future__ import annotations

import pytest

from docflow.model.stages import (
    AnalysisPage,
    DocumentAnalysis,
    FlowKind,
    FlowSection,
    PageGeometry,
    PlannedElement,
    RecognitionEvidence,
    RecognitionItem,
    RecognitionPage,
    Rect,
    ReflowLayoutPlan,
    ReflowPagePlan,
    SemanticElement,
)


def test_stage_models_preserve_order_provenance_and_immutability() -> None:
    first = RecognitionItem("raw-1", "title", Rect(0, 0, 100, 20), 1.0, attributes={"nested": [1]})
    second = RecognitionItem("raw-2", "text", Rect(0, 30, 100, 80), 2.0)
    evidence = RecognitionEvidence((RecognitionPage(0, 100, 100, (first, second)),))

    with pytest.raises(TypeError):
        first.attributes["new"] = True

    elements = (
        SemanticElement("heading", "title", first.bbox, 1.0, ("raw-1",), text="Title"),
        SemanticElement("body", "paragraph", second.bbox, 2.0, ("raw-2",), text="Body"),
    )
    analysis = DocumentAnalysis((AnalysisPage(0, 100, 100, elements),))
    section = FlowSection("main", FlowKind.SINGLE, ("heading", "body"))
    plan = ReflowLayoutPlan(
        (
            ReflowPagePlan(
                0,
                PageGeometry(595, 842, 36, 36, 36, 36),
                tuple(PlannedElement(item.element_id, item.kind, text=item.text) for item in elements),
                (section,),
                0.9,
            ),
        ),
        word_safety_factor=0.96,
    )

    assert [item["model_order"] for item in evidence.to_dict()["pages"][0]["items"]] == [1.0, 2.0]
    assert analysis.pages[0].elements[1].source_ids == ("raw-2",)
    assert plan.to_dict()["pages"][0]["sections"][0]["kind"] == "single_flow"


def test_recognition_page_rejects_reordered_model_output() -> None:
    items = (
        RecognitionItem("raw-2", "text", Rect(0, 30, 100, 80), 2.0),
        RecognitionItem("raw-1", "title", Rect(0, 0, 100, 20), 1.0),
    )
    with pytest.raises(ValueError, match="Model Order"):
        RecognitionPage(0, 100, 100, items)
