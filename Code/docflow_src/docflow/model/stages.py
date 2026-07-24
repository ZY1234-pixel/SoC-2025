"""Canonical models for the reflow pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if hasattr(value, "tolist"):
        return _freeze(value.tolist())
    return value


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _primitive(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_primitive(item) for item in value]
    return value


def _require_unique(values: Sequence[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


@dataclass(frozen=True)
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("invalid rectangle")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "Rect":
        if len(values) != 4:
            raise ValueError("bbox must contain four values")
        return cls(*(float(value) for value in values))


@dataclass(frozen=True)
class TextEvidence:
    text: str
    confidence: float = 1.0
    polygon: Tuple[Tuple[float, float], ...] = ()


@dataclass(frozen=True)
class RecognitionItem:
    evidence_id: str
    category: str
    bbox: Rect
    model_order: float
    confidence: float = 1.0
    text_lines: Tuple[TextEvidence, ...] = ()
    image_base64: Optional[str] = None
    html: Optional[str] = None
    latex: Optional[str] = None
    raw_type: Optional[str] = None
    layout_model: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "text_lines", tuple(self.text_lines))
        object.__setattr__(self, "attributes", _freeze(self.attributes))


@dataclass(frozen=True)
class RecognitionPage:
    page_index: int
    width_px: int
    height_px: int
    items: Tuple[RecognitionItem, ...] = ()
    image_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("page dimensions must be positive")
        object.__setattr__(self, "items", tuple(self.items))
        _require_unique([item.evidence_id for item in self.items], "evidence id")
        orders = [item.model_order for item in self.items]
        if orders != sorted(orders):
            raise ValueError("recognition items must preserve Model Order")


@dataclass(frozen=True)
class RecognitionEvidence:
    pages: Tuple[RecognitionPage, ...]
    source_file: Optional[str] = None
    engine: str = "PP-DocLayoutV3"
    version: str = "3.0"
    stage: str = "recognition_evidence"

    def __post_init__(self) -> None:
        object.__setattr__(self, "pages", tuple(self.pages))
        _require_unique([str(page.page_index) for page in self.pages], "page index")

    def to_dict(self) -> dict:
        return _primitive(self)


@dataclass(frozen=True)
class AnalysisDiagnostic:
    code: str
    message: str
    evidence_ids: Tuple[str, ...] = ()
    confidence: Optional[float] = None


@dataclass(frozen=True)
class TypographicRole:
    role_id: str
    font_family: str
    western_font_family: str
    font_size_pt: float
    line_spacing: float
    bold: bool = False
    italic: bool = False
    color: str = "#000000"
    space_before_pt: float = 0.0
    space_after_pt: float = 0.0


@dataclass(frozen=True)
class SemanticElement:
    element_id: str
    kind: str
    bbox: Rect
    model_order: float
    source_ids: Tuple[str, ...]
    text: str = ""
    role_id: Optional[str] = None
    child_ids: Tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.source_ids:
            raise ValueError("semantic elements require Recognition Evidence provenance")
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        object.__setattr__(self, "child_ids", tuple(self.child_ids))
        object.__setattr__(self, "payload", _freeze(self.payload))


@dataclass(frozen=True)
class AnalysisPage:
    page_index: int
    width_px: int
    height_px: int
    elements: Tuple[SemanticElement, ...]
    diagnostics: Tuple[AnalysisDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "elements", tuple(self.elements))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        _require_unique([item.element_id for item in self.elements], "semantic element id")
        orders = [item.model_order for item in self.elements]
        if orders != sorted(orders):
            raise ValueError("semantic elements must preserve Model Order")


@dataclass(frozen=True)
class DocumentAnalysis:
    pages: Tuple[AnalysisPage, ...]
    roles: Tuple[TypographicRole, ...] = ()
    source_file: Optional[str] = None
    version: str = "3.0"
    stage: str = "document_analysis"

    def __post_init__(self) -> None:
        object.__setattr__(self, "pages", tuple(self.pages))
        object.__setattr__(self, "roles", tuple(self.roles))
        _require_unique([role.role_id for role in self.roles], "typographic role id")

    def to_dict(self) -> dict:
        return _primitive(self)


class FlowKind(str, Enum):
    SINGLE = "single_flow"
    SEQUENTIAL_COLUMNS = "sequential_columns"
    GRID = "grid_flow"


@dataclass(frozen=True)
class PageGeometry:
    width_pt: float
    height_pt: float
    margin_top_pt: float
    margin_right_pt: float
    margin_bottom_pt: float
    margin_left_pt: float

    def __post_init__(self) -> None:
        if self.width_pt <= 0 or self.height_pt <= 0:
            raise ValueError("page dimensions must be positive")
        if min(self.margin_top_pt, self.margin_right_pt, self.margin_bottom_pt, self.margin_left_pt) < 0:
            raise ValueError("page margins must not be negative")
        if self.margin_left_pt + self.margin_right_pt >= self.width_pt:
            raise ValueError("horizontal margins consume the page")
        if self.margin_top_pt + self.margin_bottom_pt >= self.height_pt:
            raise ValueError("vertical margins consume the page")


@dataclass(frozen=True)
class GridCell:
    row: int
    column: int
    element_ids: Tuple[str, ...]
    row_span: int = 1
    column_span: int = 1

    def __post_init__(self) -> None:
        if self.row < 0 or self.column < 0 or self.row_span < 1 or self.column_span < 1:
            raise ValueError("invalid grid cell")
        object.__setattr__(self, "element_ids", tuple(self.element_ids))


@dataclass(frozen=True)
class FlowSection:
    section_id: str
    kind: FlowKind
    element_ids: Tuple[str, ...]
    column_widths_pt: Tuple[float, ...] = ()
    gutter_pt: float = 0.0
    grid_cells: Tuple[GridCell, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "element_ids", tuple(self.element_ids))
        object.__setattr__(self, "column_widths_pt", tuple(float(value) for value in self.column_widths_pt))
        object.__setattr__(self, "grid_cells", tuple(self.grid_cells))
        if self.gutter_pt < 0 or any(width <= 0 for width in self.column_widths_pt):
            raise ValueError("invalid flow section dimensions")
        if self.kind == FlowKind.SEQUENTIAL_COLUMNS and len(self.column_widths_pt) < 2:
            raise ValueError("sequential columns require at least two columns")
        if self.kind == FlowKind.GRID and not self.grid_cells:
            raise ValueError("grid flow requires cells")


@dataclass(frozen=True)
class PlannedElement:
    element_id: str
    kind: str
    role_id: Optional[str] = None
    text: str = ""
    payload: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(self.payload))


@dataclass(frozen=True)
class ReflowPagePlan:
    page_index: int
    geometry: PageGeometry
    elements: Tuple[PlannedElement, ...]
    sections: Tuple[FlowSection, ...]
    fit_scale: float
    header_element_ids: Tuple[str, ...] = ()
    footer_element_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.fit_scale <= 0 or self.fit_scale > 1:
            raise ValueError("fit scale must be in (0, 1]")
        object.__setattr__(self, "elements", tuple(self.elements))
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "header_element_ids", tuple(self.header_element_ids))
        object.__setattr__(self, "footer_element_ids", tuple(self.footer_element_ids))
        element_ids = [item.element_id for item in self.elements]
        _require_unique(element_ids, "planned element id")
        known = set(element_ids)
        placed = [item for section in self.sections for item in section.element_ids]
        if not set(placed).issubset(known):
            raise ValueError("flow section references an unknown element")
        if len(placed) != len(set(placed)):
            raise ValueError("an element may appear in only one flow section")
        if not set(self.header_element_ids + self.footer_element_ids).issubset(known):
            raise ValueError("page furniture references an unknown element")


@dataclass(frozen=True)
class ReflowLayoutPlan:
    pages: Tuple[ReflowPagePlan, ...]
    roles: Tuple[TypographicRole, ...] = ()
    source_file: Optional[str] = None
    word_safety_factor: float = 1.0
    version: str = "3.0"
    stage: str = "reflow_layout_plan"

    def __post_init__(self) -> None:
        if self.word_safety_factor <= 0 or self.word_safety_factor > 1:
            raise ValueError("Word safety factor must be in (0, 1]")
        object.__setattr__(self, "pages", tuple(self.pages))
        object.__setattr__(self, "roles", tuple(self.roles))

    def to_dict(self) -> dict:
        return _primitive(self)
