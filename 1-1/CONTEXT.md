# Document Reconstruction

This context describes how source documents are understood and reconstructed into editable output documents.

## Language

**Source Document**:
An image or PDF whose visible content, structure, and reading order are to be recovered.

**Recognition Evidence**:
The immutable set of detected regions, recognized text, geometry, confidence, and model metadata produced from a Source Document.
_Avoid_: Cleaned blocks, final content

**Model Order**:
The authoritative page reading order emitted by PP-DocLayoutV3 and preserved through analysis, planning, and rendering.
_Avoid_: Geometric re-sort, renderer order

**Page Layout Tree**:
A spatial containment hierarchy that assigns Model-ordered Semantic Elements to Flow Sections without changing their reading order.
_Avoid_: Reading-order tree, document-type template

**Page Geometry**:
The normalized Word page rectangle and semantic content frame derived directly from a Source Document page's aspect ratio and content boundaries.
_Avoid_: Guessed paper type, document-profile margins

**Page Relation Graph**:
A scored set of candidate semantic relationships between Recognition Evidence on one source page; it preserves PP-DocLayoutV3 reading order rather than replacing it.
_Avoid_: Reading-order sorter, ordered cleanup rules

**Document Analysis**:
A resolved representation of the source document's content, geometry, and structural relationships, with provenance back to Recognition Evidence and reasons for rejected candidates.
_Avoid_: Render result, DOCX model

**Analysis Diagnostic**:
An explicit record of a low-confidence or rejected semantic decision that does not select a different processing path.
_Avoid_: Hidden fallback, silent deletion

**Semantic Element**:
A resolved unit of document meaning, such as a paragraph, heading, figure, table, or equation.
_Avoid_: Detection region, render block

**ParagraphGroup**:
A semantic paragraph assembled from recognition lines or regions; OCR visual line endings inside it do not create output line breaks.
_Avoid_: OCR line list, fixed visual lines

**Composite Element**:
A typed Semantic Element whose related parts must be understood and planned together, such as an equation with its number or a figure with its caption.
_Avoid_: Nearby blocks, arbitrary group

**TableGroup**:
An editable Composite Element containing a structured cell grid, spans, cell content, and table presentation evidence together with its caption and notes.
_Avoid_: Table image, HTML fragment, layout grid

**EquationGroup**:
A Composite Element containing a faithful equation image and an editable equation number that are aligned and planned as one unit.
_Avoid_: Plain LaTeX text, unrelated formula caption

**FigureGroup**:
A Composite Element containing a faithful figure crop and editable external captions or descriptions; text visually embedded inside the figure remains part of the image.
_Avoid_: Duplicated chart labels, detached caption

**Typographic Role**:
A document-level style shared by Semantic Elements with the same presentation purpose, such as body text, a heading level, or a caption.
_Avoid_: Per-block font correction

**Structural Decoration**:
A visual treatment attached to a Semantic Element or Flow Section, such as shading, a border, a separator, or text color.
_Avoid_: Free-floating background shape

**Reflow Layout Plan**:
A complete, output-oriented description of page content frames, flow sections, styles, and Page Fit for a Reflow Document.
_Avoid_: Document Analysis, renderer heuristics

**Flow Section**:
An ordered part of a Reflow Layout Plan using exactly one layout primitive: Single Flow, Sequential Columns, or Grid Flow.
_Avoid_: Document-type template, absolute canvas

**Grid Flow**:
A row-and-column layout whose cells contain editable flowing content; it represents parallel page regions rather than tabular data.
_Avoid_: Data table, absolute positioning

**Reflow Document**:
An editable document whose semantic structure and reading order are preserved while content is allowed to reflow within each source page's Page Budget.
_Avoid_: Visual replica, pixel-perfect document

**Page Budget**:
The requirement that each Source Document page corresponds to exactly one output page, with all of its content retained.
_Avoid_: Approximate page count

**Page Furniture**:
Editable header, footer, and page-number content attached to one source page and excluded from its body Page Fit.
_Avoid_: Body paragraph, repeated linked section

**Page Fit**:
A page-wide adjustment that preserves content and semantic structure while making a Reflow Document satisfy its Page Budget.
_Avoid_: Bounding-box restoration

**Reference Layout Engine**:
Microsoft Word Desktop, whose pagination defines whether a Reflow Document satisfies its Page Budget.
_Avoid_: LibreOffice pagination, renderer estimate as final truth

**Word Safety Factor**:
A single conservative calibration applied to every predicted Page Fit to absorb remaining differences between the planner's metrics and the Reference Layout Engine.
_Avoid_: Document-type margin, column-count exception

**Visual Review**:
A side-by-side inspection of every reconstructed page image against its Source Document page after a structural refactor.
_Avoid_: Single aggregate image score, uninspected DOCX

**Replica Document**:
An editable document whose elements retain the source geometry as closely as possible, including position and bounding-box dimensions.
_Avoid_: Reflow document, semantic document
