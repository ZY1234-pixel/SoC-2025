---
status: accepted
---

# Use a staged deterministic reflow pipeline

The pipeline has one canonical model per stage: immutable Recognition Evidence, resolved Document Analysis with provenance and Analysis Diagnostics, a complete Reflow Layout Plan, and mechanical DOCX emission. Adapters do not delete evidence; PP-DocLayoutV3 Model Order is authoritative; a Page Relation Graph resolves semantic composites; ParagraphGroups discard OCR visual line endings; and document-level Typographic Roles replace block-specific font corrections.

The Page Layout Tree assigns ordered elements to Single Flow, Sequential Columns, or Grid Flow without reordering them. The renderer performs no semantic, topology, style, or fit inference, low confidence follows the same best-effort path with diagnostics, and legacy Block, Zone, profile, repair, and strategy-switch paths are removed rather than retained behind fallbacks.
