# DocFlow Layout

This context turns recognized document elements into a page-constrained, editable document layout.

## Language

**Page Fit**:
The document-level process that chooses common scaling and spacing adjustments so all elements remain on the source page without clipping, overlap, or overflow.
_Avoid_: element fit, per-box shrink

**Source Box**:
The geometry reported by document analysis for an element. It is evidence about preferred placement and size, not a command to crop or independently resize that element.
_Avoid_: hard box, target box

**Typographic Role**:
A document-wide semantic style class that fixes the base font family, size, weight, color, and line-spacing policy for matching text.
_Avoid_: per-element style, local font size

**Character Metrics**:
A fixed width table generated from the supported font files and used to estimate line wrapping without reading fonts at runtime.
_Avoid_: system-font lookup, character-count estimate

**Preferred Slot**:
The vertical space indicated by a Source Box for an element before Page Fit and gap adjustment are applied.
_Avoid_: fixed height, clipping height

**Cell Text Flow**:
Per-cell processing of OCR lines through the same paragraph grouping, line joining or breaking, alignment, and typographic layout rules used for body text while preserving the table cell boundary.
_Avoid_: flattened table text, table content as page flow

**Cell Paragraph Break**:
A boundary between logical paragraphs inside one table cell, inferred primarily from stable changes in vertical pitch, alignment, indentation, or typography; language cues are only tie-breakers.
_Avoid_: punctuation-only break, blank-line-only break

**Cell Paragraph Alignment**:
One horizontal alignment assigned to each logical paragraph inside a table cell; a cell may contain several paragraphs with different alignments.
_Avoid_: one alignment per cell, mixed alignment within one paragraph

**Cell Paragraph Continuation**:
Adjacent lines inside one cell remain in the same logical paragraph only when their alignment matches, their starting position is compatible, and their vertical pitch stays within the cell's normal line-pitch tolerance.
_Avoid_: merge solely because lines are adjacent, preserve every OCR line as a paragraph

**Cell Paragraph Reflow**:
Wrapped OCR lines are joined for normal renderer wrapping; source line breaks are preserved when a centered or right-aligned display block has multiple short, independent lines.
_Avoid_: hard-break every OCR line, collapse independent right-aligned lines

**Cell Content Block**:
An ordered block inside a table cell, normally a logical text paragraph with its own alignment and optional visual or formula content.
_Avoid_: flattened cell text, parallel arrays keyed only by cell position

**Cell Paragraph Typography**:
Each logical paragraph inherits the cell's baseline typography and may override size, weight, or spacing only when explicit style evidence supports the difference.
_Avoid_: force every paragraph to one uniform style, infer font size independently from every OCR line

**Cell List Item**:
A list-marker line in a table cell starts a new logical paragraph; following non-marker lines may continue that item according to the normal continuation rules.
_Avoid_: merge list items into the surrounding body paragraph, treat every wrapped line as a new item

**Cell Alignment Fallback**:
When alignment evidence is insufficient, header-like and corner cells default to center alignment and body cells default to left alignment.
_Avoid_: guessing center or right from isolated short lines

**Cell Content Order**:
Text paragraphs, formulas, and images inside one table cell retain their original vertical order.
_Avoid_: grouping all visuals before text, reordering by content type

**Cell Paragraph Spacing**:
Extra vertical separation between logical paragraphs is inferred from the source geometry relative to the cell's normal line pitch and expressed as paragraph spacing.
_Avoid_: blank paragraphs, uniform paragraph gaps
