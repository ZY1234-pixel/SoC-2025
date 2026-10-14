# Table cells preserve logical paragraphs within a cell

Table cell content is represented as ordered content blocks and processed through the normal body-text pipeline inside the cell boundary. OCR lines are grouped into logical paragraphs using alignment, indentation, vertical pitch, list markers, and typography; each paragraph gets its own alignment and source-derived paragraph spacing. Wrapped lines are joined and reflowed by the renderer, while short independent lines in centered or right-aligned display blocks keep their source line breaks. Text, formulas, and images retain their original vertical order.

This was chosen over preserving every OCR line and over assigning one alignment to the entire cell. The former prevents normal reflow; the latter cannot represent cells containing introductory text, body text, and right-aligned sign-off or date lines.
