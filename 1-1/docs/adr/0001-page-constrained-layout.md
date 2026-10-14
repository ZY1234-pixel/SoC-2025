# Page-constrained layout uses soft boxes and one page scale

DOCX conversion must preserve the source page count, prevent clipping and overlap, and keep the same typographic roles consistent across a page. Source Boxes are noisy, so they are soft placement evidence rather than hard text dimensions. The planner therefore uses column/container width for text, role-level typography, real fixed font metrics, gap compression before scaling, and one Page Fit scale per page. If the readable lower bounds cannot satisfy the page constraint, the sample fails with diagnostics instead of silently producing an unreadable document.

Considered alternatives were per-element bbox fitting, which caused squeezed text and inconsistent fonts, and whole-document scaling, which made unrelated pages unnecessarily small. Per-column scaling was rejected because it makes the same role differ across columns on one page.
