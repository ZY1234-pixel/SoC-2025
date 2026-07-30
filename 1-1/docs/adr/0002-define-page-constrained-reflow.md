---
status: accepted
---

# Define page-constrained reflow

A Reflow Document preserves all content, semantic structure, and PP-DocLayoutV3 Model Order while mapping each source page to exactly one Word page. Source aspect ratio and semantic content bounds define Page Geometry; Page Furniture uses native unlinked headers and footers, tables remain native and editable, figure-internal text stays in the figure crop, equations use a faithful image with editable numbering, and only Structural Decoration is retained.

Page Fit is one uniform per-page scale calculated before a single render pass, with no font floor or render-measure-regenerate loop. Microsoft Word Desktop is the pagination reference; the fixed SimSun, SimHei, KaiTi, FangSong, and Times New Roman catalog plus one offline-calibrated Word Safety Factor favor over-reduction over exceeding the Page Budget.
