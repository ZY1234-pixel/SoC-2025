---
status: accepted
---

# Separate outputs and validation

DOCX consumes the Reflow Layout Plan, Markdown consumes Document Analysis, and PDF is exported from the final DOCX instead of maintaining another page-layout implementation. The main Code/test.py workflow and dataset-to-test-result outputs remain stable while internal APIs and intermediate JSON may change.

Production performs one planning pass and one render. Structural refactors run the full dataset in pocr39, convert final DOCX pages to images for side-by-side Visual Review, and retain automated checks for content integrity, native editable tables, Model Order, valid DOCX, and Page Budget; offline Windows evaluation may open each final file once in Microsoft Word for pagination and safety-factor calibration without regenerating it.
