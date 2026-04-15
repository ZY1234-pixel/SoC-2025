# Changelog

## 0.5.0-dev - 2026-04-15

- Added lightweight page layout routing via `layout_profile` in the recovery pipeline.
- Added rule hit statistics and page quality metrics to pipeline page attributes.
- Added `render_plan.json` output for each sample to explain the selected render strategy.
- Expanded run/sample manifests with render-plan paths, layout profile summary, and rendering strategy statistics.
- Added nested duplicate suppression improvements for caption-family blocks such as short table numbers (`表7`) contained in longer captions.
