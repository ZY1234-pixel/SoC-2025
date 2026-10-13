# 4-dewarp project integration notes

Copy these items into the target repository:

```text
dewarp_geometry/
tests/test_estimator.py
estimate_grid_aspect.py
examples/uvdoc_dynamic_canvas.py
scripts/setup_uvdoc.sh
README.md (merge the relevant section if the project already has one)
```

Place the package at the `4-dewarp` project level, alongside individual model
implementations. Any pipeline that can provide a 3D page grid may call it:

```python
estimate = estimate_aspect_from_grid(grid3d, layout="chw")
if estimate.valid and estimate.confidence >= args.min_aspect_confidence:
    output_size = canvas_from_aspect(
        estimate.aspect_ratio,
        long_edge=args.output_long_edge,
    )
else:
    # Keep the existing output mode or skip ratio correction. Do not silently
    # force A4, because that would hide an uncertain geometric estimate.
    output_size = existing_output_size
```

The caller owns model inference, fallback policy and image resampling. Recommended
project-level CLI additions are:

```text
--aspect-mode {existing,3d-arc}
--min-aspect-confidence 0.15
--output-long-edge 1400
--save-aspect-metadata
```

For a first PR, leave `--aspect-mode existing` as the default. Switching the
project default can be discussed after benchmark results are available.
