## Summary

This PR adds an independent geometry component under the 4-dewarp project for recovering a document's free
aspect ratio from a predicted 3D page grid. It integrates horizontal and
vertical 3D arc lengths, robustly aggregates them across rows and columns, and
reports quality diagnostics. It does not assume A4 or another fixed format.

## Changes

- Add `dewarp_geometry.estimate_aspect_from_grid`.
- Add free dynamic-canvas calculation.
- Add a model-neutral NumPy-grid CLI.
- Add UVDoc only as an optional example adapter outside the core package.
- Add metadata output and low-confidence rejection.
- Add synthetic tests for planar and cylindrical pages, input layouts,
  transformation invariance, invalid grids, and canvas sizing.

## Model

The optional example adapter uses the official UVDoc pretrained model. Third-party source and
weights are intentionally not committed; `scripts/setup_uvdoc.sh` obtains the
official repository, model, license, and citation together.

## Validation

```text
python -m unittest discover -s tests -v
Ran 6 tests ... OK
```

An end-to-end smoke test with the official UVDoc checkpoint generated a dynamic
output image and `aspect_metadata.json` successfully.

## Scope and limitations

- `confidence` is a diagnostic heuristic, not a calibrated probability.
- The 3D grid must cover the complete page and use a common scale for X/Y/Z.
- This estimates dimensionless W/H, not physical size in millimetres.
- The module has no dependency on a particular detector, segmenter, or 3D reconstruction network.
- Contour-to-3D boundary lifting is deliberately left for a follow-up PR.

## Checklist

- [x] Core algorithm is independent of OpenCV and PyTorch.
- [x] No fixed paper-format prior.
- [x] No third-party model weights added to Git.
- [x] Synthetic curved-page test included.
- [x] Invalid and low-confidence estimates are handled explicitly.
