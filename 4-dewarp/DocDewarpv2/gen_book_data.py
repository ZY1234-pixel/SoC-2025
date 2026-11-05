"""
gen_book_data.py
===============

Synthesize a *double-page book* dataset that has the same directory / file
composition as the UVDoc benchmark (``data/UVDoc``), but where every sample is an
**open book spread** (two single pages side by side) instead of one page.

This version is driven by the *real book deformation fields* shipped with the
project under ``data/real_book/grid2d`` (one deformation per ``.mat`` file), so
the warped geometry of the synthesized books matches the real open-book captures
instead of being re-derived from the single-page benchmark.

Core idea (how a real open book is photographed)
------------------------------------------------
1. Take TWO flat page textures from ``textures/`` (the benchmark's flat single
   pages) and splice them into ONE continuous flat spread ``F`` (landscape,
   height = page height, width = 2*page_width + gutter).  The gutter column in
   the middle is the book spine.
2. Pick one real-book deformation field ``G`` from ``data/real_book/grid2d``.
   ``G`` is a regular mesh of node positions (2, NX, NY) in the *photographed*
   book image; think of it as: flat-grid node (i, j) is seen at pixel
   ``G[:, i, j]``.  We resample ``G`` to a fixed canonical mesh size so that the
   whole dataset shares one geometry resolution (the loader assumes a uniform
   grid size per dataset).
3. Warp ``F`` through that single mesh with ``skimage`` PiecewiseAffineTransform
   -- i.e. we stamp the flat spread onto the real-book node positions, so the two
   pages are warped *together* and stay connected through the spine.  The same
   transform is applied to a coordinate image to obtain the per-pixel backward
   map ``uvmap`` (canvas -> flat), and to a mask image to obtain ``seg``.
4. Composite the warped book onto a larger background (with a soft drop shadow),
   exactly like a photographed book on a surface.

Geometry annotations cover the whole spread and are written in the exact
conventions the UVDoc tooling expects:
    - ``grid2d`` : (2, NFX, NFY) -- node positions in PIXEL coords of ``img``
                   (read by the loader via h5py; normalized in ``crop_tight``).
    - ``grid3d`` : (3, NFX, NFY) -- a plausible open-book 3D (tent along spine).
    - ``uvmap``  : (Hc, Wc, 2)  -- per-pixel backward map, normalized [0, 1]
                   (read by the metric via hdf5storage as ``uv``).
    - ``seg``    : (Hc, Wc)     -- document mask.

``img`` / ``img_geom`` keep the (reused, real) background and the flat spread as
the un-shaded ground truth, so the dataset loads with ``UVDocDataset`` (which
infers the grid size from the .mat file).

Output layout (identical schema to ``data/UVDoc``)
--------------------------------------------------
    data/book/
        img/  img_geom/  texture_sample/  warped_textures/
        grid2d/  grid3d/  uvmap/  seg/  wc/  metadata_sample/  metadata_geom/
        textures/  split.json

Usage
-----
    python gen_book_data.py \
        --uvdoc_path ./data/UVDoc --real_book_path ./data/real_book \
        --out_path ./data/book --num_samples 50 --seed 0
"""

import argparse
import json
import os
import random
import shutil

import cv2
import h5py
import hdf5storage as h5
import numpy as np
from scipy.interpolate import griddata


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def load_mat_arr(path, key):
    with h5py.File(path, "r") as f:
        return np.array(f[key])


def fit_page(tex, page_h, page_w, pad_color=255):
    """Resize a texture to height ``page_h`` keeping aspect ratio, then
    center-crop / pad (white) to exactly (page_h, page_w)."""
    h, w = tex.shape[:2]
    scale = page_h / float(h)
    new_w = max(1, int(round(w * scale)))
    tex = cv2.resize(tex, (new_w, page_h), interpolation=cv2.INTER_AREA)
    if new_w > page_w:
        start = (new_w - page_w) // 2
        tex = tex[:, start:start + page_w]
    elif new_w < page_w:
        pad = page_w - new_w
        left = pad // 2
        tex = np.pad(tex, ((0, 0), (left, pad - left), (0, 0)),
                     mode="constant", constant_values=pad_color)
    return tex


def build_flat_spread(tex1, tex2, page_h, page_w, gutter=30,
                      spine_edge=210, spine_center=175):
    """Splice two single pages into ONE continuous flat spread (landscape).

    Returns float [0,1] array (page_h, 2*page_w + gutter, 3).  The thin gutter
    column in the middle is a light-grey fold (slightly darker at the very
    center) so the two pages read as one physical sheet joined at the spine.
    """
    p1 = fit_page(tex1, page_h, page_w)
    p2 = fit_page(tex2, page_h, page_w)
    if gutter <= 1:
        prof = np.array([spine_center], dtype=np.float32)
    else:
        half = max(1, gutter // 2)
        prof = np.concatenate([
            np.linspace(spine_edge, spine_center, half, dtype=np.float32),
            np.linspace(spine_center, spine_edge, gutter - half, dtype=np.float32),
        ])
        if len(prof) < gutter:
            prof = np.pad(prof, (0, gutter - len(prof)), mode="edge")
    spine = np.broadcast_to(prof.reshape(1, gutter, 1), (page_h, gutter, 3)).copy()
    spine = spine.astype(p1.dtype)
    F = np.concatenate([p1, spine, p2], axis=1)
    return F.astype(np.float32) / 255.0


def resample_grid_to_canonical(g, nf, nr):
    """Resample a (2, NX, NY) node grid to a fixed (2, nf, nr) canonical mesh.

    Resampling the node *positions* keeps the deformation shape while making the
    whole dataset share one geometry resolution (required by ``UVDocDataset``,
    which assumes a uniform grid size)."""
    g = np.nan_to_num(g.astype(np.float64))                    # (2, NX, NY)
    # view as an (NY, NX, 2) image so cv2 resizes the NX->width, NY->height axes
    arr = np.transpose(g, (2, 1, 0)).astype(np.float32)        # (NY, NX, 2)
    arr = cv2.resize(arr, (nf, nr), interpolation=cv2.INTER_LINEAR)   # (nr, nf, 2)
    # back to (2, nf, nr): axis order must be (2, 1, 0), NOT (2, 0, 1)
    return np.transpose(arr, (2, 1, 0)).astype(np.float64)     # (2, nf, nr)


def build_canonical_3d(nf, nr, z_amp=0.35):
    """A plausible open-book 3D: planar (x, y) over the spread plus a tent
    along the spine (pages rise away from the binding)."""
    a = np.arange(nf, dtype=np.float64) / max(1, nf - 1)       # across width
    b = np.arange(nr, dtype=np.float64) / max(1, nr - 1)       # down height
    U, V = np.meshgrid(a, b, indexing="ij")                   # (nr, nf)
    x = U
    y = V
    z = z_amp * (2.0 * np.abs(U - 0.5)) + 0.1 * np.sin(np.pi * V)
    return np.stack([x, y, z], axis=0).astype(np.float64)      # (3, nf, nr)


def make_background(H, W, value, noise, rng):
    """A neutral surface: flat colour + faint noise + subtle vignette."""
    bg = np.full((H, W, 3), value, dtype=np.float32)
    bg += rng.normal(0.0, noise, (H, W, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0
    d = np.sqrt(((xx - cx) / (W / 2.0)) ** 2 + ((yy - cy) / (H / 2.0)) ** 2)
    vig = 1.0 - 0.15 * np.clip(d - 0.6, 0.0, 1.0) / 0.4
    bg *= vig[..., None]
    return np.clip(bg, 0.0, 1.0)


def make_shadow(seg, margin):
    """Soft drop shadow band just outside the document mask."""
    k = max(3, margin // 3)
    dil = cv2.dilate(seg.astype(np.uint8), np.ones((k, k), np.uint8), iterations=2)
    band = np.clip(dil - seg.astype(np.uint8), 0, 1).astype(np.float32)
    sigma = max(4.0, margin / 3.0)
    band = cv2.GaussianBlur(band, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return band


# --------------------------------------------------------------------------- #
# Main generation
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Synthesize a double-page book dataset.")
    parser.add_argument("--uvdoc_path", default="./data/UVDoc", help="UVDoc benchmark (provides the flat page textures).")
    parser.add_argument("--real_book_path", default="./data/real_book", help="Folder with real book deformation fields (grid2d/*.mat).")
    parser.add_argument("--out_path", default="./data/book", help="Output dataset path.")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    # flat spread geometry
    parser.add_argument("--page_w", type=int, default=512, help="Width of one flat page.")
    parser.add_argument("--page_h", type=int, default=720, help="Height of one flat page.")
    parser.add_argument("--gutter", type=int, default=30, help="Pixel width of the book spine/gutter.")

    # deformation mesh resolution (canonical, shared by all samples)
    parser.add_argument("--grid_cols", type=int, default=96, help="Canonical mesh width (across spread).")
    parser.add_argument("--grid_rows", type=int, default=64, help="Canonical mesh height (down spread).")

    # canvas / background
    parser.add_argument("--book_target", type=int, default=1200, help="Rough target size (px) of the warped book's longest side.")
    parser.add_argument("--margin", type=int, default=100, help="Background border around the spread.")
    parser.add_argument("--bg_value", type=float, default=0.85, help="Background grey level [0,1].")
    parser.add_argument("--bg_noise", type=float, default=0.025, help="Std-dev of background noise.")
    parser.add_argument("--tex_dir", default="textures", help="Sub-folder of uvdoc_path holding flat pages.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = args.uvdoc_path
    rbook = args.real_book_path
    out = args.out_path

    # ---- gather flat page textures -------------------------------------- #
    tex_dir = os.path.join(root, args.tex_dir)
    tex_names = sorted(f for f in os.listdir(tex_dir) if f.lower().endswith(".png"))
    if len(tex_names) < 2:
        raise RuntimeError("Need at least two textures to build a double page.")

    # ---- gather real book deformation fields ---------------------------- #
    rbook_grid_dir = os.path.join(rbook, "grid2d")
    rbook_files = sorted(f for f in os.listdir(rbook_grid_dir) if f.endswith(".mat"))
    if not rbook_files:
        raise RuntimeError(f"No deformation fields in {rbook_grid_dir}")
    geom_names = [os.path.splitext(f)[0] for f in rbook_files]

    NFX, NFY = args.grid_cols, args.grid_rows
    page_w, page_h, gutter = args.page_w, args.page_h, args.gutter
    WF = 2 * page_w + gutter
    HF = page_h

    subdirs = ["img", "img_geom", "texture_sample", "warped_textures",
               "grid2d", "grid3d", "uvmap", "seg", "wc",
               "metadata_sample", "metadata_geom", "textures"]
    for d in subdirs:
        os.makedirs(os.path.join(out, d), exist_ok=True)

    # cache the (resampled) canonical deformation per geom
    geom_cache = {}
    used_textures = set()
    used_geoms = set()
    Hc = Wc = 0  # (re)assigned per sample inside the loop; used by the summary

    for i in range(args.num_samples):
        geom = geom_names[i % len(geom_names)]

        # ---- load + resample the real-book deformation (canonical mesh) -- #
        if geom not in geom_cache:
            g_raw = load_mat_arr(os.path.join(rbook_grid_dir, f"{geom}.mat"), "grid2d")
            g_can = resample_grid_to_canonical(g_raw, NFX, NFY)        # (2, NFX, NFY)
            geom_cache[geom] = g_can
        g_can = geom_cache[geom]                                       # (2, NFX, NFY) pixel coords (real book)

        # normalize the deformation into OUR canvas (keep shape, uniform scale)
        xs = g_can[0]
        ys = g_can[1]
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        bw, bh = max(1e-6, xmax - xmin), max(1e-6, ymax - ymin)
        s = args.book_target / max(bw, bh)
        ox = args.margin
        oy = args.margin
        # dest node positions (canvas pixel coords), shape (2, NFX, NFY)
        dest = np.stack([
            (xs - xmin) * s + ox,
            (ys - ymin) * s + oy,
        ], axis=0).astype(np.float64)

        Hc = int(round(bh * s)) + 2 * args.margin
        Wc = int(round(bw * s)) + 2 * args.margin
        # guarantee the flat spread F always fits (some deformations are narrow)
        Hc = max(Hc, HF + 2 * args.margin)
        Wc = max(Wc, WF + 2 * args.margin)

        # ---- build ONE continuous flat spread (two pages joined) --------- #
        t1, t2 = rng.sample(tex_names, 2)
        tex1 = cv2.imread(os.path.join(tex_dir, t1))
        tex2 = cv2.imread(os.path.join(tex_dir, t2))
        assert tex1 is not None, f"missing texture {t1}"
        assert tex2 is not None, f"missing texture {t2}"
        tex1 = cv2.cvtColor(tex1, cv2.COLOR_BGR2RGB)
        tex2 = cv2.cvtColor(tex2, cv2.COLOR_BGR2RGB)
        used_textures.update([t1, t2])
        F = build_flat_spread(tex1, tex2, page_h, page_w, gutter=gutter)   # (HF, WF, 3) [0,1]

        # ---- mesh warp F through the real-book deformation --------------- #
        # src node positions: regular grid over F
        ai = np.arange(NFX) / max(1, NFX - 1)
        bi = np.arange(NFY) / max(1, NFY - 1)
        A, B = np.meshgrid(ai, bi, indexing="ij")                          # (NFY, NFX)
        src_pts = np.stack([A.reshape(-1) * (WF - 1),
                            B.reshape(-1) * (HF - 1)], axis=1)             # (NFX*NFY, 2)
        dst_pts = np.stack([dest[0].reshape(-1), dest[1].reshape(-1)], axis=1)

        # dense backward map (canvas pixel -> flat position) via the node mesh.
        # griddata linearly interpolates the node correspondences; pixels outside
        # the mesh convex hull become NaN and are treated as background.
        yy, xx = np.mgrid[0:Hc, 0:Wc]
        pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        mapxy = griddata(dst_pts, src_pts, pts, method="linear")           # (Hc*Wc, 2)
        valid = ~np.isnan(mapxy).any(axis=1)
        mapx = np.where(valid, mapxy[:, 0], 0.0).reshape(Hc, Wc).astype(np.float32)
        mapy = np.where(valid, mapxy[:, 1], 0.0).reshape(Hc, Wc).astype(np.float32)
        seg = valid.reshape(Hc, Wc)

        coord_img = np.zeros((HF, WF, 2), dtype=np.float32)
        coord_img[:, :, 0] = (np.arange(WF) / max(1, WF - 1)).astype(np.float32)[None, :]
        coord_img[:, :, 1] = (np.arange(HF) / max(1, HF - 1)).astype(np.float32)[:, None]

        # sample F (color) and coord_img (normalized flat coords) through the map
        warped = cv2.remap(F, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        uvmap = cv2.remap(coord_img, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        warped = np.clip(warped, 0.0, 1.0)
        warped[~seg] = 0.0                                         # keep background clean
        uvmap[~seg] = np.nan                                       # invalid outside doc

        # ---- background plate + composite ------------------------------- #
        bg = make_background(Hc, Wc, args.bg_value, args.bg_noise,
                             np.random.default_rng(args.seed + i))
        shadow = make_shadow(seg, args.margin)
        img = bg * (1.0 - 0.35 * shadow[..., None])                # shadow darkens bg
        img[seg] = warped[seg]                                     # book on top
        img = np.clip(img, 0.0, 1.0)

        # flat ground truth placed in the same canvas (un-warped) -> img_geom
        bg2 = make_background(Hc, Wc, args.bg_value, args.bg_noise,
                             np.random.default_rng(args.seed + i + 100000))
        img_geom = bg2.copy()
        x0 = (Wc - WF) // 2
        y0 = (Hc - HF) // 2
        img_geom[y0:y0 + HF, x0:x0 + WF] = F

        # warped content only (black background) -> warped_textures
        warped_only = warped.copy()

        # ---- save image-like outputs ------------------------------------ #
        sample_id = f"{i:05d}"
        cv2.imwrite(os.path.join(out, "img", f"{sample_id}.png"),
                    (img * 255).astype(np.uint8)[..., ::-1])
        cv2.imwrite(os.path.join(out, "img_geom", f"{geom}.png"),
                    (img_geom * 255).astype(np.uint8)[..., ::-1])
        cv2.imwrite(os.path.join(out, "warped_textures", f"{sample_id}.png"),
                    (warped_only * 255).astype(np.uint8)[..., ::-1])
        cv2.imwrite(os.path.join(out, "texture_sample", f"{sample_id}.png"),
                    (F * 255).astype(np.uint8)[..., ::-1])

        # ---- composite geometry (generated once per geom) --------------- #
        if geom not in used_geoms:
            used_geoms.add(geom)
            # grid2d: node positions in pixel coords of the saved image
            with h5py.File(os.path.join(out, "grid2d", f"{geom}.mat"), "w") as f:
                f.create_dataset("grid2d", data=dest.astype(np.float64))

            # grid3d: plausible open-book 3D
            g3 = build_canonical_3d(NFX, NFY)
            with h5py.File(os.path.join(out, "grid3d", f"{geom}.mat"), "w") as f:
                f.create_dataset("grid3d", data=g3)

            # uvmap: per-pixel backward map, normalized [0,1] (metric reads it via
            # hdf5storage). Keep NaN outside the document so the metric masks it,
            # exactly like the benchmark.
            uv = uvmap.astype(np.float64)
            h5.savemat(os.path.join(out, "uvmap", f"{geom}.mat"), {"uv": uv})

            # seg: document mask
            with h5py.File(os.path.join(out, "seg", f"{geom}.mat"), "w") as f:
                f.create_dataset("seg", data=seg.astype(np.uint8))

            # metadata_geom
            mj = {
                "is_book_spread": True,
                "real_book_source": f"{geom}.mat",
                "canvas": {"height": Hc, "width": Wc},
                "grid": {"cols": NFX, "rows": NFY},
                "spread": {"width": WF, "height": HF, "gutter": gutter},
                "warp_scale": float(s), "warp_offset": [ox, oy],
            }
            with open(os.path.join(out, "metadata_geom", f"{geom}.json"), "w") as f:
                json.dump(mj, f, indent=2)

        # ---- metadata_sample ------------------------------------------- #
        meta = {
            "geom_name": geom,
            "texture_name": [t1, t2],
            "background_name": "synthetic_gray",
            "sample_id": sample_id,
            "page_size": [page_w, page_h],
        }
        with open(os.path.join(out, "metadata_sample", f"{sample_id}.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ---- textures pool + split.json ------------------------------------ #
    for t in sorted(used_textures):
        shutil.copy(os.path.join(tex_dir, t), os.path.join(out, "textures", t))
    split = {
        "textures": sorted(used_textures),
        "backgrounds": ["synthetic_gray"],
        "geoms": sorted(used_geoms),
        "note": (
            "Double-page book dataset. Each sample splices two flat single-page "
            "textures (from UVDoc's textures/) into one flat spread and warps it "
            "with a real-book deformation field (data/real_book/grid2d). grid2d / "
            "grid3d are (2,NFX,NFY)/(3,NFX,NFY) in pixel coords of img; uvmap is "
            "(Hc,Wc,2) in [0,1]; seg is (Hc,Wc). wc/ is intentionally empty."
        ),
    }
    with open(os.path.join(out, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

    print(f"Generated {args.num_samples} book samples into {out}")
    print(f"  reused geoms : {len(used_geoms)} (canonical mesh {NFX}x{NFY})")
    print(f"  used textures: {len(used_textures)}")
    print(f"  canvas size  : {Wc} x {Hc}")
    print("  directories  :", ", ".join(subdirs))


if __name__ == "__main__":
    main()
