#!/usr/bin/env python3
"""depth_estimate: shaded render / photo -> 16-bit height map for depth2stl.

Runs Depth Anything V2 locally (GPU via the torch in the venv that runs this).
White = nearest, black = background floor. The background is taken from the
source: pixels darker than --bg are the floor; everything else is the figure,
whose depth range is stretched to [--gap, 1] so it stands proud of the floor.

Hybrid: the network gives the large-scale layering (what is in front of what)
but smooths away scales, hair and faces; those live in the render's own
shading. --detail adds the high-pass of the render luminance back on top.
--base-file uses an existing depth map (e.g. one Gemini converted) as the
layering instead of running the network.

  depth_estimate.py render.png out.png [--model large|base|small] [--res 1036]
                    [--detail 0.5 --radius 12] [--base-file depth.png]
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

MODELS = {"small": "depth-anything/Depth-Anything-V2-Small-hf",
          "base": "depth-anything/Depth-Anything-V2-Base-hf",
          "large": "depth-anything/Depth-Anything-V2-Large-hf"}


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth2stl import shading_detail  # noqa: E402  (same folder, numpy-only)


def estimate(img, model="large", res=1036, device=None):
    """Relative depth (bigger = nearer) at the image's own resolution, float32."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForDepthEstimation
    name = MODELS[model]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()
    # preprocessing done by hand (the HF image processor drags in torchvision):
    # fit inside res x res keeping aspect, dims multiple of 14, ImageNet normalise
    w, h = img.size
    k = res / max(w, h)
    nw, nh = max(14, round(w * k / 14) * 14), max(14, round(h * k / 14) * 14)
    x = np.asarray(img.convert("RGB").resize((nw, nh), Image.BICUBIC), dtype=np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        d = net(pixel_values=x).predicted_depth  # (1, h, w), relative inverse depth
    d = F.interpolate(d[None], size=(h, w), mode="bicubic", align_corners=False)[0, 0]
    return d.float().cpu().numpy()


def to_heightmap(img, depth, bg=0.1, gap=0.08, detail=0.5, radius=12):
    """Mask the floor from the source's dark background, rescale the figure to
    [gap, 1], and add `detail` x the render's shading detail (see
    depth2stl.shading_detail) so fine sculpted texture survives."""
    lum = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    fig = lum > bg
    lo, hi = np.percentile(depth[fig], 0.5), np.percentile(depth[fig], 99.5)
    h = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    if detail > 0:
        h = np.clip(h + detail * 0.5 * shading_detail(lum, fig, radius), 0, 1)
    h = gap + (1 - gap) * h
    h[~fig] = 0.0
    return h.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("out")
    ap.add_argument("--model", choices=MODELS, default="large")
    ap.add_argument("--res", type=int, default=1036, help="network input size (long side); more = finer detail")
    ap.add_argument("--bg", type=float, default=0.1, help="source luminance below this is background floor")
    ap.add_argument("--gap", type=float, default=0.08, help="height of the figure's lowest point above the floor (0..1)")
    ap.add_argument("--detail", type=float, default=0.5, help="how much render shading detail to add (0 = none)")
    ap.add_argument("--radius", type=float, default=12, help="px: detail finer than this is taken from the render")
    ap.add_argument("--base-file", help="use this depth map for the layering instead of the network")
    ap.add_argument("--no-mask", action="store_true", help="keep the raw estimate, no floor masking")
    a = ap.parse_args()
    t0 = time.time()
    img = Image.open(a.image)
    if a.base_file:
        b = Image.open(a.base_file)
        depth = np.asarray(b.convert("F" if b.mode.startswith("I") else "L").resize(img.size, Image.LANCZOS), dtype=np.float32)
        print(f"layering from {a.base_file}", file=sys.stderr)
    else:
        depth = estimate(img, a.model, a.res)
        print(f"estimated {img.size[0]}x{img.size[1]} with {a.model} @ {a.res}px in {time.time() - t0:.1f} s", file=sys.stderr)
    if a.no_mask:
        h = (depth - depth.min()) / max(np.ptp(depth), 1e-6)
    else:
        h = to_heightmap(img, depth, a.bg, a.gap, detail=a.detail, radius=a.radius)
    Image.fromarray((h * 65535).round().astype(np.uint16)).save(a.out)
    print(f"wrote {a.out}  (16-bit gray, figure {100 * (h > 0).mean():.0f}% of frame)", file=sys.stderr)


if __name__ == "__main__":
    main()
