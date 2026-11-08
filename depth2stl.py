#!/usr/bin/env python3
"""depth2stl: depth-map PNG -> upscaled, smoothed relief STL. GUI or CLI.

GUI:  depth2stl [image.png]
CLI:  depth2stl image.png -o out.stl [--scale 2 --passes 10 --remove-canvas ...]
Test: depth2stl --check
"""
import argparse
import json
import math
import os
import queue
import shutil
import struct
import subprocess
import tempfile
import threading
import time

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
os.environ.setdefault("PYOPENGL_PLATFORM", "x11")  # Tk is X11 (XWayland); EGL auto-pick breaks GLX

# Visual defaults for the 3D view. Edit here, not inline.
VIEW = dict(bg=(0.13, 0.13, 0.14), color=(0.80, 0.72, 0.58), fov_deg=40.0,
            yaw=-20.0, pitch=-55.0, dist=2.8, light=(0.3, 0.4, 1.0, 0.0),  # dist 1/tan(fov/2) fits all
            rotate_speed=0.5, zoom_step=1.15)


# ---------------------------------------------------------------- pipeline
def load_height(path):
    """Return float32 heightmap in 0..1 plus a note. Height = luminance x alpha
    (composited over black), so a white-on-transparent depth map reads its
    alpha and an opaque grayscale reads its luminance. 16-bit sources keep
    their precision when ImageMagick is installed; PIL falls back to 8-bit."""
    if shutil.which("magick"):
        with tempfile.TemporaryDirectory() as d:
            g = os.path.join(d, "g16.png")
            subprocess.run(["magick", path, "-background", "black", "-alpha", "remove",
                            "-grayscale", "Rec709Luma", "-depth", "16", "png:" + g], check=True)
            a = np.asarray(Image.open(g), dtype=np.float32)
        return a / 65535.0, "16-bit (magick)"
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(im, dtype=np.float32)
        return a / (65535.0 if a.max() > 255 else 255.0), "16-bit gray"
    rgba = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
    lum = rgba[..., :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return lum * rgba[..., 3], "8-bit (PIL)"


def gaussian(h, sigma):
    """Separable Gaussian blur, edge-clamped. numpy only (scipy costs 100 MB)."""
    r = int(3 * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-x * x / (2 * sigma * sigma)); k /= k.sum()
    for axis in (0, 1):
        p = np.pad(h, [(r, r) if a == axis else (0, 0) for a in (0, 1)], mode="edge")
        out = np.zeros_like(h)
        for i, w in enumerate(k):
            out += w * (p[i:i + h.shape[0]] if axis == 0 else p[:, i:i + h.shape[1]])
        h = out
    return h


def diffuse(h, edge, dt=0.2):
    """One Perona-Malik anisotropic-diffusion pass. Smooths steps smaller than
    `edge` (fraction of full height) and leaves larger steps (silhouettes,
    scale edges) sharp. Repeated passes = more smoothing, edges still kept."""
    p = np.pad(h, 1, mode="edge")
    k2 = np.float32(edge * edge)
    flux = np.zeros_like(h)
    for d in (p[:-2, 1:-1] - h, p[2:, 1:-1] - h, p[1:-1, :-2] - h, p[1:-1, 2:] - h):
        flux += d / (1.0 + d * d / k2)
    return h + np.float32(dt) * flux


def erode(m):
    p = np.pad(m, 1, constant_values=False)
    return m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def shading_detail(lum, fig, radius):
    """High-pass of a shaded render's luminance inside the figure, ~[-1, 1].
    Normalised blur so the black background can't halo the silhouette; tapered
    to 0 over `radius` px inside the edge so the render's dark rim doesn't
    carve a groove."""
    f = fig.astype(np.float32)
    local = gaussian(lum * f, radius) / np.maximum(gaussian(f, radius), 1e-6)
    hp = (lum - local) * f
    hp /= max(np.percentile(np.abs(hp[fig]), 99), 1e-6) if fig.any() else 1.0
    inner = np.zeros(fig.shape, np.float32); m = fig
    for _ in range(int(radius)):
        m = erode(m); inner += m
    return hp * inner / max(int(radius), 1)


def stages(h, scale=1, smooth=0.0, invert=False, floor=0.0, passes=0, edge=0.05,
           boost=0.0, boost_radius=8, render=None, render_amt=0.5):
    """Generator yielding (label, heightmap) after every step so callers can
    show progress. The last yield is the finished map.
    boost: unsharp-style gain on the height map's own detail finer than
    boost_radius px. render: luminance array of the shaded render this depth
    map came from; its shading detail is added at render_amt strength."""
    if invert:
        h = 1.0 - h
    if floor > 0:
        h = np.clip((h - floor) / (1.0 - floor), 0.0, 1.0)
    yield "floor cut", h
    if scale != 1:
        im = Image.fromarray(h, mode="F")
        w, ht = im.size
        h = np.asarray(im.resize((round(w * scale), round(ht * scale)), Image.LANCZOS), dtype=np.float32)
        h = np.clip(h, 0.0, 1.0)  # lanczos rings slightly past the range
        yield f"upscale ×{scale}", h
    if smooth > 0:
        h = gaussian(h, smooth)
        yield f"gaussian σ={smooth:g}", h
    for i in range(passes):
        h = diffuse(h, edge)
        yield f"diffusion pass {i + 1}/{passes}", h
    if render is not None and render_amt > 0:
        lum = np.asarray(Image.fromarray(render, mode="F").resize(h.shape[::-1], Image.LANCZOS), dtype=np.float32)
        lum = gaussian(lum, 1.0)  # kill JPEG grain before it becomes texture
        h = np.clip(h + render_amt * 0.5 * shading_detail(lum, h > 0, boost_radius), 0.0, 1.0)
        yield f"render detail ×{render_amt:g}", h
    if boost > 0:
        fig = h > 0
        h = np.clip(h + boost * (h - gaussian(h, boost_radius)) * fig, 0.0, 1.0)
        yield f"detail boost ×{boost:g}", h


def n_stages(scale=1, smooth=0.0, passes=0, boost=0.0, render=None, render_amt=0.5, **_):
    return 1 + (scale != 1) + (smooth > 0) + passes + (render is not None and render_amt > 0) + (boost > 0)


def load_render(path):
    """Luminance of a shaded render as float32 0..1 (for stages(render=...))."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def process(h, **kw):
    for _, h in stages(h, **kw):
        pass
    return h


def dilate(m):
    p = np.pad(m, 1)
    out = m.copy()
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            out |= p[di:di + m.shape[0], dj:dj + m.shape[1]]
    return out


def remove_canvas(h, coarse=8, fade=0):
    """Drop the leftover canvas: keep only the island holding the highest
    ground (the figure) and crop to it. Returns (h, mask, zmul).
    fade > 0 adds a skirt `fade` px wide around the figure whose total height
    (base + relief) ramps down to the floor, so the silhouette has no vertical
    wall; zmul is that ramp (1 inside the figure) or None when fade == 0.
    Island search runs on a coarse grid so a 3000-px figure fills in <1 s."""
    H, W = h.shape
    margin = fade + 1
    ph, pw = -H % coarse, -W % coarse
    hp = np.pad(h, ((0, ph), (0, pw)))
    blocks = hp.reshape(H // coarse + (ph > 0), coarse, W // coarse + (pw > 0), coarse)
    cmask = blocks.max(axis=(1, 3)) > 0
    cmean = blocks.mean(axis=(1, 3))
    seed = np.zeros_like(cmask); seed[np.unravel_index(cmean.argmax(), cmean.shape)] = True
    while True:  # geodesic flood fill: 8-neighbour dilation clipped to the mask
        grown = dilate(seed) & cmask
        if (grown == seed).all():
            break
        seed = grown
    keep = np.kron(seed, np.ones((coarse, coarse), bool))[:H, :W]
    mask = (h > 0) & keep
    rows, cols = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
    if not len(rows):
        return h, np.ones_like(mask), None
    r0, r1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, H)
    c0, c1 = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, W)
    h, mask = h[r0:r1, c0:c1] * mask[r0:r1, c0:c1], mask[r0:r1, c0:c1].copy()
    zmul = None
    if fade > 0:
        # ponytail: Chebyshev (8-neighbour) distance by repeated dilation;
        # fine for a fade, swap for a Euclidean transform if corners look square
        dist = np.full(mask.shape, fade + 1, np.float32); dist[mask] = 0
        grown = mask
        for k in range(1, fade + 1):
            nxt = dilate(grown)
            dist[nxt & ~grown] = k
            grown = nxt
        mask = grown
        r = np.clip(1.0 - dist / (fade + 1), 0.0, 1.0)
        zmul = (r * r * (3 - 2 * r)).astype(np.float32)  # smoothstep
    mask[[0, -1], :] = False; mask[:, [0, -1]] = False  # ring so walls never sit on the crop edge
    return h, mask, zmul


def height_mm(h, relief, base, zmul=None):
    z = base + h * relief
    return z if zmul is None else z * zmul


def merged_blocks(z, C, tol, max_level=6):
    """Quadtree merge: square blocks of cells (2..2^max_level wide) whose
    heights all lie within `tol` of the bilinear patch through the block's
    corners collapse to one quad. Returns [(i, j, size), ...] arrays covering
    every cell of C exactly once (leftover cells come back with size 1).
    ponytail: neighbours of different size leave T-junctions with cracks <= tol;
    CAM ignores them. tol=0 in grid_mesh keeps the strict watertight grid."""
    Hc, Wc = C.shape
    covered = np.zeros((Hc, Wc), bool)
    out = []
    for lvl in range(max_level, 0, -1):
        B = 1 << lvl
        nb_i, nb_j = Hc // B, Wc // B
        if nb_i == 0 or nb_j == 0:
            continue
        zi = z[:nb_i * B + 1:B, :nb_j * B + 1:B]
        z00, z01, z10, z11 = zi[:-1, :-1], zi[:-1, 1:], zi[1:, :-1], zi[1:, 1:]
        t = np.arange(B + 1, dtype=np.float32) / B
        dev = np.zeros((nb_i, nb_j), np.float32)
        for ai, a in enumerate(t):
            for bi_, b in enumerate(t):
                pred = z00 * (1 - a) * (1 - b) + z01 * (1 - a) * b + z10 * a * (1 - b) + z11 * a * b
                actual = z[ai:ai + nb_i * B:B, bi_:bi_ + nb_j * B:B]
                np.maximum(dev, np.abs(actual - pred), out=dev)
        blocks = lambda m: m[:nb_i * B, :nb_j * B].reshape(nb_i, B, nb_j, B)
        ok = (dev <= tol) & blocks(C).all(axis=(1, 3)) & ~blocks(covered).any(axis=(1, 3))
        bi, bj = np.nonzero(ok)
        if len(bi):
            out.append((bi * B, bj * B, B))
            covered[:nb_i * B, :nb_j * B] |= np.kron(ok, np.ones((B, B), bool))
    bi, bj = np.nonzero(C & ~covered)
    out.append((bi, bj, 1))
    return out


def grid_mesh(h, mask, sx, sy, relief, base, zmul=None, tol=0.0):
    """Indexed mesh of the heightmap: top surface, walls down to z=0, bottom.
    mask=None: whole canvas (walls on the border, 2-triangle bottom).
    mask=bool array: only cells whose 4 corners are in the mask, with a wall
    along the silhouette and a matching bottom, so the solid is watertight.
    tol > 0 (mm): merge flat blocks of the top (see merged_blocks) and mesh the
    bottom as row runs; far fewer triangles, cracks no larger than tol.
    Returns (verts (N,3) float32, tris (M,3) uint32), CCW = outward."""
    H, W = h.shape
    X = np.arange(W, dtype=np.float32) * sx
    Y = (H - 1 - np.arange(H, dtype=np.float32)) * sy  # image top -> +Y
    top = np.stack([*np.meshgrid(X, Y), height_mm(h, relief, base, zmul)], axis=-1).reshape(-1, 3).astype(np.float32)
    bot = top.copy(); bot[:, 2] = 0
    verts = np.concatenate([top, bot])
    ids = np.arange(H * W, dtype=np.uint32).reshape(H, W); B = np.uint32(H * W)
    C = np.ones((H - 1, W - 1), bool) if mask is None else \
        mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]
    tris = []
    if tol > 0:
        for bi, bj, S in merged_blocks(top[:, 2].reshape(H, W), C, tol):
            a, b, c, d = ids[bi, bj], ids[bi + S, bj], ids[bi + S, bj + S], ids[bi, bj + S]
            tris += [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
    else:
        a, b, c, d = ids[:-1, :-1][C], ids[1:, :-1][C], ids[1:, 1:][C], ids[:-1, 1:][C]
        tris += [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
    if mask is None:
        # ponytail: 2-triangle bottom leaves T-junctions on the rim; CAM
        # ignores them and it halves the file. Masked meshes get a full bottom.
        c00, c01, c10, c11 = B + ids[0, 0], B + ids[0, -1], B + ids[-1, 0], B + ids[-1, -1]
        tris.append(np.array([[c00, c01, c11], [c00, c11, c10]], np.uint32))
    elif tol > 0:
        # flat bottom as one quad per row run of cells (coplanar T-junctions only)
        Cc = np.pad(C, ((0, 0), (1, 1)))
        i0, j0 = np.nonzero(~Cc[:, :-1] & Cc[:, 1:]); i1, j1 = np.nonzero(Cc[:, :-1] & ~Cc[:, 1:])
        a, b, c, d = ids[i0, j0], ids[i0 + 1, j0], ids[i1 + 1, j1], ids[i1, j1]
        tris += [np.stack([B + a, B + c, B + b], 1), np.stack([B + a, B + d, B + c], 1)]
    else:
        a, b, c, d = ids[:-1, :-1][C], ids[1:, :-1][C], ids[1:, 1:][C], ids[:-1, 1:][C]
        tris += [np.stack([B + a, B + c, B + b], 1), np.stack([B + a, B + d, B + c], 1)]
    Cp = np.pad(C, 1)
    # walls on edges along rows: cell above (+Y) vs below. Winding (P,Pb,Qb),(P,Qb,Q)
    # faces -Y, which is outward when the inside cell is the one above.
    above, below = Cp[:-1, 1:-1], Cp[1:, 1:-1]
    i, j = np.nonzero(above != below)
    P, Q, ins = ids[i, j], ids[i, j + 1], above[i, j][:, None]
    tris += [np.where(ins, np.stack([P, B + P, B + Q], 1), np.stack([P, B + Q, B + P], 1)),
             np.where(ins, np.stack([P, B + Q, Q], 1), np.stack([P, Q, B + Q], 1))]
    # walls on edges along columns: cell left vs right (+X). Same winding faces -X,
    # outward when the inside cell is the one on the right.
    left, right = Cp[1:-1, :-1], Cp[1:-1, 1:]
    i, j = np.nonzero(left != right)
    P, Q, ins = ids[i, j], ids[i + 1, j], right[i, j][:, None]
    tris += [np.where(ins, np.stack([P, B + P, B + Q], 1), np.stack([P, B + Q, B + P], 1)),
             np.where(ins, np.stack([P, B + Q, Q], 1), np.stack([P, Q, B + Q], 1))]
    return verts, np.concatenate(tris).astype(np.uint32)


def vertex_normals(verts, tris):
    fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    n = np.zeros_like(verts)
    for k in range(3):
        for c in range(3):
            n[:, c] += np.bincount(tris[:, k], weights=fn[:, c], minlength=len(verts))
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return n.astype(np.float32)


def shade(z, px_mm, light=(-0.6, 0.6, 0.55)):
    """Lambert-shaded uint8 preview of a height array in mm."""
    gy, gx = np.gradient(z, px_mm)
    n = np.stack([-gx, gy, np.ones_like(z)], axis=-1)  # gy sign: image row 0 is +Y
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    l = np.array(light, dtype=np.float32)
    l /= np.linalg.norm(l)
    d = np.clip(n @ l, 0, 1)
    return (25 + 230 * d).astype(np.uint8)


def to_stl(h, mask, path, x_mm, y_mm, relief_mm, base_mm, zmul=None, tol=0.0):
    """Mesh the heightmap and write a binary STL. Returns triangle count."""
    H, W = h.shape
    verts, tris = grid_mesh(h, mask, x_mm / (W - 1), y_mm / (H - 1), relief_mm, base_mm, zmul, tol)
    return write_stl(verts, tris, path)


def write_stl(verts, tris, path):
    soup = verts[tris]  # M,3,3
    n = np.cross(soup[:, 1] - soup[:, 0], soup[:, 2] - soup[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, soup
    with open(path, "wb") as f:
        f.write(b"depth2stl".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        rec.tofile(f)
    return len(tris)


def tri_count(h, mask):
    """Exact triangle count without building the mesh."""
    H, W = h.shape
    if mask is None:
        return 2 * (H - 1) * (W - 1) + 4 * (H - 1) + 4 * (W - 1) + 2
    C = mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]
    Cp = np.pad(C, 1)
    walls = (Cp[:-1, 1:-1] != Cp[1:, 1:-1]).sum() + (Cp[1:-1, :-1] != Cp[1:-1, 1:]).sum()
    return 4 * int(C.sum()) + 2 * int(walls)


def stl_bytes(n_tris):
    return 84 + 50 * n_tris


def fmt_bytes(n):
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def preview_mesh(h, mask, sx, sy, relief, base, max_px, zmul=None):
    """Decimated indexed mesh + smooth normals for the 3D view."""
    k = max(1, math.ceil(h.shape[1] / max_px), math.ceil(h.shape[0] / max_px))
    hd = h[::k, ::k]; md = None if mask is None else mask[::k, ::k]
    zd = None if zmul is None else zmul[::k, ::k]
    verts, tris = grid_mesh(hd, md, sx * k, sy * k, relief, base, zd)
    return verts, vertex_normals(verts, tris), tris


# ---------------------------------------------------------------- GUI
def make_viewer_class():
    """Built lazily so the CLI never imports OpenGL."""
    from OpenGL import GL, GLX
    from OpenGL.arrays import vbo
    import pyopengltk.linux as L
    from pyopengltk import OpenGLFrame
    if GLX.GLX_DEPTH_SIZE not in L.fbatt:  # library asks for no depth buffer
        L.fbatt[-1:-1] = [GLX.GLX_DEPTH_SIZE, 24]

    class Viewer(OpenGLFrame):
        def __init__(self, master, **kw):
            super().__init__(master, **kw)
            self.buf = None; self.n_idx = 0; self.pending = None
            self.center = np.zeros(3, np.float32); self.radius = 1.0
            self.yaw, self.pitch, self.dist = VIEW["yaw"], VIEW["pitch"], VIEW["dist"]
            self.pan = [0.0, 0.0]; self.last = None
            self.bind("<ButtonPress-1>", self._down); self.bind("<B1-Motion>", self._rotate)
            self.bind("<ButtonPress-3>", self._down); self.bind("<B3-Motion>", self._pan)
            self.bind("<Button-4>", lambda e: self._zoom(1 / VIEW["zoom_step"]))
            self.bind("<Button-5>", lambda e: self._zoom(VIEW["zoom_step"]))
            self.bind("<MouseWheel>", lambda e: self._zoom(1 / VIEW["zoom_step"] if e.delta > 0 else VIEW["zoom_step"]))
            self.bind("<Double-Button-1>", lambda e: self.home())

        def initgl(self):  # also re-run by the library on every resize: keep it idempotent
            self.tkMakeCurrent()
            GL.glClearColor(*VIEW["bg"], 1.0)
            GL.glEnable(GL.GL_DEPTH_TEST); GL.glEnable(GL.GL_LIGHTING); GL.glEnable(GL.GL_LIGHT0)
            GL.glEnable(GL.GL_COLOR_MATERIAL); GL.glEnable(GL.GL_NORMALIZE)
            GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
            GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, GL.GL_TRUE)
            if self.pending:
                self._upload(*self.pending); self.pending = None

        def set_mesh(self, verts, normals, tris):
            if not self.context_created:
                self.pending = (verts, normals, tris); return
            self.tkMakeCurrent(); self._upload(verts, normals, tris); self.tkExpose(None)

        def _upload(self, verts, normals, tris):
            self.buf = (vbo.VBO(verts), vbo.VBO(normals),
                        vbo.VBO(tris, target=GL.GL_ELEMENT_ARRAY_BUFFER))
            self.n_idx = tris.size
            used = verts[np.unique(tris)]
            lo, hi = used.min(0), used.max(0)
            self.center, self.radius = (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)) / 2, 1e-6)

        def home(self):
            self.yaw, self.pitch, self.dist, self.pan = VIEW["yaw"], VIEW["pitch"], VIEW["dist"], [0.0, 0.0]
            self.tkExpose(None)

        def redraw(self):
            w, h = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
            GL.glViewport(0, 0, w, h)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glMatrixMode(GL.GL_PROJECTION); GL.glLoadIdentity()
            near, far = self.radius * 0.02, self.radius * 50
            f = near * math.tan(math.radians(VIEW["fov_deg"]) / 2)
            GL.glFrustum(-f * w / h, f * w / h, -f, f, near, far)
            GL.glMatrixMode(GL.GL_MODELVIEW); GL.glLoadIdentity()
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, VIEW["light"])  # camera-fixed light
            GL.glTranslatef(self.pan[0], self.pan[1], -self.dist * self.radius)
            GL.glRotatef(self.pitch, 1, 0, 0); GL.glRotatef(self.yaw, 0, 0, 1)
            GL.glTranslatef(*(-self.center))
            if not self.n_idx:
                return
            vb, nb, ib = self.buf
            vb.bind(); GL.glEnableClientState(GL.GL_VERTEX_ARRAY); GL.glVertexPointer(3, GL.GL_FLOAT, 0, vb)
            nb.bind(); GL.glEnableClientState(GL.GL_NORMAL_ARRAY); GL.glNormalPointer(GL.GL_FLOAT, 0, nb)
            ib.bind(); GL.glColor3f(*VIEW["color"])
            GL.glDrawElements(GL.GL_TRIANGLES, self.n_idx, GL.GL_UNSIGNED_INT, None)
            ib.unbind(); nb.unbind(); vb.unbind()
            GL.glDisableClientState(GL.GL_NORMAL_ARRAY); GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

        def _down(self, e):
            self.last = (e.x, e.y)

        def _rotate(self, e):
            if self.last is None:  # drag that started outside the widget
                self.last = (e.x, e.y); return
            dx, dy = e.x - self.last[0], e.y - self.last[1]; self.last = (e.x, e.y)
            self.yaw += dx * VIEW["rotate_speed"]; self.pitch += dy * VIEW["rotate_speed"]
            self.tkExpose(None)

        def _pan(self, e):
            if self.last is None:
                self.last = (e.x, e.y); return
            dx, dy = e.x - self.last[0], e.y - self.last[1]; self.last = (e.x, e.y)
            s = 2 * self.dist * self.radius * math.tan(math.radians(VIEW["fov_deg"]) / 2) / max(self.winfo_height(), 1)
            self.pan[0] += dx * s; self.pan[1] -= dy * s
            self.tkExpose(None)

        def _zoom(self, f):
            self.dist *= f; self.tkExpose(None)

    return Viewer


def pick_file(save, start_dir, default_name, filt_name, patterns):
    """Native file dialog: KDE's kdialog, then GTK's zenity, else Tk's own.
    Returns a path or None."""
    if shutil.which("kdialog"):
        arg = ["--getsavefilename", os.path.join(start_dir, default_name)] if save else ["--getopenfilename", start_dir]
        cmd = ["kdialog", "--title", "depth2stl", *arg, f"{filt_name} ({' '.join(patterns)})"]
    elif shutil.which("zenity"):
        cmd = ["zenity", "--file-selection", "--title=depth2stl", f"--filename={os.path.join(start_dir, default_name if save else '')}",
               f"--file-filter={filt_name} | {' '.join(patterns)}"] + (["--save", "--confirm-overwrite"] if save else [])
    else:
        from tkinter import filedialog
        types = [(filt_name, " ".join(patterns)), ("All", "*")]
        if save:
            return filedialog.asksaveasfilename(defaultextension=patterns[0][1:], filetypes=types,
                                                initialdir=start_dir, initialfile=default_name) or None
        return filedialog.askopenfilename(filetypes=types, initialdir=start_dir) or None
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip() or None if r.returncode == 0 else None


DEFAULTS = dict(scale=2, floor=2, smooth=1.0, passes=10, edge=5, fade=20, boost=0.0, boost_radius=8,
                render_amt=0.5, x=300, y=147, z=8, base=3, tol=0.02, detail=1600, invert=False, rm=True, lock=True)
PRESETS = {  # built-in; user presets live in PRESET_FILE and override on name clash
    "Defaults": {},
    "Fast draft": dict(scale=1, passes=0, smooth=1.0, detail=800),
    "Max detail": dict(scale=3, smooth=0.5, passes=10, edge=3, boost=1.0, boost_radius=8, render_amt=0.8, detail=2400),
    "Cutout, soft edge": dict(rm=True, fade=60, passes=15),
    "Full-canvas plaque": dict(rm=False, fade=0),
    "Deep carve 15 mm": dict(z=15, base=5, passes=15, edge=3),
}
PRESET_FILE = os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")), "depth2stl", "presets.json")


def load_user_presets():
    try:
        with open(PRESET_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_user_preset(name, values):
    user = load_user_presets(); user[name] = values
    os.makedirs(os.path.dirname(PRESET_FILE), exist_ok=True)
    with open(PRESET_FILE, "w") as f:
        json.dump(user, f, indent=1)


def ask_text(prompt, default=""):
    if shutil.which("kdialog"):
        r = subprocess.run(["kdialog", "--title", "depth2stl", "--inputbox", prompt, default], capture_output=True, text=True)
    elif shutil.which("zenity"):
        r = subprocess.run(["zenity", "--entry", "--title=depth2stl", f"--text={prompt}", f"--entry-text={default}"],
                           capture_output=True, text=True)
    else:
        from tkinter import simpledialog
        return simpledialog.askstring("depth2stl", prompt, initialvalue=default)
    return r.stdout.strip() or None if r.returncode == 0 else None


def estimator_cmd():
    """External depth estimator (depth_estimate.py in a torch venv), or None."""
    env = os.environ.get("DEPTH2STL_ESTIMATOR")
    if env:
        return env.split()
    exe = shutil.which("depth2stl-estimate")
    return [exe] if exe else None


def gui(path=None):
    import tkinter as tk
    from tkinter import messagebox, ttk
    from PIL import ImageTk
    Viewer = make_viewer_class()

    root = tk.Tk()
    root.title("depth2stl")
    st = {"h": None, "path": None, "proc": None, "mask": None, "zmul": None, "shaded": None, "photo": None,
          "render": None, "render_path": None, "mesh": None,
          "job": None, "mjob": None, "gen": 0, "mgen": 0, "busy": False, "syncing": False,
          "last": {}}
    q = queue.Queue()

    left = ttk.Frame(root, padding=8); left.grid(row=0, column=0, sticky="ns")
    pane = ttk.PanedWindow(root, orient="vertical"); pane.grid(row=0, column=1, sticky="nsew")
    viewer = Viewer(pane, width=900, height=420); pane.add(viewer, weight=3)
    canvas = tk.Canvas(pane, bg="#222", height=220); pane.add(canvas, weight=1)
    right = ttk.LabelFrame(root, text="Statistics", padding=8); right.grid(row=0, column=2, sticky="ns")
    root.columnconfigure(1, weight=1); root.rowconfigure(0, weight=1)
    status = ttk.Label(root, anchor="w"); status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8)
    prog = ttk.Progressbar(root, mode="determinate"); prog.grid(row=1, column=2, sticky="ew", padx=8, pady=4)

    v = {k: tk.DoubleVar(value=d) for k, d in DEFAULTS.items() if not isinstance(d, bool)}
    invert = tk.BooleanVar(value=DEFAULTS["invert"]); rm_canvas = tk.BooleanVar(value=DEFAULTS["rm"])
    lock = tk.BooleanVar(value=DEFAULTS["lock"])
    flags = {"invert": invert, "rm": rm_canvas, "lock": lock}
    PROC = ("scale", "floor", "smooth", "passes", "edge", "fade", "boost", "boost_radius", "render_amt")

    def row(r, label, var, lo, hi, res):
        ttk.Label(left, text=label).grid(row=r, column=0, sticky="w")
        ttk.Spinbox(left, textvariable=var, from_=lo, to=hi, increment=res, width=8
                    ).grid(row=r, column=1, sticky="e", pady=2)

    ttk.Button(left, text="Open depth map…", command=lambda: open_()).grid(row=0, column=0, columnspan=2, sticky="ew")
    est_frame = ttk.Frame(left); est_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 8))
    est_btn = ttk.Button(est_frame, text="Estimate from render…", command=lambda: estimate_())
    est_btn.pack(side="left", expand=True, fill="x")
    est_detail = tk.DoubleVar(value=0.5)
    ttk.Spinbox(est_frame, textvariable=est_detail, from_=0, to=1, increment=0.1, width=4).pack(side="left", padx=(4, 0))
    if not estimator_cmd():
        est_btn.state(["disabled"])
    preset_var = tk.StringVar(value="Defaults")
    preset_box = ttk.Combobox(left, textvariable=preset_var, state="readonly", width=18)
    preset_box.grid(row=2, column=0, columnspan=2, sticky="ew")
    pbtns = ttk.Frame(left); pbtns.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 6))
    ttk.Button(pbtns, text="Reset", width=7, command=lambda: apply_preset("Defaults")).pack(side="left", expand=True, fill="x")
    ttk.Button(pbtns, text="Save preset…", command=lambda: save_preset()).pack(side="left", expand=True, fill="x")
    left_row = [4]

    def nrow():
        left_row[0] += 1; return left_row[0] - 1
    ttk.Label(left, text="Processing", font="TkHeadingFont").grid(row=nrow(), column=0, columnspan=2, sticky="w")
    row(nrow(), "Upscale ×", v["scale"], 1, 4, 1)
    row(nrow(), "Floor cutoff %", v["floor"], 0, 50, 0.5)
    row(nrow(), "Gaussian σ px", v["smooth"], 0, 20, 0.5)
    row(nrow(), "Diffusion passes", v["passes"], 0, 200, 1)
    row(nrow(), "Edge keep % height", v["edge"], 0.5, 50, 0.5)
    ttk.Checkbutton(left, text="Invert (black = high)", variable=invert).grid(row=nrow(), column=0, columnspan=2, sticky="w")
    ttk.Checkbutton(left, text="Remove leftover canvas", variable=rm_canvas).grid(row=nrow(), column=0, columnspan=2, sticky="w")
    row(nrow(), "Edge fade px", v["fade"], 0, 400, 2)
    row(nrow(), "Detail boost", v["boost"], 0, 3, 0.1)
    row(nrow(), "Detail radius px", v["boost_radius"], 2, 100, 1)
    rframe = ttk.Frame(left); rframe.grid(row=nrow(), column=0, columnspan=2, sticky="ew", pady=2)
    render_btn = ttk.Button(rframe, text="Detail from render…", command=lambda: pick_render())
    render_btn.pack(side="left", expand=True, fill="x")
    ttk.Spinbox(rframe, textvariable=v["render_amt"], from_=0, to=1, increment=0.1, width=4).pack(side="left", padx=(4, 0))
    ttk.Label(left, text="Size", font="TkHeadingFont").grid(row=nrow(), column=0, columnspan=2, sticky="w", pady=(8, 0))
    row(nrow(), "X mm", v["x"], 1, 5000, 10)
    row(nrow(), "Y mm", v["y"], 1, 5000, 10)
    row(nrow(), "Z relief mm", v["z"], 0.1, 500, 0.5)
    row(nrow(), "Base mm", v["base"], 0, 100, 0.5)
    ttk.Checkbutton(left, text="Lock X : Y : Z", variable=lock).grid(row=nrow(), column=0, columnspan=2, sticky="w")
    row(nrow(), "Simplify tol mm", v["tol"], 0, 0.5, 0.01)
    row(nrow(), "3D preview px", v["detail"], 200, 6000, 200)
    ttk.Button(left, text="Export STL…", command=lambda: export()).grid(row=nrow(), column=0, columnspan=2, sticky="ew", pady=(12, 2))
    ttk.Label(left, text="drag: orbit · right-drag: pan\nwheel: zoom · double-click: reset",
              foreground="#777").grid(row=nrow(), column=0, columnspan=2, sticky="w", pady=(6, 0))

    # ---- presets
    def all_presets():
        return {**PRESETS, **load_user_presets()}

    def refresh_presets():
        preset_box["values"] = list(all_presets())

    def current_values():
        try:
            return {k: (flags[k].get() if k in flags else v[k].get()) for k in DEFAULTS}
        except tk.TclError:
            return None

    def apply_preset(name):
        vals = {**DEFAULTS, **all_presets().get(name, {})}
        st["syncing"] = True  # bypass the X:Y:Z lock while loading
        for k, val in vals.items():
            (flags[k] if k in flags else v[k]).set(val)
        st["syncing"] = False
        st["last"].update({o: v[o].get() for o in ("x", "y", "z")})
        preset_var.set(name if name in all_presets() else "Defaults")
        if st["h"] is not None:
            if lock.get() and st["proc"] is not None:
                fit_y_to_pixels()
            soon()
        say(f"preset: {name}")

    def save_preset():
        vals = current_values()
        if vals is None:
            say("fix the highlighted value first"); return
        name = ask_text("Preset name:", preset_var.get() if preset_var.get() not in PRESETS else "")
        if not name:
            return
        save_user_preset(name, vals); refresh_presets(); preset_var.set(name)
        say(f"saved preset '{name}' to {PRESET_FILE}")

    preset_box.bind("<<ComboboxSelected>>", lambda e: apply_preset(preset_var.get()))
    refresh_presets()

    stat_keys = ["Source", "Source size", "Source px", "Channel", "Render", "Canvas", "Edge fade", "Mesh px", "mm / pixel",
                 "Model mm", "Z range mm", "Triangles", "STL estimate", "Process time",
                 "Last export", "Export size", "Export time"]
    stats = {}
    for i, k in enumerate(stat_keys):
        ttk.Label(right, text=k + ":").grid(row=i, column=0, sticky="ne", padx=(0, 6), pady=1)
        stats[k] = ttk.Label(right, text="—", wraplength=260, justify="left")
        stats[k].grid(row=i, column=1, sticky="w", pady=1)

    def set_stat(k, val):
        stats[k].config(text=str(val))

    def say(msg):
        status.config(text=msg)

    def params():
        try:
            p = {k: x.get() for k, x in v.items()}
        except tk.TclError:
            return None  # half-typed number
        p.update(scale=int(p["scale"]), passes=int(p["passes"]), detail=int(p["detail"]), fade=int(p["fade"]),
                 floor=p["floor"] / 100, edge=p["edge"] / 100, invert=invert.get(), rm=rm_canvas.get())
        return p

    def draw(img):
        cw, ch = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        img = img.copy(); img.thumbnail((cw, ch), Image.LANCZOS)
        st["photo"] = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=st["photo"])

    def quick_shade(arr, px_mm, p, cw, zmul=None):
        # ponytail: intermediate previews shade a decimated copy so a pass shows
        # in ~50 ms; the final result is shaded at full resolution.
        k = max(1, arr.shape[1] // max(cw, 1))
        zd = None if zmul is None else zmul[::k, ::k]
        return Image.fromarray(shade(height_mm(arr[::k, ::k], p["z"], p["base"], zd), px_mm * k))

    # ---- processing worker: floor/upscale/smooth/diffuse (+ canvas removal)
    def pick_render():
        """Attach (or detach) the shaded render whose shading feeds the detail layer."""
        if st["render"] is not None:
            st["render"] = st["render_path"] = None
            render_btn.config(text="Detail from render…"); set_stat("Render", "—"); soon(); return
        start = os.path.dirname(st["path"]) if st["path"] else os.path.expanduser("~")
        p = pick_file(False, start, "", "Render or photo", ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.tif", "*.tiff"])
        if not p:
            return
        try:
            st["render"] = load_render(p); st["render_path"] = p
        except Exception as e:
            say("render load failed: " + repr(e)); return
        render_btn.config(text=f"✕ {os.path.basename(p)[:18]}"); set_stat("Render", p); soon()

    def worker(gen, h, p, cw, render):
        t0 = time.time()
        kw = {k: p[k] for k in ("scale", "smooth", "invert", "floor", "passes", "edge", "boost", "boost_radius", "render_amt")}
        kw["render"] = render
        total = n_stages(**kw) + p["rm"]
        try:
            for i, (label, arr) in enumerate(stages(h, **kw)):
                if st["gen"] != gen:
                    return
                px_mm = p["x"] / (arr.shape[1] - 1)
                q.put(("stage", gen, label, i + 1, total, time.time() - t0,
                       quick_shade(arr, px_mm, p, cw)))
            mask = zmul = None
            if p["rm"]:
                arr, mask, zmul = remove_canvas(arr, fade=p["fade"])
                q.put(("stage", gen, "remove leftover canvas", total, total, time.time() - t0,
                       quick_shade(arr, px_mm, p, cw, zmul)))
            full = Image.fromarray(shade(height_mm(arr, p["z"], p["base"], zmul), px_mm))
            q.put(("done", gen, arr, mask, zmul, full, time.time() - t0))
        except Exception as e:  # surfaced in the status bar instead of a dead thread
            q.put(("error", gen, repr(e)))

    def start():
        st["job"] = None
        p = params()
        if st["h"] is None or p is None:
            return
        st["gen"] += 1; st["busy"] = True; prog["value"] = 0
        threading.Thread(target=worker, args=(st["gen"], st["h"], p, canvas.winfo_width(), st["render"]), daemon=True).start()

    def soon(*_):  # debounce: spinbox keystrokes fire many events
        if st["job"]: root.after_cancel(st["job"])
        st["job"] = root.after(350, start)

    # ---- mesh worker: geometry only (X/Y/Z/base/detail), no reprocessing
    def mesh_worker(mgen, arr, mask, zmul, p):
        try:
            H, W = arr.shape
            sx, sy = p["x"] / (W - 1), p["y"] / (H - 1)
            m = preview_mesh(arr, mask, sx, sy, p["z"], p["base"], p["detail"], zmul)
            q.put(("mesh", mgen, m))
            t0 = time.time()  # the real export mesh, cached so Export is instant and the stats are exact
            verts, tris = grid_mesh(arr, mask, sx, sy, p["z"], p["base"], zmul, p["tol"])
            q.put(("fullmesh", mgen, verts, tris, tri_count(arr, mask), time.time() - t0))
        except Exception as e:
            q.put(("error", st["gen"], repr(e)))

    def rebuild_mesh():
        st["mjob"] = None
        p = params()
        if st["proc"] is None or p is None:
            return
        st["mgen"] += 1; st["mesh"] = None
        update_geom_stats(p)
        threading.Thread(target=mesh_worker, args=(st["mgen"], st["proc"], st["mask"], st["zmul"], p), daemon=True).start()

    def soon_mesh(*_):
        if st["mjob"]: root.after_cancel(st["mjob"])
        st["mjob"] = root.after(250, rebuild_mesh)

    def update_geom_stats(p):
        arr, mask = st["proc"], st["mask"]
        H, W = arr.shape
        set_stat("Mesh px", f"{W} × {H}")
        set_stat("mm / pixel", f"{p['x'] / (W - 1):.4f} × {p['y'] / (H - 1):.4f}")
        set_stat("Model mm", f"{p['x']:.1f} × {p['y']:.1f} × {p['base'] + p['z']:.2f}")
        z = height_mm(arr, p["z"], p["base"], st["zmul"])
        set_stat("Z range mm", f"{z.min():.3f} … {z.max():.3f}")
        set_stat("Edge fade", "—" if mask is None else
                 f"{p['fade']} px = {p['fade'] * p['x'] / (W - 1):.2f} mm" if p["fade"] else "none (vertical wall)")
        set_stat("Triangles", "meshing…"); set_stat("STL estimate", "…")

    # ---- X/Y/Z lock: editing one field scales the other two by the same factor
    def on_xyz(k):
        if st["syncing"]:
            return
        try:
            val = v[k].get()
        except tk.TclError:
            return
        if lock.get() and st["last"].get(k):
            f = val / st["last"][k]
            st["syncing"] = True
            for o in ("x", "y", "z"):
                if o != k:
                    v[o].set(round(st["last"][o] * f, 3))
            st["syncing"] = False
        st["last"].update({o: v[o].get() for o in ("x", "y", "z")})
        soon_mesh()

    def fit_y_to_pixels():
        """Locked: Y follows X so pixels stay square when the mesh dims change."""
        H, W = st["proc"].shape
        st["syncing"] = True
        v["y"].set(round(v["x"].get() * (H - 1) / (W - 1), 3))
        st["syncing"] = False
        st["last"].update({o: v[o].get() for o in ("x", "y", "z")})

    def on_done(arr, mask, zmul, full, dt):
        p = params()
        st["proc"], st["mask"], st["zmul"], st["shaded"], st["busy"] = arr, mask, zmul, full, False
        draw(full)
        if lock.get():
            fit_y_to_pixels()
        cov = "kept (full canvas)" if mask is None else f"removed → {mask.mean() * 100:.0f}% of crop is figure"
        set_stat("Canvas", cov)
        set_stat("Process time", f"{dt:.2f} s")
        say(f"ready — processed in {dt:.2f} s, building 3D preview…")
        prog["value"] = 100
        rebuild_mesh()

    def poll():
        try:
            while True:
                m = q.get_nowait()
                if m[0] in ("stage", "done", "error") and m[1] != st["gen"]:
                    continue  # stale generation
                if m[0] == "stage":
                    _, _, label, i, total, dt, img = m
                    draw(img); prog["value"] = 100 * i / total
                    say(f"{label}  ({i}/{total}, {dt:.2f} s)")
                elif m[0] == "done":
                    on_done(*m[2:])
                elif m[0] == "mesh":
                    if m[1] == st["mgen"]:
                        viewer.set_mesh(*m[2])
                        say(f"ready — 3D preview {m[2][2].shape[0]:,} triangles")
                elif m[0] == "fullmesh":
                    if m[1] == st["mgen"]:
                        _, _, verts, tris, full_n, dt = m
                        st["mesh"] = (verts, tris)
                        n = len(tris)
                        set_stat("Triangles", f"{n:,}" + (f"  (from {full_n:,}, −{100 * (1 - n / full_n):.0f}%)" if n < full_n else ""))
                        set_stat("STL estimate", fmt_bytes(stl_bytes(n)))
                        say(f"ready — export mesh {n:,} triangles, meshed in {dt:.1f} s")
                elif m[0] == "error":
                    st["busy"] = False; say("error: " + m[2])
                elif m[0] == "exported":
                    _, path, n, size, dt = m
                    set_stat("Last export", path); set_stat("Export size", fmt_bytes(size))
                    set_stat("Export time", f"{dt:.2f} s")
                    say(f"wrote {path}: {n:,} triangles, {fmt_bytes(size)}")
                elif m[0] == "export_error":
                    say("export failed: " + m[1])
                elif m[0] == "estimated":
                    _, out, rc, last = m
                    est_btn.state(["!disabled"])
                    if rc == 0:
                        load(out); say(f"estimated -> {out}")
                    else:
                        say("estimate failed: " + last[0])
        except queue.Empty:
            pass
        root.after(40, poll)

    def load(p):
        try:
            h, src = load_height(p)
        except Exception as e:
            messagebox.showerror("Open failed", str(e)); return
        st.update(h=h, path=p)
        root.title(f"depth2stl — {os.path.basename(p)}")
        set_stat("Source", p); set_stat("Source size", fmt_bytes(os.path.getsize(p)))
        set_stat("Source px", f"{h.shape[1]} × {h.shape[0]}"); set_stat("Channel", src)
        start()

    def open_():
        start = os.path.dirname(st["path"]) if st["path"] else os.path.expanduser("~")
        p = pick_file(False, start, "", "Images", ["*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"])
        if p: load(p)

    def estimate_():
        """Shaded render -> depth map via the external GPU estimator, then load it."""
        start = os.path.dirname(st["path"]) if st["path"] else os.path.expanduser("~")
        p = pick_file(False, start, "", "Render or photo", ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.tif", "*.tiff"])
        if not p:
            return
        out = os.path.splitext(p)[0] + "_depth.png"
        try:
            detail = est_detail.get()
        except tk.TclError:
            detail = 0.5
        say(f"estimating depth for {os.path.basename(p)} (GPU)…"); est_btn.state(["disabled"])

        def run():
            r = subprocess.run(estimator_cmd() + [p, out, "--detail", str(detail)], capture_output=True, text=True)
            q.put(("estimated", out, r.returncode, r.stderr.strip().splitlines()[-1:] or [""]))
        threading.Thread(target=run, daemon=True).start()

    def export():
        if st["proc"] is None or st["busy"]:
            say("nothing processed yet" if st["proc"] is None else "still processing — wait for 'ready'")
            return
        p = params()
        default = os.path.splitext(os.path.basename(st["path"]))[0] + ".stl"
        out = pick_file(True, os.path.dirname(st["path"]), default, "STL", ["*.stl"])
        if not out:
            return
        if not out.lower().endswith(".stl"):
            out += ".stl"
        say(f"writing {out} …")
        arr, mask, zmul, mesh = st["proc"], st["mask"], st["zmul"], st["mesh"]

        def run():
            t0 = time.time()
            try:
                if mesh is not None:
                    n = write_stl(*mesh, out)
                else:
                    n = to_stl(arr, mask, out, p["x"], p["y"], p["z"], p["base"], zmul, p["tol"])
                q.put(("exported", out, n, os.path.getsize(out), time.time() - t0))
            except Exception as e:
                q.put(("export_error", repr(e)))
        threading.Thread(target=run, daemon=True).start()

    for k in PROC:
        v[k].trace_add("write", soon)
    invert.trace_add("write", soon); rm_canvas.trace_add("write", soon)
    for k in ("x", "y", "z"):
        v[k].trace_add("write", lambda *_, k=k: on_xyz(k))
    for k in ("base", "detail", "tol"):
        v[k].trace_add("write", soon_mesh)
    lock.trace_add("write", lambda *_: (fit_y_to_pixels(), soon_mesh()) if lock.get() and st["proc"] is not None else None)
    st["last"].update({o: v[o].get() for o in ("x", "y", "z")})
    canvas.bind("<Configure>", lambda e: draw(st["shaded"]) if st["shaded"] is not None else None)
    poll()
    if path:
        root.after(100, lambda: load(path))
    root.mainloop()


# ---------------------------------------------------------------- CLI / check
def check():
    h = np.zeros((4, 6), np.float32); h[1:3, 2:4] = 1.0
    p = process(h, scale=2, smooth=0.5)
    assert p.shape == (8, 12) and 0 <= p.min() and p.max() <= 1
    assert process(np.full((2, 2), 0.05, np.float32), floor=0.1).max() == 0, "floor cut"
    # diffusion: kills small noise, keeps a big step
    rng = np.random.default_rng(0)
    step = (np.arange(64) >= 32).astype(np.float32)[None].repeat(16, 0)
    noisy = step + rng.normal(0, 0.01, step.shape).astype(np.float32)
    d = process(noisy, passes=20, edge=0.05)
    assert np.abs(d - step)[:, :28].max() < 0.005, "noise not smoothed"
    assert d[:, 32].min() - d[:, 31].max() > 0.9, "edge not preserved"
    assert n_stages(scale=2, smooth=0.5, passes=3) == 6
    # detail boost raises fine-scale energy, leaves the floor at 0
    bumpy = np.clip(0.5 + 0.05 * np.sin(np.arange(64) * 0.8), 0, 1).astype(np.float32)[None].repeat(16, 0)
    bumpy[:, :8] = 0
    bb = process(bumpy, boost=1.0, boost_radius=4)
    assert np.ptp(bb[:, 16:]) > np.ptp(bumpy[:, 16:]) * 1.3 and bb[:, :8].max() == 0
    assert n_stages(boost=1.0, render=bumpy, render_amt=0.5) == 3
    # render detail: a stripe in the render lifts the height there
    rd = process(np.full((32, 64), 0.5, np.float32), render=bumpy, render_amt=1.0, boost_radius=4)
    assert rd.shape == (32, 64) and np.ptp(rd[:, 16:]) > 0.05

    def closed(verts, tris):
        """Every directed edge has exactly one reverse twin, no duplicates."""
        e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        key = e[:, 0].astype(np.int64) * len(verts) + e[:, 1]
        rev = e[:, 1].astype(np.int64) * len(verts) + e[:, 0]
        return len(np.unique(key)) == len(key) and np.array_equal(np.sort(key), np.sort(rev))

    def volume(verts, tris):
        s = verts[tris].astype(np.float64)
        return np.einsum("ij,ij->i", s[:, 0], np.cross(s[:, 1], s[:, 2])).sum() / 6

    # remove_canvas: a blob plus a far speck -> speck dropped, crop to blob, mask ring
    big = np.zeros((60, 90), np.float32)
    yy, xx = np.mgrid[:60, :90]
    big[(yy - 30) ** 2 + (xx - 40) ** 2 < 15 ** 2] = 0.8
    big[3, 85] = 1.0  # speck brighter than the blob, must still be dropped
    hc, mc, zm = remove_canvas(big, coarse=4)
    assert zm is None
    assert hc.shape[0] <= 34 and hc.shape[1] <= 34 and mc.any() and hc[mc].min() > 0, hc.shape
    assert not mc[0].any() and not mc[-1].any() and not mc[:, 0].any() and not mc[:, -1].any()
    verts, tris = grid_mesh(hc, mc, 0.5, 0.5, 2, 1)
    assert closed(verts, tris), "masked mesh not watertight"
    assert volume(verts, tris) > 0, "masked mesh inside out"
    assert len(tris) == tri_count(hc, mc)
    # simplification: covers every cell exactly once, shrinks the flat-ish blob a lot,
    # keeps the volume within tolerance, and tol=0 is the exact grid
    Cc = mc[:-1, :-1] & mc[1:, :-1] & mc[1:, 1:] & mc[:-1, 1:]
    zmm = 1 + hc * 2
    blocks = merged_blocks(zmm, Cc, 0.05)
    cover = np.zeros(Cc.shape, int)
    for bi, bj, S in blocks:
        for i, j in zip(bi, bj):
            cover[i:i + S, j:j + S] += 1
    assert (cover == Cc.astype(int)).all(), "blocks must tile the masked cells exactly once"
    vs, ts = grid_mesh(hc, mc, 0.5, 0.5, 2, 1, tol=0.05)
    assert len(ts) < 0.5 * len(tris), (len(ts), len(tris))
    assert abs(volume(vs, ts) - volume(verts, tris)) < 0.05 * volume(verts, tris)
    assert len(grid_mesh(hc, mc, 0.5, 0.5, 2, 1, tol=0)[1]) == len(tris)
    flat = np.zeros((70, 70), np.float32); flat[3:-3, 3:-3] = 0.5; fm = flat > 0
    tf = grid_mesh(flat, fm, 1, 1, 2, 1, tol=0.001)[1]
    assert (tf < 70 * 70).all(1).sum() < 0.1 * 2 * 63 * 63, "flat plateau top must collapse >90%"
    # fade: skirt of `fade` px around the figure, ramp 1 inside -> ~0 at the rim, still watertight
    hf, mf, zf = remove_canvas(big, coarse=4, fade=6)
    assert hf.shape[0] == hc.shape[0] + 12 and hf.shape[1] == hc.shape[1] + 12, (hf.shape, hc.shape)
    assert (zf[hf > 0] == 1).all() and 0 < zf[mf & (hf == 0)].max() < 1 and zf[mf & (hf == 0)].min() < 0.1
    vf, tf = grid_mesh(hf, mf, 0.5, 0.5, 2, 1, zf)
    assert closed(vf, tf) and volume(vf, tf) > volume(verts, tris), "fade skirt must add volume, stay closed"
    # unmasked: top+walls watertight apart from the deliberate 2-tri bottom
    verts, tris = grid_mesh(p, None, 1, 1, 2, 1)
    assert volume(verts, tris) > 0 and len(tris) == tri_count(p, None)
    top_walls = tris[~(tris >= p.size).all(1)]  # everything but the 2-tri bottom
    e = np.concatenate([top_walls[:, [0, 1]], top_walls[:, [1, 2]], top_walls[:, [2, 0]]])
    key = e[:, 0].astype(np.int64) * len(verts) + e[:, 1]; rev = e[:, 1].astype(np.int64) * len(verts) + e[:, 0]
    assert (~np.isin(rev, key)).sum() == 2 * (7 + 11), "only the bottom rim may be open"
    assert vertex_normals(verts, tris)[:p.size, 2].min() > 0, "top normals must point up"
    with tempfile.TemporaryDirectory() as dd:
        f = os.path.join(dd, "t.stl")
        n = to_stl(p, None, f, 10, 5, 2, 1)
        assert os.path.getsize(f) == stl_bytes(n) and n == tri_count(p, None)
        with open(f, "rb") as fh:
            fh.seek(80); assert struct.unpack("<I", fh.read(4))[0] == n
            rec = np.fromfile(fh, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        vz = rec["v"][..., 2]
        assert vz.min() == 0 and 1 < vz.max() <= 3 + 1e-3, "z range must be 0..base+relief"
        assert abs(rec["v"][..., 0].max() - 10) < 1e-4 and abs(rec["v"][..., 1].max() - 5) < 1e-4, "X/Y independent"
        cn = np.cross(rec["v"][:, 1] - rec["v"][:, 0], rec["v"][:, 2] - rec["v"][:, 0])
        assert ((cn * rec["n"]).sum(1) > 0).all()
    print("ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("-o", "--out", help="write STL and exit (no GUI)")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--smooth", type=float, default=1.0, help="gaussian sigma in px after upscale")
    ap.add_argument("--floor", type=float, default=2, help="%% cutoff clamped to the floor")
    ap.add_argument("--passes", type=int, default=10, help="edge-preserving diffusion passes")
    ap.add_argument("--edge", type=float, default=5, help="%% of height a step must exceed to be kept")
    ap.add_argument("--boost", type=float, default=0.0, help="gain on the map's own fine detail (0 = off)")
    ap.add_argument("--boost-radius", type=float, default=8, help="px: detail finer than this is boosted")
    ap.add_argument("--render", help="shaded render to lift extra detail from")
    ap.add_argument("--render-amt", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=0.02, help="mm: mesh simplification tolerance (0 = exact grid)")
    ap.add_argument("--remove-canvas", action="store_true", help="crop to the figure, mesh only the figure")
    ap.add_argument("--fade", type=int, default=20, help="skirt width px ramping the cutout edge to the floor")
    ap.add_argument("--x", type=float, default=300, help="model width mm")
    ap.add_argument("--y", type=float, default=None, help="model height mm (default: square pixels)")
    ap.add_argument("--z", type=float, default=8, help="relief height mm")
    ap.add_argument("--base", type=float, default=3)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    if a.out:
        t0 = time.time()
        h, src = load_height(a.image)
        print(f"loaded {src} {h.shape[1]}x{h.shape[0]} in {time.time() - t0:.2f} s")
        render = load_render(a.render) if a.render else None
        for label, p in stages(h, a.scale, a.smooth, a.invert, a.floor / 100, a.passes, a.edge / 100,
                               a.boost, a.boost_radius, render, a.render_amt):
            print(f"  {label:<24s} {time.time() - t0:6.2f} s")
        mask = zmul = None
        if a.remove_canvas:
            p, mask, zmul = remove_canvas(p, fade=a.fade)
            print(f"  {'remove leftover canvas':<24s} {time.time() - t0:6.2f} s  -> {p.shape[1]}x{p.shape[0]} px")
        H, W = p.shape
        y = a.y if a.y is not None else a.x * (H - 1) / (W - 1)
        n = to_stl(p, mask, a.out, a.x, y, a.z, a.base, zmul, a.tol)
        print(f"{W}x{H} px, {a.x:g}x{y:.1f}x{a.base + a.z:g} mm, {n:,} tris, {fmt_bytes(os.path.getsize(a.out))} "
              f"-> {a.out}  ({time.time() - t0:.2f} s total)")
        return
    gui(a.image)


if __name__ == "__main__":
    main()
