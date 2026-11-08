"""depth2stl's plugin API: how another program (Cut Craft) drives the compiled binary
without its window. Plain JSON over the standard streams, so any language can call it.

  depth2stl api describe        -> one JSON object on stdout: the API version, this
                                   program's version and a JSON Schema of every setting
                                   (labels, ranges, units, the GUI's tooltips as descriptions)
  depth2stl api run < job.json  -> JSON lines on stdout: {"stage", "step", "of"} while it
                                   works, then {"ok": true, ...} or {"ok": false, "error"}

A job:
  {"api": 1,
   "input": "depth.png",           grey depth map, white = high (16-bit kept)
   "render": "photo.png",          optional: the picture the map came from, for fine detail
   "heightmap": "out.png",         optional: 16-bit grey PNG of the finished model, value =
                                   height / (base + relief), 0 outside the kept figure
   "stl": "out.stl",               optional: the binary STL (at least one output is needed)
   "settings": {...}}              any of describe's settings; the rest take their defaults

Nothing here opens a window or imports Tk, so it runs on a headless machine.
"""
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

API_VERSION = 1

# Every setting a job can carry, in the order a form shows them: key, JSON type, default,
# minimum, maximum, unit, label, the GUI hint that explains it, and the form group.
# Defaults follow the GUI's (DEFAULTS), not the bare CLI's, so a plugin user gets the same
# result as someone in the depth2stl window.
SETTINGS = [
    ("fit_to_bit", "boolean", True, None, None, None, "Fit to bit", "fit", "Bit"),
    ("tip", "number", 0.397, 0.01, 25.0, "mm", "Tip diameter", "tip", "Bit"),
    ("taper", "number", 10.0, 0.0, 45.0, None, "Taper (degrees per side)", "taper", "Bit"),
    ("simulate_bit", "boolean", False, None, None, None, "Keep only what the bit can carve", "sim", "Bit"),
    ("width", "number", 300.0, 1.0, 5000.0, "mm", "Width", "x", "Size"),
    ("relief", "number", 8.0, 0.01, 500.0, "mm", "Relief height", "z", "Size"),
    ("base", "number", 3.0, 0.0, 500.0, "mm", "Base", "base", "Size"),
    ("invert", "boolean", False, None, None, None, "Invert (black is high)", "invert", "Size"),
    ("remove_canvas", "boolean", True, None, None, None, "Remove background", "rm", "Shape"),
    ("fade", "integer", 20, 0, 400, None, "Edge skirt (px)", "fade", "Shape"),
    ("floor", "number", 2.0, 0.0, 90.0, None, "Floor cut-off (%)", "floor", "Shape"),
    ("scale", "integer", 2, 1, 4, None, "Upscale", "scale", "Detail"),
    ("smooth", "number", 1.0, 0.0, 10.0, None, "Smoothing (px)", "smooth", "Detail"),
    ("passes", "integer", 10, 0, 100, None, "Edge-keeping passes", "passes", "Detail"),
    ("edge", "number", 5.0, 0.0, 100.0, None, "Edge threshold (%)", "edge", "Detail"),
    ("boost", "number", 0.0, 0.0, 5.0, None, "Detail boost", "boost", "Detail"),
    ("boost_radius", "number", 8.0, 1.0, 200.0, None, "Detail size (px)", "boost_radius", "Detail"),
    ("render_amount", "number", 0.5, 0.0, 1.0, None, "Detail from picture", "render", "Detail"),
    ("tolerance", "number", 0.02, 0.0, 1.0, "mm", "STL simplification", "tol", "Detail"),
]
FIT_NOTE = (" With Fit to bit on, upscale, smoothing, edge-keeping, detail boost and detail size "
            "are worked out from the bit and the width, and those settings are ignored.")


def schema(hints):
    props = {}
    for key, typ, default, lo, hi, unit, label, hint, group in SETTINGS:
        p = {"type": typ, "title": label, "default": default, "x-group": group,
             "description": hints[hint] + (FIT_NOTE if key == "fit_to_bit" else "")}
        if lo is not None:
            p["minimum"], p["maximum"] = lo, hi
        if unit:
            p["x-unit"] = unit
        props[key] = p
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "properties": props, "additionalProperties": False}


def settings_of(given):
    """The job's settings over the defaults, checked against SETTINGS."""
    out = {key: default for key, _, default, *_ in SETTINGS}
    for key, value in (given or {}).items():
        spec = next((s for s in SETTINGS if s[0] == key), None)
        if spec is None:
            raise ValueError(f"unknown setting {key!r}")
        _, typ, _, lo, hi, *_ = spec
        if typ == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a number")
            if typ == "integer":
                if value != int(value):
                    raise ValueError(f"{key} must be a whole number")
                value = int(value)
            if not lo <= value <= hi:
                raise ValueError(f"{key} must be from {lo:g} to {hi:g}")
        out[key] = value
    return out


def fit(tip, width_mm, columns):
    """Fit to bit: about 4 mesh points across the tip, detail boosted up to ~1 mm,
    simplification well under the tip radius. Same rule as the GUI's Fit to bit."""
    raw = width_mm / max(columns - 1, 1)
    scale = min(4, max(1, math.ceil(raw / (tip / 4))))
    pitch = raw / scale
    return dict(scale=scale, smooth=0.5, passes=10, edge=3.0, boost=1.0,
                boost_radius=float(max(2, round(1.0 / pitch))), render_amount=0.8,
                tolerance=round(tip / 20, 3))


def run(job, d2s, say):
    """Does one job (a parsed request) with the pipeline module d2s; say(dict) reports
    progress. Returns the final report."""
    if job.get("api", API_VERSION) != API_VERSION:
        raise ValueError(f"this depth2stl speaks API {API_VERSION}, not {job.get('api')}")
    src = job.get("input")
    if not src:
        raise ValueError("the job names no input picture")
    if not (job.get("heightmap") or job.get("stl")):
        raise ValueError("the job asks for no output: give heightmap, stl or both")
    s = settings_of(job.get("settings"))
    t0 = time.time()
    h, note = d2s.load_height(src)
    if s["fit_to_bit"]:
        # the width is the kept figure's, so with the background removed the pitch comes
        # from the figure's columns, not the whole canvas's
        columns = h.shape[1]
        if s["remove_canvas"]:
            pre = 1.0 - h if s["invert"] else h
            floor = s["floor"] / 100
            if floor > 0:
                pre = np.clip((pre - floor) / (1.0 - floor), 0.0, 1.0)
            columns = d2s.remove_canvas(pre)[0].shape[1]
        s.update(fit(s["tip"], s["width"], columns))
    render = d2s.load_render(job["render"]) if job.get("render") else None
    kw = dict(scale=s["scale"], smooth=s["smooth"], invert=s["invert"], floor=s["floor"] / 100,
              passes=s["passes"], edge=s["edge"] / 100, boost=s["boost"],
              boost_radius=s["boost_radius"], render=render, render_amt=s["render_amount"])
    total = d2s.n_stages(**kw) + s["remove_canvas"] + s["simulate_bit"] + 1
    step = 0
    for label, p in d2s.stages(h, **kw):
        step += 1
        say({"stage": label, "step": step, "of": total})
    mask = zmul = None
    if s["remove_canvas"]:
        p, mask, zmul = d2s.remove_canvas(p, fade=s["fade"])
        step += 1
        say({"stage": "remove background", "step": step, "of": total})
    rows, columns = p.shape
    if rows < 2 or columns < 2:
        raise ValueError("nothing is left of the picture: lower the floor cut-off")
    loss = None
    if s["simulate_bit"]:
        cut = d2s.simulate_bit(p, s["width"] / (columns - 1), s["relief"], s["tip"], s["taper"])
        loss = d2s.bit_loss(p, cut, s["relief"], mask)
        p = cut
        step += 1
        say({"stage": "bit simulation", "step": step, "of": total})
    height = s["width"] * (rows - 1) / (columns - 1)
    top = s["base"] + s["relief"]
    if job.get("heightmap"):
        z = d2s.height_mm(p, s["relief"], s["base"], zmul)
        if mask is not None:
            z = np.where(mask, z, 0.0)
        grey = np.round(np.clip(z / top, 0.0, 1.0) * 65535).astype(np.uint16)
        Image.fromarray(grey).save(job["heightmap"])
    triangles = None
    if job.get("stl"):
        triangles = d2s.to_stl(p, mask, job["stl"], s["width"], height, s["relief"], s["base"], zmul,
                               s["tolerance"])
    step += 1
    say({"stage": "written", "step": step, "of": total})
    return {"ok": True, "columns": columns, "rows": rows, "width_mm": s["width"],
            "height_mm": round(height, 4), "top_mm": top, "base_mm": s["base"],
            "pixel_mm": s["width"] / (columns - 1), "source": note, "bit": loss,
            "triangles": triangles, "settings": s, "seconds": round(time.time() - t0, 2)}


def main(argv, d2s, version):
    """`depth2stl api ...`; d2s is the depth2stl module. Returns the exit status."""
    out = sys.stdout

    def emit(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

    if argv == ["describe"]:
        emit({"api": API_VERSION, "id": "depth2stl", "name": "Depth to STL", "version": version,
              "outputs": ["heightmap", "stl"], "settings": schema(d2s.HINTS)})
        return 0
    if argv == ["run"]:
        try:
            job = json.load(sys.stdin)
            if not isinstance(job, dict):
                raise ValueError("the job is not a JSON object")
            emit(run(job, d2s, emit))
            return 0
        except Exception as e:  # noqa: BLE001  every failure is reported to the caller as JSON
            emit({"ok": False, "error": str(e) or type(e).__name__})
            return 1
    sys.stderr.write("usage: depth2stl api describe | depth2stl api run < job.json\n")
    return 2


def check(d2s):
    """Self-test for --check: describe, a run to both outputs, and refusals."""
    import io
    import tempfile
    sc = schema(d2s.HINTS)
    assert all(p["description"] for p in sc["properties"].values()), "every setting explained"
    assert settings_of({})["remove_canvas"] is True and settings_of({"scale": 3.0})["scale"] == 3
    for bad in ({"scale": 9}, {"nope": 1}, {"invert": 1}, {"scale": 1.5}, {"width": float("nan")}):
        try:
            settings_of(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    assert fit(0.397, 300, 6001)["scale"] == 1 and fit(0.397, 300, 301)["scale"] == 4
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.png")
        yy, xx = np.mgrid[:60, :90]
        blob = ((yy - 30) ** 2 + (xx - 45) ** 2 < 20 ** 2) * (40000 + 100 * xx)
        Image.fromarray(blob.astype(np.uint16)).save(src)
        hm, stl = os.path.join(d, "h.png"), os.path.join(d, "o.stl")
        said = []
        r = run({"input": src, "heightmap": hm, "stl": stl,
                 "settings": {"width": 50, "fit_to_bit": False, "scale": 1, "passes": 2, "fade": 4}},
                d2s, said.append)
        assert r["ok"] and said[-1]["step"] == said[-1]["of"] == len(said), said
        g = np.asarray(Image.open(hm))
        assert g.dtype == np.uint16 and g.shape == (r["rows"], r["columns"]) and g.max() > 45000, g.max()
        assert g[0].max() == 0 and r["columns"] < 90, "background cut away, cropped"
        assert os.path.getsize(stl) == d2s.stl_bytes(r["triangles"])
        r2 = run({"input": src, "heightmap": hm, "settings": {"width": 50}}, d2s, lambda _: None)
        assert r2["settings"]["scale"] > 1 and r2["triangles"] is None, "fit to bit upscales a small map"
        # a small figure on a wide canvas: fit to bit sizes for the figure, not the canvas
        wide = np.zeros((60, 900), np.uint16); wide[20:40, 440:460] = 50000
        Image.fromarray(wide).save(src)
        r3 = run({"input": src, "heightmap": hm, "settings": {"width": 20}}, d2s, lambda _: None)
        assert r3["settings"]["scale"] == 4, r3["settings"]["scale"]
        for job in ({"input": src}, {"heightmap": hm}, {"api": 2, "input": src, "heightmap": hm}):
            try:
                run(job, d2s, lambda _: None)
            except ValueError:
                continue
            raise AssertionError(f"ran {job}")
        old = sys.stdin, sys.stdout
        try:
            sys.stdin, sys.stdout = io.StringIO("[]"), io.StringIO()
            assert main(["run"], d2s, "t") == 1 and '"ok": false' in sys.stdout.getvalue()
            sys.stdout = io.StringIO()
            assert main(["describe"], d2s, "t") == 0
            assert json.loads(sys.stdout.getvalue())["api"] == API_VERSION
        finally:
            sys.stdin, sys.stdout = old
