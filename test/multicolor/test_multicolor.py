"""
Tests for the multicolor export. Needs no Earth Engine or internet:
uses the repo's sample DEM and a small hand-made features file.

    python -m pytest test/multicolor      (or)      python test/multicolor/test_multicolor.py
"""
import math
import os
import sys
import tempfile

import manifold3d
import numpy as np
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from touchterrain.common import multicolor as mc  # noqa: E402

DEM = os.path.join(ROOT, "stuff", "geotiff_example.tif")
FEATURES = os.path.join(HERE, "test_features.geojson")
QUIET = lambda *a: None  # noqa: E731

_cache = {}


def _job(**kw):
    j = {"tilewidth": 120, "zscale": 2.0, "basethick": 2.0, "zip_file_name": "t",
         "features_file": FEATURES}
    j.update(kw)
    return j


def _build(**kw):
    """Build once per set of settings; returns (summary, out_dir)."""
    key = tuple(sorted(kw.items()))
    if key not in _cache:
        out = tempfile.mkdtemp()
        _cache[key] = (mc.make_multicolor(DEM, _job(**kw), out, log=QUIET), out)
    return _cache[key]


def _read_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), np.uint32)[0])
        rec = np.frombuffer(f.read(), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n)
    return rec["v"].astype(np.float64)


def _edge_check(tris):
    """Every edge must be used exactly twice with opposite directions
    (closed and consistently oriented)."""
    v = np.round(tris.reshape(-1, 3), 5)
    _, inv = np.unique(v, axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    fwd = {}
    for a, b in map(tuple, e):
        fwd[(a, b)] = fwd.get((a, b), 0) + 1
    bad = sum(1 for (a, b), c in fwd.items() if fwd.get((b, a), 0) != c)
    return bad, len(fwd)


def _volume(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _solid(path):
    """Load an STL back as a manifold3d solid."""
    tris = _read_stl(path).reshape(-1, 3)
    v, inv = np.unique(np.round(tris, 6), axis=0, return_inverse=True)
    m = manifold3d.Manifold(manifold3d.Mesh(vert_properties=v.astype(np.float32),
                                            tri_verts=inv.reshape(-1, 3).astype(np.uint32)))
    assert m.status() == manifold3d.Error.NoError, path
    return m


def test_build_and_check():
    s, out = _build()
    assert set(s["files"]) == {"terrain", "roads", "water", "towns"}
    vols = {}
    for layer, fn in s["files"].items():
        tris = _read_stl(os.path.join(out, fn))
        bad, n = _edge_check(tris)
        assert bad == 0, f"{layer}: {bad} of {n} edges not closed/consistent"
        vols[layer] = _volume(tris)
        assert vols[layer] > 0, f"{layer}: inside-out (negative volume)"
    # toolhead numbers in file names follow the default mapping
    assert s["files"]["roads"].endswith("_T2.stl") and s["files"]["water"].endswith("_T3.stl")
    return s, vols


def test_alignment_same_footprint():
    """All parts share one coordinate system: every feature part lies
    inside the terrain's xy extent."""
    s, out = _build()
    terr = _read_stl(os.path.join(out, s["files"]["terrain"])).reshape(-1, 3)
    lo, hi = terr.min(0), terr.max(0)
    for layer in ("roads", "water", "towns"):
        p = _read_stl(os.path.join(out, s["files"][layer])).reshape(-1, 3)
        assert (p[:, :2] >= lo[:2] - 1e-4).all() and (p[:, :2] <= hi[:2] + 1e-4).all(), layer


def test_parts_never_overlap():
    """Every pair of parts shares at most a surface, never a volume. Parts
    that share a surface can show a rounding-level overlap once written as
    float32 STL; it must stay under 0.0001 mm averaged over the shared area."""
    s, out = _build()
    solids = {k: _solid(os.path.join(out, fn)) for k, fn in s["files"].items()}
    names = list(solids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = (solids[a] ^ solids[b]).volume()
            shared = min(s["area_mm2"].get(a, 1e9), s["area_mm2"].get(b, 1e9))
            assert v < 1e-4 * shared + 1e-3, f"{a} and {b} overlap by {v:.4f} mm3"


def test_layer_volumes_match_outlines():
    """Each layer is its outline times a constant thickness (inlay plus its
    height above the terrain), so the pockets are exactly filled. Flush
    roads add nothing above the terrain; only the bridges stand up."""
    s, _ = _build()
    st = s["settings"]
    inlay = st["inlay_depth_mm"]
    a, v = s["area_mm2"], s["volume_mm3"]
    for layer, h in (("water", st["water_height_mm"]), ("towns", st["town_height_mm"])):
        want = a[layer] * (inlay + h)
        assert abs(v[layer] - want) < 0.005 * want + 0.5, (layer, v[layer], want)
    ab = s["bridge_area_mm2"]
    want = (a["roads"] - ab) * (inlay + st["road_height_mm"]) + ab * (inlay + st["bridge_height_mm"])
    assert abs(v["roads"] - want) < 0.005 * want + 0.5, (v["roads"], want)


def test_flush_roads_with_bridges():
    """Defaults: roads are flush, and raised only where they cross water."""
    s, _ = _build()
    assert s["settings"]["road_height_mm"] == 0.0
    assert s["bridges"] >= 1, "the sample stream crosses the roads; expected a raised crossing"
    assert 0 < s["bridge_area_mm2"] < 0.2 * s["area_mm2"]["roads"]


def _road_centerlines_mm(s, tilewidth):
    """Sample road centerlines and min width, in the same mm frame as the outlines."""
    from osgeo import gdal, osr
    st = mc.settings_with_defaults(_job(tilewidth=tilewidth, layers=["roads"]))
    src = gdal.Open(DEM)
    m_per_mm = src.RasterXSize * abs(src.GetGeoTransform()[1]) / tilewidth
    out = tempfile.mkdtemp()
    rs = mc.resample_dem(DEM, st["feature_res_mm"] * m_per_mm, os.path.join(out, "r.tif"))
    ds, _ = mc.read_dem(rs)
    feats = [f for f in mc.load_features_file(FEATURES, st) if f["layer"] == "roads"]
    shapes, _, _ = mc.features_to_shapes(feats, ds, st, m_per_mm, log=QUIET)
    gt = ds.GetGeoTransform()
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    proj.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(mc._wgs84(), proj)
    y0 = gt[3] + ds.RasterYSize * gt[5]
    lines = []
    for f in feats:
        g = f["geom"].Clone()
        g.Transform(tr)
        lines.append(shapely.geometry.LineString(
            [((x - gt[0]) / m_per_mm, (y - y0) / m_per_mm) for x, y in g.GetPoints()]))
    return shapes["roads"], lines, st, m_per_mm


def test_minimum_width_scales():
    """Roads are never narrower than road_min_width_mm on the print: widened
    at small scale (1:125,000), true width at large scale (1:31,000)."""
    for tilewidth in (120, 480):
        roads, lines, st, m_per_mm = _road_centerlines_mm(None, tilewidth)
        box = roads.envelope.buffer(-1.0)
        for ln in lines:
            ln = ln.intersection(box)
            if ln.is_empty:
                continue
            band = ln.buffer(st["road_min_width_mm"] / 2.0 - 0.025, cap_style="flat")
            missing = band.difference(roads).area
            assert missing < 1e-3 * band.area, \
                f"1:{m_per_mm * 1000:,.0f}: a road is narrower than {st['road_min_width_mm']} mm"


def test_outlines_are_smooth():
    """Outlines follow the real road directions instead of stair-stepping
    along the grid: most of the outline length is not axis-parallel."""
    roads, _, _, _ = _road_centerlines_mm(None, 120)
    axis, total = 0.0, 0.0
    for p in mc._parts(roads):
        c = np.asarray(p.exterior.coords)
        d = np.diff(c, axis=0)
        L = np.hypot(d[:, 0], d[:, 1])
        total += L.sum()
        axis += L[(np.abs(d[:, 0]) < 1e-6) | (np.abs(d[:, 1]) < 1e-6)].sum()
    assert total > 0 and axis / total < 0.2, f"{axis / total:.0%} of the road outline is axis-parallel"


def test_settings_rules():
    s = mc.settings_with_defaults({"road_min_width_mm": 0.5})
    w = mc.check_settings(s, basethick=2, log=QUIET)
    assert any("nozzle" in x for x in w)
    s = mc.settings_with_defaults({})
    w = mc.check_settings(s, basethick=0.5, log=QUIET)
    assert any("basethick" in x for x in w)
    s = mc.settings_with_defaults({"water_height_mm": -0.6})
    try:
        mc.check_settings(s, basethick=2, log=QUIET)
        raise AssertionError("a layer recessed deeper than its inlay should be refused")
    except ValueError:
        pass


def test_overpass_parsing():
    """OSM json (as Overpass returns it) is classified into layers."""
    osm = {"elements": [
        {"type": "way", "tags": {"highway": "primary"},
         "geometry": [{"lat": 44.55, "lon": -108.2}, {"lat": 44.56, "lon": -108.1}]},
        {"type": "way", "tags": {"highway": "residential"},
         "geometry": [{"lat": 44.55, "lon": -108.2}, {"lat": 44.56, "lon": -108.1}]},
        {"type": "way", "tags": {"waterway": "river", "width": "45"},
         "geometry": [{"lat": 44.55, "lon": -108.2}, {"lat": 44.56, "lon": -108.1}]},
        {"type": "relation", "tags": {"natural": "water", "type": "multipolygon"}, "members": [
            {"type": "way", "role": "outer", "geometry": [
                {"lat": 44.55, "lon": -108.10}, {"lat": 44.55, "lon": -108.05}, {"lat": 44.58, "lon": -108.05}]},
            {"type": "way", "role": "outer", "geometry": [
                {"lat": 44.58, "lon": -108.05}, {"lat": 44.58, "lon": -108.10}, {"lat": 44.55, "lon": -108.10}]}]},
        {"type": "way", "tags": {"landuse": "retail"},
         "geometry": [{"lat": 44.55, "lon": -108.2}, {"lat": 44.56, "lon": -108.2},
                      {"lat": 44.56, "lon": -108.19}, {"lat": 44.55, "lon": -108.2}]},
    ]}
    s = mc.settings_with_defaults({})
    f = mc.osm_to_features(osm, s)
    kinds = sorted((x["layer"], x["kind"]) for x in f)
    assert kinds == [("roads", "line"), ("towns", "area"), ("water", "area"), ("water", "line")]
    river = [x for x in f if x["kind"] == "line" and x["layer"] == "water"][0]
    assert river["width_m"] == 45
    lake = [x for x in f if x["kind"] == "area" and x["layer"] == "water"][0]
    assert lake["geom"].GetArea() > 0
    assert "highway" in mc.overpass_query((44.5, -108.3, 44.7, -108.0), s)


if __name__ == "__main__":
    s, vols = test_build_and_check()
    print("closed + oriented: OK;  volumes (mm3):", {k: round(v) for k, v in vols.items()})
    test_alignment_same_footprint(); print("alignment: OK")
    test_parts_never_overlap(); print("no overlaps between parts: OK")
    test_layer_volumes_match_outlines(); print("pockets exactly filled: OK")
    test_flush_roads_with_bridges(); print(f"flush roads, {s['bridges']} raised crossing(s): OK")
    test_minimum_width_scales(); print("minimum width at two scales: OK")
    test_outlines_are_smooth(); print("smooth outlines: OK")
    test_settings_rules(); print("settings rules: OK")
    test_overpass_parsing(); print("OpenStreetMap parsing: OK")
