"""
Multi-color (multi-extruder) export for TouchTerrain.

Builds the terrain plus feature layers (water, roads, towns) as separate,
perfectly aligned solids so a multi-toolhead printer can print each layer
in its own color. Layers sit in shallow pockets ("inlays") cut into the
terrain, so the parts touch but never overlap.

Pipeline:
    elevation GeoTIFF (projected, meters)
      -> resample to the terrain grid (feature_res_mm on the print)
      -> load features (OpenStreetMap via Overpass, or a local GeoJSON file)
      -> smooth outlines in print mm: roads and rivers as bands along their
         true centerlines (at least the minimum printed width), areas with
         their real outlines; overlaps settled by priority
      -> solids with manifold3d: terrain minus the exact pocket under each
         outline; each layer = its outline x a slab that follows the terrain
      -> write one binary STL per layer (+ summary json)

Edges follow the real map outlines, not the terrain grid, so diagonal roads
and curved shorelines are smooth rather than stair-stepped.

Part of TouchTerrain_for_MPH (a fork of TouchTerrain), GPL v3.
"""

import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from osgeo import gdal, ogr, osr  # first: loads the conda C++ runtime the others share
import manifold3d
import shapely
import shapely.affinity
import shapely.geometry
import shapely.ops
import shapely.wkb

gdal.UseExceptions()
ogr.UseExceptions()

# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------

LAYERS = ("roads", "water", "towns")

DEFAULTS = {
    "multicolor": True,
    "layers": ["roads", "water", "towns"],     # which feature layers to build
    "road_min_width_mm": 0.8,                  # narrowest a road may print
    "water_min_width_mm": 1.0,                 # narrowest a river/stream may print
    "min_feature_size_mm": 2.0,                # lakes/towns smaller than this across are dropped
    "road_classes": ["motorway", "trunk", "primary", "secondary", "tertiary"],
    "town_landuse": ["residential", "commercial", "retail", "industrial"],
    "inlay_depth_mm": 0.6,                     # how deep each layer sits into the terrain
    "road_height_mm": 0.0,                     # road surface relative to terrain (0 = flush)
    "bridge_height_mm": 0.4,                   # roads stand this far up where they cross water
    "bridge_extend_mm": 0.5,                   # raised part reaches this far onto each bank
    "water_height_mm": 0.0,                    # water surface relative to terrain (negative = recessed)
    "town_height_mm": 0.0,                     # town surface relative to terrain
    "feature_res_mm": 0.25,                    # terrain grid size on the print
    "outline_tolerance_mm": 0.02,              # outlines simplified to within this (keeps files small)
    "feature_priority": ["roads", "water", "towns"],  # first wins where layers overlap
    "toolheads": {"terrain": 1, "roads": 2, "water": 3, "towns": 4},
    "nozzle_mm": 0.4,                          # only used for warnings
    "features_file": None,                     # optional local GeoJSON instead of OpenStreetMap
    "overpass_url": "https://overpass-api.de/api/interpreter",
    "output": "stl",
}

# typical real-world widths (m) when OSM has no usable width tag
ROAD_WIDTH_M = {"motorway": 25, "trunk": 20, "primary": 15, "secondary": 12,
                "tertiary": 10, "unclassified": 7, "residential": 7, "service": 5}
WATERWAY_WIDTH_M = {"river": 30, "canal": 15, "stream": 4}


def settings_with_defaults(job):
    """Return a dict of all multicolor settings, job values over defaults."""
    s = dict(DEFAULTS)
    for k in DEFAULTS:
        if k in job and job[k] is not None:
            s[k] = job[k]
    s["toolheads"] = {**DEFAULTS["toolheads"], **(job.get("toolheads") or {})}
    bad = [l for l in s["layers"] if l not in LAYERS]
    if bad:
        raise ValueError(f"unknown layer(s) {bad}; allowed: {LAYERS}")
    return s


def check_settings(s, basethick, log=print):
    """Apply the sanity rules; return list of warnings (also logged)."""
    warns = []
    if "roads" in s["layers"]:
        mins = [s["road_min_width_mm"]]
    else:
        mins = []
    if "water" in s["layers"]:
        mins.append(s["water_min_width_mm"])
    for m in mins:
        if m < 2 * s["nozzle_mm"]:
            warns.append(f"minimum width {m} mm is under twice the {s['nozzle_mm']} mm nozzle; "
                         f"it may not print as a solid line")
    for k in ("road_height_mm", "bridge_height_mm", "water_height_mm", "town_height_mm"):
        if s[k] <= -s["inlay_depth_mm"]:
            raise ValueError(f"{k} {s[k]} must be above -inlay_depth_mm ({-s['inlay_depth_mm']})")
    if basethick <= s["inlay_depth_mm"] + 0.2:
        warns.append(f"basethick {basethick} mm is too thin for a {s['inlay_depth_mm']} mm inlay; "
                     f"use at least {s['inlay_depth_mm'] + 0.6:.1f} mm")
    for w in warns:
        log("WARNING:", w)
    return warns


# --------------------------------------------------------------------------
# features: OpenStreetMap (Overpass) or local GeoJSON
# A feature is a dict: {"layer": str, "geom": ogr.Geometry (WGS84 lon/lat),
#                       "kind": "line"|"area", "width_m": float|None}
# --------------------------------------------------------------------------

def _wgs84():
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return sr


def _parse_width(tags):
    w = tags.get("width")
    if not w:
        return None
    try:
        return float(str(w).replace("m", "").strip().split(";")[0])
    except ValueError:
        return None


def overpass_query(bbox, s):
    """bbox = (south, west, north, east) in degrees."""
    b = "{:.6f},{:.6f},{:.6f},{:.6f}".format(*bbox)
    parts = []
    if "roads" in s["layers"]:
        cls = "|".join(s["road_classes"])
        parts.append(f'way["highway"~"^({cls})(_link)?$"]({b});')
    if "water" in s["layers"]:
        parts += [f'way["natural"="water"]({b});', f'relation["natural"="water"]({b});',
                  f'way["waterway"~"^(river|stream|canal|riverbank)$"]({b});',
                  f'relation["waterway"="riverbank"]({b});']
    if "towns" in s["layers"]:
        lu = "|".join(s["town_landuse"])
        parts += [f'way["landuse"~"^({lu})$"]({b});', f'relation["landuse"~"^({lu})$"]({b});']
    return "[out:json][timeout:180];(" + "".join(parts) + ");out geom;"


def fetch_overpass(bbox, s, log=print):
    q = overpass_query(bbox, s)
    data = urllib.parse.urlencode({"data": q}).encode()
    req = urllib.request.Request(s["overpass_url"], data=data,
                                 headers={"User-Agent": "TouchTerrain_for_MPH multicolor"})
    log("downloading OpenStreetMap features from", s["overpass_url"])
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=240) as r:
        osm = json.load(r)
    log(f" got {len(osm.get('elements', []))} OSM elements in {time.time() - t0:.1f} s")
    return osm_to_features(osm, s)


def _line_from_coords(coords):
    g = ogr.Geometry(ogr.wkbLineString)
    for lon, lat in coords:
        g.AddPoint_2D(lon, lat)
    return g


def _poly_from_ring(coords):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for lon, lat in coords:
        ring.AddPoint_2D(lon, lat)
    ring.CloseRings()
    p = ogr.Geometry(ogr.wkbPolygon)
    p.AddGeometry(ring)
    return p


def _classify(tags, s):
    """Return (layer, kind, width_m) for an OSM tag set, or None."""
    hw = tags.get("highway")
    if hw:
        base = hw.replace("_link", "")
        if base in s["road_classes"]:
            w = _parse_width(tags) or ROAD_WIDTH_M.get(base, 8) * (0.6 if hw.endswith("_link") else 1)
            return "roads", "line", w
    if tags.get("natural") == "water" or tags.get("waterway") == "riverbank":
        return "water", "area", None
    ww = tags.get("waterway")
    if ww in WATERWAY_WIDTH_M:
        return "water", "line", _parse_width(tags) or WATERWAY_WIDTH_M[ww]
    if tags.get("landuse") in s["town_landuse"]:
        return "towns", "area", None
    return None


def osm_to_features(osm, s):
    feats = []
    for el in osm.get("elements", []):
        c = _classify(el.get("tags", {}), s)
        if not c or c[0] not in s["layers"]:
            continue
        layer, kind, width = c
        try:
            if el["type"] == "way" and "geometry" in el:
                coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
                if len(coords) < 2:
                    continue
                if kind == "area":
                    if len(coords) < 4 or coords[0] != coords[-1]:
                        continue
                    geom = _poly_from_ring(coords)
                else:
                    geom = _line_from_coords(coords)
            elif el["type"] == "relation" and kind == "area":
                geom = _relation_polygon(el)
                if geom is None:
                    continue
            else:
                continue
        except Exception:
            continue  # skip broken geometry rather than failing the whole job
        feats.append({"layer": layer, "geom": geom, "kind": kind, "width_m": width})
    return feats


def _relation_polygon(el):
    """Assemble an OSM multipolygon relation (out geom) into an OGR polygon."""
    outer = ogr.Geometry(ogr.wkbMultiLineString)
    inner = ogr.Geometry(ogr.wkbMultiLineString)
    for m in el.get("members", []):
        if m.get("type") != "way" or "geometry" not in m:
            continue
        line = _line_from_coords([(p["lon"], p["lat"]) for p in m["geometry"]])
        (inner if m.get("role") == "inner" else outer).AddGeometry(line)
    if outer.GetGeometryCount() == 0:
        return None
    o = outer.Polygonize()
    if o is None or o.IsEmpty():
        return None
    o = o.UnionCascaded() if o.GetGeometryCount() > 1 else o.GetGeometryRef(0).Clone()
    if inner.GetGeometryCount():
        i = inner.Polygonize()
        if i is not None and not i.IsEmpty():
            o = o.Difference(i)
    return o


def load_features_file(path, s):
    """Local GeoJSON in lon/lat. Each feature needs either a 'layer' property
    (roads/water/towns, optional 'width_m') or OSM-style tags as properties."""
    with open(path) as f:
        gj = json.load(f)
    feats = []
    for ft in gj["features"]:
        props = ft.get("properties") or {}
        geom = ogr.CreateGeometryFromJson(json.dumps(ft["geometry"]))
        gtype = ogr.GT_Flatten(geom.GetGeometryType())
        kind = "area" if gtype in (ogr.wkbPolygon, ogr.wkbMultiPolygon) else "line"
        if "layer" in props:
            layer, width = props["layer"], props.get("width_m")
            if width is None and kind == "line":
                width = ROAD_WIDTH_M["tertiary"] if layer == "roads" else WATERWAY_WIDTH_M["stream"]
        else:
            c = _classify(props, s)
            if not c:
                continue
            layer, _, width = c
        if layer in s["layers"]:
            feats.append({"layer": layer, "geom": geom, "kind": kind, "width_m": width})
    return feats


# --------------------------------------------------------------------------
# elevation grid
# --------------------------------------------------------------------------

def read_dem(path):
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)
    a = band.ReadAsArray().astype(np.float64)
    nd = band.GetNoDataValue()
    if nd is not None:
        a[a == nd] = np.nan
    a[a < -1e30] = np.nan
    return ds, a


def resample_dem(src_path, cell_m, out_path):
    """Resample the DEM to a square grid of cell_m meters (bilinear)."""
    gdal.Warp(out_path, src_path, xRes=cell_m, yRes=cell_m, resampleAlg="bilinear",
              dstNodata=-3.4e38, format="GTiff", creationOptions=["COMPRESS=DEFLATE"])
    return out_path


def dem_bbox_wgs84(ds):
    """(south, west, north, east) of a projected dataset."""
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    src = osr.SpatialReference(wkt=ds.GetProjection())
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(src, _wgs84())
    xs, ys = [], []
    for px, py in ((0, 0), (nx, 0), (0, ny), (nx, ny), (nx / 2, 0), (nx / 2, ny), (0, ny / 2), (nx, ny / 2)):
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        lon, lat, _ = tr.TransformPoint(x, y)
        xs.append(lon)
        ys.append(lat)
    return (min(ys), min(xs), max(ys), max(xs))


def clean_diagonals(valid):
    """Remove cells that touch the rest of the model only at a corner
    (a 2x2 checkerboard), which would make a pinched, non-manifold solid."""
    v = valid.copy()
    while True:
        a, b, c, d = v[:-1, :-1], v[:-1, 1:], v[1:, :-1], v[1:, 1:]
        k1 = a & d & ~b & ~c          # cells a and d touch only at a corner
        k2 = b & c & ~a & ~d
        if not (k1.any() or k2.any()):
            return v
        v[:-1, :-1][k1] = False
        v[:-1, 1:][k2] = False


# --------------------------------------------------------------------------
# feature outlines (smooth, in print millimeters)
# --------------------------------------------------------------------------

def _to_shapely(g):
    return shapely.wkb.loads(bytes(g.ExportToWkb()))


def _polygonal(g):
    """Keep only the area parts of a geometry (drops stray lines/points)."""
    if g is None or g.is_empty:
        return shapely.geometry.Polygon()
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    if g.geom_type == "GeometryCollection":
        parts = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
        return shapely.ops.unary_union(parts) if parts else shapely.geometry.Polygon()
    return shapely.geometry.Polygon()


def _parts(g):
    g = _polygonal(g)
    if g.is_empty:
        return []
    return list(g.geoms) if g.geom_type == "MultiPolygon" else [g]


def features_to_shapes(feats, ds, s, m_per_mm, log=print):
    """Turn features into one smooth outline per layer, in millimeters on the
    print (x from the west edge, y from the south edge of the grid).

    Lines (roads, rivers, streams) become bands along their true centerlines,
    at their real width or the minimum printed width, whichever is larger.
    Areas keep their real outlines; a water area too thin everywhere for the
    minimum is grown to it. Overlaps are settled by feature_priority (first
    wins), then slivers and features under min_feature_size_mm are removed.

    Returns (shapes, raw_water, stats): shapes[layer] is a shapely geometry,
    raw_water is all water before roads were cut out of it (used for bridges).
    """
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    x0 = gt[0]
    y0 = gt[3] + ny * gt[5]                       # south edge (gt[5] < 0)
    to_mm = [1.0 / m_per_mm, 0, 0, 1.0 / m_per_mm, -x0 / m_per_mm, -y0 / m_per_mm]
    cell_mm = abs(gt[1]) / m_per_mm
    domain = shapely.geometry.box(0, 0, nx * cell_mm, ny * cell_mm)

    proj = osr.SpatialReference(wkt=ds.GetProjection())
    proj.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(_wgs84(), proj)
    min_w = {"roads": s["road_min_width_mm"], "water": s["water_min_width_mm"]}
    stats = {l: {"features": 0, "widened": 0} for l in s["layers"]}
    pieces = {l: [] for l in s["layers"]}
    # only features near the model matter; skip the rest before buffering
    near = domain.buffer(5.0)
    for f in feats:
        g = f["geom"].Clone()
        g.Transform(tr)
        g = shapely.affinity.affine_transform(_to_shapely(g), to_mm)
        if not g.intersects(near):
            continue
        if not g.is_valid:
            g = shapely.make_valid(g)
        layer = f["layer"]
        stats[layer]["features"] += 1
        if f["kind"] == "line":
            true_w = (f["width_m"] or 0) / m_per_mm
            w = max(true_w, min_w.get(layer, 0))
            if min_w.get(layer, 0) > true_w:
                stats[layer]["widened"] += 1
            g = g.buffer(w / 2.0, quad_segs=8)
        elif layer == "water":
            mw = min_w["water"]
            if g.buffer(-mw / 2.0).is_empty:
                g = g.buffer(mw / 2.0, quad_segs=8)
                stats[layer]["widened"] += 1
        g = _polygonal(g)
        if not g.is_empty:
            pieces[layer].append(g)

    # simplify before settling overlaps, so neighbouring layers share exact edges
    merged = {l: _polygonal(_polygonal(shapely.ops.unary_union(pieces[l])).intersection(domain)
                            .simplify(s["outline_tolerance_mm"], preserve_topology=True))
              for l in s["layers"]}
    raw_water = merged.get("water", shapely.geometry.Polygon())

    order = [l for l in s["feature_priority"] if l in s["layers"]] + \
            [l for l in s["layers"] if l not in s["feature_priority"]]
    shapes, taken = {}, shapely.geometry.Polygon()
    sliver = s["nozzle_mm"] / 4.0                     # strips thinner than half a line are dropped
    min_area = math.pi / 4.0 * s["min_feature_size_mm"] ** 2
    for layer in order:
        g = _polygonal(merged[layer].difference(taken))
        g = _polygonal(g.buffer(-sliver).buffer(sliver, quad_segs=8).intersection(g))
        if layer in ("water", "towns"):
            keep = [p for p in _parts(g) if p.area >= min_area]
            dropped = len(_parts(g)) - len(keep)
            if dropped:
                log(f" {layer}: dropped {dropped} pieces under {s['min_feature_size_mm']} mm across")
            g = shapely.geometry.MultiPolygon(keep) if keep else shapely.geometry.Polygon()
        shapes[layer] = g
        taken = taken.union(g)
    for layer in s["layers"]:
        log(f" {layer}: {stats[layer]['features']} features, "
            f"{stats[layer]['widened']} widened to the minimum, {shapes[layer].area:,.0f} mm2")
    return shapes, raw_water, stats


def bridge_shape(roads, raw_water, s):
    """Parts of the roads that cross water, reaching bridge_extend_mm onto each bank."""
    if roads.is_empty or raw_water.is_empty:
        return shapely.geometry.Polygon()
    reach = raw_water.buffer(s["bridge_extend_mm"]) if s["bridge_extend_mm"] > 0 else raw_water
    return _polygonal(roads.intersection(reach))


# --------------------------------------------------------------------------
# solids (manifold3d)
# --------------------------------------------------------------------------

def corner_heights(z):
    """Cell-centre heights (ny,nx, NaN outside) -> corner heights (ny+1,nx+1),
    each the mean of the valid cells touching that corner."""
    ny, nx = z.shape
    p = np.full((ny + 2, nx + 2), np.nan)
    p[1:-1, 1:-1] = z
    stack = np.stack([p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]])
    with np.errstate(invalid="ignore"):
        cnt = np.sum(~np.isnan(stack), axis=0)
        s = np.nansum(stack, axis=0)
        h = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return h


def heightfield_solid(valid, hc, cell_mm, top_off, bottom=None, bot_off=None):
    """Closed solid over the valid cells: top surface = terrain + top_off,
    bottom = flat at `bottom`, or terrain + bot_off. Every solid built here
    shares the same corner grid and triangle split, so surfaces at the same
    offset coincide exactly."""
    ny, nx = valid.shape
    used = np.zeros((ny + 1, nx + 1), bool)
    for di in (0, 1):
        for dj in (0, 1):
            used[di:di + ny, dj:dj + nx] |= valid
    idx = -np.ones(used.shape, np.int64)
    nc = int(used.sum())
    idx[used] = np.arange(nc)
    ci, cj = np.nonzero(used)
    x = cj * cell_mm
    y = (ny - ci) * cell_mm                       # row 0 is the north edge
    zt = hc[ci, cj] + top_off
    zb = np.full(nc, float(bottom)) if bottom is not None else hc[ci, cj] + bot_off
    verts = np.concatenate([np.stack([x, y, zt], 1), np.stack([x, y, zb], 1)]).astype(np.float32)

    I, J = np.nonzero(valid)
    A, B = idx[I, J], idx[I, J + 1]               # NW, NE
    C, D = idx[I + 1, J + 1], idx[I + 1, J]       # SE, SW
    faces = [np.stack([A, D, C], 1), np.stack([A, C, B], 1),           # top
             np.stack([A, C, D], 1) + nc, np.stack([A, B, C], 1) + nc]  # bottom
    vp = np.zeros((ny + 2, nx + 2), bool)
    vp[1:-1, 1:-1] = valid
    for p, q, ni, nj in ((A, D, I, J - 1), (D, C, I + 1, J), (C, B, I, J + 1), (B, A, I - 1, J)):
        k = ~vp[ni + 1, nj + 1]                   # outer edge: wall from top down to bottom
        pp, qq = p[k], q[k]
        faces += [np.stack([pp + nc, qq + nc, qq], 1), np.stack([pp + nc, qq, pp], 1)]
    tri = np.concatenate(faces).astype(np.uint32)
    m = manifold3d.Manifold(manifold3d.Mesh(vert_properties=verts, tri_verts=tri))
    if m.status() != manifold3d.Error.NoError:
        raise RuntimeError(f"terrain solid is not closed: {m.status()}")
    return m


def prism(shape, z_lo, z_hi):
    """Vertical prism with the outline of `shape` (shapely, mm)."""
    rings = []
    for p in _parts(shape):
        rings.append(np.asarray(p.exterior.coords, np.float64)[:-1])
        rings += [np.asarray(r.coords, np.float64)[:-1] for r in p.interiors]
    if not rings:
        return manifold3d.Manifold()
    cs = manifold3d.CrossSection(rings, manifold3d.FillRule.EvenOdd)
    return manifold3d.Manifold.extrude(cs, z_hi - z_lo).translate((0, 0, z_lo))


def manifold_triangles(m):
    """(n,3,3) float32 triangles of a manifold, outward facing."""
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    t = np.asarray(mesh.tri_verts)
    return v[t].astype(np.float32)


def write_stl(path, tris, name="touchterrain"):
    tris = np.asarray(tris, np.float32)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = (n / np.where(ln == 0, 1, ln)).astype(np.float32)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = tris
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b" "))
        f.write(np.uint32(len(tris)).tobytes())
        f.write(rec.tobytes())


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------

def make_multicolor(dem_path, job, out_dir, log=print):
    """Build multicolor STLs from a projected (meters) DEM GeoTIFF.

    job: TouchTerrain-style settings (tilewidth, zscale, basethick, ...) plus
         the multicolor settings in DEFAULTS.
    Returns a summary dict; files are written into out_dir.
    """
    s = settings_with_defaults(job)
    tilewidth = float(job.get("tilewidth", 120))
    zscale = float(job.get("zscale", 1.0))
    basethick = float(job.get("basethick", 2.0))
    warns = check_settings(s, basethick, log)
    os.makedirs(out_dir, exist_ok=True)

    src = gdal.Open(dem_path)
    gt = src.GetGeoTransform()
    if osr.SpatialReference(wkt=src.GetProjection()).IsGeographic():
        raise ValueError("DEM must be in a projected (meter) coordinate system, e.g. UTM")
    width_m = src.RasterXSize * abs(gt[1])
    m_per_mm = width_m / tilewidth
    cell_m = s["feature_res_mm"] * m_per_mm
    log(f"model {tilewidth} mm wide = {width_m:.0f} m -> scale 1:{m_per_mm * 1000:,.0f}")
    log(f"terrain grid {s['feature_res_mm']} mm = {cell_m:.1f} m cells")
    src = None

    rs_path = os.path.join(out_dir, "_dem_resampled.tif")
    resample_dem(dem_path, cell_m, rs_path)
    ds, elev = read_dem(rs_path)
    valid = clean_diagonals(~np.isnan(elev))
    if not valid.any():
        raise ValueError("DEM has no valid cells")
    cell_mm = abs(ds.GetGeoTransform()[1]) / m_per_mm
    log(f"grid {elev.shape[1]} x {elev.shape[0]} cells")

    # features, as smooth outlines
    if s["features_file"]:
        log("reading features from", s["features_file"])
        feats = load_features_file(s["features_file"], s)
    else:
        feats = fetch_overpass(dem_bbox_wgs84(ds), s, log)
    shapes, raw_water, fstats = features_to_shapes(feats, ds, s, m_per_mm, log)
    bridges = shapely.geometry.Polygon()
    if "roads" in shapes and s["bridge_height_mm"] != s["road_height_mm"]:
        bridges = bridge_shape(shapes["roads"], raw_water, s)
        if not bridges.is_empty:
            log(f" roads: {len(_parts(bridges))} crossings over water raised "
                f"{s['bridge_height_mm']} mm")

    # heights in mm on the print
    min_el = float(job["min_elev"]) if job.get("min_elev") is not None else float(np.nanmin(elev))
    z = (elev - min_el) / m_per_mm * zscale + basethick
    hc = corner_heights(z)
    z_lo = -1.0
    z_hi = float(np.nanmax(hc)) + max(5.0, s["bridge_height_mm"], s["road_height_mm"]) + 1.0

    inlay = s["inlay_depth_mm"]
    offsets = {"roads": s["road_height_mm"], "water": s["water_height_mm"],
               "towns": s["town_height_mm"]}
    slabs = {}

    def slab(top_off):
        if top_off not in slabs:
            slabs[top_off] = heightfield_solid(valid, hc, cell_mm, top_off, bot_off=-inlay)
        return slabs[top_off]

    t0 = time.time()
    terrain = heightfield_solid(valid, hc, cell_mm, 0.0, bottom=0.0)
    all_feat = shapely.ops.unary_union([g for g in shapes.values() if not g.is_empty])
    bodies = {}
    if not all_feat.is_empty:
        # pocket = everything inside the feature outlines above (terrain - inlay)
        above = heightfield_solid(valid, hc, cell_mm, z_hi, bot_off=-inlay)
        terrain = terrain - (prism(all_feat, z_lo, z_hi) ^ above)
    bodies["terrain"] = terrain
    for layer in s["layers"]:
        g = shapes[layer]
        if g.is_empty:
            log(f" {layer}: nothing in this area, no file written")
            continue
        if layer == "roads" and not bridges.is_empty:
            flat = _polygonal(g.difference(bridges))
            body = prism(flat, z_lo, z_hi) ^ slab(offsets["roads"])
            body = body + (prism(bridges, z_lo, z_hi) ^ slab(s["bridge_height_mm"]))
        else:
            body = prism(g, z_lo, z_hi) ^ slab(offsets[layer])
        bodies[layer] = body
    for name, b in bodies.items():
        if b.status() != manifold3d.Error.NoError:
            raise RuntimeError(f"{name}: solid could not be built ({b.status()})")
    log(f" solids built in {time.time() - t0:.1f} s")

    base = job.get("zip_file_name", "terrain")
    files = {}
    for name, body in bodies.items():
        tris = manifold_triangles(body)
        th = s["toolheads"].get(name, "")
        fn = f"{base}_{name}_T{th}.stl" if th else f"{base}_{name}.stl"
        path = os.path.join(out_dir, fn)
        write_stl(path, tris, name)
        files[name] = fn
        log(f" wrote {fn}: {len(tris):,} triangles, {os.path.getsize(path) / 1048576:.1f} MB")

    os.remove(rs_path)
    summary = {
        "scale": f"1:{m_per_mm * 1000:,.0f}",
        "model_mm": [round(elev.shape[1] * cell_mm, 2), round(elev.shape[0] * cell_mm, 2)],
        "terrain_grid_mm": round(cell_mm, 4),
        "settings": {k: s[k] for k in DEFAULTS if k not in ("overpass_url",)},
        "features": fstats,
        "area_mm2": {name: round(g.area, 1) for name, g in shapes.items()},
        "bridges": len(_parts(bridges)),
        "bridge_area_mm2": round(bridges.area, 1),
        "volume_mm3": {name: round(b.volume(), 1) for name, b in bodies.items()},
        "files": files,
        "warnings": warns,
    }
    with open(os.path.join(out_dir, f"{base}_multicolor_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
