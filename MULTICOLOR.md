# Multicolor export (TouchTerrain_for_MPH)

Exports the terrain plus **roads, water and towns as separate, aligned STL parts**
so a multi-toolhead printer (built for the Flashforge Creator 5 + Flash Studio)
can print each layer in its own color.

## How it works
1. Elevation comes from Earth Engine (same sources as TouchTerrain) or a local GeoTIFF (`importedDEM`).
2. Roads, water and town areas come from OpenStreetMap (Overpass API), or a local GeoJSON (`features_file`).
3. Features become **smooth outlines** in print millimeters: roads and rivers are bands along their
   real centerlines, areas keep their real shorelines and edges. Nothing is snapped to the terrain grid,
   so diagonal roads and curved shores are not stair-stepped.
4. Every road and thin river is widened to a **minimum printed width** in mm, whatever the map scale.
5. Each layer sits in an exact pocket cut into the terrain (`inlay_depth_mm`), so parts touch but never
   overlap (solids built with manifold3d).
6. Roads are **flush** with the terrain and stand up only where they **cross water** (bridges and causeways).
7. One closed STL per layer is written, named `<name>_<layer>_T<toolhead>.stl`.

## Run
```
docker compose run --rm touchterrain python TouchTerrain_multicolor.py /jobs/<job>.json
```
Offline check (no Earth Engine or internet): `jobs/offline-sample.json`.
Real test area: `jobs/stansbury-test.json`.

## Settings (job file)
| Setting | Default | Meaning |
|---|---|---|
| layers | roads, water, towns | layers to build |
| road_min_width_mm | 0.8 | narrowest a road prints |
| water_min_width_mm | 1.0 | narrowest a river/stream prints |
| min_feature_size_mm | 2.0 | lakes/towns smaller than this across are dropped |
| road_classes | motorway..tertiary | OSM road types included |
| inlay_depth_mm | 0.6 | depth of each layer's pocket |
| road_height_mm / water_height_mm / town_height_mm | 0 / 0 / 0 | surface above terrain (0 = flush) |
| bridge_height_mm | 0.4 | how far roads stand up where they cross water |
| bridge_extend_mm | 0.5 | raised part reaches this far onto each bank |
| feature_res_mm | 0.25 | terrain grid size on the print (outlines no longer depend on it) |
| outline_tolerance_mm | 0.02 | outlines simplified to within this, keeps files small |
| feature_priority | roads, water, towns | first wins where layers overlap |
| toolheads | terrain 1, roads 2, water 3, towns 4 | used in file names (3MF later) |
| features_file | none | local GeoJSON instead of OpenStreetMap |

Widened roads now print at the minimum width itself (0.8 mm), not 0.9-1.2 mm as with the old
grid-painted layers.

## In Flash Studio
Select all layer STLs in one Import dialog and answer **Yes** when asked to load them as a single
object with multiple parts. Loaded as separate objects, every part drops to the bed and the
terrain covers the rest. Then set each part's filament to the toolhead in its file name.

## Tests
`python test/multicolor/test_multicolor.py` - closed meshes, alignment, no overlaps, pockets exactly
filled, flush roads with raised crossings, minimum widths, smooth outlines, OSM parsing.
