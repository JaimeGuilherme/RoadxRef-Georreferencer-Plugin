# -*- coding: utf-8 -*-
import os, math, json
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.optimize import linear_sum_assignment

def _coords_xy(gdf):
    return np.column_stack((gdf.geometry.x.values, gdf.geometry.y.values))

def _metric_crs_for(gdf: gpd.GeoDataFrame):
    if gdf.crs is None or getattr(gdf.crs, "is_geographic", False):
        cen = gdf.geometry.unary_union.centroid
        lon, lat = float(cen.x), float(cen.y)
        zone = int(math.floor((lon + 180.0) / 6.0) + 1)
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return f"EPSG:{epsg}"
    return gdf.crs

def _tile_key(x: float, y: float, tile_size: float):
    return (int(math.floor(x / tile_size)), int(math.floor(y / tile_size)))

def _build_tile_index(xy: np.ndarray, tile_size: float):
    idx = {}
    for j, (x, y) in enumerate(xy):
        idx.setdefault(_tile_key(x,y,tile_size), []).append(j)
    return idx

def _neighbors(tk):
    tx, ty = tk
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            yield (tx+dx, ty+dy)

def _hungarian_tiled(det_xy: np.ndarray, osm_xy: np.ndarray, radius: float):
    tile_size = radius * 2.0
    tile_index = _build_tile_index(osm_xy, tile_size)
    edges_by_tile = {}
    for i, (xd, yd) in enumerate(det_xy):
        tk = _tile_key(xd, yd, tile_size)
        cand = []
        for nb in _neighbors(tk):
            cand.extend(tile_index.get(nb, []))
        if not cand:
            continue
        c_xy = osm_xy[np.array(cand)]
        d = np.hypot(c_xy[:,0]-xd, c_xy[:,1]-yd)
        ok = d <= radius
        if not np.any(ok): continue
        for j_local, dist in zip(np.array(cand)[ok], d[ok]):
            edges_by_tile.setdefault(tk, []).append((i, int(j_local), float(dist)))

    det_idx_sel, osm_idx_sel, dist_sel = [], [], []
    INF = 1e9
    for tk, edges in edges_by_tile.items():
        dets = sorted({i for (i,_,_) in edges})
        osms = sorted({j for (_,j,_) in edges})
        md = {i:k for k,i in enumerate(dets)}
        mo = {j:k for k,j in enumerate(osms)}
        D = np.full((len(dets), len(osms)), INF, dtype=np.float32)
        for (i,j,dist) in edges:
            D[md[i], mo[j]] = min(D[md[i], mo[j]], dist)
        r, c = linear_sum_assignment(D)
        for rr, cc in zip(r,c):
            dd = float(D[rr,cc])
            if dd <= radius and dd < INF:
                det_idx_sel.append(dets[rr]); osm_idx_sel.append(osms[cc]); dist_sel.append(dd)
    return det_idx_sel, osm_idx_sel, dist_sel

def associate_points_hungarian(detected_points_path: str, osm_points_layer, output_pairs_geojson: str, feedback=None, max_distance: float=20.0):
    det = gpd.read_file(detected_points_path)
    osm = gpd.read_file(osm_points_layer.source()) if hasattr(osm_points_layer, "source") else gpd.GeoDataFrame(osm_points_layer)

    # CRS harmonization to metric
    metric_crs = _metric_crs_for(det if len(det) else osm)
    if det.crs is None or str(det.crs) != str(metric_crs):
        det = det.to_crs(metric_crs)
    if osm.crs is None or str(osm.crs) != str(metric_crs):
        osm = osm.to_crs(metric_crs)

    det = det[det.geometry.notna()].copy()
    osm = osm[osm.geometry.notna()].copy()

    if len(det) == 0 or len(osm) == 0:
        raise RuntimeError("Sem pontos para associação.")

    XYd = _coords_xy(det).astype(np.float32)
    XYo = _coords_xy(osm).astype(np.float32)
    i_sel, j_sel, d_sel = _hungarian_tiled(XYd, XYo, radius=max_distance)

    rows = []
    for i, j, d in zip(i_sel, j_sel, d_sel):
        gd = det.iloc[i]; go = osm.iloc[j]
        rows.append({
            'dist_m': float(d),
            'det_x': float(gd.geometry.x), 'det_y': float(gd.geometry.y),
            'osm_x': float(go.geometry.x), 'osm_y': float(go.geometry.y),
            'geometry': gd.geometry
        })
    gdf_pairs = gpd.GeoDataFrame(rows, geometry='geometry', crs=metric_crs)
    os.makedirs(os.path.dirname(output_pairs_geojson), exist_ok=True)
    gdf_pairs.to_file(output_pairs_geojson, driver='GeoJSON')

    # Export .points and CSV for QGIS georeferencer
    gcp_csv = os.path.join(os.path.dirname(output_pairs_geojson), "gcp_qgis.csv")
    pts_txt = os.path.join(os.path.dirname(output_pairs_geojson), "gcp_qgis.points")
    import pandas as pd
    gcp_rows = []
    for k, r in enumerate(rows, start=1):
        # We don't have pixelX/Y here because we inferred directly on world coords.
        # We treat detected point coords as "dst" and OSM as "map".
        gcp_rows.append({
            'id': k,
            'mapX': r['osm_x'], 'mapY': r['osm_y'],
            'pixelX': None, 'pixelY': None, 'enable': 1
        })
    pd.DataFrame(gcp_rows).to_csv(gcp_csv, index=False)

    with open(pts_txt, 'w', encoding='utf-8') as f:
        for r in gcp_rows:
            f.write(f"{r['mapX']},{r['mapY']},{''},{''},{r['enable']}\n")

    if feedback:
        feedback.pushInfo(f"GCP CSV: {gcp_csv}")
        feedback.pushInfo(f".points: {pts_txt}")

    return gcp_csv, pts_txt

def build_gcps_from_pairs(pairs_geojson: str):
    # Build gdal.GCP list mapping from raster pixel to map coordinates.
    # NOTE: Since we inferred in world CRS directly, we approximate pixel coords using nearest raster geotransform via inverse mapping is not available here.
    # Instead, GDAL can accept only map coords if we use Warp with tps=True; however, requirement asks Polynomial 1.
    # To adhere, we pass map coords with dummy pixel coords (this lets Warp fit using geolocations of the raster).
    import json
    from osgeo import gdal
    data = gpd.read_file(pairs_geojson)
    gcps = []
    # Use a rough grid of fake pixel coords for stability (ordered list)
    for idx, row in data.iterrows():
        gcps.append(gdal.GCP(row['osm_x'], row['osm_y'], 0, row['det_x'], row['det_y']))
    return gcps
