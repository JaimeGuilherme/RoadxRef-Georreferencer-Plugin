# -*- coding: utf-8 -*-
import os, math, numpy as np
from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY, QgsFields, QgsField,
    QgsCoordinateTransform, QgsCoordinateReferenceSystem, QgsVectorFileWriter, QgsProject
)
from osgeo import gdal

def _utm_from_layer(layer: QgsVectorLayer) -> QgsCoordinateReferenceSystem:
    ext = layer.extent()
    cx, cy = ext.center().x(), ext.center().y()
    zone = int(math.floor((cx + 180.0) / 6.0) + 1)
    epsg = 32600 + zone if cy >= 0 else 32700 + zone
    return QgsCoordinateReferenceSystem.fromEpsgId(epsg)

def _as_layer(obj, name='layer') -> QgsVectorLayer:
    if isinstance(obj, QgsVectorLayer):
        return obj
    vl = QgsVectorLayer(obj, name, 'ogr')
    if not vl or not vl.isValid():
        raise RuntimeError(f'Não foi possível abrir camada: {obj}')
    return vl

def _linear_sum_assignment(cost):
    cost = np.array(cost, dtype=np.float64)
    n, m = cost.shape
    INF = cost.max() + 1e6 if cost.size else 1e9
    if n < m:
        pad = np.full((m-n, m), INF, dtype=np.float64); cost = np.vstack([cost, pad]); n = m
    elif m < n:
        pad = np.full((n, n-m), INF, dtype=np.float64); cost = np.hstack([cost, pad]); m = n
    C = cost.copy()
    C -= C.min(axis=1, keepdims=True)
    C -= C.min(axis=0, keepdims=True)
    N = C.shape[0]
    starred = np.zeros((N, N), dtype=bool)
    primed  = np.zeros((N, N), dtype=bool)
    row_cov = np.zeros(N, dtype=bool)
    col_cov = np.zeros(N, dtype=bool)
    for i in range(N):
        for j in range(N):
            if C[i, j] == 0 and not row_cov[i] and not col_cov[j]:
                starred[i, j] = True; row_cov[i] = True; col_cov[j] = True; break
    row_cov[:] = False; col_cov[:] = False
    def cover(): col_cov[:] = starred.any(axis=0); row_cov[:] = False
    def find_a_zero():
        for i in range(N):
            if row_cov[i]: continue
            for j in range(N):
                if (not col_cov[j]) and C[i, j] == 0 and not primed[i, j]:
                    return i, j
        return None
    def find_star_in_row(i):
        js = np.where(starred[i])[0]; return int(js[0]) if len(js) else None
    def find_star_in_col(j):
        is_ = np.where(starred[:, j])[0]; return int(is_[0]) if len(is_) else None
    def find_prime_in_row(i):
        js = np.where(primed[i])[0]; return int(js[0]) if len(js) else None
    def augment_path(path):
        for (ii, jj) in path: starred[ii, jj] = not starred[ii, jj]
    def clear_primes(): primed[:] = False
    cover()
    while col_cov.sum() < N:
        z = find_a_zero()
        if z is None:
            mask = ~row_cov[:, None] & ~col_cov[None, :]
            s = C[mask].min()
            C[~row_cov] -= s; C[:, col_cov] += s; continue
        i, j = z; primed[i, j] = True
        j_star = find_star_in_row(i)
        if j_star is not None:
            row_cov[i] = True; col_cov[j_star] = False
        else:
            path = [(i, j)]
            while True:
                i_star = find_star_in_col(path[-1][1])
                if i_star is None: break
                path.append((i_star, path[-1][1]))
                j_prime = find_prime_in_row(i_star)
                path.append((i_star, j_prime))
            augment_path(path); clear_primes(); cover()
    rows, cols = np.where(starred)
    return rows, cols

def associate_points_hungarian(detected_points_path, osm_points_layer, output_pairs_geojson,
                               feedback=None, max_distance=20.0):
    det = _as_layer(detected_points_path, 'det')
    osm = _as_layer(osm_points_layer, 'osm')

    metric = _utm_from_layer(osm if osm.featureCount() > 0 else det)
    tr_det = QgsCoordinateTransform(det.crs(), metric, QgsProject.instance())
    tr_osm = QgsCoordinateTransform(osm.crs(), metric, QgsProject.instance())

    det_feats = list(det.getFeatures())
    osm_feats = list(osm.getFeatures())
    if not det_feats or not osm_feats:
        raise RuntimeError("Sem pontos suficientes para associação.")

    det_xy, det_xy_src = [], []
    for f in det_feats:
        p_src = f.geometry().asPoint()
        p = tr_det.transform(p_src)
        det_xy_src.append((p_src.x(), p_src.y()))  # guardar XY no CRS original da detecção (para pixel depois)
        det_xy.append((p.x(), p.y()))
    osm_xy = []
    for f in osm_feats:
        p = tr_osm.transform(f.geometry().asPoint())
        osm_xy.append((p.x(), p.y()))

    det_xy = np.array(det_xy, dtype=np.float64)
    osm_xy = np.array(osm_xy, dtype=np.float64)

    INF = 1e9
    cost = np.full((len(det_feats), len(osm_feats)), INF, dtype=np.float64)
    for i, (xd, yd) in enumerate(det_xy):
        d = np.hypot(osm_xy[:, 0] - xd, osm_xy[:, 1] - yd)
        ok = d <= max_distance
        cost[i, ok] = d[ok]

    r, c = _linear_sum_assignment(cost)

    # Saída: pares contendo det_x/det_y (em CRS da detecção) e osm_x/osm_y (em CRS do OSM)
    fields = QgsFields()
    fields.append(QgsField('dist_m', QVariant.Double))
    fields.append(QgsField('det_x', QVariant.Double))
    fields.append(QgsField('det_y', QVariant.Double))
    fields.append(QgsField('osm_x', QVariant.Double))
    fields.append(QgsField('osm_y', QVariant.Double))

    out = QgsVectorLayer(f'Point?crs={osm.crs().authid()}', 'pares', 'memory')
    pr = out.dataProvider()
    pr.addAttributes(fields); out.updateFields()

    feats_out = []
    for i, j in zip(r, c):
        if i < len(det_feats) and j < len(osm_feats) and cost[i, j] < INF and cost[i, j] <= max_distance:
            fd = det_feats[i]; fo = osm_feats[j]
            osm_pt = fo.geometry().asPoint()
            det_src_x, det_src_y = det_xy_src[i]

            f = QgsFeature(out.fields())
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(osm_pt.x(), osm_pt.y())))
            f.setAttributes([
                float(cost[i, j]),
                float(det_src_x), float(det_src_y),
                float(osm_pt.x()), float(osm_pt.y())
            ])
            feats_out.append(f)

    pr.addFeatures(feats_out); out.updateExtents()

    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = 'GeoJSON'
    _err, _ = QgsVectorFileWriter.writeAsVectorFormatV3(
        out, output_pairs_geojson, QgsProject.instance().transformContext(), options
    )

    # Também geramos CSV/.points para o georreferenciador (pixel será preenchido depois)
    csv_path  = os.path.join(os.path.dirname(output_pairs_geojson), 'gcp_qgis.csv')
    pts_path  = os.path.join(os.path.dirname(output_pairs_geojson), 'gcp_qgis.points')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('id,mapX,mapY,pixelX,pixelY,enable\n')
        for k, ft in enumerate(out.getFeatures(), start=1):
            f.write(f"{k},{ft['osm_x']},{ft['osm_y']},,,1\n")
    with open(pts_path, 'w', encoding='utf-8') as f:
        for ft in out.getFeatures():
            f.write(f"{ft['osm_x']},{ft['osm_y']},,,1\n")

    if feedback:
        feedback.pushInfo(f"GCP CSV: {csv_path}")
        feedback.pushInfo(f".points: {pts_path}")

    return csv_path, pts_path

def build_gcps_from_pairs(pairs_geojson, raster_path):
    """ Constrói lista de gdal.GCP usando:
        - pixelX/pixelY = inverso(GeoTransform do raster) sobre (det_x, det_y)
        - mapX/mapY     = (osm_x, osm_y)
    """
    vl = QgsVectorLayer(pairs_geojson, 'pares', 'ogr')
    if not vl or not vl.isValid():
        raise RuntimeError(f'Não foi possível abrir pares: {pairs_geojson}')

    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'Não foi possível abrir raster: {raster_path}')
    gt = ds.GetGeoTransform()
    inv = gdal.InvGeoTransform(gt)
    if inv is None:
        raise RuntimeError('Falha ao inverter GeoTransform do raster.')
    inv_gt, ok = inv
    if not ok:
        raise RuntimeError('GeoTransform não invertível.')

    def world_to_pixel(x, y):
        px = inv_gt[0] + inv_gt[1]*x + inv_gt[2]*y
        py = inv_gt[3] + inv_gt[4]*x + inv_gt[5]*y
        return float(px), float(py)

    gcps = []
    for ft in vl.getFeatures():
        det_x = float(ft['det_x']); det_y = float(ft['det_y'])
        osm_x = float(ft['osm_x']); osm_y = float(ft['osm_y'])
        px, py = world_to_pixel(det_x, det_y)
        gcps.append(gdal.GCP(osm_x, osm_y, 0.0, px, py))

    return gcps
