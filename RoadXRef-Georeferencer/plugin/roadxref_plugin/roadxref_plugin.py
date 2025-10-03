
# -*- coding: utf-8 -*-
from qgis.PyQt.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                 QPushButton, QFileDialog, QLineEdit, QCheckBox, QSpinBox,
                                 QGroupBox, QTextEdit, QComboBox, QProgressBar)
from qgis.PyQt.QtCore import Qt
from qgis.core import QgsProject, QgsRasterLayer
from qgis.utils import iface

from osgeo import gdal, osr
import os, sys, traceback, math
import geopandas as gpd
import numpy as np

def try_import_torch():
    errs = []
    torch = None
    try:
        import torch as _torch
        torch = _torch
    except Exception as e:
        errs.append("torch: " + str(e))
    return torch, errs

def try_import_smp():
    try:
        import segmentation_models_pytorch as smp
        return smp
    except Exception:
        return None

from .infer_utils import build_model, adapt_first_conv_if_needed, infer_patches_on_raster, hungarian_pairing

class RoadXRefDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("RoadXRef Georeferencer", parent)
        self.setObjectName("RoadXRefDock")
        self.widget = QWidget(self)
        self.setWidget(self.widget)
        self.v = QVBoxLayout(self.widget)

        # Inputs
        self.le_road = QLineEdit(); btn_road = QPushButton("...")
        btn_road.clicked.connect(lambda: self.pick_file(self.le_road, "Camada OSM (linha)", "GPKG (*.gpkg);;Shapefile (*.shp);;GeoPackage/Vector (*.*)"))
        self.v.addLayout(self.row("Malha rodoviária (linha):", self.le_road, btn_road))

        self.le_raster = QLineEdit(); btn_raster = QPushButton("...")
        btn_raster.clicked.connect(lambda: self.pick_file(self.le_raster, "Imagem de satélite (TIFF)", "GeoTIFF (*.tif *.tiff)"))
        self.v.addLayout(self.row("Imagem a georreferenciar (TIFF):", self.le_raster, btn_raster))

        self.le_model = QLineEdit(); btn_model = QPushButton("...")
        btn_model.clicked.connect(lambda: self.pick_file(self.le_model, "Modelo .pth", "PyTorch (*.pth)"))
        self.v.addLayout(self.row("Modelo treinado (.pth):", self.le_model, btn_model))

        # Advanced
        gb = QGroupBox("Configurações avançadas"); lv = QVBoxLayout(gb)
        # Batch
        self.sp_batch = QSpinBox(); self.sp_batch.setRange(1, 4096); self.sp_batch.setValue(32)
        lv.addLayout(self.row("Batch (padrão 32):", self.sp_batch))
        # Associação radius
        self.sp_radius = QSpinBox(); self.sp_radius.setRange(1, 1000); self.sp_radius.setValue(20)
        lv.addLayout(self.row("Raio de associação (m):", self.sp_radius))
        # Resampling method
        self.cb_resample = QComboBox()
        self.cb_resample.addItems(["nearest", "bilinear", "cubic", "cubicspline", "lanczos"])
        self.cb_resample.setCurrentText("cubic")  # 4x4 kernel
        lv.addLayout(self.row("Reamostragem (GDAL):", self.cb_resample))
        # Polynomial order
        self.cb_poly = QComboBox()
        self.cb_poly.addItems(["1","2","3"])
        self.cb_poly.setCurrentText("1")
        lv.addLayout(self.row("Polinômio (ordem):", self.cb_poly))
        # Exports
        self.cb_export_points = QCheckBox("Exportar .points e .csv de GCPs")
        self.cb_export_points.setChecked(True)
        lv.addWidget(self.cb_export_points)
        self.cb_export_inferred = QCheckBox("Exportar pontos inferidos (GeoJSON)")
        self.cb_export_inferred.setChecked(False)
        lv.addWidget(self.cb_export_inferred)
        self.v.addWidget(gb)

        # Progress bars
        self.pb_infer = QProgressBar(); self.pb_infer.setFormat("Inferência: %p%")
        self.pb_assoc = QProgressBar(); self.pb_assoc.setFormat("Associação: %p%")
        self.pb_warp  = QProgressBar(); self.pb_warp.setFormat("Warp: %p%")
        self.v.addWidget(self.pb_infer); self.v.addWidget(self.pb_assoc); self.v.addWidget(self.pb_warp)

        # Logs
        self.logs = QTextEdit(); self.logs.setReadOnly(True)
        self.v.addWidget(self.logs)

        # Run
        btn_run = QPushButton("Executar georreferenciamento")
        btn_run.clicked.connect(self.run_all)
        self.v.addWidget(btn_run)

    def row(self, label, w1, w2=None):
        h=QHBoxLayout(); h.addWidget(QLabel(label)); h.addWidget(w1)
        if w2: h.addWidget(w2)
        return h

    def pick_file(self, lineedit, title, filter):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter)
        if path: lineedit.setText(path)

    def log(self, msg): 
        self.logs.append(msg); 
        self.logs.verticalScrollBar().setValue(self.logs.verticalScrollBar().maximum())
        QApplication = type(self).parent  # no-op to keep lints quiet

    def set_progress(self, bar, value):
        bar.setValue(int(max(0, min(100, value))))

    def run_all(self):
        road_path = self.le_road.text().strip()
        raster_path = self.le_raster.text().strip()
        model_path = self.le_model.text().strip()
        export_points = self.cb_export_points.isChecked()
        export_inferred = self.cb_export_inferred.isChecked()
        batch = self.sp_batch.value()
        radius = float(self.sp_radius.value())
        resample = self.cb_resample.currentText()
        poly_order = int(self.cb_poly.currentText())

        if not (os.path.isfile(road_path) and os.path.isfile(raster_path) and os.path.isfile(model_path)):
            self.log("⚠️ Verifique os caminhos de entrada (.gpkg/.shp, .tif, .pth).")
            return

        # Reset progress
        self.set_progress(self.pb_infer, 0)
        self.set_progress(self.pb_assoc, 0)
        self.set_progress(self.pb_warp, 0)

        torch, errs = try_import_torch()
        if errs:
            self.log("⚠️ Dependências ausentes: " + "; ".join(errs))
            self.log("   Dica: use um Python/venv com torch instalado e configure o QGIS para usá-lo (Opções→Sistema→Ambiente).")
            return

        try:
            import rasterio
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log(f"🖥️ Dispositivo: {device}")

            # 1) Filtrar malha e extrair interseções
            roads = gpd.read_file(road_path)
            if 'fclass' in roads.columns:
                roads = roads[roads['fclass'].isin(['primary','secondary','tertiary','residential'])].copy()
                self.log(f"🛣️ Rodovias filtradas: {len(roads)}")
            else:
                self.log("⚠️ Coluna 'fclass' não encontrada. Prosseguindo com todas as linhas.")

            if roads.crs is None:
                self.log("⚠️ CRS ausente na malha. Defina um CRS apropriado antes de prosseguir.")
                return

            roads = roads.explode(index_parts=False).reset_index(drop=True)

            # Interseções (com progresso raso baseado no loop)
            sindex = roads.sindex
            nodes = []
            total = len(roads)
            for i, geom in enumerate(roads.geometry):
                for j in sindex.intersection(geom.bounds):
                    if j <= i: continue
                    g2 = roads.geometry.iloc[j]
                    inter = geom.intersection(g2)
                    if not inter.is_empty:
                        if inter.geom_type == 'Point':
                            nodes.append(inter)
                        elif inter.geom_type.startswith('Multi'):
                            nodes += [g for g in inter.geoms if g.geom_type=='Point']
                if total>0 and i % max(1, total//20) == 0:
                    self.set_progress(self.pb_assoc, (i/total)*25)  # reservar 25% para este sub-passo
            gdf_nodes = gpd.GeoDataFrame(geometry=nodes, crs=roads.crs).drop_duplicates()
            self.log(f"✳️ Interseções encontradas: {len(gdf_nodes)}")

            # 2) Inferência na imagem → pontos detectados
            in_ch = 4  # preferência RGBNIR
            smp = try_import_smp()
            arch = "smp_unet" if smp is not None else "custom"
            if smp is None:
                self.log("ℹ️ 'segmentation_models_pytorch' não encontrado — usando UNet própria embutida.")
            model = build_model(arch, in_ch).to(device)
            ckpt = torch.load(model_path, map_location=device)
            adapt_first_conv_if_needed(model, ckpt if isinstance(ckpt, dict) else {})

            thr = 0.5
            if isinstance(ckpt, dict) and 'best_threshold' in ckpt:
                thr = float(ckpt['best_threshold'])
            self.log(f"🧠 Threshold: {thr}")

            # Inferência com progresso grosseiro: varrer raster duas vezes (1ª contagem de tiles, 2ª inferência)
            with rasterio.open(raster_path) as src:
                width, height = src.width, src.height
                tile = 256
                ntiles = ((width + tile - 1)//tile) * ((height + tile - 1)//tile)
            done = 0
            def infer_with_progress(*args, **kwargs):
                nonlocal done, ntiles
                # *Wrap* do generator de tiles para reportar progresso
                import rasterio as _rio
                from rasterio.windows import Window
                preds_centers = []
                with _rio.open(kwargs['tif_path']) as src:
                    crs = src.crs
                    width, height = src.width, src.height
                    tile = 256
                    for top in range(0, height, tile):
                        for left in range(0, width, tile):
                            w = min(tile, width-left); h = min(tile, height-top)
                            window = Window(left, top, w, h)
                            arr = src.read(window=window)
                            transform = src.window_transform(window)
                            from .infer_utils import select_bands, norm01_uint8
                            arr_sel = select_bands(arr, mode="rgbnir")
                            arr_sel = norm01_uint8(arr_sel)
                            import torch as _torch, numpy as _np
                            ten = _torch.from_numpy(arr_sel.astype(_np.float32) / 255.0).unsqueeze(0).to(kwargs['device'])
                            with _torch.no_grad():
                                prob = _torch.sigmoid(kwargs['model'](ten))[0,0].detach().cpu().numpy()
                            mask = (prob > kwargs['threshold']).astype(np.uint8)
                            from scipy.ndimage import label, center_of_mass
                            labeled, numf = label(mask)
                            if numf>0:
                                centers = center_of_mass(mask, labeled, range(1, numf+1))
                                for c in centers:
                                    y, x = c if len(c)==2 else c[-2:]
                                    lon, lat = _rio.transform.xy(transform, int(round(y)), int(round(x)))
                                    preds_centers.append((lon, lat))
                            done += 1
                            self.set_progress(self.pb_infer, min(100, 100*done/max(1,ntiles)))
                if not preds_centers:
                    return gpd.GeoDataFrame(geometry=[], crs=crs)
                from shapely.geometry import Point
                pts = [Point(lon, lat) for lon, lat in preds_centers]
                return gpd.GeoDataFrame(geometry=pts, crs=crs)

            pts_detect = infer_with_progress(tif_path=raster_path, model=model, device=device, threshold=thr)
            self.log(f"🔎 Pontos inferidos: {len(pts_detect)}")
            self.set_progress(self.pb_infer, 100)

            # 3) Associação (Húngaro) com progresso
            cand = gdf_nodes
            # reprojetar para CRS métrico
            if cand.crs.is_geographic:
                cen = cand.unary_union.centroid
                lon, lat = float(cen.x), float(cen.y)
                zone = int(math.floor((lon+180.0)/6.0)+1)
                epsg = 32600+zone if lat>=0 else 32700+zone
                cand = cand.to_crs(f"EPSG:{epsg}")
                pts_detect = pts_detect.to_crs(cand.crs)
            else:
                pts_detect = pts_detect.to_crs(cand.crs)

            # progresso aproximado por blocos
            self.set_progress(self.pb_assoc, 30)
            det_idx, osm_idx, dist = hungarian_pairing(pts_detect, cand, max_distance=radius)
            self.set_progress(self.pb_assoc, 100)
            self.log(f"🤝 Pares formados: {len(det_idx)} (raio={radius} m)")

            # 4) GCPs + Warp
            if len(det_idx) < max(3, poly_order+2):
                self.log(f"⚠️ GCPs insuficientes para polinômio {poly_order}. Necessários ≥ {max(3, poly_order+2)}.")
                return

            import rasterio
            with rasterio.open(raster_path) as src:
                transform = src.transform
                inv = ~transform

            work_dir = os.path.join(os.path.dirname(raster_path), "roadxref_outputs")
            os.makedirs(work_dir, exist_ok=True)

            # Exportações
            if export_inferred:
                out_infer = os.path.join(work_dir, "pontos_inferidos.geojson")
                pts_detect.to_file(out_infer, driver="GeoJSON"); self.log(f"💾 {out_infer}")

            # Criar listas de GCPs e (opcional) export CSV/.points
            gcps = []
            for i_d, i_o in zip(det_idx, osm_idx):
                det_pt = pts_detect.iloc[i_d].geometry
                osm_pt = cand.iloc[i_o].geometry
                px, py = inv * (det_pt.x, det_pt.y)
                gcp = gdal.GCP()
                gcp.GCPPixel = float(px); gcp.GCPLine = float(py)
                gcp.GCPX = float(osm_pt.x); gcp.GCPY = float(osm_pt.y); gcp.GCPZ = 0.0
                gcps.append(gcp)

            if export_points:
                pts_csv = os.path.join(work_dir, "gcp_qgis.csv")
                with open(pts_csv, "w", encoding="utf-8") as f:
                    f.write("id,mapX,mapY,pixelX,pixelY,enable,image\n")
                    for i,(i_d,i_o) in enumerate(zip(det_idx, osm_idx), start=1):
                        det_pt = pts_detect.iloc[i_d].geometry
                        osm_pt = cand.iloc[i_o].geometry
                        px, py = inv * (det_pt.x, det_pt.y)
                        f.write(f"{i},{osm_pt.x},{osm_pt.y},{px},{py},1,\n")
                pts_points = os.path.join(work_dir, "gcp_qgis.points")
                with open(pts_points, "w", encoding="utf-8") as f:
                    for i,(i_d,i_o) in enumerate(zip(det_idx, osm_idx), start=1):
                        det_pt = pts_detect.iloc[i_d].geometry
                        osm_pt = cand.iloc[i_o].geometry
                        px, py = inv * (det_pt.x, det_pt.y)
                        f.write(f"{osm_pt.x},{osm_pt.y},{px},{py},1\n")
                self.log(f"💾 {pts_csv}\n💾 {pts_points}")

            # Criar VRT com GCPs
            ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(int(cand.crs.to_epsg()))
            vrt_path = os.path.join(work_dir, "with_gcps.vrt")
            gdal.Translate(vrt_path, ds, GCPs=gcps, outputSRS=srs.ExportToWkt())

            # Warp com progresso (GDAL não dá callback simples aqui; simulamos)
            out_tif = os.path.join(work_dir, os.path.basename(raster_path).replace(".tif", "_georef.tif"))
            warp_opts = gdal.WarpOptions(
                dstSRS=srs,
                polynomialOrder=poly_order,
                resampleAlg=resample
            )
            self.set_progress(self.pb_warp, 10)
            gdal.Warp(out_tif, vrt_path, options=warp_opts)
            self.set_progress(self.pb_warp, 100)
            self.log(f"✅ Georreferenciado: {out_tif}  (resample={resample}, poly={poly_order})")

            # carregar no QGIS
            rlayer = QgsRasterLayer(out_tif, os.path.basename(out_tif))
            if rlayer.isValid(): QgsProject.instance().addMapLayer(rlayer)

        except Exception as e:
            self.log("❌ Erro: " + str(e))
            self.log(traceback.format_exc())

class RoadXRefPlugin:
    def __init__(self, iface_):
        self.iface = iface_
        self.dock = None

    def initGui(self):
        self.dock = RoadXRefDock(self.iface.mainWindow())
        self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock)

    def unload(self):
        if self.dock is not None:
            self.iface.removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = None
