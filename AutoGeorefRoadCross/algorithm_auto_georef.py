# -*- coding: utf-8 -*-
import os, tempfile
from datetime import datetime

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm, QgsProcessingException,
    QgsProcessingParameterVectorLayer, QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFile, QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber, QgsProcessingParameterEnum,
    QgsProcessingParameterFolderDestination, QgsProcessingParameterFileDestination,
    QgsProcessingContext, QgsProcessingFeedback, QgsProcessingParameterDefinition,
    QgsVectorLayer, QgsRasterLayer, QgsFields, QgsField, QgsFeature, QgsGeometry, QgsPointXY
)
import processing

# Local components (mantém seu layout com subpasta "components")
from .components.infer_helpers import TorchInferencer
from .components.associate_helpers import associate_points_hungarian, build_gcps_from_pairs
from .components.requirements_check import ensure_requirements

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(PLUGIN_DIR, "models", "best_model.pth")

def _fallback_intersections_by_vertices(lines_layer: QgsVectorLayer, feedback: QgsProcessingFeedback = None) -> QgsVectorLayer:
    if feedback:
        feedback.pushInfo("Aplicando fallback por vértices (grau ≥ 2)…")

    key_to_ids, key_to_xy = {}, {}
    for feat in lines_layer.getFeatures():
        geom = feat.geometry()
        if not geom or geom.isEmpty():
            continue
        for v in geom.vertices():
            x, y = float(v.x()), float(v.y())
            rx, ry = round(x, 6), round(y, 6)
            key = (rx, ry)
            if key not in key_to_ids:
                key_to_ids[key] = set()
                key_to_xy[key] = (x, y)
            key_to_ids[key].add(feat.id())

    out = QgsVectorLayer(f"Point?crs={lines_layer.crs().authid()}", "inters_fallback", "memory")
    pr = out.dataProvider()
    fields = QgsFields()
    fields.append(QgsField("degree", QVariant.Int))
    pr.addAttributes(fields); out.updateFields()

    feats = []
    for key, ids in key_to_ids.items():
        if len(ids) >= 2:
            x, y = key_to_xy[key]
            f = QgsFeature(out.fields())
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
            f.setAttributes([len(ids)])
            feats.append(f)
    pr.addFeatures(feats); out.updateExtents()
    if feedback:
        feedback.pushInfo(f"Fallback gerou {len(feats)} interseções.")
    return out


class AlgoAutoGeoref(QgsProcessingAlgorithm):
    P_ROADS = "roads"
    P_RASTER = "raster"
    P_MODEL = "pth_model"
    P_USE_INTERNAL = "use_internal_model"
    P_BANDSMODE = "bands_mode"
    P_BATCH = "batch_size"
    P_EXPORT_POINTS = "export_points"
    P_EXPORT_INFERRED = "export_inferred_points"
    P_TRANSFORM = "transform_order"
    P_RESAMPLE  = "resample_alg"
    P_OUTPUT_DIR = "output_folder"
    P_OUT_RASTER = "output_raster"

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.P_ROADS, "Malha rodoviária (linhas, com coluna 'fclass' ou 'highway')", [QgsProcessing.TypeVectorLine]
        ))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.P_RASTER, "Imagem de satélite (TIFF)"
        ))

        param_model = QgsProcessingParameterFile(
            self.P_MODEL, "Modelo PyTorch (.pth)",
            behavior=QgsProcessingParameterFile.File
        )
        param_model.setFlags(param_model.flags()
            | QgsProcessingParameterDefinition.FlagOptional
            | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param_model)

        self.addParameter(QgsProcessingParameterEnum(
            self.P_BANDSMODE, "Bandas", options=["rgb", "rgbnir"], defaultValue=1
        ))

        p_batch = QgsProcessingParameterNumber(
            self.P_BATCH, "Batch size (PyTorch)",
            type=QgsProcessingParameterNumber.Integer, defaultValue=32
        )
        p_batch.setFlags(p_batch.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_batch)

        p_exp = QgsProcessingParameterBoolean(
            self.P_EXPORT_POINTS, "Exportar .points / .csv dos homólogos (antes do warp)", defaultValue=True
        )
        p_exp.setFlags(p_exp.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_exp)

        p_inf = QgsProcessingParameterBoolean(
            self.P_EXPORT_INFERRED, "Exportar pontos inferidos (GPKG)", defaultValue=False
        )
        p_inf.setFlags(p_inf.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_inf)

        p_use = QgsProcessingParameterBoolean(
            self.P_USE_INTERNAL, "Usar modelo interno (embutido no plugin)", defaultValue=True
        )
        p_use.setFlags(p_use.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_use)

        p = QgsProcessingParameterEnum(
            self.P_TRANSFORM, "Transformação (GDAL Warp)",
            options=["Polynomial 1", "Polynomial 2", "Polynomial 3", "Thin Plate Spline (TPS)"], defaultValue=0
        )
        p.setFlags(p.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p)

        p = QgsProcessingParameterEnum(
            self.P_RESAMPLE, "Interpolação (GDAL Warp)",
            options=["near","bilinear","cubic (4x4 kernel)","cubicspline","lanczos","average","mode","max","min","med","q1","q3"],
            defaultValue=2
        )
        p.setFlags(p.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p)

        self.addParameter(QgsProcessingParameterFolderDestination(
            self.P_OUTPUT_DIR, "Pasta de saída"
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.P_OUT_RASTER, "Imagem georreferenciada (TIFF)", 'TIFF (*.tif *.tiff)'
        ))

    def name(self): return "autogeoref_roadcross"
    def displayName(self): return "AutoGeoref RoadCross (OSM + UNet + Georef)"
    def group(self): return "Georreferenciamento"
    def groupId(self): return "georeferencing"

    def shortHelpString(self):
        return self.tr("""
Fluxo:
1) Filtra 'fclass'/'highway' (case-insensitive) em {'primary','secondary','tertiary','residential'}; se der 0, usa TODAS as feições.
2) Interseções: native:lineintersections; se 0, fallback por vértices com grau ≥ 2.
3) Inferência (patches 256) com PyTorch (.pth interno/externo).
4) Associação (método húngaro) gera pares det↔OSM.
5) GCPs (pixel←det_x/det_y, mapa←osm_x/osm_y) + GDAL Warp (Polynomial 1 por padrão, cubic 4×4).
        """)

    def tr(self, s): return QCoreApplication.translate("AlgoAutoGeoref", s)

    def processAlgorithm(self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback):
        ensure_requirements(feedback)

        vlayer: QgsVectorLayer = self.parameterAsVectorLayer(parameters, self.P_ROADS, context)
        rlayer: QgsRasterLayer = self.parameterAsRasterLayer(parameters, self.P_RASTER, context)
        bands_mode = ["rgb", "rgbnir"][ self.parameterAsEnum(parameters, self.P_BANDSMODE, context) ]
        batch_size = int(self.parameterAsInt(parameters, self.P_BATCH, context))
        export_points = self.parameterAsBool(parameters, self.P_EXPORT_POINTS, context)
        export_inferred = self.parameterAsBool(parameters, self.P_EXPORT_INFERRED, context)
        transform_idx = self.parameterAsEnum(parameters, self.P_TRANSFORM, context)
        resample_idx  = self.parameterAsEnum(parameters, self.P_RESAMPLE,  context)
        out_folder = self.parameterAsFileOutput(parameters, self.P_OUTPUT_DIR, context)
        out_georef  = self.parameterAsFileOutput(parameters, self.P_OUT_RASTER, context)
        use_internal = self.parameterAsBool(parameters, self.P_USE_INTERNAL, context)

        if vlayer is None or rlayer is None:
            raise QgsProcessingException("Camadas de entrada inválidas.")

        # Normalização de caminhos
        raster_path = rlayer.dataProvider().dataSourceUri().split("|")[0]
        if not out_folder or out_folder.strip() in (".", "./"):
            base_dir = os.path.dirname(raster_path) or tempfile.gettempdir()
            out_folder = os.path.join(base_dir, f"autogeoref_{datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(out_folder, exist_ok=True)

        if not out_georef or out_georef.strip() in (".", "./"):
            stem = os.path.splitext(os.path.basename(raster_path))[0]
            out_georef = os.path.join(out_folder, f"{stem}_georef.tif")
        else:
            if not os.path.isabs(out_georef):
                out_georef = os.path.join(out_folder, out_georef)
            root, ext = os.path.splitext(out_georef)
            if ext.lower() not in (".tif", ".tiff"):
                out_georef = root + ".tif"

        feedback.pushInfo(f"Pasta de saída: {out_folder}")
        feedback.pushInfo(f"Arquivo georreferenciado: {out_georef}")

        # Modelo (.pth)
        pth = DEFAULT_MODEL_PATH if use_internal else os.path.normpath(self.parameterAsFile(parameters, self.P_MODEL, context) or "")
        feedback.pushInfo(("Usando modelo interno: " if use_internal else "Usando modelo externo: ") + str(pth))
        if not pth or not os.path.isfile(pth):
            raise QgsProcessingException("Arquivo .pth não encontrado. Ative 'Usar modelo interno' ou selecione um .pth válido.")
        if not pth.lower().endswith(".pth"):
            raise QgsProcessingException("O arquivo selecionado não tem extensão .pth")

        # 1) Filtro por classe (case-insensitive) — se zerar, usa todas as feições
        feedback.pushInfo("1) Filtrando malha rodoviária por classe…")
        field_names = [f.name().lower() for f in vlayer.fields()]
        classes = "'primary','secondary','tertiary','residential'"
        if 'fclass' in field_names:
            expr = f"lower(\"fclass\") IN ({classes})"
        elif 'highway' in field_names:
            expr = f"lower(\"highway\") IN ({classes})"
        else:
            expr = None
            feedback.pushWarning("Campo 'fclass'/'highway' não encontrado — usando todas as feições.")

        if expr:
            filt = processing.run("native:extractbyexpression", {
                'INPUT': vlayer, 'EXPRESSION': expr, 'OUTPUT': 'memory:'
            }, context=context, feedback=feedback)['OUTPUT']
            if filt.featureCount() == 0:
                feedback.pushWarning("Filtro por classe retornou 0. Usando todas as feições como fallback.")
                filt = vlayer
        else:
            filt = vlayer

        feedback.pushInfo(f"Linhas após filtro: {filt.featureCount()}")

        # Explode e corrige
        single = processing.run("native:multiparttosingleparts", {
            'INPUT': filt, 'OUTPUT': 'memory:'
        }, context=context, feedback=feedback)['OUTPUT']
        fixed = processing.run("native:fixgeometries", {
            'INPUT': single, 'OUTPUT': 'memory:'
        }, context=context, feedback=feedback)['OUTPUT']
        feedback.pushInfo(f"Linhas (single+fix): {fixed.featureCount()}")

        # 2) Interseções
        feedback.pushInfo("2) Calculando interseções (lineintersections)…")
        inters = processing.run("native:lineintersections", {
            'INPUT': fixed, 'INPUT_FIELDS': [], 'INTERSECT': fixed, 'INTERSECT_FIELDS': [],
            'INPUT_FIELDS_PREFIX': '', 'INTERSECT_FIELDS_PREFIX': '', 'OUTPUT': 'memory:'
        }, context=context, feedback=feedback)['OUTPUT']
        inters_uniq = processing.run("native:deleteduplicategeometries", {
            'INPUT': inters, 'OUTPUT': 'memory:'
        }, context=context, feedback=feedback)['OUTPUT']

        cnt = inters_uniq.featureCount()
        feedback.pushInfo(f"Interseções (lineintersections): {cnt}")

        if cnt == 0:
            inters_uniq = _fallback_intersections_by_vertices(fixed, feedback)
            if inters_uniq.featureCount() == 0:
                raise QgsProcessingException("Nenhuma interseção foi encontrada (filtro e fallback retornaram 0).")

        # 3) Inferência
        feedback.pushInfo("3) Inferindo cruzamentos na imagem…")
        inferencer = TorchInferencer(model_path=pth, bands_mode=bands_mode, batch_size=batch_size, feedback=feedback)
        detected_points_gpkg = os.path.join(out_folder, "pontos_inferidos.gpkg")
        inferencer.run_on_raster(raster_path, out_points_gpkg=detected_points_gpkg)
        if export_inferred:
            feedback.pushInfo(f"Pontos inferidos exportados: {detected_points_gpkg}")

        # 4) Associação
        feedback.pushInfo("4) Associando pontos (método húngaro)…")
        pairs_geojson = os.path.join(out_folder, "pares_homologos.geojson")
        gcp_csv, points_txt = associate_points_hungarian(
            detected_points_path=detected_points_gpkg,
            osm_points_layer=inters_uniq,
            output_pairs_geojson=pairs_geojson,
            feedback=feedback
        )
        feedback.pushInfo(f"Pares salvos: {pairs_geojson}")

        # 5) GCPs (pixel de det_x/det_y) + Warp
        from osgeo import gdal
        gdal.UseExceptions()

        gcps = build_gcps_from_pairs(pairs_geojson, raster_path)
        if len(gcps) < 3:
            raise QgsProcessingException("GCPs insuficientes (<3) para Polynomial 1/2/3. (TPS requer ≥3).")

        tmp_dir = tempfile.mkdtemp(prefix="autogeoref_", dir=out_folder)
        tmp_gcpraster = os.path.join(tmp_dir, "with_gcps.tif")
        ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        translate_opts = gdal.TranslateOptions(GCPs=gcps)
        _ = gdal.Translate(tmp_gcpraster, ds, options=translate_opts); ds = None

        _transform_opts = ["poly1", "poly2", "poly3", "tps"]
        _resample_opts  = ["near","bilinear","cubic","cubicspline","lanczos","average","mode","max","min","med","q1","q3"]
        chosen_transform = _transform_opts[transform_idx]
        chosen_resample  = _resample_opts[resample_idx]

        feedback.pushInfo(f"5) Georreferenciando raster (transform: {chosen_transform}; resample: {chosen_resample})")
        dst_wkt = inters_uniq.crs().toWkt()

        if chosen_transform == "tps":
            warp_opts = gdal.WarpOptions(tps=True, resampleAlg=chosen_resample, dstSRS=dst_wkt)
        else:
            order = 1 if chosen_transform == "poly1" else (2 if chosen_transform == "poly2" else 3)
            warp_opts = gdal.WarpOptions(tps=False, order=order, resampleAlg=chosen_resample, dstSRS=dst_wkt)

        _ = gdal.Warp(out_georef, tmp_gcpraster, options=warp_opts)

        if export_points and gcp_csv and points_txt:
            feedback.pushInfo(f"GCP CSV: {gcp_csv}")
            feedback.pushInfo(f".points: {points_txt}")

        feedback.pushInfo("✅ Finalizado.")
        return { self.P_OUTPUT_DIR: out_folder, self.P_OUT_RASTER: out_georef }

    def createInstance(self):
        return AlgoAutoGeoref()
