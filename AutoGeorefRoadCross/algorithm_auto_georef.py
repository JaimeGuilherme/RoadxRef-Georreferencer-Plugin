# -*- coding: utf-8 -*-
import os, tempfile, math, json, sys, shutil
from typing import List, Tuple

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm, QgsProcessingException,
    QgsProcessingParameterVectorLayer, QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFile, QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber, QgsProcessingParameterEnum,
    QgsProcessingParameterFolderDestination, QgsProcessingParameterRasterDestination,
    QgsProcessingParameterFileDestination, QgsProcessingContext, QgsProcessingFeedback,
    QgsVectorLayer, QgsRasterLayer, QgsFields, QgsField, QgsFeature, QgsGeometry,
    QgsProject, QgsCoordinateTransformContext, QgsWkbTypes, QgsCoordinateReferenceSystem,
    QgsProcessingParameterDefinition
)
import processing

from .components.dataset import read_raster_window_select_bands
from .components.infer_helpers import TorchInferencer
from .components.associate_helpers import associate_points_hungarian, build_gcps_from_pairs
from .components.requirements_check import ensure_requirements

class AlgoAutoGeoref(QgsProcessingAlgorithm):
    P_ROADS = "roads"
    P_RASTER = "raster"
    P_MODEL = "pth_model"
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
            self.P_ROADS, "Malha rodoviária (linhas, com coluna 'fclass')", [QgsProcessing.TypeVectorLine]
        ))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.P_RASTER, "Imagem de satélite (TIFF)"
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.P_MODEL, "Modelo PyTorch (.pth)", extension="pth", behavior=QgsProcessingParameterFile.File, fileFilter="*.pth"
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.P_BANDSMODE, "Bandas", options=["rgb", "rgbnir"], defaultValue=1
        ))
        p_batch = QgsProcessingParameterNumber(self.P_BATCH, "Batch size (PyTorch)", 
                                               type=QgsProcessingParameterNumber.Integer, defaultValue=32)
        p_batch.setFlags(p_batch.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_batch)

        p_exp = QgsProcessingParameterBoolean(self.P_EXPORT_POINTS, "Exportar .points / .csv dos homólogos (antes do warp)", defaultValue=True)
        p_exp.setFlags(p_exp.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_exp)

        p_inf = QgsProcessingParameterBoolean(self.P_EXPORT_INFERRED, "Exportar pontos inferidos (GPKG)", defaultValue=False)
        p_inf.setFlags(p_inf.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_inf)

        p = QgsProcessingParameterEnum(
            self.P_TRANSFORM,
            "Transformação (GDAL Warp)",
            options=["Polynomial 1", "Polynomial 2", "Polynomial 3", "Thin Plate Spline (TPS)"],
            defaultValue=0
        )
        p.setFlags(p.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p)

        p = QgsProcessingParameterEnum(
            self.P_RESAMPLE,
            "Interpolação (GDAL Warp)",
            options=["near","bilinear","cubic (4x4 kernel)","cubicspline","lanczos","average","mode","max","min","med","q1","q3"],
            defaultValue=2
        )
        p.setFlags(p.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p)

        self.addParameter(QgsProcessingParameterFolderDestination(
            self.P_OUTPUT_DIR, "Pasta de saída"
        ))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.P_OUT_RASTER, "Imagem georreferenciada (TIFF)"
        ))

    def name(self):
        return "autogeoref_roadcross"

    def displayName(self):
        return "AutoGeoref RoadCross (OSM + UNet + Georef)"

    def group(self):
        return "Georreferenciamento"

    def groupId(self):
        return "georeferencing"

    def shortHelpString(self):
        return self.tr("""
Fluxo:
1) Filtra 'fclass' em {'primary','secondary','tertiary','residential'} e calcula interseções de linhas.
2) Tile 256x256 do raster e inferência com PyTorch (.pth).
3) Associação detectados↔OSM (método húngaro) para gerar GCPs.
4) Georreferencia com Polynomial 1 (ordem 1) e reamostragem Cúbica (4x4) por padrão.
(Opções avançadas permitem escolher Polynomial 2/3 ou TPS e outros métodos de interpolação.)

Opções avançadas: batch (padrão 32), export de GCPs (.points/.csv) e dos pontos inferidos.
        """)

    def tr(self, string):
        return QCoreApplication.translate("AlgoAutoGeoref", string)

    def processAlgorithm(self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback):
        ensure_requirements(feedback)

        vlayer: QgsVectorLayer = self.parameterAsVectorLayer(parameters, self.P_ROADS, context)
        rlayer: QgsRasterLayer = self.parameterAsRasterLayer(parameters, self.P_RASTER, context)
        pth = self.parameterAsFile(parameters, self.P_MODEL, context)
        bands_mode_idx = self.parameterAsEnum(parameters, self.P_BANDSMODE, context)
        bands_mode = ["rgb", "rgbnir"][bands_mode_idx]
        batch_size = int(self.parameterAsInt(parameters, self.P_BATCH, context))
        export_points = self.parameterAsBool(parameters, self.P_EXPORT_POINTS, context)
        export_inferred = self.parameterAsBool(parameters, self.P_EXPORT_INFERRED, context)
        transform_idx = self.parameterAsEnum(parameters, self.P_TRANSFORM, context)
        resample_idx  = self.parameterAsEnum(parameters, self.P_RESAMPLE,  context)
        out_folder = self.parameterAsFileOutput(parameters, self.P_OUTPUT_DIR, context)
        out_georef = self.parameterAsOutputLayer(parameters, self.P_OUT_RASTER, context)

        if vlayer is None or rlayer is None:
            raise QgsProcessingException("Camadas de entrada inválidas.")
        if not os.path.exists(pth):
            raise QgsProcessingException("Arquivo .pth não encontrado.")

        os.makedirs(out_folder, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="autogeoref_", dir=out_folder)

        feedback.pushInfo("1) Filtrando malha rodoviária por fclass…")
        expr = "\"fclass\" IN ('primary','secondary','tertiary','residential')"
        filt = processing.run("native:extractbyexpression", {
            'INPUT': vlayer,
            'EXPRESSION': expr,
            'OUTPUT': 'memory:'
        }, context=context, feedback=feedback)['OUTPUT']

        feedback.pushInfo("2) Calculando interseções…")
        inters = processing.run("native:lineintersections", {
            'INPUT': filt,
            'INPUT_FIELDS': [],
            'INTERSECT': filt,
            'INTERSECT_FIELDS': [],
            'INPUT_FIELDS_PREFIX': '',
            'INTERSECT_FIELDS_PREFIX': '',
            'OUTPUT': 'memory:'
        }, context=context, feedback=feedback)['OUTPUT']

        inters_uniq = processing.run("native:deleteduplicategeometries", {
            'INPUT': inters, 'OUTPUT':'memory:'
        }, context=context, feedback=feedback)['OUTPUT']
        feedback.pushInfo(f"Interseções únicas: {inters_uniq.featureCount()}")

        feedback.pushInfo("3) Inferindo cruzamentos na imagem…")
        inferencer = TorchInferencer(model_path=pth, bands_mode=bands_mode, batch_size=batch_size, feedback=feedback)
        raster_path = rlayer.dataProvider().dataSourceUri().split("|")[0]
        detected_points_gpkg = os.path.join(out_folder, "pontos_inferidos.gpkg")
        inferencer.run_on_raster(raster_path, out_points_gpkg=detected_points_gpkg)

        if export_inferred:
            feedback.pushInfo(f"Pontos inferidos exportados: {detected_points_gpkg}")

        feedback.pushInfo("4) Associando pontos (método húngaro)…")
        pairs_geojson = os.path.join(out_folder, "pares_homologos.geojson")
        gcp_csv, points_txt = associate_points_hungarian(
            detected_points_path=detected_points_gpkg,
            osm_points_layer=inters_uniq,
            output_pairs_geojson=pairs_geojson,
            feedback=feedback
        )
        feedback.pushInfo(f"Pares salvos: {pairs_geojson}")

        gcps = build_gcps_from_pairs(pairs_geojson)
        if len(gcps) < 3:
            raise QgsProcessingException("GCPs insuficientes (<3) para Polynomial 1/2/3. Para TPS, ainda são necessários >=3 GCPs.")

        from osgeo import gdal
        gdal.UseExceptions()

        tmp_gcpraster = os.path.join(tmp_dir, "with_gcps.tif")
        ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        translate_opts = gdal.TranslateOptions(GCPs=gcps)
        _ = gdal.Translate(tmp_gcpraster, ds, options=translate_opts)
        ds = None

        _transform_opts = ["poly1", "poly2", "poly3", "tps"]
        _resample_opts  = ["near","bilinear","cubic","cubicspline","lanczos","average","mode","max","min","med","q1","q3"]

        chosen_transform = _transform_opts[transform_idx]
        chosen_resample  = _resample_opts[resample_idx]

        feedback.pushInfo(f"5) Georreferenciando raster (transform: {chosen_transform}; resample: {chosen_resample})")

        dst_wkt = inters_uniq.crs().toWkt()

        if chosen_transform == "tps":
            warp_opts = gdal.WarpOptions(
                tps=True,
                resampleAlg=chosen_resample,
                dstSRS=dst_wkt
            )
        else:
            order = 1 if chosen_transform == "poly1" else (2 if chosen_transform == "poly2" else 3)
            warp_opts = gdal.WarpOptions(
                tps=False,
                order=order,
                resampleAlg=chosen_resample,
                dstSRS=dst_wkt
            )

        _ = gdal.Warp(out_georef, tmp_gcpraster, options=warp_opts)

        if export_points and gcp_csv and points_txt:
            feedback.pushInfo(f"GCP CSV: {gcp_csv}")
            feedback.pushInfo(f".points: {points_txt}")

        feedback.pushInfo("✅ Finalizado.")
        return {
            self.P_OUTPUT_DIR: out_folder,
            self.P_OUT_RASTER: out_georef
        }

    def createInstance(self):
        return AlgoAutoGeoref()
