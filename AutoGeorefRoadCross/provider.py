# -*- coding: utf-8 -*-
from qgis.core import QgsProcessingProvider
from .algorithm_auto_georef import AlgoAutoGeoref

class AutoGeorefProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(AlgoAutoGeoref())

    def id(self):
        return "autogeoref_roadcross"

    def name(self):
        return "AutoGeoref RoadCross"

    def longName(self):
        return "AutoGeoref RoadCross (PyTorch + OSM)"
