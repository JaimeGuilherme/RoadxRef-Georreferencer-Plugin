# -*- coding: utf-8 -*-
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import QgsProcessingProvider
from .provider import AutoGeorefProvider

class AutoGeorefRoadCrossPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initGui(self):
        from qgis.core import QgsApplication
        self.provider = AutoGeorefProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        from qgis.core import QgsApplication
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)
