# -*- coding: utf-8 -*-
from .plugin import AutoGeorefRoadCrossPlugin

def classFactory(iface):  # QGIS calls this
    return AutoGeorefRoadCrossPlugin(iface)
