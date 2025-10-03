
# -*- coding: utf-8 -*-
from .roadxref_plugin import RoadXRefPlugin

def classFactory(iface):
    return RoadXRefPlugin(iface)
