# -*- coding: utf-8 -*-
def ensure_requirements(feedback=None):
    # Basic runtime checks
    import sys
    pyv = sys.version_info
    if feedback:
        feedback.pushInfo(f"Python: {pyv.major}.{pyv.minor}.{pyv.micro}")
    try:
        import torch
        if feedback:
            feedback.pushInfo(f"PyTorch: {torch.__version__} | CUDA: {'OK' if torch.cuda.is_available() else 'CPU'}")
    except Exception as e:
        if feedback:
            feedback.reportError("PyTorch não está disponível no Python do QGIS. Instale torch/torchvision compatíveis com Python 3.9.", fatalError=False)
    # Optional SMP
    try:
        import segmentation_models_pytorch
        if feedback:
            feedback.pushInfo("segmentation_models_pytorch disponível.")
    except Exception:
        if feedback:
            feedback.pushInfo("segmentation_models_pytorch NÃO encontrado (usando UNet simples).")
    try:
        import geopandas, rasterio, shapely, scipy
    except Exception as e:
        if feedback:
            feedback.reportError("Dependências geoespaciais faltando: geopandas/rasterio/shapely/scipy.", fatalError=False)
