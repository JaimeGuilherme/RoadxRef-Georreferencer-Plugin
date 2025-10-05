# -*- coding: utf-8 -*-
import rasterio
import numpy as np
from rasterio.windows import Window

def read_raster_window_select_bands(path, col_off, row_off, width, height, bands_mode="rgbnir"):
    with rasterio.open(path) as src:
        w = Window(col_off, row_off, width, height)
        arr = src.read(window=w)  # (C,H,W)
        transform = src.window_transform(w)
        crs = src.crs
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    # select bands
    C, H, W = arr.shape
    if bands_mode == "rgb":
        if C >= 3:
            out = arr[:3]
        else:
            out = np.zeros((3, H, W), dtype=arr.dtype)
            out[:C] = arr
    else:  # rgbnir
        if C >= 4:
            out = arr[:4]
        elif C == 3:
            nir_zero = np.zeros((1, H, W), dtype=arr.dtype)
            out = np.concatenate([arr, nir_zero], axis=0)
        else:
            tmp3 = np.zeros((3, H, W), dtype=arr.dtype)
            tmp3[:C] = arr
            nir_zero = np.zeros((1, H, W), dtype=arr.dtype)
            out = np.concatenate([tmp3, nir_zero], axis=0)
    # normalize to 0..1 float32 (per-patch min-max)
    out = out.astype(np.float32)
    for b in range(out.shape[0]):
        vmin, vmax = out[b].min(), out[b].max()
        if vmax > vmin:
            out[b] = (out[b] - vmin) / (vmax - vmin)
        else:
            out[b] = 0.0
    return out, transform, crs
