
# -*- coding: utf-8 -*-
"""
Utilitários de inferência e associação, adaptados dos seus scripts.
- Carregamento de modelo (UNet custom ou SMP Unet)
- Adaptação do 1º conv para 3/4 canais
- Inferência em patches 256x256
- Extração de centros de massa
- Associação via Húngaro (em tiles)
"""

import os, math, numpy as np
from shapely.geometry import Point
import geopandas as gpd
from scipy.ndimage import label, center_of_mass
import rasterio
import torch

# ===== Modelo =====

class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_channels, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)

        self.conv_last = torch.nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x); x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x); x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x); x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        x = self.upsample(x); x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x); x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x); x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        return self.conv_last(x)

def build_model(arch:str, in_ch:int):
    if arch == "custom":
        return UNet(in_channels=in_ch, out_channels=1)
    elif arch == "smp_unet":
        import importlib
        smp = importlib.import_module("segmentation_models_pytorch")
        return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=in_ch, classes=1)
    else:
        raise ValueError("ARCH inválido")

def adapt_first_conv_if_needed(model, checkpoint_state):
    state = checkpoint_state.get('model_state_dict', checkpoint_state)
    model_state = model.state_dict()
    possible_keys = [k for k in model_state.keys() if k.endswith("weight") and model_state[k].dim() == 4]
    if not possible_keys:
        model.load_state_dict(state, strict=False); return
    first_conv_key = possible_keys[0]
    if first_conv_key not in state:
        model.load_state_dict(state, strict=False); return
    w_ckpt = state[first_conv_key]
    w_model = model_state[first_conv_key]
    in_ckpt, in_model = w_ckpt.shape[1], w_model.shape[1]
    if in_ckpt == in_model:
        model.load_state_dict(state, strict=False); return
    if in_ckpt == 3 and in_model == 4:
        w_new = w_model.clone()
        w_new[:, :3, :, :] = w_ckpt
        w_new[:, 3:4, :, :] = w_ckpt.mean(dim=1, keepdim=True)
        state[first_conv_key] = w_new
    elif in_ckpt == 4 and in_model == 3:
        state[first_conv_key] = w_ckpt[:, :3, :, :].contiguous()
    model.load_state_dict(state, strict=False)

# ===== Patches / Inferência =====

def select_bands(image_all, mode="rgbnir"):
    # image_all: (C,H,W)
    C,H,W = image_all.shape
    if mode == "rgb":
        if C >= 3: return image_all[:3]
        if C == 2:
            out = np.zeros((3,H,W), dtype=image_all.dtype); out[:2]=image_all; return out
        if C == 1:
            return np.repeat(image_all, 3, axis=0)
    else:
        if C >= 4: return image_all[:4]
        if C == 3:
            out = np.zeros((4,H,W), dtype=image_all.dtype); out[:3]=image_all; return out
        if C == 2:
            out3 = np.zeros((3,H,W), dtype=image_all.dtype); out3[:2]=image_all
            out4 = np.zeros((4,H,W), dtype=image_all.dtype); out4[:3]=out3; return out4
        if C == 1:
            out3 = np.repeat(image_all, 3, axis=0)
            out4 = np.zeros((4,H,W), dtype=image_all.dtype); out4[:3]=out3; return out4
    return image_all

def norm01_uint8(arr):
    arr = arr.astype(np.float32)
    for b in range(arr.shape[0]):
        mn, mx = arr[b].min(), arr[b].max()
        if mx > mn:
            arr[b] = (arr[b] - mn) / (mx - mn) * 255.0
        else:
            arr[b] = 0
    return arr.astype(np.uint8)

def tile_raster_to_array(src, tile=256):
    """Yield (window, arr[C,H,W], transform)."""
    width, height = src.width, src.height
    for top in range(0, height, tile):
        for left in range(0, width, tile):
            w = min(tile, width-left); h = min(tile, height-top)
            window = rasterio.windows.Window(left, top, w, h)
            arr = src.read(window=window)  # (C,H,W)
            transform = src.window_transform(window)
            yield (window, arr, transform)

def infer_patches_on_raster(tif_path, model, device, bands="rgbnir", batch=32, threshold=0.5, tile=256):
    preds_centers = []
    with rasterio.open(tif_path) as src:
        crs = src.crs
        for window, arr, transform in tile_raster_to_array(src, tile=tile):
            arr_sel = select_bands(arr, mode=bands)
            arr_sel = norm01_uint8(arr_sel)
            # normalize 0..1 and to tensor
            ten = torch.from_numpy(arr_sel.astype(np.float32) / 255.0).unsqueeze(0) # (1,C,H,W)
            ten = ten.to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(ten))[0,0].cpu().numpy()  # (H,W)
            mask = (prob > threshold).astype(np.uint8)
            labeled, numf = label(mask)
            if numf == 0: 
                continue
            centers = center_of_mass(mask, labeled, range(1, numf+1))
            for c in centers:
                y, x = c if len(c)==2 else c[-2:]
                # map to CRS
                lon, lat = rasterio.transform.xy(transform, int(round(y)), int(round(x)))
                preds_centers.append((lon, lat))
    # build GDF
    if not preds_centers:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    pts = [Point(lon, lat) for lon, lat in preds_centers]
    return gpd.GeoDataFrame(geometry=pts, crs=crs)

# ===== Associação (Húngaro em tiles) =====

from scipy.optimize import linear_sum_assignment

def _coords_xy(gdf):
    return np.column_stack((gdf.geometry.x.values, gdf.geometry.y.values))

def _tile_key(x: float, y: float, tile_size: float): return (int(math.floor(x/tile_size)), int(math.floor(y/tile_size)))
def _build_index(xy: np.ndarray, tile_size: float):
    idx={}
    for i,(x,y) in enumerate(xy):
        tk=_tile_key(x,y,tile_size)
        idx.setdefault(tk, []).append(i)
    return idx
def _neighbors(tk):
    tx,ty=tk
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            yield (tx+dx, ty+dy)

def hungarian_pairing(det_gdf, osm_gdf, max_distance=20.0):
    # Assume CRS métrico
    det = _coords_xy(det_gdf).astype(np.float32)
    osm = _coords_xy(osm_gdf).astype(np.float32)
    tile_size = max_distance*2.0
    idx = _build_index(osm, tile_size)
    edges = {}
    for i,(xd,yd) in enumerate(det):
        tk=_tile_key(xd,yd,tile_size)
        cand=[]
        for nb in _neighbors(tk):
            cand += idx.get(nb, [])
        if not cand: continue
        cxy = osm[np.array(cand)]
        d = np.hypot(cxy[:,0]-xd, cxy[:,1]-yd)
        ok = d <= max_distance
        if not np.any(ok): continue
        for j,dist in zip(np.array(cand)[ok], d[ok]):
            edges.setdefault(tk, []).append((i,int(j),float(dist)))
    det_idx_sel, osm_idx_sel, dist_sel = [], [], []
    INF=1e9
    for tk, e in edges.items():
        dets = sorted({i for (i,_,_) in e}); osms = sorted({j for (_,j,_) in e})
        mdet = {i:k for k,i in enumerate(dets)}; mosm = {j:k for k,j in enumerate(osms)}
        D = np.full((len(dets), len(osms)), INF, dtype=np.float32)
        for (i,j,d) in e:
            D[mdet[i], mosm[j]] = min(D[mdet[i], mosm[j]], d)
        r,c = linear_sum_assignment(D)
        for rr,cc in zip(r,c):
            dd = float(D[rr,cc])
            if dd <= max_distance and dd < INF:
                det_idx_sel.append(dets[rr]); osm_idx_sel.append(osms[cc]); dist_sel.append(dd)
    return det_idx_sel, osm_idx_sel, dist_sel
