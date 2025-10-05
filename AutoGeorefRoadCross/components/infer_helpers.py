# -*- coding: utf-8 -*-
import os, math
import numpy as np
import torch
from shapely.geometry import Point
import geopandas as gpd
import rasterio
from rasterio.transform import xy

from .unet import UNet
try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except Exception:
    HAS_SMP = False

from .utils import load_checkpoint_raw

class TorchInferencer:
    def __init__(self, model_path: str, bands_mode: str="rgbnir", batch_size: int=32, feedback=None):
        self.model_path = model_path
        self.bands_mode = bands_mode
        self.batch_size = batch_size
        self.feedback = feedback
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_ch = 3 if bands_mode == "rgb" else 4
        self.model = self._build_model()
        self._load_weights()

    def _build_model(self):
        # Prefer SMP UNet if present
        if HAS_SMP:
            model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=self.in_ch, classes=1)
        else:
            model = UNet(in_channels=self.in_ch, out_channels=1)
        return model.to(self.device)

    def _adapt_first_conv_if_needed(self, checkpoint_state):
        state = checkpoint_state.get('model_state_dict', checkpoint_state)
        model_state = self.model.state_dict()
        keys = [k for k in model_state.keys() if k.endswith("weight") and model_state[k].dim() == 4]
        if not keys:
            self.model.load_state_dict(state, strict=False); return
        first_key = keys[0]
        if first_key not in state:
            self.model.load_state_dict(state, strict=False); return
        w_ckpt = state[first_key]; w_model = model_state[first_key]
        in_ckpt, in_model = w_ckpt.shape[1], w_model.shape[1]
        if in_ckpt == in_model:
            self.model.load_state_dict(state, strict=False); return
        if in_ckpt == 3 and in_model == 4:
            w_new = w_model.clone()
            w_new[:, :3] = w_ckpt
            w_new[:, 3:4] = w_ckpt.mean(dim=1, keepdim=True)
            state[first_key] = w_new
        elif in_ckpt == 4 and in_model == 3:
            state[first_key] = w_ckpt[:, :3].contiguous()
        self.model.load_state_dict(state, strict=False)

    def _load_weights(self):
        ckpt = load_checkpoint_raw(self.model_path, map_location=self.device)
        self._adapt_first_conv_if_needed(ckpt)
        self.model.eval()
        self.threshold = float(ckpt.get('best_threshold', 0.5))

    def _infer_batch(self, batch):
        with torch.no_grad():
            x = torch.from_numpy(np.stack(batch, axis=0)).to(self.device)  # (B,C,H,W)
            y = torch.sigmoid(self.model(x))
            y = (y > self.threshold).float().cpu().numpy()  # (B,1,H,W)
        return y

    def run_on_raster(self, raster_path: str, out_points_gpkg: str):
        from .dataset import read_raster_window_select_bands
        os.makedirs(os.path.dirname(out_points_gpkg), exist_ok=True)
        with rasterio.open(raster_path) as src:
            W, H = src.width, src.height
            transform = src.transform
            crs = src.crs
        patch = 256
        batch = []
        tiles = []
        points = []
        for row_off in range(0, H, patch):
            for col_off in range(0, W, patch):
                w = min(patch, W - col_off); h = min(patch, H - row_off)
                if w <=0 or h <=0:
                    continue
                arr, win_transform, _ = read_raster_window_select_bands(raster_path, col_off, row_off, w, h, self.bands_mode)
                batch.append(arr)
                tiles.append((row_off, col_off, h, w, win_transform))
                if len(batch) == self.batch_size:
                    preds = self._infer_batch(batch)
                    points.extend(self._centers_from_preds(preds, tiles))
                    batch, tiles = [], []
        if batch:
            preds = self._infer_batch(batch)
            points.extend(self._centers_from_preds(preds, tiles))

        if len(points):
            gdf = gpd.GeoDataFrame({'id': list(range(1, len(points)+1))}, geometry=[p for p in points], crs=crs)
            gdf.to_file(out_points_gpkg, driver='GPKG')
        return out_points_gpkg

    def _centers_from_preds(self, preds, tiles):
        from scipy.ndimage import label, center_of_mass
        pts = []
        for i in range(len(tiles)):
            mask = preds[i,0].astype(np.uint8)
            labeled, n = label(mask)
            if n == 0:
                continue
            centers = center_of_mass(mask, labeled, range(1, n+1))
            row_off, col_off, h, w, win_transform = tiles[i]
            for c in centers:
                y, x = c if len(c)==2 else (c[1], c[2])
                rx = int(round(x)); ry = int(round(y))
                X, Y = xy(win_transform, ry, rx)  # pixel -> CRS coordinates
                pts.append(Point(X, Y))
        return pts
