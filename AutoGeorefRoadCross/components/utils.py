# -*- coding: utf-8 -*-
import os, torch

def load_checkpoint_raw(caminho, map_location='cpu'):
    if not os.path.exists(caminho):
        raise FileNotFoundError(caminho)
    return torch.load(caminho, map_location=map_location, weights_only=False)
