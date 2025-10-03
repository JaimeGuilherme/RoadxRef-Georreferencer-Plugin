
# RoadXRef Georeferencer (QGIS Plugin)

Detecta cruzamentos de vias em imagens de satélite, associa com interseções de malha OSM e georreferencia o raster usando GCPs calculados automaticamente.

## ✨ Recursos
- Filtra malha OSM por `fclass ∈ {primary, secondary, tertiary, residential}` e extrai **interseções** (pontos).
- Quebra o **TIFF** em **tiles 256×256** e roda um **modelo `.pth`** (SMP Unet *resnet34* por padrão; fallback para **UNet própria**).
- **Associação Húngara** entre pontos detectados e interseções OSM (raio configurável).
- **Georreferenciamento GDAL** com **polinômio (1/2/3)** e **reamostragem** (nearest/bilinear/**cubic**/cubicspline/lanczos).
- Exporta **GCPs** em `.points` e `.csv` e (opcional) **pontos inferidos** em GeoJSON.
- **Barras de progresso** para inferência, associação e warp.

## 📦 Estrutura do repositório
```
RoadXRef-Georeferencer/
├─ LICENSE
├─ README.md
├─ CHANGELOG.md
├─ .gitignore
└─ plugin/
   ├─ build_zip.py
   └─ roadxref_plugin/
      ├─ metadata.txt
      ├─ __init__.py
      ├─ roadxref_plugin.py
      ├─ infer_utils.py
      ├─ README.md
      └─ requirements.txt
```

## 🧰 Requisitos (QGIS 3.40, Python 3.11)
Instale estas dependências **no Python do QGIS** (ou em um venv apontado pelo QGIS):

```
albumentations>=1.4.10
geopandas>=1.0.1
numpy>=1.26.4
pandas>=2.2.2
pillow>=11.0.0
pyproj>=3.7.0
rasterio>=1.4.3
scikit-learn>=1.5.2
scipy>=1.13.1
shapely>=2.0.4
tensorboard>=2.17.0
tensorboard-data-server>=0.7.2
torch>=2.2.2
torchvision>=0.17.2
tqdm>=4.66.5
segmentation_models_pytorch>=0.3.3
```

> Dica: se `segmentation_models_pytorch` não estiver disponível, o plugin usa a **UNet própria** automaticamente.

### Windows (OSGeo4W / Standalone)
```bat
"C:\Program Files\QGIS 3.40pps\Python311\python.exe" -m pip install --upgrade pip setuptools wheel
"C:\Program Files\QGIS 3.40pps\Python311\python.exe" -m pip install rasterio shapely geopandas scipy numpy pillow
"C:\Program Files\QGIS 3.40pps\Python311\python.exe" -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
"C:\Program Files\QGIS 3.40pps\Python311\python.exe" -m pip install segmentation_models_pytorch albumentations scikit-learn tensorboard tqdm pyproj
```

### Linux
```bash
python3.11 -m pip install --upgrade pip
python3.11 -m pip install rasterio shapely geopandas scipy numpy pillow pyproj
python3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3.11 -m pip install segmentation_models_pytorch albumentations scikit-learn tensorboard tqdm
```

## 🚀 Instalação no QGIS
1. **Baixe o ZIP do plugin** (ou gere com `python plugin/build_zip.py`).
2. QGIS → **Complementos → Gerenciar e Instalar Complementos → Instalar a partir de ZIP**.
3. Se necessário, ajuste **Preferências → Opções → Sistema → Ambiente** para apontar para seu venv.

## 🖥️ Uso
1. Abra *RoadXRef Georeferencer* (dock à direita).
2. Informe:
   - **Malha OSM (linha)** (GPKG/SHP). O plugin filtra `fclass` automaticamente.
   - **TIFF** a georreferenciar (imagem “deslocada”).
   - **Modelo `.pth`** treinado (SMP preferido, UNet fallback).
3. **Configurações avançadas**:
   - **Batch** (padrão 32)
   - **Raio de associação (m)** (padrão 20)
   - **Polinômio** (1/2/3)
   - **Reamostragem** (nearest, bilinear, cubic, cubicspline, lanczos)
   - **Exportar** `.points`/`.csv` e **pontos inferidos** (GeoJSON)
4. Clique **Executar georreferenciamento** → saída: `roadxref_outputs/<imagem>_georef.tif`

## ⚙️ Atributos do plugin
- **Entrada**: camada vetorial de linhas (OSM), raster GeoTIFF, checkpoint `.pth`.
- **Saída**: GeoTIFF georreferenciado, `.points`/`.csv` (opcional), GeoJSON de pontos inferidos (opcional).
- **CRS**: reprojeta interseções para CRS métrico UTM automaticamente quando necessário.
- **Modelo**: SMP Unet (resnet34) ou UNet interna; adaptação 3↔4 canais do 1º conv quando necessário.
- **Associação**: método Húngaro com *tiling* espacial; raio configurável (m).

## 🧪 Diagnóstico (opcional)
Abra o **Console Python** do QGIS e teste:
```python
import sys, torch, rasterio, geopandas, shapely, scipy, numpy
print(sys.executable)
print(torch.__version__, rasterio.__version__)
```

## 📄 Licença
[MIT](LICENSE)
