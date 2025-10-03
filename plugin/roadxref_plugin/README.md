
# RoadXRef Georeferencer (QGIS Plugin)

Fluxo:
1. Filtra malha OSM por `fclass in {primary, secondary, tertiary, residential}` e extrai interseções.
2. Quebra a imagem TIFF em tiles 256x256 em memória e roda o modelo `.pth` para detectar cruzamentos.
3. Associa com método Húngaro (limite 20 m) para gerar pontos homólogos.
4. Georreferencia a imagem com polinomial de ordem 1 (resampling cúbico 4x4), produzindo um novo GeoTIFF.

## Instalação
- QGIS 3.22+
- Python com `torch` e (opcional) `segmentation_models_pytorch` instalados e visíveis para o QGIS.
  - Configure em **QGIS → Opções → Sistema → Variáveis de ambiente** o caminho do seu `venv` com **torch**.

## Uso
- Abra o plugin (Dock à direita).
- Informe: malha (gpkg/shp), imagem `.tif`, modelo `.pth`.
- (Avançado) Batch, exportações.
- Clique **Executar georreferenciamento**.

## Observações
- O plugin tenta adaptar o 1º conv caso seu `.pth` tenha sido treinado com 3 canais e a imagem tenha 4 (ou vice-versa).
- Threshold padrão recuperado de `best_threshold` salvo no `.pth` (se existir), senão 0.5.
- Exporta (`roadxref_outputs/`): GeoTIFF georreferenciado e, se marcado, `gcp_qgis.points`, `gcp_qgis.csv` e `pontos_inferidos.geojson`.



## Dependências (Windows / QGIS 3.34, Python 3.9)
- Use **um Python 3.9** com `torch`, `rasterio`, `geopandas`, `shapely`, `scipy`, `segmentation_models_pytorch`.
- Aponte o QGIS para esse Python em **Preferências → Opções → Sistema → Ambiente** (PYTHONHOME/PYTHONPATH/PATH).
- Alternativa: instale pacotes direto no **Python do QGIS** via OSGeo4W Shell.

### Comandos (Python do QGIS)
```bat
"C:\Program Files\QGIS 3.34.4\apps\Python39\python.exe" -m pip install --upgrade pip
"C:\Program Files\QGIS 3.34.4\apps\Python39\python.exe" -m pip install rasterio==1.4.3 shapely==2.1.1 geopandas==1.1.1 scipy==1.16.1 numpy==2.3.2 pillow==11.0.0
"C:\Program Files\QGIS 3.34.4\apps\Python39\python.exe" -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
"C:\Program Files\QGIS 3.34.4\apps\Python39\python.exe" -m pip install segmentation_models_pytorch==0.5.0
```
Se SMP falhar, o plugin usará a UNet interna automaticamente.
