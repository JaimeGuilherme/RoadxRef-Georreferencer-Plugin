# AutoGeoref RoadCross (QGIS 3.34)

Detecta **cruzamentos de vias em imagens de satélite** com **UNet (PyTorch / SMP)**, encontra os **homólogos** nos cruzamentos de uma malha OSM (via método Húngaro) e **georreferencia** a imagem automaticamente (Polynomial 1, interpolação **cubic 4×4**).

> **Status:** testado no **QGIS 3.34.4 LTR (Prizren)** em Windows com **Python 3.9.18** (embutido no QGIS).

---

## Índice

- [Arquitetura](#arquitetura)
- [Pré-requisitos](#pré-requisitos)
- [Instalação de dependências](#instalação-de-dependências)
  - [A) PyTorch + SMP (CPU ou CUDA)](#a-pytorch--smp-cpu-ou-cuda)
  - [B) (Opcional) Pacotes OSGeo4W](#b-opcional-pacotes-osgeo4w)
- [Instalação do plugin](#instalação-do-plugin)
- [Como usar](#como-usar)
- [Configurações avançadas](#configurações-avançadas)
- [Formato dos dados de entrada](#formato-dos-dados-de-entrada)
- [Saídas](#saídas)
- [Solução de problemas](#solução-de-problemas)
- [Desenvolvimento](#desenvolvimento)
- [Licença](#licença)

---

## Arquitetura

Este plugin registra um **algoritmo de Processing** chamado:

> **Georreferenciamento → AutoGeoref RoadCross (OSM + UNet + Georef)**

Pipeline do algoritmo:

1. **Filtro OSM**: mantém apenas `fclass ∈ {'primary','secondary','tertiary','residential'}` e encontra os **pontos de interseção** (line-line).
2. **Inferência**: quebra o raster em **patches 256×256**, faz **UNet** (PyTorch) para gerar **máscara binária** (cruzamentos) e extrai **centros de massa (pontos)**.
3. **Associação**: associa **pontos inferidos ↔ interseções OSM** (método **Húngaro**, custo = distância em metros, limite padrão `20 m`).
4. **Georreferenciamento**: gera GCPs e aplica **Polynomial 1** com **cubic** (GDAL **gdal_translate + gdal_warp**) para salvar a **imagem georreferenciada**.

> Internamente usa **GDAL (osgeo)** do QGIS. Não depende de `rasterio/geopandas` (mantém compatibilidade direta com o QGIS).

---

## Pré-requisitos

- **QGIS 3.34.x LTR** (testado em **3.34.4 Prizren**).
- **Windows** (testado; Linux/Mac podem funcionar mas não foram validados).
- **Python 3.9 do QGIS** (embutido). Versão vista: **3.9.18**.
- **Modelo `.pth`** treinado para detecção de cruzamentos (SMP UNet ou compatível).

### Versões recomendadas de ML
- `torch == 2.2.*`
- `torchvision == 0.17.*`
- `torchaudio == 2.2.*` (opcional)
- `segmentation-models-pytorch == 0.3.3`
- `timm == 0.9.*`

> Use **CPU** se não tiver CUDA configurado. Para **CUDA** verifique sua versão de drivers e escolha a build compatível (ex.: `cu118`).

---

## Instalação de dependências

> Sempre execute os comandos no **OSGeo4W Shell** do seu QGIS (Menu Iniciar → *OSGeo4W / QGIS 3.34 Shell*). Assim você garante que o `pip` usa o **Python do QGIS**.

### A) PyTorch + SMP (CPU ou CUDA)

**CPU-only (mais simples):**
```bat
python -m pip install --upgrade pip wheel setuptools
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.* torchvision==0.17.* torchaudio==2.2.*
python -m pip install segmentation-models-pytorch==0.3.3 timm==0.9.*
```

**CUDA 11.8 (se sua GPU/driver suportar):**
```bat
python -m pip install --upgrade pip wheel setuptools
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.2.* torchvision==0.17.* torchaudio==2.2.*
python -m pip install segmentation-models-pytorch==0.3.3 timm==0.9.*
```

Teste no **Console Python** do QGIS:
```python
import torch, segmentation_models_pytorch as smp, timm
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("smp:", smp.__version__)
```

### B) (Opcional) Pacotes OSGeo4W

Se você precisar de binários extra do ecossistema GDAL (não obrigatório para o plugin):

1. Abra o **OSGeo4W Network Installer** → *Advanced Install*.
2. Procure por pacotes `python3-...` compatíveis (ex.: `python3-pandas`, etc.).  
   > **Não instale `rasterio` via pip** fora do OSGeo4W; no Windows, `rasterio` precisa casar com o **GDAL** do QGIS.

---

## Instalação do plugin

Há duas formas:

### 1) Via ZIP (recomendado para usuários)
No QGIS: **Plugins → Manage and Install… → Install from ZIP** e selecione o arquivo `.zip` deste repositório (pasta `dist/` ou GitHub Release).

### 2) Via código fonte (desenvolvedores)
Copie a pasta `AutoGeorefRoadCross/` para:
```
%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\AutoGeorefRoadCross
```
Reinicie o QGIS.

---

## Como usar

1. Carregue uma **malha rodoviária (linhas)** com campo **`fclass`** (OSM ou equivalente).
2. Carregue a **imagem de satélite** a ser georreferenciada (TIFF).
3. Abra a **Caixa de Processamento** → **Georreferenciamento** → **AutoGeoref RoadCross (OSM + UNet + Georef)**.
4. Preencha os parâmetros:
   - **Malha rodoviária (linha)**: camada com `fclass`.
   - **Imagem de satélite (TIFF)**.
   - **Modelo PyTorch (.pth)**: checkpoint treinado.
   - **Bandas**: `rgb` (3 canais) ou `rgbnir` (4 canais) — precisa bater com o treino.
   - **Pasta de saída** e **Arquivo TIFF de saída** (georreferenciado).
5. (Opcional) Ajuste **Configurações Avançadas** (ver abaixo).
6. Execute. O log mostrará as etapas: filtro OSM, interseções, inferência, associação, georreferenciamento.

---

## Configurações avançadas

- **Batch size (PyTorch)**: padrão `32`. Reduza se faltar memória.
- **Exportar .points/.csv dos homólogos**: salva GCPs para uso no Georreferenciador do QGIS.
- **Exportar pontos inferidos (GPKG)**: salva os pontos detectados pela rede.

---

## Formato dos dados de entrada

- **Linhas OSM**: precisa do campo **`fclass`** contendo, ao menos, algumas destas classes:  
  `primary`, `secondary`, `tertiary`, `residential`  
  O plugin aplica internamente:
  ```sql
  "fclass" IN ('primary','secondary','tertiary','residential')
  ```

- **Raster**: TIFF com 3 (RGB) ou 4 (RGBNIR) bandas, **alinhado ao treino do `.pth`** (mesma ordem e normalização básica).

- **Modelo `.pth`**: checkpoint compatível com UNet (SMP ou UNet “vanilla”). O plugin adapta o **conv1** se os canais não baterem (ex.: treino em 3 ch e inferência em 4 ch).

---

## Saídas

Na pasta escolhida:
- `pontos_inferidos.gpkg` *(opcional)* — pontos detectados pela rede.
- `pares_homologos.geojson` — pares inferido↔OSM usados como GCPs.
- `gcp_qgis.csv` — GCPs (mapX, mapY, pixelX, pixelY) para o Georreferenciador.
- `gcp_qgis.points` — formato de pontos do Georreferenciador.
- **TIFF georreferenciado** — arquivo final (Polynomial 1 + cubic).

---

## Solução de problemas

### 1) `NotImplementedError: QgsProcessingAlgorithm.createInstance()`
Você está com uma versão sem `createInstance()`. Atualize o plugin ou adicione no `algorithm_auto_georef.py`:
```python
def createInstance(self):
    return AlgoAutoGeoref()
```

### 2) `SyntaxError: invalid syntax` (na linha do `expr`)
A expressão precisa de aspas corretamente escapadas. Use:
```python
expr = "\"fclass\" IN ('primary','secondary','tertiary','residential')"
# ou
expr = '"fclass" IN (\'primary\', \'secondary\', \'tertiary\', \'residential\')'
```

### 3) `ModuleNotFoundError: No module named 'segmentation_models_pytorch'`
Instale PyTorch/SMP conforme a seção [Instalação de dependências](#a-pytorch--smp-cpu-ou-cuda). Sempre pelo **OSGeo4W Shell** do QGIS.

### 4) CUDA: `torch.cuda.is_available() == False`
- Verifique drivers NVIDIA e se a build do `torch` corresponde à sua **CUDA**.
- Use a versão **CPU** se não precisar de GPU.

### 5) Problemas de GDAL
O plugin usa **osgeo.gdal** do QGIS. Evite misturar instalações externas de GDAL. Use o Python do QGIS.

---

## Desenvolvimento

### Estrutura do projeto
```
AutoGeorefRoadCross/
  __init__.py
  metadata.txt
  plugin.py
  provider.py
  algorithm_auto_georef.py
  components/
    dataset.py
    infer_helpers.py
    associate_helpers.py
    cc_utils.py
    unet.py
    utils.py
    requirements_check.py
```

### Build de ZIP para distribuição
Compacte a pasta `AutoGeorefRoadCross/` (sem diretórios de nível acima) e publique em **Releases** do GitHub.

### Testes rápidos no QGIS
- Recarregue plugins: **Plugins → Plugin Reloader** (se tiver instalado) ou reinicie o QGIS.
- Console Python:
  ```python
  import torch, segmentation_models_pytorch as smp
  print(torch.__version__, smp.__version__)
  ```

---

## Licença

MIT © 2025 — Contribuições são bem-vindas via Pull Requests.
