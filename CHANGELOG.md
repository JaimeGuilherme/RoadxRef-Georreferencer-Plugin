
# Changelog

## [0.2.0] - 2025-10-03
- Compatível com QGIS 3.40 (Python 3.11)
- Config avançadas: polinômio (1/2/3), resampling (nearest/bilinear/cubic/cubicspline/lanczos), raio de associação (padrão 20 m)
- Barras de progresso: inferência, associação e warp
- SMP como padrão; fallback para UNet própria se SMP ausente
- Exportação opcional de GCPs (.points/.csv) e pontos inferidos (GeoJSON)

## [0.1.0] - 2025-10-02
- Primeira versão funcional (pipeline completo, QGIS 3.34/py3.9)
