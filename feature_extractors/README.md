# Feature Extractors

Este directorio contiene implementaciones de diferentes extractores de características para imágenes.

## Extractores Disponibles

### 1. ResNetExtractor
- **Modelo**: ResNet50 pre-entrenado en ImageNet
- **Dimensión**: 2048
- **Ventajas**: Excelente para similitud semántica, captura características de alto nivel
- **Uso recomendado**: Cuando necesitas entender el contenido semántico de las imágenes

### 2. VGGExtractor
- **Modelo**: VGG16 pre-entrenado en ImageNet
- **Dimensión**: 512
- **Ventajas**: Vector más compacto que ResNet, buena precisión
- **Uso recomendado**: Balance entre tamaño y precisión

### 3. ColorTextureExtractor
- **Componentes**: Histograma HSV + LBP (Local Binary Patterns)
- **Dimensión**: 512 + 26 = 538
- **Ventajas**: Rápido, captura patrones de color y textura
- **Uso recomendado**: Cuando el color y la textura son características discriminativas

### 4. HOGExtractor
- **Método**: Histogram of Oriented Gradients
- **Dimensión**: 3,024 (configurable)
- **Ventajas**: Excelente para detectar formas y bordes
- **Uso recomendado**: Cuando la forma del objeto es importante

### 5. ColorShapeExtractor
- **Componentes**: Histograma RGB + Momentos de Hu
- **Dimensión**: 96 + 7 = 103
- **Ventajas**: Muy compacto, invariante a transformaciones
- **Uso recomendado**: Cuando necesitas un descriptor pequeño y rápido

## Uso

```python
from feature_extractors import ResNetExtractor

# Inicializar extractor
extractor = ResNetExtractor()

# Extraer características de una imagen
features = extractor.extract("path/to/image.jpg")

# El vector retornado está normalizado (L2 norm)
print(f"Dimensión: {len(features)}")
print(f"Norma: {np.linalg.norm(features)}")  # ~1.0
```

## Características Comunes

Todos los extractores:
- Heredan de `BaseExtractor`
- Retornan vectores **normalizados** con L2 norm
- Aceptan rutas de archivo, arrays numpy o imágenes PIL
- Implementan `extract()` y `get_feature_dim()`

## Instalación de Dependencias

```bash
pip install -r requirements_extractors.txt
```

## Testing

Ejecuta el script de prueba para verificar que todos los extractores funcionan:

```bash
python test_extractors.py
```
