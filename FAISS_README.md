# Sistema de Recuperación de Imágenes con FAISS

Este proyecto implementa un sistema completo de recuperación de imágenes basado en contenido (CBIR) utilizando múltiples extractores de características y diferentes índices FAISS.

## 📁 Estructura del Proyecto

```
Proyecto_CIBR/
├── images/                      # Dataset original (960 imágenes)
├── features/                    # Características extraídas
│   ├── ResNet50_features.npy
│   ├── VGG16_features.npy
│   ├── ColorTexture_features.npy
│   ├── HOG_features.npy
│   └── ColorShape_features.npy
├── faiss_indices/              # Índices FAISS (20 índices totales)
│   ├── ResNet50/
│   │   ├── index_flat.index
│   │   ├── index_ivf.index
│   │   ├── index_ivfpq.index
│   │   ├── index_hnsw.index
│   │   └── indices_info.json
│   ├── VGG16/
│   ├── ColorTexture/
│   ├── HOG/
│   └── ColorShape/
└── feature_extractors/         # Extractores de características
```

## 🎯 Extractores de Características

### 1. **ResNet50** (2048 dim)
- Red neuronal convolucional profunda pre-entrenada
- Mejor para similitud semántica
- Velocidad: ~0.5s por imagen
- **Casos de uso**: Cuando necesitas entender el contenido de alto nivel

### 2. **VGG16** (512 dim)
- CNN más compacta que ResNet
- Balance entre tamaño y precisión
- Velocidad: ~0.4s por imagen
- **Casos de uso**: Cuando necesitas un buen compromiso

### 3. **ColorTexture** (538 dim)
- Combina histograma HSV + Local Binary Patterns (LBP)
- Muy rápido
- Velocidad: ~0.02s por imagen
- **Casos de uso**: Instrumentos con colores/texturas distintivas

### 4. **HOG** (8100 dim)
- Histogram of Oriented Gradients
- Excelente para formas y contornos
- Velocidad: ~0.006s por imagen
- **Casos de uso**: Cuando la forma del instrumento es importante

### 5. **ColorShape** (103 dim)
- Combina histograma RGB + Momentos de Hu
- Más compacto y rápido
- Velocidad: ~0.005s por imagen
- **Casos de uso**: Búsquedas rápidas con descriptor pequeño

##  Tipos de Índices FAISS

### 1. **IndexFlatL2** (Búsqueda Exacta)
- **Precisión**: 100% (búsqueda exacta)
- **Velocidad**: Lenta para datasets grandes
- **Memoria**: Alta (vectores completos sin comprimir)
- **Cuándo usar**: Datasets pequeños (<10K), cuando necesitas precisión garantizada

### 2. **IndexIVFFlat** (Inverted File)
- **Precisión**: ~95-99% (configurable con nprobe)
- **Velocidad**: Rápida
- **Memoria**: Media
- **Cuándo usar**: Datasets medianos/grandes (10K-1M), balance velocidad/precisión

### 3. **IndexIVFPQ** (Product Quantization)
- **Precisión**: ~90-95% (aproximada)
- **Velocidad**: Muy rápida
- **Memoria**: Baja (compresión 100x-1000x)
- **Cuándo usar**: Datasets muy grandes (>1M), memoria limitada

### 4. **IndexHNSWFlat** (Hierarchical NSW)
- **Precisión**: ~98-99%
- **Velocidad**: Muy rápida
- **Memoria**: Media-Alta
- **Cuándo usar**: Cuando necesitas alta precisión Y velocidad

## 📊 Resumen de Archivos Generados

### Características Extraídas
- **Total imágenes procesadas**: 944-960 (algunas imágenes corruptas fueron omitidas)
- **Tamaño total**: ~41 MB
- **Archivos por extractor**: 
  - `*_features.npy` - Matriz de características
  - `*_metadata.pkl` - Metadata con paths y categorías
  - `*_info.json` - Información resumida

### Índices FAISS
- **Total índices**: 20 (5 extractores × 4 tipos)
- **Tamaño promedio**: ~1-5 MB por índice
- **Metadata**: `indices_info.json` en cada directorio

##  Uso Rápido

### Cargar un índice y buscar

```python
import faiss
import numpy as np
import pickle

# 1. Cargar índice
index = faiss.read_index("faiss_indices/ResNet50/index_flat.index")

# 2. Cargar metadata
with open("features/ResNet50_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# 3. Extraer características de imagen query
from feature_extractors import ResNetExtractor
extractor = ResNetExtractor()
query_features = extractor.extract("mi_imagen.jpg")

# 4. Buscar las 5 imágenes más similares
query_vector = query_features.reshape(1, -1)
distances, indices = index.search(query_vector, k=5)

# 5. Mostrar resultados
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    img_path = metadata['paths'][idx]
    category = metadata['metadata'][idx]['category']
    print(f"{i+1}. {img_path} (distancia: {dist:.4f}, categoría: {category})")
```

## 🎯 Recomendaciones por Caso de Uso

### Para tu dataset de instrumentos musicales:

1. **Mejor precisión global**: `ResNet50 + IndexHNSWFlat`
2. **Más rápido**: `ColorShape + IndexFlatL2`
3. **Balance óptimo**: `VGG16 + IndexIVFFlat`
4. **Para producción**: `ResNet50 + IndexIVFPQ` (si memoria es limitada)
5. **Para formas similares**: `HOG + IndexHNSWFlat`

## 📝 Scripts Disponibles

- `extract_all_features.py` - Extrae características de todas las imágenes
- `build_faiss_indices.py` - Construye todos los índices FAISS
- `test_extractors.py` - Prueba los extractores con una imagen

## ⚙️ Configuración y Parámetros

### Ajustar precisión de IVF
```python
index.nprobe = 20  # Más nprobe = más precisión, más lento
```

### Ajustar HNSW
```python
index.efSearch = 64  # Más efSearch = más precisión, más lento
```

## 📈 Próximos Pasos

1. Implementar sistema de búsqueda con interfaz web
2. Evaluación de precisión (mAP, Precision@K, Recall@K)
3. Visualización de resultados
4. Comparación entre diferentes combinaciones extractor+índice

##  Dependencias

Ver `requirements.txt` y `requirements_extractors.txt`
