# Tests - Evaluación del Sistema CBIR

Este directorio contiene scripts para evaluar el rendimiento del sistema CBIR con imágenes de test.

## 📋 Scripts Disponibles

### evaluate_system.py

Script principal de evaluación que prueba todas las combinaciones de extractores e índices.

**Uso:**
```bash
# Desde el directorio raíz del proyecto
python tests/evaluate_system.py
```

**Qué hace:**
1. ✅ Carga todas las imágenes del conjunto de test (60 imágenes, 10 por categoría)
2. ✅ Prueba todas las combinaciones:
   - 5 extractores × 4 tipos de índice = 20 combinaciones
   - 60 imágenes × 20 combinaciones = 1,200 búsquedas
3. ✅ Para cada búsqueda, encuentra las 10 imágenes más similares
4. ✅ Calcula métricas de evaluación:
   - **Precisión:** % de resultados de la misma categoría
   - **Similitud promedio:** Qué tan similares son los resultados
   - **Tiempo de búsqueda:** Velocidad de cada combinación
5. ✅ Genera reporte Excel con 5 hojas
6. ✅ Genera matrices de confusión visuales (guardadas en `results/`)
7. ✅ Genera 8 gráficos estadísticos (guardados en `tests/graphics/`)

---

## 📊 Archivos de Salida

### 1. Excel: evaluation_YYYYMMDD_HHMMSS.xlsx (en `tests/`)

El script genera un archivo con 5 hojas:

### Hoja 1: Resultados_Detallados

Cada fila = 1 búsqueda individual

| Columna | Descripción |
|---------|-------------|
| query_image | Nombre de la imagen de consulta |
| query_category | Categoría real de la imagen |
| extractor | Extractor usado (ResNet50, VGG16, etc.) |
| index_type | Tipo de índice (flat, ivf, ivfpq, hnsw) |
| total_results | Número de resultados encontrados (10) |
| correct_matches | Cuántos resultados son de la categoría correcta |
| incorrect_matches | Cuántos resultados son de otra categoría |
| precision | Precisión (correct/total) |
| avg_distance | Distancia promedio de los resultados |
| avg_similarity | Similitud promedio (0-1, más alto = mejor) |
| search_time_sec | Tiempo de búsqueda en segundos |
| categories_found | Distribución de categorías en resultados |

**Total:** ~1,200 filas (60 imágenes × 20 combinaciones)

---

### Hoja 2: Resumen_Combinaciones

Estadísticas agregadas por cada combinación extractor + índice

| Columna | Descripción |
|---------|-------------|
| extractor | Nombre del extractor |
| index_type | Tipo de índice |
| num_tests | Número de imágenes evaluadas |
| avg_precision | Precisión promedio (% correctos) |
| std_precision | Desviación estándar de precisión |
| min_precision | Peor precisión |
| max_precision | Mejor precisión |
| avg_similarity | Similitud promedio general |
| std_similarity | Desviación estándar de similitud |
| avg_distance | Distancia promedio general |
| std_distance | Desviación estándar de distancia |

**Ordenado por:** Precisión promedio (mejor primero)
**Total:** 20 filas (una por combinación)

---

### Hoja 3: Matriz_Confusion

Análisis de con qué clases se confunde cada categoría

| Columna | Descripción |
|---------|-------------|
| extractor | Nombre del extractor |
| index_type | Tipo de índice |
| true_class | Categoría real de la imagen de consulta |
| most_confused_with | Clase con la que más se confunde |
| confusion_count | Cuántas veces se confundió con esa clase |
| total_errors | Total de errores para esta clase en esta combinación |
| confusion_rate | Porcentaje de ese tipo de error (confusion_count/total_errors) |
| all_confusions | Diccionario con todas las confusiones |

**Ejemplo:**
- true_class: "guitar"
- most_confused_with: "violin" 
- confusion_count: 15
- total_errors: 25
- confusion_rate: 0.60 (60% de los errores fueron con violin)

**Total:** Varía según errores, ordenado por extractor, clase verdadera y cantidad de confusiones

---

### Hoja 4: Errores_Por_Clase

Análisis de qué modelo se equivoca más con cada clase

| Columna | Descripción |
|---------|-------------|
| class | Categoría analizada |
| extractor | Nombre del extractor |
| index_type | Tipo de índice |
| total_queries | Número de imágenes de test de esta clase |
| total_errors | Total de errores (resultados incorrectos) |
| avg_errors_per_query | Errores promedio por consulta |
| avg_precision | Precisión promedio para esta clase |
| error_rate | Tasa de error (1 - precision) |

**Uso:** Identificar qué combinaciones tienen más problemas con cada clase específica

**Ejemplo:**
- class: "accordion"
- extractor: "HOG"
- total_errors: 45
- avg_precision: 0.55 (55%)
- error_rate: 0.45 (45% de errores)

**Total:** 120 filas (6 clases × 20 combinaciones), ordenado por clase y total de errores

---

### Hoja 5: Estadisticas_Generales

Resumen global de toda la evaluación

| Métrica | Descripción |
|---------|-------------|
| total_tests | Total de búsquedas realizadas |
| total_combinations | Total de combinaciones evaluadas |
| overall_avg_precision | Precisión promedio de todo el sistema |
| overall_std_precision | Desviación estándar global |
| overall_avg_similarity | Similitud promedio global |
| overall_std_similarity | Desviación estándar de similitud |
| best_extractor | Mejor extractor |
| best_index | Mejor tipo de índice |
| best_precision | Mejor precisión alcanzada |
| worst_extractor | Peor extractor |
| worst_index | Peor tipo de índice |
| worst_precision | Peor precisión |

---

## 📈 Métricas Explicadas

### Precisión (Precision)

**Definición:** Porcentaje de resultados que pertenecen a la categoría correcta.

**Fórmula:** `Precisión = (Resultados correctos / Total resultados) × 100%`

**Ejemplo:**
- Imagen de consulta: guitarra
- Top 10 resultados: 8 guitarras, 2 violines
- Precisión = 8/10 = 80%

**Interpretación:**
- 100% = Perfecto, todos los resultados son de la categoría correcta
- 50% = Mitad correctos, mitad incorrectos
- 10% = Solo 1 de cada 10 es correcto

---

### Similitud Promedio (Average Similarity)

**Definición:** Qué tan similares son los resultados encontrados.

**Fórmula:** `Similitud = 1 / (1 + distancia)`

**Rango:** 0.0 a 1.0
- 1.0 = Idénticos (distancia = 0)
- 0.5 = Moderadamente similares (distancia = 1)
- 0.1 = Muy diferentes (distancia = 9)

**Interpretación:**
- > 0.8 = Muy similares
- 0.5 - 0.8 = Similares
- < 0.5 = Poco similares

---

### Distancia Promedio (Average Distance)

**Definición:** Distancia euclidiana promedio en el espacio de características.

**Interpretación:**
- Distancia menor = Más similares
- Distancia mayor = Menos similares
- Valores absolutos dependen del extractor

---

### 2. Matrices de Confusión (en `results/`)

Se generan 20 imágenes PNG (una por cada combinación extractor + índice):
- Archivos: `confusion_matrix_<Extractor>_<Index>.png`
- Formato: Heatmap 6×6 (una celda por categoría)
- **Diagonal:** Clasificaciones correctas (valores altos = bueno)
- **Fuera diagonal:** Confusiones entre clases (valores altos = problemático)

**Ejemplo:**
```
confusion_matrix_ResNet50_flat.png
confusion_matrix_VGG16_hnsw.png
...
```

**Interpretación:**
- **Diagonal oscura:** Muchos aciertos ✓
- **Celdas fuera diagonal oscuras:** Confusiones frecuentes ⚠️
- Identifica patrones: ¿guitar se confunde con violin? ¿drum con otros instrumentos?

---

### 3. Gráficos Estadísticos (en `tests/graphics/`)

El sistema genera 8 gráficos automáticamente:

#### 1. `precision_by_extractor.png`
- **Tipo:** Gráfico de barras
- **Muestra:** Precisión promedio de cada extractor (agregado sobre todos los índices)
- **Uso:** Identificar el mejor extractor general

#### 2. `precision_by_index.png`
- **Tipo:** Gráfico de barras
- **Muestra:** Precisión promedio de cada tipo de índice (agregado sobre todos los extractores)
- **Uso:** Comparar rendimiento entre flat, ivf, ivfpq, hnsw

#### 3. `top10_combinations.png`
- **Tipo:** Gráfico de barras
- **Muestra:** Las 10 mejores combinaciones extractor+índice
- **Uso:** Vista rápida de las mejores configuraciones

#### 4. `heatmap_extractor_vs_index.png`
- **Tipo:** Heatmap 5×4
- **Muestra:** Precisión de cada combinación (filas=extractores, columnas=índices)
- **Uso:** Ver de un vistazo qué combinaciones funcionan mejor
- **Colores:** Verde = mejor, Rojo = peor

#### 5. `precision_by_category.png`
- **Tipo:** Gráfico de barras
- **Muestra:** Precisión promedio por categoría (agregado sobre todas las combinaciones)
- **Uso:** Identificar clases difíciles de clasificar

#### 6. `search_time_by_combination.png`
- **Tipo:** Gráfico de barras (Top 15 más lentos)
- **Muestra:** Tiempo promedio de búsqueda en segundos
- **Uso:** Identificar combinaciones lentas vs rápidas

#### 7. `errors_class_vs_extractor.png`
- **Tipo:** Heatmap 6×5
- **Muestra:** Errores promedio por consulta (filas=clases, columnas=extractores)
- **Uso:** Identificar qué extractor tiene problemas con qué clase
- **Colores:** Amarillo = pocos errores, Rojo = muchos errores

#### 8. `precision_distribution_boxplot.png`
- **Tipo:** Box plot
- **Muestra:** Distribución de precisión por extractor
- **Uso:** Ver variabilidad y outliers en el rendimiento
- **Interpretación:**
  - Caja = rango intercuartil (50% de los datos)
  - Línea = mediana
  - Bigotes = rango completo (excluyendo outliers)
  - Círculos = outliers

---

## 🎯 Cómo Interpretar Resultados

### 1. ¿Qué combinación es mejor?

Mira la hoja **Resumen_Combinaciones**, ordenada por `avg_precision`:
- Primera fila = Mejor combinación
- Última fila = Peor combinación

**Busca:**
- Precisión promedio > 80% = Excelente
- Precisión promedio > 60% = Bueno
- Precisión promedio < 40% = Mejorar

**Visualización:** Ver `top10_combinations.png` y `heatmap_extractor_vs_index.png`

### 2. ¿Qué extractor funciona mejor?

Agrupa por `extractor` en la hoja de resumen y compara precisiones.

**Visualización:** Ver `precision_by_extractor.png` y `precision_distribution_boxplot.png`

### 3. ¿Qué tipo de índice es mejor?

Agrupa por `index_type` y compara:
- **Precisión:** flat suele ser el mejor (búsqueda exacta)
- **Velocidad:** ivfpq o hnsw son más rápidos

**Visualización:** Ver `precision_by_index.png` y `search_time_by_combination.png`

### 4. ¿Hay categorías problemáticas?

Filtra **Resultados_Detallados** por `query_category` y mira `precision`:
- Si una categoría tiene baja precisión → Es difícil de distinguir
- Si todas tienen alta precisión → Sistema funciona bien

**Visualización:** Ver `precision_by_category.png`

### 5. ¿Con qué clases se confunde cada categoría?

Mira la hoja **Matriz_Confusion**:
- Filtra por `true_class` para ver con qué se confunde cada categoría
- `most_confused_with` te dice la clase más problemática
- `confusion_rate` indica qué porcentaje de errores son de ese tipo

**Visualización:** Ver matrices en `results/confusion_matrix_*.png`

**Ejemplo:**
```
true_class: guitar
most_confused_with: violin
confusion_count: 20
confusion_rate: 0.67 (67%)
```
→ Cuando el sistema se equivoca con guitarras, 67% de las veces confunde con violines

### 6. ¿Qué modelo se equivoca más en cada clase?

Mira la hoja **Errores_Por_Clase**:
- Filtra por `class` para ver qué extractores tienen más problemas
- Ordena por `total_errors` o `error_rate` descendente
- Identifica combinaciones problemáticas para clases específicas

**Visualización:** Ver `errors_class_vs_extractor.png`

**Ejemplo:**
```
class: accordion
extractor: HOG
total_errors: 58
error_rate: 0.58 (58%)
```
→ HOG tiene dificultades con acordeones (58% de tasa de error)

---

## 🔍 Ejemplo de Análisis

```
TOP 5 MEJORES COMBINACIONES:
──────────────────────────────────────────────────────────────────────
ResNet50        + flat   | Precisión: 92.50% | Similitud: 0.7845
VGG16           + flat   | Precisión: 89.33% | Similitud: 0.7621
ResNet50        + hnsw   | Precisión: 91.17% | Similitud: 0.7799
VGG16           + hnsw   | Precisión: 88.00% | Similitud: 0.7534
ColorTexture    + flat   | Precisión: 75.83% | Similitud: 0.6892
```

**Interpretación:**
- ✅ ResNet50 con índice flat es el mejor (92.5% precisión)
- ✅ CNNs (ResNet50, VGG16) superan a descriptores clásicos
- ✅ Índice flat da mejor precisión que aproximados
- ⚠️ ColorTexture tiene precisión moderada (75.8%)

### Análisis de Confusiones Comunes

**Hoja Matriz_Confusion revelará patrones como:**
- 🎸 Guitarras ↔️ Violines: Instrumentos de cuerda similares visualmente
- 🎷 Saxofones ↔️ Clarinetes: Forma similar
- 🥁 Tambores ↔️ Percusión: Categoría visual amplia

**Hoja Errores_Por_Clase mostrará:**
- HOG puede tener problemas con instrumentos complejos
- ResNet50/VGG16 distinguen mejor detalles finos
- Algunas clases son inherentemente más difíciles que otras

---

## ⚡ Tiempos de Ejecución Estimados

| Dataset | Tiempo Estimado |
|---------|-----------------|
| 60 imágenes test | ~10-15 minutos |
| Por combinación | ~30-45 segundos |

**Factores que afectan:**
- Velocidad del extractor (ResNet50 más lento que ColorShape)
- Tamaño del índice (960 imágenes)
- Hardware disponible

---

## 🛠️ Requisitos

```bash
pip install pandas openpyxl tqdm
```

Asegúrate de haber ejecutado antes:
```bash
python extract_all_features.py  # Extraer características
python build_faiss_indices.py    # Construir índices
```

---

## 📝 Notas

- Los resultados excluyen la imagen de consulta si está en el índice
- Se usa `k=10` (top 10 resultados) por defecto
- Las imágenes de test NO deben estar en el índice de entrenamiento
- El script maneja errores y continúa si una búsqueda falla

---

## 🎓 Usos

1. **Comparar extractores:** ¿ResNet50 o VGG16?
2. **Seleccionar índice:** ¿Precisión (flat) o velocidad (hnsw)?
3. **Validar sistema:** ¿Funciona bien el CBIR?
4. **Identificar problemas:** ¿Qué categorías confunde?
5. **Optimizar parámetros:** ¿Qué configuración es mejor?

---

## 📧 Análisis de Resultados

Para análisis más profundo, abre el Excel y usa:
- **Tablas dinámicas:** Agrupa por extractor/índice/categoría
- **Gráficos:** Visualiza precisiones y similitudes
- **Filtros:** Identifica casos de baja precisión
- **Estadística:** Calcula correlaciones y tendencias
