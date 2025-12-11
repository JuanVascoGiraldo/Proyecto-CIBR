"""
    API de búsqueda de imágenes similares usando FastAPI.
    Permite buscar las 10 imágenes más parecidas a una imagen de entrada.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import faiss
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
import io
import os

from feature_extractors import (
    ResNetExtractor,
    VGGExtractor,
    ColorTextureExtractor,
    HOGExtractor,
    ColorShapeExtractor
)

# Crear aplicación FastAPI
app = FastAPI(
    title="API de Búsqueda de Imágenes Similares",
    description="Sistema de recuperación de imágenes basado en contenido (CBIR) con múltiples extractores",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (imágenes)
if Path("images").exists():
    app.mount(
                "/images",
                StaticFiles(directory="images"),
                name="images")

# Diccionario de extractores disponibles
EXTRACTORS = {
    'ResNet50': ResNetExtractor(),
    'VGG16': VGGExtractor(),
    'ColorTexture': ColorTextureExtractor(),
    'HOG': HOGExtractor(),
    'ColorShape': ColorShapeExtractor()
}

# Diccionario de índices disponibles
INDEX_TYPES = ['flat', 'ivf', 'ivfpq', 'hnsw']

# Cache de índices y metadata
indices_cache = {}
metadata_cache = {}


def load_index(extractor_name: str, index_type: str):
    """
    Carga un índice FAISS y su metadata.
    
    Args:
        extractor_name: Nombre del extractor
        index_type: Tipo de índice (flat, ivf, ivfpq, hnsw)
        
    Returns:
        tuple: (index, metadata)
    """
    cache_key = f"{extractor_name}_{index_type}"
    
    # Verificar si ya está en cache
    if cache_key in indices_cache:
        return indices_cache[cache_key], metadata_cache[cache_key]
    
    # Cargar índice
    index_path = Path(f"faiss_indices/{extractor_name}/index_{index_type}.index")
    if not index_path.exists():
        raise ValueError(f"Índice no encontrado: {index_path}")
    
    index = faiss.read_index(str(index_path))
    
    # Cargar metadata
    metadata_path = Path(f"features/{extractor_name}_metadata.pkl")
    if not metadata_path.exists():
        raise ValueError(f"Metadata no encontrada: {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Guardar en cache
    indices_cache[cache_key] = index
    metadata_cache[cache_key] = metadata
    
    return index, metadata


@app.get("/")
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "API de Búsqueda de Imágenes Similares",
        "version": "1.0.0",
        "endpoints": {
            "/search": "Buscar imágenes similares",
            "/extractors": "Listar extractores disponibles",
            "/indices": "Listar tipos de índices disponibles",
            "/health": "Verificar estado de la API"
        }
    }


@app.get("/health")
async def health_check():
    """Verifica el estado de la API."""
    return {
        "status": "healthy",
        "extractors_loaded": len(EXTRACTORS),
        "indices_cached": len(indices_cache)
    }


@app.get("/extractors")
async def list_extractors():
    """Lista todos los extractores disponibles con sus características."""
    extractors_info = {}
    
    for name, extractor in EXTRACTORS.items():
        extractors_info[name] = {
            "name": name,
            "dimension": extractor.get_feature_dim(),
            "description": extractor.__doc__.strip().split('\n')[0] if extractor.__doc__ else ""
        }
    
    return {
        "extractors": extractors_info,
        "count": len(extractors_info)
    }


@app.get("/indices")
async def list_index_types():
    """Lista todos los tipos de índices disponibles."""
    indices_info = {
        "flat": {
            "name": "IndexFlatL2",
            "description": "Búsqueda exacta con distancia L2",
            "precision": "100%",
            "speed": "Lenta para datasets grandes"
        },
        "ivf": {
            "name": "IndexIVFFlat",
            "description": "Búsqueda aproximada con clustering",
            "precision": "95-99%",
            "speed": "Rápida"
        },
        "ivfpq": {
            "name": "IndexIVFPQ",
            "description": "Búsqueda aproximada con compresión",
            "precision": "90-95%",
            "speed": "Muy rápida"
        },
        "hnsw": {
            "name": "IndexHNSWFlat",
            "description": "Búsqueda aproximada con grafos",
            "precision": "98-99%",
            "speed": "Muy rápida"
        }
    }
    
    return {
        "indices": indices_info,
        "count": len(indices_info)
    }


@app.post("/search")
async def search_similar_images(
    file: UploadFile = File(..., description="Imagen de consulta"),
    extractor: str = Query(..., description="Nombre del extractor a usar"),
    index_type: str = Query("flat", description="Tipo de índice FAISS"),
    k: int = Query(10, ge=1, le=50, description="Número de imágenes similares a retornar")
):
    """
    Busca las K imágenes más similares a la imagen de entrada.
    
    Args:
        file: Archivo de imagen
        extractor: Nombre del extractor (ResNet50, VGG16, ColorTexture, HOG, ColorShape)
        index_type: Tipo de índice (flat, ivf, ivfpq, hnsw)
        k: Número de resultados a retornar (1-50)
        
    Returns:
        JSON con las imágenes más similares y sus distancias
    """
    try:
        # Validar extractor
        if extractor not in EXTRACTORS:
            raise HTTPException(
                status_code=400,
                detail=f"Extractor '{extractor}' no válido. Opciones: {list(EXTRACTORS.keys())}"
            )
        
        # Validar tipo de índice
        if index_type not in INDEX_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de índice '{index_type}' no válido. Opciones: {INDEX_TYPES}"
            )
        
        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extraer características
        extractor_obj = EXTRACTORS[extractor]
        query_features = extractor_obj.extract(np.array(image))
        
        # Cargar índice y metadata
        index, metadata = load_index(extractor, index_type)
        
        # Ajustar k si es mayor que el número de vectores en el índice
        k = min(k, index.ntotal)
        
        # Buscar en el índice
        query_vector = query_features.reshape(1, -1).astype(np.float32)
        distances, indices = index.search(query_vector, k)
        
        # Preparar resultados
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            img_path = metadata['paths'][idx]
            img_metadata = metadata['metadata'][idx]
            
            # Convertir path absoluto a relativo
            rel_path = os.path.relpath(img_path)
            
            results.append({
                "rank": i + 1,
                "image_path": rel_path,
                "distance": float(dist),
                "category": img_metadata['category'],
                "split": img_metadata['split'],
                "filename": img_metadata['filename']
            })
        
        return {
            "success": True,
            "query_image": file.filename,
            "extractor": extractor,
            "index_type": index_type,
            "feature_dimension": extractor_obj.get_feature_dim(),
            "k_requested": k,
            "results": results
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")


@app.get("/stats")
async def get_statistics():
    """Obtiene estadísticas del sistema."""
    stats = {
        "extractors": {},
        "total_images": 0
    }
    
    # Obtener estadísticas de cada extractor
    for extractor_name in EXTRACTORS.keys():
        metadata_path = Path(f"features/{extractor_name}_metadata.pkl")
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            stats["extractors"][extractor_name] = {
                "num_images": metadata['num_images'],
                "feature_dim": metadata['feature_dim']
            }
            
            if stats["total_images"] == 0:
                stats["total_images"] = metadata['num_images']
    
    return stats


if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("API DE BÚSQUEDA DE IMÁGENES SIMILARES")
    print("="*60)
    print("\nExtractores disponibles:")
    for name in EXTRACTORS.keys():
        print(f"  - {name}")
    print("\nTipos de índices disponibles:")
    for idx_type in INDEX_TYPES:
        print(f"  - {idx_type}")
    print("\n" + "="*60)
    print("Iniciando servidor en http://localhost:8000")
    print("Documentación interactiva: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
