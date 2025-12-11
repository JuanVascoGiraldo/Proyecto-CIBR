"""
Script para extraer características de todas las imágenes del dataset.
Extrae características usando todos los extractores disponibles y las guarda.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

from feature_extractors import (
    ResNetExtractor,
    VGGExtractor,
    ColorTextureExtractor,
    HOGExtractor,
    ColorShapeExtractor
)


def get_all_images(base_path: str = "images"):
    """
    Obtiene todas las imágenes del dataset.
    
    Args:
        base_path: Ruta base del dataset
        
    Returns:
        list: Lista con información de las imágenes
    """
    base_path = Path(base_path)
    categories = ['accordion', 'drum', 'flute', 'guitar', 'saxophone', 'violin']
    
    images_info = []
    
    for category in categories:
        category_path = base_path / category
        
        if not category_path.exists():
            continue
        
        # Buscar todas las imágenes
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            for img_path in category_path.glob(ext):
                # Determinar split basado en el nombre del archivo
                # Asumimos que las primeras 80 son train y las últimas son test
                img_num = int(''.join(filter(str.isdigit, img_path.stem)))
                split = 'train' if img_num <= 80 else 'test'
                
                images_info.append({
                    'path': str(img_path),
                    'category': category,
                    'split': split,
                    'filename': img_path.name
                })
    
    return images_info


def extract_features_for_all(images_info: list, extractor, extractor_name: str):
    """
    Extrae características para todas las imágenes con un extractor.
    
    Args:
        images_info: Lista de información de imágenes
        extractor: Instancia del extractor
        extractor_name: Nombre del extractor
        
    Returns:
        tuple: (features_matrix, image_paths, metadata)
    """
    features_list = []
    paths_list = []
    metadata_list = []
    
    print(f"\nExtrayendo características con {extractor_name}...")
    
    for img_info in tqdm(images_info, desc=extractor_name):
        try:
            # Extraer características
            features = extractor.extract(img_info['path'])
            
            features_list.append(features)
            paths_list.append(img_info['path'])
            metadata_list.append({
                'category': img_info['category'],
                'split': img_info['split'],
                'filename': img_info['filename']
            })
            
        except Exception as e:
            print(f"\nError procesando {img_info['path']}: {e}")
            continue
    
    # Convertir a matriz numpy
    features_matrix = np.array(features_list, dtype=np.float32)
    
    return features_matrix, paths_list, metadata_list


def save_features(features_matrix, paths_list, metadata_list, extractor_name: str, 
                  output_dir: str = "features"):
    """
    Guarda las características extraídas.
    
    Args:
        features_matrix: Matriz de características
        paths_list: Lista de rutas de imágenes
        metadata_list: Lista de metadata
        extractor_name: Nombre del extractor
        output_dir: Directorio de salida
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Guardar matriz de características
    features_file = output_dir / f"{extractor_name}_features.npy"
    np.save(features_file, features_matrix)
    
    # Guardar metadata
    metadata = {
        'extractor': extractor_name,
        'num_images': len(paths_list),
        'feature_dim': features_matrix.shape[1],
        'paths': paths_list,
        'metadata': metadata_list
    }
    
    metadata_file = output_dir / f"{extractor_name}_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Guardar también en JSON (sin paths para que sea más legible)
    metadata_json = {
        'extractor': extractor_name,
        'num_images': len(paths_list),
        'feature_dim': features_matrix.shape[1],
        'categories': {}
    }
    
    # Contar imágenes por categoría
    for meta in metadata_list:
        cat = meta['category']
        if cat not in metadata_json['categories']:
            metadata_json['categories'][cat] = {'train': 0, 'test': 0}
        metadata_json['categories'][cat][meta['split']] += 1
    
    json_file = output_dir / f"{extractor_name}_info.json"
    with open(json_file, 'w') as f:
        json.dump(metadata_json, f, indent=2)
    
    print(f"  ✓ Guardado en {output_dir}/")
    print(f"    - Features: {features_file.name}")
    print(f"    - Metadata: {metadata_file.name}")
    print(f"    - Info: {json_file.name}")


def main():
    """
    Función principal.
    """
    print("="*60)
    print("EXTRACCIÓN DE CARACTERÍSTICAS DEL DATASET")
    print("="*60)
    
    # Obtener todas las imágenes
    print("\nBuscando imágenes en el dataset...")
    images_info = get_all_images()
    
    if not images_info:
        print("\n✗ Error: No se encontraron imágenes.")
        print("  Por favor, ejecuta preprocess_images.py primero.")
        return
    
    print(f"✓ Encontradas {len(images_info)} imágenes")
    
    # Contar por categoría y split
    categories = {}
    for img in images_info:
        cat = img['category']
        split = img['split']
        if cat not in categories:
            categories[cat] = {'train': 0, 'test': 0}
        categories[cat][split] += 1
    
    print("\nDistribución del dataset:")
    for cat, counts in sorted(categories.items()):
        print(f"  {cat:12s}: train={counts['train']:3d}, test={counts['test']:3d}")
    
    # Inicializar extractores
    extractors = {
        'ResNet50': ResNetExtractor(),
        'VGG16': VGGExtractor(),
        'ColorTexture': ColorTextureExtractor(),
        'HOG': HOGExtractor(),
        'ColorShape': ColorShapeExtractor()
    }
    
    print(f"\n{'='*60}")
    print("EXTRACCIÓN DE CARACTERÍSTICAS")
    print(f"{'='*60}")
    
    # Extraer características con cada extractor
    for extractor_name, extractor in extractors.items():
        print(f"\n{extractor_name} (dimensión: {extractor.get_feature_dim()})")
        print("-" * 60)
        
        features_matrix, paths_list, metadata_list = extract_features_for_all(
            images_info, extractor, extractor_name
        )
        
        print(f"\n  Características extraídas:")
        print(f"    - Shape: {features_matrix.shape}")
        print(f"    - Dtype: {features_matrix.dtype}")
        print(f"    - Memoria: {features_matrix.nbytes / (1024*1024):.2f} MB")
        
        # Guardar
        save_features(features_matrix, paths_list, metadata_list, extractor_name)
    
    print(f"\n{'='*60}")
    print("✓ EXTRACCIÓN COMPLETADA")
    print(f"{'='*60}")
    print("\nArchivos generados en el directorio 'features/'")
    print("Ahora puedes construir los índices FAISS con build_faiss_indices.py")


if __name__ == "__main__":
    main()
