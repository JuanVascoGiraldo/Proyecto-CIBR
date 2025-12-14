"""
Script para construir múltiples índices FAISS con diferentes métodos de indexación.
Crea índices Flat, IVF, IVFPQ y HNSW para cada extractor de características.
"""

import os
import sys
import json
import numpy as np
import faiss
from pathlib import Path
from typing import Tuple
import time


class FAISSIndexBuilder:
    """
    Constructor de índices FAISS con diferentes métodos de indexación.
    Usa IndexIDMap para asignar IDs personalizados.
    """
    
    def __init__(self, features: np.ndarray, ids: np.ndarray, dimension: int):
        """
        Inicializa el constructor.
        
        Args:
            features: Matriz de características (N x D)
            ids: Array de IDs personalizados para cada vector
            dimension: Dimensión de los vectores
        """
        self.features = features.astype(np.float32)
        self.ids = ids.astype(np.int64)
        self.dimension = dimension
        self.num_vectors = features.shape[0]
        
        print(f"  Dataset: {self.num_vectors} vectores de dimensión {self.dimension}")
        print(f"  Usando IDs personalizados (rango: {self.ids.min()} - {self.ids.max()})")
    
    def build_flat_index(self) -> Tuple[faiss.Index, dict]:
        """
        Construye un índice Flat (búsqueda exacta con L2) con IDs personalizados.
        
        Returns:
            tuple: (índice, metadata)
        """
        print("\n  [1/4] Construyendo IndexFlatL2 (búsqueda exacta)...")
        start_time = time.time()
        
        # Crear índice base
        base_index = faiss.IndexFlatL2(self.dimension)
        
        # Envolver con IndexIDMap para IDs personalizados
        index = faiss.IndexIDMap(base_index)
        
        # Agregar vectores con IDs específicos
        index.add_with_ids(self.features, self.ids)
        
        build_time = time.time() - start_time
        
        metadata = {
            'type': 'IndexFlatL2',
            'description': 'Búsqueda exacta con distancia L2',
            'dimension': self.dimension,
            'num_vectors': index.ntotal,
            'build_time': build_time,
            'uses': 'Datasets pequeños, búsqueda exacta garantizada'
        }
        
        print(f"     Completado en {build_time:.2f}s")
        print(f"     Vectores indexados: {index.ntotal}")
        
        return index, metadata
    
    def build_ivf_index(self, nlist: int = None) -> Tuple[faiss.Index, dict]:
        """
        Construye un índice IVF (Inverted File) con IDs personalizados.
        
        Args:
            nlist: Número de clusters (por defecto: sqrt(N))
            
        Returns:
            tuple: (índice, metadata)
        """
        if nlist is None:
            nlist = min(int(np.sqrt(self.num_vectors)), 100)
        
        print(f"\n  [2/4] Construyendo IndexIVFFlat (nlist={nlist})...")
        start_time = time.time()
        
        # Crear índice base (quantizer)
        quantizer = faiss.IndexFlatL2(self.dimension)
        
        # Crear índice IVF
        base_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        # Envolver con IndexIDMap
        index = faiss.IndexIDMap(base_index)
        
        # Entrenar el índice
        print(f"    - Entrenando con {self.num_vectors} vectores...")
        base_index.train(self.features)
        
        # Agregar vectores con IDs
        print(f"    - Agregando vectores al índice...")
        index.add_with_ids(self.features, self.ids)
        
        # Configurar nprobe (número de clusters a buscar)
        base_index.nprobe = min(10, nlist)
        
        build_time = time.time() - start_time
        
        metadata = {
            'type': 'IndexIVFFlat',
            'description': 'Búsqueda aproximada con clustering',
            'dimension': self.dimension,
            'num_vectors': index.ntotal,
            'nlist': nlist,
            'nprobe': base_index.nprobe,
            'build_time': build_time,
            'uses': 'Datasets medianos/grandes, búsqueda rápida'
        }
        
        print(f"     Completado en {build_time:.2f}s")
        print(f"     Vectores indexados: {index.ntotal}")
        print(f"     Clusters: {nlist}, nprobe: {base_index.nprobe}")
        
        return index, metadata
    
    def build_ivfpq_index(self, nlist: int = None, m: int = 8, 
                          nbits: int = 8) -> Tuple[faiss.Index, dict]:
        """
        Construye un índice IVFPQ (IVF + Product Quantization).
        
        Args:
            nlist: Número de clusters
            m: Número de sub-vectores para PQ
            nbits: Bits por sub-vector
            
        Returns:
            tuple: (índice, metadata)
        """
        if nlist is None:
            nlist = min(int(np.sqrt(self.num_vectors)), 100)
        
        # Ajustar m para que divida la dimensión
        if self.dimension % m != 0:
            m = 8 if self.dimension >= 64 else 4
            while self.dimension % m != 0 and m > 1:
                m -= 1
        
        print(f"\n  [3/4] Construyendo IndexIVFPQ (nlist={nlist}, m={m}, nbits={nbits})...")
        start_time = time.time()
        
        # Crear índice base
        quantizer = faiss.IndexFlatL2(self.dimension)
        
        # Crear índice IVFPQ
        base_index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits)
        
        # Envolver con IndexIDMap
        index = faiss.IndexIDMap(base_index)
        
        # Entrenar
        print(f"    - Entrenando con {self.num_vectors} vectores...")
        base_index.train(self.features)
        
        # Agregar vectores con IDs
        print(f"    - Agregando vectores al índice...")
        index.add_with_ids(self.features, self.ids)
        
        # Configurar nprobe
        base_index.nprobe = min(10, nlist)
        
        build_time = time.time() - start_time
        
        # Calcular compresión
        original_size = self.num_vectors * self.dimension * 4  # float32
        compressed_size = index.ntotal * m * nbits / 8
        compression_ratio = original_size / compressed_size
        
        metadata = {
            'type': 'IndexIVFPQ',
            'description': 'Búsqueda aproximada con compresión',
            'dimension': self.dimension,
            'num_vectors': index.ntotal,
            'nlist': nlist,
            'nprobe': base_index.nprobe,
            'm': m,
            'nbits': nbits,
            'build_time': build_time,
            'compression_ratio': compression_ratio,
            'uses': 'Datasets grandes, memoria limitada'
        }
        
        print(f"     Completado en {build_time:.2f}s")
        print(f"     Vectores indexados: {index.ntotal}")
        print(f"     Compresión: {compression_ratio:.1f}x")
        
        return index, metadata
    
    def build_hnsw_index(self, M: int = 32) -> Tuple[faiss.Index, dict]:
        """
        Construye un índice HNSW (Hierarchical Navigable Small World) con IDs personalizados.
        
        Args:
            M: Número de conexiones por nodo
            
        Returns:
            tuple: (índice, metadata)
        """
        print(f"\n  [4/4] Construyendo IndexHNSWFlat (M={M})...")
        start_time = time.time()
        
        # Crear índice HNSW base
        base_index = faiss.IndexHNSWFlat(self.dimension, M)
        
        # Envolver con IndexIDMap
        index = faiss.IndexIDMap(base_index)
        
        # Agregar vectores con IDs (HNSW no requiere entrenamiento)
        print(f"    - Agregando vectores al índice...")
        index.add_with_ids(self.features, self.ids)
        
        build_time = time.time() - start_time
        
        metadata = {
            'type': 'IndexHNSWFlat',
            'description': 'Búsqueda aproximada con grafos',
            'dimension': self.dimension,
            'num_vectors': index.ntotal,
            'M': M,
            'build_time': build_time,
            'uses': 'Alta precisión, búsqueda muy rápida'
        }
        
        print(f"     Completado en {build_time:.2f}s")
        print(f"     Vectores indexados: {index.ntotal}")
        
        return index, metadata


def build_all_indices_for_extractor(extractor_name: str, 
                                    features_dir: str = "features",
                                    output_dir: str = "faiss_indices"):
    """
    Construye todos los tipos de índices para un extractor.
    
    Args:
        extractor_name: Nombre del extractor
        features_dir: Directorio de características
        output_dir: Directorio de salida
    """
    print(f"\n{'='*60}")
    print(f"PROCESANDO: {extractor_name}")
    print(f"{'='*60}")
    
    # Cargar características
    features_file = Path(features_dir) / f"{extractor_name}_features.npy"
    ids_file = Path(features_dir) / f"{extractor_name}_ids.npy"
    
    if not features_file.exists():
        print(f" Error: No se encontró {features_file}")
        return
    
    if not ids_file.exists():
        print(f" Error: No se encontró {ids_file}")
        return
    
    print(f"\nCargando características desde {features_file}...")
    features = np.load(features_file)
    
    print(f"Cargando IDs desde {ids_file}...")
    ids = np.load(ids_file)
    
    print(f"   Cargadas {features.shape[0]} características")
    print(f"   Dimensión: {features.shape[1]}")
    print(f"   IDs: {len(ids)} (rango: {ids.min()} - {ids.max()})")
    
    # Crear constructor
    builder = FAISSIndexBuilder(features, ids, features.shape[1])
    
    # Crear directorio de salida
    output_path = Path(output_dir) / extractor_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Construir todos los índices
    indices_info = {}
    
    # 1. Flat Index
    index, metadata = builder.build_flat_index()
    index_file = output_path / "index_flat.index"
    faiss.write_index(index, str(index_file))
    indices_info['flat'] = metadata
    print(f"Guardado: {index_file}")
    
    # 2. IVF Index
    index, metadata = builder.build_ivf_index()
    index_file = output_path / "index_ivf.index"
    faiss.write_index(index, str(index_file))
    indices_info['ivf'] = metadata
    print(f"Guardado: {index_file}")
    
    # 3. IVFPQ Index
    index, metadata = builder.build_ivfpq_index()
    index_file = output_path / "index_ivfpq.index"
    faiss.write_index(index, str(index_file))
    indices_info['ivfpq'] = metadata
    print(f"Guardado: {index_file}")
    
    # 4. HNSW Index
    index, metadata = builder.build_hnsw_index()
    index_file = output_path / "index_hnsw.index"
    faiss.write_index(index, str(index_file))
    indices_info['hnsw'] = metadata
    print(f"Guardado: {index_file}")
    
    # Guardar metadata de todos los índices
    metadata_file = output_path / "indices_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(indices_info, f, indent=2)
    
    print(f"\n   Todos los índices creados para {extractor_name}")
    print(f"   Metadata guardada en {metadata_file}")


def main():
    """
    Función principal.
    """
    print("="*60)
    print("CONSTRUCCIÓN DE ÍNDICES FAISS")
    print("="*60)
    print("\nTipos de índices a construir:")
    print("  1. IndexFlatL2    - Búsqueda exacta")
    print("  2. IndexIVFFlat   - Búsqueda aproximada rápida")
    print("  3. IndexIVFPQ     - Búsqueda aproximada con compresión")
    print("  4. IndexHNSWFlat  - Búsqueda aproximada con grafos")
    
    # Verificar directorio de características
    features_dir = Path("features")
    if not features_dir.exists():
        print("\n Error: Directorio 'features/' no encontrado")
        print("  Ejecuta extract_all_features.py primero")
        return
    
    # Buscar archivos de características
    feature_files = list(features_dir.glob("*_features.npy"))
    
    if not feature_files:
        print("\n Error: No se encontraron archivos de características")
        print("  Ejecuta extract_all_features.py primero")
        return
    
    # Extraer nombres de extractores
    extractors = [f.stem.replace('_features', '') for f in feature_files]
    
    print(f"\n Encontrados {len(extractors)} extractores:")
    for ext in extractors:
        print(f"  - {ext}")
    
    # Construir índices para cada extractor
    for extractor_name in extractors:
        build_all_indices_for_extractor(extractor_name)
    
    print(f"\n{'='*60}")
    print(" CONSTRUCCIÓN DE ÍNDICES COMPLETADA")
    print(f"{'='*60}")
    print("\nResumen:")
    print(f"  - Extractores procesados: {len(extractors)}")
    print(f"  - Índices por extractor: 4")
    print(f"  - Total de índices: {len(extractors) * 4}")
    print("\nArchivos generados en el directorio 'faiss_indices/'")
    print("  Estructura: faiss_indices/<extractor>/index_<tipo>.index")


if __name__ == "__main__":
    main()
