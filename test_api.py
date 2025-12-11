"""
Script de ejemplo para probar la API de búsqueda de imágenes.
"""

import requests
import json
from pathlib import Path


API_URL = "http://localhost:8000"


def test_health():
    """Verifica que la API esté funcionando."""
    print("\n" + "="*60)
    print("1. VERIFICANDO ESTADO DE LA API")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print("✓ API funcionando correctamente")
        print(f"  Estado: {data['status']}")
        print(f"  Extractores cargados: {data['extractors_loaded']}")
        print(f"  Índices en cache: {data['indices_cached']}")
        return True
    else:
        print("✗ Error conectando con la API")
        return False


def list_extractors():
    """Lista los extractores disponibles."""
    print("\n" + "="*60)
    print("2. EXTRACTORES DISPONIBLES")
    print("="*60)
    
    response = requests.get(f"{API_URL}/extractors")
    if response.status_code == 200:
        data = response.json()
        print(f"Total: {data['count']} extractores\n")
        
        for name, info in data['extractors'].items():
            print(f"  {name}:")
            print(f"    - Dimensión: {info['dimension']}")
            print(f"    - Descripción: {info['description']}")
        
        return list(data['extractors'].keys())
    return []


def list_indices():
    """Lista los tipos de índices disponibles."""
    print("\n" + "="*60)
    print("3. TIPOS DE ÍNDICES DISPONIBLES")
    print("="*60)
    
    response = requests.get(f"{API_URL}/indices")
    if response.status_code == 200:
        data = response.json()
        print(f"Total: {data['count']} tipos\n")
        
        for idx_type, info in data['indices'].items():
            print(f"  {idx_type}:")
            print(f"    - Nombre: {info['name']}")
            print(f"    - Descripción: {info['description']}")
            print(f"    - Precisión: {info['precision']}")
            print(f"    - Velocidad: {info['speed']}")
        
        return list(data['indices'].keys())
    return []


def search_similar_images(image_path: str, extractor: str = "ResNet50", 
                         index_type: str = "flat", k: int = 10):
    """
    Busca imágenes similares.
    
    Args:
        image_path: Ruta a la imagen de consulta
        extractor: Nombre del extractor
        index_type: Tipo de índice
        k: Número de resultados
    """
    print("\n" + "="*60)
    print("4. BUSCANDO IMÁGENES SIMILARES")
    print("="*60)
    print(f"  Imagen: {image_path}")
    print(f"  Extractor: {extractor}")
    print(f"  Índice: {index_type}")
    print(f"  K: {k}")
    print()
    
    # Verificar que la imagen existe
    if not Path(image_path).exists():
        print(f"✗ Error: Imagen no encontrada: {image_path}")
        return
    
    # Preparar request
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        params = {
            'extractor': extractor,
            'index_type': index_type,
            'k': k
        }
        
        response = requests.post(
            f"{API_URL}/search",
            files=files,
            params=params
        )
    
    if response.status_code == 200:
        data = response.json()
        
        print("✓ Búsqueda exitosa\n")
        print(f"Extractor usado: {data['extractor']}")
        print(f"Dimensión de características: {data['feature_dimension']}")
        print(f"Resultados encontrados: {len(data['results'])}\n")
        
        print("Top 10 imágenes más similares:")
        print("-" * 60)
        
        for result in data['results']:
            print(f"\n{result['rank']}. {result['filename']}")
            print(f"   Ruta: {result['image_path']}")
            print(f"   Categoría: {result['category']}")
            print(f"   Split: {result['split']}")
            print(f"   Distancia: {result['distance']:.4f}")
        
        return data
    else:
        print(f"✗ Error en la búsqueda: {response.status_code}")
        print(f"  Detalle: {response.text}")
        return None


def get_statistics():
    """Obtiene estadísticas del sistema."""
    print("\n" + "="*60)
    print("5. ESTADÍSTICAS DEL SISTEMA")
    print("="*60)
    
    response = requests.get(f"{API_URL}/stats")
    if response.status_code == 200:
        data = response.json()
        
        print(f"Total de imágenes en el dataset: {data['total_images']}\n")
        print("Por extractor:")
        
        for extractor, info in data['extractors'].items():
            print(f"  {extractor}:")
            print(f"    - Imágenes: {info['num_images']}")
            print(f"    - Dimensión: {info['feature_dim']}")


def main():
    """Función principal de prueba."""
    print("\n" + "="*60)
    print("PRUEBA DE LA API DE BÚSQUEDA DE IMÁGENES")
    print("="*60)
    
    # 1. Verificar estado
    if not test_health():
        print("\n⚠️  La API no está disponible. Asegúrate de iniciarla primero:")
        print("   python api.py")
        return
    
    # 2. Listar extractores
    extractors = list_extractors()
    
    # 3. Listar índices
    indices = list_indices()
    
    # 4. Buscar imagen de ejemplo
    # Buscar una imagen en el dataset
    test_image = None
    for img in Path("images/accordion").glob("*.jpg"):
        test_image = str(img)
        break
    
    if test_image:
        # Prueba con diferentes extractores
        print("\n" + "="*60)
        print("PRUEBAS DE BÚSQUEDA")
        print("="*60)
        
        # Prueba 1: ResNet50 con índice flat
        search_similar_images(test_image, extractor="ResNet50", index_type="flat", k=5)
        
        # Prueba 2: ColorTexture con índice hnsw
        search_similar_images(test_image, extractor="ColorTexture", index_type="hnsw", k=5)
    
    # 5. Estadísticas
    get_statistics()
    
    print("\n" + "="*60)
    print("✓ PRUEBAS COMPLETADAS")
    print("="*60)
    print("\nPara más pruebas:")
    print("  - Visita http://localhost:8000/docs para la documentación interactiva")
    print("  - Usa Postman o curl para hacer requests personalizadas")
    print("\nEjemplo con curl:")
    print('  curl -X POST "http://localhost:8000/search?extractor=ResNet50&index_type=flat&k=10" \\')
    print('       -F "file=@images/accordion/0001.jpg"')


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: No se pudo conectar con la API")
        print("  Asegúrate de que la API está ejecutándose:")
        print("  python api.py")
