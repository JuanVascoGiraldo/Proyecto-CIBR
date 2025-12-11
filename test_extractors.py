"""
Script para probar todos los extractores de características.
Prueba cada extractor con una imagen de ejemplo y muestra las dimensiones.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extractors import (
    ResNetExtractor,
    VGGExtractor,
    ColorTextureExtractor,
    HOGExtractor,
    ColorShapeExtractor
)


def test_extractor(extractor, image_path: str):
    """
    Prueba un extractor con una imagen.
    
    Args:
        extractor: Instancia del extractor
        image_path: Ruta a la imagen de prueba
    """
    print(f"\n{'='*60}")
    print(f"Probando: {extractor}")
    print(f"{'='*60}")
    
    try:
        # Medir tiempo de extracción
        start_time = time.time()
        features = extractor.extract(image_path)
        elapsed_time = time.time() - start_time
        
        # Mostrar resultados
        print(f"✓ Extracción exitosa")
        print(f"  - Dimensión del vector: {len(features)}")
        print(f"  - Dimensión esperada: {extractor.get_feature_dim()}")
        print(f"  - Tiempo de extracción: {elapsed_time:.4f} segundos")
        print(f"  - Rango de valores: [{features.min():.4f}, {features.max():.4f}]")
        print(f"  - Norma L2: {np.linalg.norm(features):.4f}")
        print(f"  - Primeros 10 valores: {features[:10]}")
        
        return True, features
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """
    Función principal para probar todos los extractores.
    """
    print("="*60)
    print("PRUEBA DE EXTRACTORES DE CARACTERÍSTICAS")
    print("="*60)
    
    # Usar imagen específica
    test_image = "images/accordion/0001.jpg"
    
    if not Path(test_image).exists():
        print(f"\n✗ Error: No se encontró la imagen {test_image}")
        return
    
    print(f"\nImagen de prueba: {test_image}")
    
    # Inicializar extractores
    extractors = [
        ResNetExtractor(),
        VGGExtractor(),
        ColorTextureExtractor(),
        HOGExtractor(),
        ColorShapeExtractor()
    ]
    
    # Probar cada extractor
    results = {}
    for extractor in extractors:
        success, features = test_extractor(extractor, test_image)
        if success:
            results[extractor.name] = {
                'features': features,
                'dim': len(features)
            }
    
    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN DE RESULTADOS")
    print(f"{'='*60}")
    print(f"Extractores exitosos: {len(results)}/{len(extractors)}")
    
    if results:
        print("\nComparación de dimensiones:")
        for name, data in results.items():
            print(f"  {name:20s}: {data['dim']:5d} dimensiones")
        
        print("\nRecomendaciones:")
        print("  - ResNet50 y VGG16: Mejores para similitud semántica")
        print("  - ColorTexture: Rápido, bueno para patrones visuales")
        print("  - HOG: Excelente para formas y bordes")
        print("  - ColorShape: Combina color global con forma")
        
        print("\n✓ Todos los extractores están listos para usar")
        print("  Puedes importarlos con:")
        print("  from feature_extractors import ResNetExtractor, VGGExtractor, ...")


if __name__ == "__main__":
    main()
