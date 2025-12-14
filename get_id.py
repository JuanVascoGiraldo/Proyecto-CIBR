"""
Módulo para generar IDs únicos y consistentes para imágenes.
Utiliza hash MD5 del path para garantizar que la misma imagen siempre tenga el mismo ID.
"""

import hashlib
from pathlib import Path


def get_image_id(image_path: str) -> int:
    """
    Genera un ID único y consistente para una imagen basado en su path.
    
    El ID es determinístico: el mismo path siempre genera el mismo ID.
    Esto hace que el sistema sea robusto ante reordenamientos.
    
    Args:
        image_path: Ruta de la imagen (puede ser relativa o absoluta)
        
    Returns:
        int: ID único de 32 bits (0 a 4,294,967,295)
        
    Examples:
        >>> get_image_id("images/guitar/train/001.jpg")
        1234567890
        >>> get_image_id("images/guitar/train/001.jpg")  # Mismo resultado
        1234567890
    """
    # Normalizar path (convertir a relativo desde images/)
    path = Path(image_path)
    
    # Si es path absoluto, hacerlo relativo desde 'images'
    if path.is_absolute():
        try:
            # Buscar 'images' en el path
            parts = path.parts
            if 'images' in parts:
                idx = parts.index('images')
                relative_path = Path(*parts[idx:])
            else:
                relative_path = path
        except:
            relative_path = path
    else:
        relative_path = path
    
    # Normalizar separadores (usar / siempre)
    normalized_path = str(relative_path).replace('\\', '/')
    
    # Generar hash MD5
    hash_object = hashlib.md5(normalized_path.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    
    # Tomar primeros 8 caracteres y convertir a entero (32 bits)
    # Esto da IDs en rango 0 a 4,294,967,295
    image_id = int(hash_hex[:8], 16)
    
    return image_id


def get_batch_ids(image_paths: list) -> list:
    """
    Genera IDs para un lote de imágenes.
    
    Args:
        image_paths: Lista de rutas de imágenes
        
    Returns:
        list: Lista de IDs en el mismo orden
        
    Examples:
        >>> paths = ["guitar/001.jpg", "drum/002.jpg"]
        >>> get_batch_ids(paths)
        [1234567890, 9876543210]
    """
    return [get_image_id(path) for path in image_paths]


def verify_id(image_path: str, expected_id: int) -> bool:
    """
    Verifica que un path genera el ID esperado.
    
    Útil para validar integridad después de cargar datos.
    
    Args:
        image_path: Ruta de la imagen
        expected_id: ID que debería generar
        
    Returns:
        bool: True si el ID coincide, False si no
        
    Examples:
        >>> verify_id("guitar/001.jpg", 1234567890)
        True
        >>> verify_id("guitar/001.jpg", 9999999999)
        False
    """
    actual_id = get_image_id(image_path)
    return actual_id == expected_id
