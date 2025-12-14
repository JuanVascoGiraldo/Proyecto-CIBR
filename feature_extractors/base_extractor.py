"""
Clase base abstracta para todos los extractores de características.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union
from PIL import Image


class BaseExtractor(ABC):
    """
    Clase base para todos los extractores de características.
    Define la interfaz común que todos los extractores deben implementar.
    """
    
    def __init__(self, name: str):
        """
        Inicializa el extractor.
        
        Args:
            name: Nombre identificador del extractor
        """
        self.name = name
    
    @abstractmethod
    def extract(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extrae características de una imagen.
        
        Args:
            image: Puede ser:
                - str: Ruta a la imagen
                - np.ndarray: Array de numpy con la imagen
                - PIL.Image: Objeto Image de PIL
        
        Returns:
            np.ndarray: Vector de características normalizado
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Retorna la dimensión del vector de características.
        
        Returns:
            int: Dimensión del vector
        """
        pass
    
    def load_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Carga y convierte una imagen a formato numpy array RGB.
        Convierte automáticamente imágenes en escala de grises o con canal alpha.
        
        Args:
            image: Imagen en cualquier formato soportado
            
        Returns:
            np.ndarray: Imagen como array de numpy en formato RGB
        """
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        elif isinstance(image, np.ndarray):
            return self._ensure_rgb(image)
        else:
            raise ValueError(f"Formato de imagen no soportado: {type(image)}")
        
        # Convertir a RGB si es necesario
        if img.mode == 'L':  # Grayscale
            img = img.convert('RGB')
        elif img.mode == 'RGBA':  # Con canal alpha
            # Crear fondo blanco y pegar la imagen con alpha
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Usar canal alpha como máscara
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img)
    
    def _ensure_rgb(self, img_array: np.ndarray) -> np.ndarray:
        """
        Asegura que la imagen es RGB (3 canales).
        
        Args:
            img_array: Array de numpy con la imagen
            
        Returns:
            np.ndarray: Imagen en formato RGB
        """
        # Si es grayscale (2D o 3D con 1 canal), convertir a RGB
        if len(img_array.shape) == 2:
            return np.stack([img_array] * 3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            return np.concatenate([img_array] * 3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA: quitar canal alpha y usar fondo blanco
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3:4] / 255.0
            return (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
        return img_array
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normaliza el vector de características a L2 norm.
        
        Args:
            features: Vector de características sin normalizar
            
        Returns:
            np.ndarray: Vector normalizado
        """
        norm = np.linalg.norm(features)
        if norm == 0:
            return features
        return features / norm
    
    def __str__(self) -> str:
        return f"{self.name} (dim={self.get_feature_dim()})"
