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
        Carga y convierte una imagen a formato numpy array.
        
        Args:
            image: Imagen en cualquier formato soportado
            
        Returns:
            np.ndarray: Imagen como array de numpy
        """
        if isinstance(image, str):
            img = Image.open(image)
            return np.array(img)
        elif isinstance(image, Image.Image):
            return np.array(image)
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise ValueError(f"Formato de imagen no soportado: {type(image)}")
    
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
