"""
Extractor de características usando ResNet50 pre-entrenado.
"""

import numpy as np
from typing import Union
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from .base_extractor import BaseExtractor


class ResNetExtractor(BaseExtractor):
    """
    Extrae características usando la red ResNet50 pre-entrenada en ImageNet.
    Utiliza la salida de la capa avg_pool (antes de la clasificación).
    """
    
    def __init__(self):
        super().__init__("ResNet50")
        # Cargar modelo sin la capa de clasificación final
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.feature_dim = 2048
        
    def extract(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extrae características usando ResNet50.
        
        Args:
            image: Imagen a procesar
            
        Returns:
            np.ndarray: Vector de características de dimensión 2048, normalizado
        """
        # Cargar y preprocesar imagen
        img = self.load_image(image)
        img = Image.fromarray(img)
        
        # Redimensionar a 224x224 (tamaño requerido por ResNet)
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extraer características
        features = self.model.predict(img_array, verbose=0)
        features = features.flatten()
        
        # Normalizar
        features = self.normalize_features(features)
        
        return features
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
