"""
Extractor de características usando VGG16 pre-entrenado.
"""

import numpy as np
from typing import Union
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from .base_extractor import BaseExtractor


class VGGExtractor(BaseExtractor):
    """
    Extrae características usando la red VGG16 pre-entrenada en ImageNet.
    Utiliza la salida de la última capa convolucional con pooling global.
    """
    
    def __init__(self):
        super().__init__("VGG16")
        # Cargar modelo sin la capa de clasificación final
        self.model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        self.feature_dim = 512
        
    def extract(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extrae características usando VGG16.
        
        Args:
            image: Imagen a procesar
            
        Returns:
            np.ndarray: Vector de características de dimensión 512, normalizado
        """
        # Cargar y preprocesar imagen
        img = self.load_image(image)
        img = Image.fromarray(img)
        
        # Redimensionar a 224x224 (tamaño requerido por VGG)
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
