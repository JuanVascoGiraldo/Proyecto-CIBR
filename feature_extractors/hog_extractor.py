"""
Extractor de características usando HOG (Histogram of Oriented Gradients).
"""

import numpy as np
from typing import Union
from PIL import Image
import cv2
from skimage.feature import hog

from .base_extractor import BaseExtractor


class HOGExtractor(BaseExtractor):
    """
    Extrae características usando Histogram of Oriented Gradients (HOG).
    Útil para capturar formas y bordes en las imágenes.
    """
    
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), 
                 cells_per_block=(2, 2), resize=(128, 128)):
        """
        Inicializa el extractor HOG.
        
        Args:
            orientations: Número de bins de orientación
            pixels_per_cell: Tamaño de celda en píxeles
            cells_per_block: Número de celdas por bloque
            resize: Tamaño al que redimensionar la imagen
        """
        super().__init__("HOG")
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.resize = resize
        
        # Calcular dimensión del vector HOG
        cells_x = resize[0] // pixels_per_cell[0]
        cells_y = resize[1] // pixels_per_cell[1]
        blocks_x = cells_x - cells_per_block[0] + 1
        blocks_y = cells_y - cells_per_block[1] + 1
        self.feature_dim = (blocks_x * blocks_y * 
                           cells_per_block[0] * cells_per_block[1] * 
                           orientations)
        
    def extract(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extrae características HOG de la imagen.
        
        Args:
            image: Imagen a procesar
            
        Returns:
            np.ndarray: Vector de características HOG normalizado
        """
        # Cargar imagen
        img = self.load_image(image)
        
        # Convertir a escala de grises
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        # Redimensionar
        img_resized = cv2.resize(img_gray, self.resize)
        
        # Extraer características HOG
        features = hog(
            img_resized,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        
        # Normalizar
        features = self.normalize_features(features)
        
        return features
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
