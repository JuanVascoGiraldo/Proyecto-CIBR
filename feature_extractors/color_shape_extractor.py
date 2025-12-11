"""
Extractor combinado de características de Color (RGB Histogram) y Forma (Hu Moments).
"""

import numpy as np
from typing import Union
from PIL import Image
import cv2

from .base_extractor import BaseExtractor


class ColorShapeExtractor(BaseExtractor):
    """
    Combina características de color (histograma RGB) y forma (momentos de Hu).
    """
    
    def __init__(self, color_bins=32):
        """
        Inicializa el extractor.
        
        Args:
            color_bins: Número de bins por canal de color
        """
        super().__init__("ColorShape")
        self.color_bins = color_bins
        
        # Dimensión: RGB histogram (3 canales) + 7 momentos de Hu
        self.color_dim = color_bins * 3
        self.shape_dim = 7
        self.feature_dim = self.color_dim + self.shape_dim
        
    def extract_color_histogram(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Extrae histograma de color RGB.
        
        Args:
            img_bgr: Imagen en formato BGR
            
        Returns:
            np.ndarray: Histograma concatenado de R, G, B normalizado
        """
        histograms = []
        
        # Calcular histograma para cada canal
        for i in range(3):
            hist = cv2.calcHist(
                [img_bgr], 
                [i], 
                None, 
                [self.color_bins], 
                [0, 256]
            )
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)
            histograms.append(hist)
        
        # Concatenar histogramas
        color_features = np.concatenate(histograms)
        
        return color_features
    
    def extract_hu_moments(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Extrae momentos de Hu (invariantes a rotación, escala y traslación).
        
        Args:
            img_gray: Imagen en escala de grises
            
        Returns:
            np.ndarray: 7 momentos de Hu
        """
        # Calcular momentos
        moments = cv2.moments(img_gray)
        
        # Calcular momentos de Hu
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Aplicar transformación logarítmica para estabilizar
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        return hu_moments
    
    def extract(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extrae características combinadas de color y forma.
        
        Args:
            image: Imagen a procesar
            
        Returns:
            np.ndarray: Vector de características normalizado
        """
        # Cargar imagen
        img = self.load_image(image)
        
        # Convertir a BGR para OpenCV
        if len(img.shape) == 2:  # Escala de grises
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_gray = img
        elif img.shape[2] == 4:  # RGBA
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:  # RGB
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Extraer características
        color_features = self.extract_color_histogram(img_bgr)
        shape_features = self.extract_hu_moments(img_gray)
        
        # Normalizar shape features individualmente
        shape_features = shape_features / (np.linalg.norm(shape_features) + 1e-7)
        
        # Concatenar (color ya está normalizado)
        features = np.concatenate([color_features, shape_features])
        
        # Normalizar vector completo
        features = self.normalize_features(features)
        
        return features
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
