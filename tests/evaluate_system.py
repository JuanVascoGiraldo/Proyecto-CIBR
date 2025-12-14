"""
Script de evaluación del sistema CBIR.
Prueba todas las combinaciones de extractores e índices con imágenes de test.
Genera reportes detallados en formato Excel.
"""

import os
import sys
import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar matplotlib para no mostrar ventanas
plt.switch_backend('Agg')

# Importar extractores
sys.path.append(str(Path(__file__).parent.parent))
from feature_extractors import (
    ResNetExtractor,
    VGGExtractor,
    ColorTextureExtractor,
    HOGExtractor,
    ColorShapeExtractor
)


class CBIRTester:
    """
    Clase para evaluar el sistema CBIR con imágenes de test.
    """
    
    def __init__(self, images_dir="images", features_dir="features", indices_dir="faiss_indices"):
        """
        Inicializa el evaluador.
        
        Args:
            images_dir: Directorio con imágenes
            features_dir: Directorio con características
            indices_dir: Directorio con índices FAISS
        """
        self.images_dir = Path(images_dir)
        self.features_dir = Path(features_dir)
        self.indices_dir = Path(indices_dir)
        
        # Extractores disponibles
        self.extractors = {
            'ResNet50': ResNetExtractor(),
            'VGG16': VGGExtractor(),
            'ColorTexture': ColorTextureExtractor(),
            'HOG': HOGExtractor(),
            'ColorShape': ColorShapeExtractor()
        }
        
        # Tipos de índices
        self.index_types = ['flat', 'ivf', 'ivfpq', 'hnsw']
        
        # Categorías
        self.categories = ['accordion', 'drum', 'flute', 'guitar', 'saxophone', 'violin']
        
        print("="*70)
        print("SISTEMA DE EVALUACIÓN CBIR")
        print("="*70)
        print(f"\nExtractores: {list(self.extractors.keys())}")
        print(f"Índices: {self.index_types}")
        print(f"Categorías: {self.categories}")
    
    def get_test_images(self):
        """
        Obtiene todas las imágenes de test.
        
        Returns:
            list: Lista de diccionarios con info de imágenes de test
        """
        test_images = []
        
        for category in self.categories:
            test_path = self.images_dir / category / 'test'
            
            if not test_path.exists():
                continue
            
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in test_path.glob(ext):
                    test_images.append({
                        'path': str(img_path),
                        'category': category,
                        'filename': img_path.name
                    })
        
        return test_images
    
    def load_index_and_mapping(self, extractor_name, index_type):
        """
        Carga un índice FAISS y su ID mapping.
        
        Args:
            extractor_name: Nombre del extractor
            index_type: Tipo de índice
            
        Returns:
            tuple: (index, id_mapping)
        """
        # Cargar índice
        index_path = self.indices_dir / extractor_name / f"index_{index_type}.index"
        if not index_path.exists():
            raise FileNotFoundError(f"Índice no encontrado: {index_path}")
        
        index = faiss.read_index(str(index_path))
        
        # Cargar ID mapping
        mapping_path = self.features_dir / f"{extractor_name}_id_mapping.pkl"
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping no encontrado: {mapping_path}")
        
        with open(mapping_path, 'rb') as f:
            id_mapping = pickle.load(f)
        
        return index, id_mapping
    
    def search_similar(self, query_image_path, extractor_name, index_type, k=10):
        """
        Busca las K imágenes más similares.
        
        Args:
            query_image_path: Path de la imagen de consulta
            extractor_name: Nombre del extractor
            index_type: Tipo de índice
            k: Número de resultados
            
        Returns:
            list: Lista de resultados con distancias y categorías
        """
        # Extraer características de la imagen de consulta
        extractor = self.extractors[extractor_name]
        query_features = extractor.extract(query_image_path)
        
        # Cargar índice y mapping
        index, id_mapping = self.load_index_and_mapping(extractor_name, index_type)
        
        # Buscar
        query_vector = query_features.reshape(1, -1).astype(np.float32)
        distances, ids = index.search(query_vector, k)
        
        # Preparar resultados
        results = []
        for dist, img_id in zip(distances[0], ids[0]):
            if img_id == -1:  # FAISS retorna -1 si no encuentra suficientes resultados
                continue
            
            img_info = id_mapping[img_id]
            results.append({
                'image_id': int(img_id),
                'distance': float(dist),
                'category': img_info['category'],
                'split': img_info['split'],
                'path': img_info['path'],
                'filename': img_info['filename']
            })
        
        return results
    
    def evaluate_results(self, query_category, results):
        """
        Evalúa los resultados de una búsqueda.
        
        Args:
            query_category: Categoría de la imagen de consulta
            results: Lista de resultados
            
        Returns:
            dict: Métricas de evaluación
        """
        total = len(results)
        correct = sum(1 for r in results if r['category'] == query_category)
        
        # Precision@K
        precision = correct / total if total > 0 else 0
        
        # Similitud promedio (convertir distancia a similitud: 1 / (1 + distancia))
        avg_distance = np.mean([r['distance'] for r in results]) if results else 0
        avg_similarity = 1 / (1 + avg_distance) if avg_distance >= 0 else 0
        
        # Categorías encontradas
        categories_found = {}
        for r in results:
            cat = r['category']
            categories_found[cat] = categories_found.get(cat, 0) + 1
        
        # Errores (categorías incorrectas)
        errors = []
        for r in results:
            if r['category'] != query_category:
                errors.append(r['category'])
        
        return {
            'total_results': total,
            'correct_category': correct,
            'incorrect_category': total - correct,
            'precision': precision,
            'avg_distance': avg_distance,
            'avg_similarity': avg_similarity,
            'categories_found': categories_found,
            'errors': errors
        }
    
    def run_full_evaluation(self, output_file='evaluation_results.xlsx'):
        """
        Ejecuta evaluación completa con todas las combinaciones.
        
        Args:
            output_file: Archivo Excel de salida
        """
        print("\n" + "="*70)
        print("INICIANDO EVALUACIÓN COMPLETA")
        print("="*70)
        
        # Obtener imágenes de test
        test_images = self.get_test_images()
        print(f"\nImágenes de test encontradas: {len(test_images)}")
        
        if not test_images:
            print(" No se encontraron imágenes de test")
            return
        
        # Preparar almacenamiento de resultados
        detailed_results = []
        summary_results = []
        confusion_data = []  # Para matriz de confusión
        error_analysis = []  # Para análisis de errores por clase
        
        # Total de combinaciones
        total_combinations = len(self.extractors) * len(self.index_types)
        total_tests = len(test_images) * total_combinations
        
        print(f"\nCombinaciones a evaluar: {total_combinations}")
        print(f"Total de pruebas: {total_tests}")
        print(f"\nIniciando evaluación...")
        print("="*70)
        
        # Barra de progreso global
        with tqdm(total=total_tests, desc="Progreso global") as pbar:
            
            # Para cada combinación de extractor e índice
            for extractor_name in self.extractors.keys():
                for index_type in self.index_types:
                    
                    combination_name = f"{extractor_name}_{index_type}"
                    
                    # Métricas acumuladas para esta combinación
                    combo_precisions = []
                    combo_similarities = []
                    combo_distances = []
                    combo_confusion = {}  # Confusiones: {true_class: {predicted_class: count}}
                    
                    # Para cada imagen de test
                    for test_img in test_images:
                        try:
                            # Buscar similares
                            start_time = time.time()
                            results = self.search_similar(
                                test_img['path'],
                                extractor_name,
                                index_type,
                                k=10
                            )
                            search_time = time.time() - start_time
                            
                            # Evaluar resultados
                            metrics = self.evaluate_results(test_img['category'], results)
                            
                            # Guardar resultado detallado
                            detailed_results.append({
                                'query_image': test_img['filename'],
                                'query_category': test_img['category'],
                                'extractor': extractor_name,
                                'index_type': index_type,
                                'total_results': metrics['total_results'],
                                'correct_matches': metrics['correct_category'],
                                'incorrect_matches': metrics['incorrect_category'],
                                'precision': metrics['precision'],
                                'avg_distance': metrics['avg_distance'],
                                'avg_similarity': metrics['avg_similarity'],
                                'search_time_sec': search_time,
                                'categories_found': str(metrics['categories_found'])
                            })
                            
                            # Acumular métricas
                            combo_precisions.append(metrics['precision'])
                            combo_similarities.append(metrics['avg_similarity'])
                            combo_distances.append(metrics['avg_distance'])
                            
                            # Registrar confusiones para esta consulta
                            true_class = test_img['category']
                            if true_class not in combo_confusion:
                                combo_confusion[true_class] = {}
                            
                            for error_class in metrics['errors']:
                                if error_class not in combo_confusion[true_class]:
                                    combo_confusion[true_class][error_class] = 0
                                combo_confusion[true_class][error_class] += 1
                            
                        except Exception as e:
                            print(f"\n Error con {test_img['filename']} - {combination_name}: {e}")
                        
                        pbar.update(1)
                    
                    # Calcular estadísticas para esta combinación
                    if combo_precisions:
                        summary_results.append({
                            'extractor': extractor_name,
                            'index_type': index_type,
                            'num_tests': len(combo_precisions),
                            'avg_precision': np.mean(combo_precisions),
                            'std_precision': np.std(combo_precisions),
                            'min_precision': np.min(combo_precisions),
                            'max_precision': np.max(combo_precisions),
                            'avg_similarity': np.mean(combo_similarities),
                            'std_similarity': np.std(combo_similarities),
                            'avg_distance': np.mean(combo_distances),
                            'std_distance': np.std(combo_distances)
                        })
                        
                        # Analizar confusiones para esta combinación
                        for true_class, confused_with in combo_confusion.items():
                            if confused_with:
                                # Encontrar la clase más común con la que se confunde
                                most_confused = max(confused_with.items(), key=lambda x: x[1])
                                total_errors = sum(confused_with.values())
                                
                                confusion_data.append({
                                    'extractor': extractor_name,
                                    'index_type': index_type,
                                    'true_class': true_class,
                                    'most_confused_with': most_confused[0],
                                    'confusion_count': most_confused[1],
                                    'total_errors': total_errors,
                                    'confusion_rate': most_confused[1] / total_errors if total_errors > 0 else 0,
                                    'all_confusions': str(confused_with)
                                })
        
        # Analizar errores por clase y modelo
        error_analysis = self.analyze_class_errors(detailed_results)
        
        # Guardar resultados en Excel
        print(f"\n\n{'='*70}")
        print("GUARDANDO RESULTADOS")
        print("="*70)
        
        output_path = Path('tests') / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Hoja 1: Resultados detallados
            df_detailed = pd.DataFrame(detailed_results)
            df_detailed.to_excel(writer, sheet_name='Resultados_Detallados', index=False)
            
            # Hoja 2: Resumen por combinación
            df_summary = pd.DataFrame(summary_results)
            df_summary = df_summary.sort_values('avg_precision', ascending=False)
            df_summary.to_excel(writer, sheet_name='Resumen_Combinaciones', index=False)
            
            # Hoja 3: Matriz de confusión
            df_confusion = pd.DataFrame(confusion_data)
            if not df_confusion.empty:
                df_confusion = df_confusion.sort_values(['extractor', 'true_class', 'confusion_count'], ascending=[True, True, False])
                df_confusion.to_excel(writer, sheet_name='Matriz_Confusion', index=False)
            
            # Hoja 4: Análisis de errores por clase
            df_errors = pd.DataFrame(error_analysis)
            if not df_errors.empty:
                df_errors = df_errors.sort_values(['class', 'total_errors'], ascending=[True, False])
                df_errors.to_excel(writer, sheet_name='Errores_Por_Clase', index=False)
            
            # Hoja 5: Estadísticas generales
            general_stats = self.calculate_general_statistics(df_detailed, df_summary)
            df_stats = pd.DataFrame([general_stats])
            df_stats.to_excel(writer, sheet_name='Estadisticas_Generales', index=False)
        
        print(f"\n Resultados guardados en: {output_path}")
        
        # Generar visualizaciones
        if not df_confusion.empty:
            self.generate_confusion_matrices(df_confusion)
        
        if not df_errors.empty:
            self.generate_statistics_graphs(df_summary, df_detailed, df_errors)
        
        print(f"\n{'='*70}")
        print("EVALUACIÓN COMPLETADA")
        print("="*70)
        
        # Mostrar resumen
        self.print_summary(df_summary)
    
    def analyze_class_errors(self, detailed_results):
        """
        Analiza errores por clase y modelo.
        
        Args:
            detailed_results: Lista de resultados detallados
            
        Returns:
            list: Lista de diccionarios con análisis de errores
        """
        error_analysis = []
        
        # Agrupar por clase
        for category in self.categories:
            # Filtrar resultados de esta clase
            class_results = [r for r in detailed_results if r['query_category'] == category]
            
            if not class_results:
                continue
            
            # Para cada combinación extractor/índice
            for extractor_name in self.extractors.keys():
                for index_type in self.index_types:
                    # Filtrar por esta combinación
                    combo_results = [r for r in class_results 
                                   if r['extractor'] == extractor_name and r['index_type'] == index_type]
                    
                    if not combo_results:
                        continue
                    
                    # Calcular errores
                    total_queries = len(combo_results)
                    total_errors = sum(r['incorrect_matches'] for r in combo_results)
                    avg_precision = np.mean([r['precision'] for r in combo_results])
                    
                    error_analysis.append({
                        'class': category,
                        'extractor': extractor_name,
                        'index_type': index_type,
                        'total_queries': total_queries,
                        'total_errors': total_errors,
                        'avg_errors_per_query': total_errors / total_queries if total_queries > 0 else 0,
                        'avg_precision': avg_precision,
                        'error_rate': 1 - avg_precision
                    })
        
        return error_analysis
    
    def calculate_general_statistics(self, df_detailed, df_summary):
        """
        Calcula estadísticas generales.
        
        Args:
            df_detailed: DataFrame con resultados detallados
            df_summary: DataFrame con resumen
            
        Returns:
            dict: Estadísticas generales
        """
        return {
            'total_tests': len(df_detailed),
            'total_combinations': len(df_summary),
            'overall_avg_precision': df_detailed['precision'].mean(),
            'overall_std_precision': df_detailed['precision'].std(),
            'overall_avg_similarity': df_detailed['avg_similarity'].mean(),
            'overall_std_similarity': df_detailed['avg_similarity'].std(),
            'best_extractor': df_summary.iloc[0]['extractor'],
            'best_index': df_summary.iloc[0]['index_type'],
            'best_precision': df_summary.iloc[0]['avg_precision'],
            'worst_extractor': df_summary.iloc[-1]['extractor'],
            'worst_index': df_summary.iloc[-1]['index_type'],
            'worst_precision': df_summary.iloc[-1]['avg_precision']
        }
    
    def print_summary(self, df_summary):
        """
        Imprime resumen de resultados.
        
        Args:
            df_summary: DataFrame con resumen
        """
        print("\n TOP 5 MEJORES COMBINACIONES:")
        print("-" * 70)
        for i, row in df_summary.head(5).iterrows():
            print(f"{row['extractor']:15s} + {row['index_type']:6s} | "
                  f"Precisión: {row['avg_precision']:.2%} | "
                  f"Similitud: {row['avg_similarity']:.4f}")
        
        print(f"\n TOP 5 PEORES COMBINACIONES:")
        print("-" * 70)
        for i, row in df_summary.tail(5).iterrows():
            print(f"{row['extractor']:15s} + {row['index_type']:6s} | "
                  f"Precisión: {row['avg_precision']:.2%} | "
                  f"Similitud: {row['avg_similarity']:.4f}")
    
    def generate_confusion_matrices(self, df_confusion, output_dir='results'):
        """
        Genera matrices de confusión visuales.
        
        Args:
            df_confusion: DataFrame con datos de confusión
            output_dir: Directorio de salida
        """
        print(f"\n{'='*70}")
        print("GENERANDO MATRICES DE CONFUSIÓN")
        print("="*70)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Para cada combinación
        combinations = df_confusion[['extractor', 'index_type']].drop_duplicates()
        
        for _, combo in tqdm(combinations.iterrows(), total=len(combinations), desc="Matrices de confusión"):
            extractor = combo['extractor']
            index_type = combo['index_type']
            
            # Filtrar datos de esta combinación
            combo_data = df_confusion[
                (df_confusion['extractor'] == extractor) & 
                (df_confusion['index_type'] == index_type)
            ]
            
            # Crear matriz de confusión
            conf_matrix = np.zeros((len(self.categories), len(self.categories)))
            
            for _, row in combo_data.iterrows():
                true_idx = self.categories.index(row['true_class'])
                
                # Parsear all_confusions
                import ast
                confusions = ast.literal_eval(row['all_confusions'])
                
                for confused_class, count in confusions.items():
                    confused_idx = self.categories.index(confused_class)
                    conf_matrix[true_idx][confused_idx] = count
            
            # Agregar diagonal (aciertos) - estimado
            # Asumimos que si hay 10 imágenes de test por clase
            for i in range(len(self.categories)):
                # Calcular aciertos basados en los errores
                total_errors = conf_matrix[i].sum()
                # 10 resultados por imagen * 10 imágenes = 100 total
                conf_matrix[i][i] = 100 - total_errors
            
            # Crear figura
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='.0f', cmap='Blues',
                       xticklabels=self.categories, yticklabels=self.categories,
                       cbar_kws={'label': 'Cantidad'})
            plt.title(f'Matriz de Confusión: {extractor} + {index_type}', fontsize=14, fontweight='bold')
            plt.xlabel('Clase Predicha', fontsize=12)
            plt.ylabel('Clase Verdadera', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Guardar
            filename = f"confusion_matrix_{extractor}_{index_type}.png"
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f" Matrices guardadas en: {output_path}")
    
    def generate_statistics_graphs(self, df_summary, df_detailed, df_errors, output_dir='tests/graphics'):
        """
        Genera gráficos estadísticos.
        
        Args:
            df_summary: DataFrame con resumen por combinación
            df_detailed: DataFrame con resultados detallados
            df_errors: DataFrame con errores por clase
            output_dir: Directorio de salida
        """
        print(f"\n{'='*70}")
        print("GENERANDO GRÁFICOS ESTADÍSTICOS")
        print("="*70)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Precisión por Extractor
        print("  • Gráfico: Precisión por extractor...")
        plt.figure(figsize=(12, 6))
        extractor_precision = df_summary.groupby('extractor')['avg_precision'].mean().sort_values(ascending=False)
        colors = plt.cm.viridis(np.linspace(0, 1, len(extractor_precision)))
        bars = plt.bar(extractor_precision.index, extractor_precision.values, color=colors, edgecolor='black', linewidth=1.5)
        plt.title('Precisión Promedio por Extractor', fontsize=14, fontweight='bold')
        plt.xlabel('Extractor', fontsize=12)
        plt.ylabel('Precisión Promedio', fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        # Agregar valores en las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'precision_by_extractor.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Precisión por Tipo de Índice
        print("  • Gráfico: Precisión por tipo de índice...")
        plt.figure(figsize=(10, 6))
        index_precision = df_summary.groupby('index_type')['avg_precision'].mean().sort_values(ascending=False)
        colors = plt.cm.plasma(np.linspace(0, 1, len(index_precision)))
        bars = plt.bar(index_precision.index, index_precision.values, color=colors, edgecolor='black', linewidth=1.5)
        plt.title('Precisión Promedio por Tipo de Índice', fontsize=14, fontweight='bold')
        plt.xlabel('Tipo de Índice', fontsize=12)
        plt.ylabel('Precisión Promedio', fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'precision_by_index.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Comparación de Top 10 Combinaciones
        print("  • Gráfico: Top 10 combinaciones...")
        plt.figure(figsize=(14, 7))
        top10 = df_summary.nlargest(10, 'avg_precision')
        combo_names = [f"{row['extractor']}\n{row['index_type']}" for _, row in top10.iterrows()]
        colors = plt.cm.coolwarm(np.linspace(0.3, 1, len(top10)))
        bars = plt.bar(range(len(top10)), top10['avg_precision'].values, color=colors, edgecolor='black', linewidth=1.5)
        plt.xticks(range(len(top10)), combo_names, rotation=0, ha='center')
        plt.title('Top 10 Mejores Combinaciones', fontsize=14, fontweight='bold')
        plt.xlabel('Extractor + Índice', fontsize=12)
        plt.ylabel('Precisión Promedio', fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'top10_combinations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Heatmap de Precisión (Extractor vs Índice)
        print("  • Gráfico: Heatmap extractor vs índice...")
        plt.figure(figsize=(10, 8))
        pivot_table = df_summary.pivot(index='extractor', columns='index_type', values='avg_precision')
        sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap='RdYlGn', center=0.5,
                   cbar_kws={'label': 'Precisión'}, linewidths=1, linecolor='gray')
        plt.title('Precisión: Extractor vs Tipo de Índice', fontsize=14, fontweight='bold')
        plt.xlabel('Tipo de Índice', fontsize=12)
        plt.ylabel('Extractor', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path / 'heatmap_extractor_vs_index.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Distribución de Precisión por Categoría
        print("  • Gráfico: Precisión por categoría...")
        plt.figure(figsize=(12, 6))
        category_precision = df_detailed.groupby('query_category')['precision'].mean().sort_values(ascending=False)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(category_precision)))
        bars = plt.bar(category_precision.index, category_precision.values, color=colors, edgecolor='black', linewidth=1.5)
        plt.title('Precisión Promedio por Categoría', fontsize=14, fontweight='bold')
        plt.xlabel('Categoría', fontsize=12)
        plt.ylabel('Precisión Promedio', fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'precision_by_category.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Tiempo de Búsqueda por Combinación
        print("  • Gráfico: Tiempo de búsqueda...")
        plt.figure(figsize=(14, 7))
        avg_time = df_detailed.groupby(['extractor', 'index_type'])['search_time_sec'].mean().reset_index()
        avg_time['combo'] = avg_time['extractor'] + '\n' + avg_time['index_type']
        avg_time = avg_time.sort_values('search_time_sec', ascending=False).head(15)
        colors = plt.cm.autumn(np.linspace(0, 1, len(avg_time)))
        bars = plt.bar(range(len(avg_time)), avg_time['search_time_sec'].values, color=colors, edgecolor='black', linewidth=1.5)
        plt.xticks(range(len(avg_time)), avg_time['combo'].values, rotation=45, ha='right')
        plt.title('Tiempo Promedio de Búsqueda (Top 15 más lentos)', fontsize=14, fontweight='bold')
        plt.xlabel('Extractor + Índice', fontsize=12)
        plt.ylabel('Tiempo (segundos)', fontsize=12)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}s', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path / 'search_time_by_combination.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Errores por Clase y Extractor
        print("  • Gráfico: Errores por clase y extractor...")
        plt.figure(figsize=(14, 8))
        error_pivot = df_errors.pivot_table(index='class', columns='extractor', 
                                            values='avg_errors_per_query', aggfunc='mean')
        sns.heatmap(error_pivot, annot=True, fmt='.2f', cmap='YlOrRd',
                   cbar_kws={'label': 'Errores Promedio por Consulta'}, 
                   linewidths=1, linecolor='gray')
        plt.title('Errores Promedio: Clase vs Extractor', fontsize=14, fontweight='bold')
        plt.xlabel('Extractor', fontsize=12)
        plt.ylabel('Clase', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path / 'errors_class_vs_extractor.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 8. Box Plot de Precisión por Extractor
        print("  • Gráfico: Distribución de precisión...")
        plt.figure(figsize=(12, 7))
        df_detailed_sorted = df_detailed.sort_values('extractor')
        box_data = [df_detailed_sorted[df_detailed_sorted['extractor'] == ext]['precision'].values 
                   for ext in sorted(df_detailed['extractor'].unique())]
        bp = plt.boxplot(box_data, labels=sorted(df_detailed['extractor'].unique()),
                        patch_artist=True, notch=True, showmeans=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        plt.title('Distribución de Precisión por Extractor', fontsize=14, fontweight='bold')
        plt.xlabel('Extractor', fontsize=12)
        plt.ylabel('Precisión', fontsize=12)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'precision_distribution_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Gráficos guardados en: {output_path}")


def main():
    """
    Función principal.
    """
    # Crear evaluador
    tester = CBIRTester()
    
    # Generar nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_{timestamp}.xlsx"
    
    # Ejecutar evaluación
    tester.run_full_evaluation(output_file)


if __name__ == "__main__":
    main()
