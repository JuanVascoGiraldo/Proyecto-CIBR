"""
Script para detectar imágenes con canales especiales (grayscale o RGBA).
Genera un reporte en image_channels_report.txt
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm


def check_images(base_path="images"):
    """
    Verifica el modo de color de todas las imágenes.
    Solo reporta las que NO son RGB estándar.
    
    Args:
        base_path: Directorio base con las imágenes
    """
    base_path = Path(base_path)
    categories = ['accordion', 'drum', 'flute', 'guitar', 'saxophone', 'violin']
    
    special_images = []
    total_images = 0
    
    print("="*70)
    print("ANÁLISIS DE CANALES DE IMAGEN")
    print("="*70)
    print()
    
    # Recopilar todas las rutas de imágenes
    all_images = []
    for category in categories:
        category_path = base_path / category
        if not category_path.exists():
            continue
        
        for split in ['train', 'test']:
            split_path = category_path / split
            if split_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    all_images.extend(list(split_path.glob(ext)))
    
    print(f"Total de imágenes a analizar: {len(all_images)}")
    print()
    
    # Analizar cada imagen
    for img_path in tqdm(all_images, desc="Analizando imágenes"):
        total_images += 1
        
        try:
            with Image.open(img_path) as img:
                mode = img.mode
                
                # Solo guardar las que NO son RGB estándar
                if mode in ['L', 'LA', 'RGBA', 'P', '1']:
                    description = {
                        'L': 'Grayscale (1 canal)',
                        'LA': 'Grayscale con Alpha (2 canales)',
                        'RGBA': 'RGB con Alpha (4 canales)',
                        'P': 'Paleta de colores',
                        '1': 'Blanco y negro (1 bit)'
                    }
                    
                    special_images.append({
                        'path': str(img_path.relative_to(base_path)),
                        'mode': mode,
                        'description': description.get(mode, mode),
                        'size': img.size
                    })
        except Exception as e:
            special_images.append({
                'path': str(img_path.relative_to(base_path)),
                'mode': 'ERROR',
                'description': f'Error al leer: {e}',
                'size': (0, 0)
            })
    
    # Generar reporte
    output_file = "image_channels_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE CANALES DE IMAGEN\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total de imágenes analizadas: {total_images}\n")
        f.write(f"Imágenes con canales especiales: {len(special_images)}\n")
        f.write(f"Imágenes RGB estándar: {total_images - len(special_images)}\n\n")
        
        if special_images:
            f.write("="*70 + "\n")
            f.write("IMÁGENES CON CANALES ESPECIALES\n")
            f.write("="*70 + "\n\n")
            
            # Agrupar por modo
            by_mode = {}
            for img in special_images:
                mode = img['mode']
                if mode not in by_mode:
                    by_mode[mode] = []
                by_mode[mode].append(img)
            
            # Escribir por categoría
            for mode, images in sorted(by_mode.items()):
                f.write(f"\n{'='*70}\n")
                f.write(f"MODO: {mode}\n")
                f.write(f"Descripción: {images[0]['description']}\n")
                f.write(f"Total: {len(images)} imágenes\n")
                f.write(f"{'='*70}\n\n")
                
                for img in sorted(images, key=lambda x: x['path']):
                    f.write(f"  {img['path']}\n")
                    f.write(f"    Tamaño: {img['size'][0]}x{img['size'][1]}\n\n")
        else:
            f.write("\n Todas las imágenes están en formato RGB estándar\n")
    
    # Resumen en consola
    print()
    print("="*70)
    print("RESUMEN")
    print("="*70)
    print(f"\nTotal de imágenes analizadas: {total_images}")
    print(f"Imágenes RGB estándar: {total_images - len(special_images)}")
    print(f"Imágenes con canales especiales: {len(special_images)}")
    
    if special_images:
        print("\nPor tipo:")
        by_mode = {}
        for img in special_images:
            mode = img['mode']
            by_mode[mode] = by_mode.get(mode, 0) + 1
        
        for mode, count in sorted(by_mode.items()):
            desc = {
                'L': 'Grayscale',
                'LA': 'Grayscale + Alpha',
                'RGBA': 'RGB + Alpha',
                'P': 'Paleta',
                '1': 'Blanco y negro (1-bit)',
                'ERROR': 'Error al leer'
            }
            print(f"  {mode:6s} ({desc.get(mode, mode):20s}): {count:3d} imágenes")
    
    print(f"\n Reporte guardado en: {output_file}")
    print()


if __name__ == "__main__":
    check_images()
