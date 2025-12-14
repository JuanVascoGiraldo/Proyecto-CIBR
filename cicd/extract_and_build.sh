#!/bin/bash
# Script para Mac/Linux: Extracción de características y construcción de índices FAISS
# Uso: ./cicd/extract_and_build.sh

set -e  # Detener en caso de error

echo "======================================================================"
echo "EXTRACCIÓN DE CARACTERÍSTICAS Y CONSTRUCCIÓN DE ÍNDICES FAISS"
echo "======================================================================"
echo ""


# Verificar/Crear entorno virtual
if [ ! -d ".venv" ]; then
    echo "Entorno virtual no encontrado. Creando .venv..."
    python3 -m venv .venv || python -m venv .venv
    echo "Entorno virtual creado"
fi

# Activar entorno virtual
echo " Activando entorno virtual..."
source .venv/bin/activate

# Verificar/Instalar dependencias
echo "Verificando dependencias..."
python -c "import tensorflow, faiss, cv2, skimage" 2>/dev/null || {
    echo "Instalando dependencias desde requirements.txt..."
    pip install -r requirements.txt
    echo "Dependencias instaladas"
}

echo "Dependencias verificadas"
echo ""

# Paso 1: Extraer características
echo "======================================================================"
echo "EXTRAYENDO CARACTERÍSTICAS"
echo "======================================================================"
echo ""

if [ -d "features" ] && [ "$(ls -A features)" ]; then
    echo "El directorio features/ ya contiene archivos"
    read -p "¿Deseas regenerar las características? (s/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[SsYy]$ ]]; then
        echo "⏭️  Saltando extracción de características"
    else
        echo "🗑️  Eliminando características anteriores..."
        rm -rf features/*
        echo "Extrayendo características de todas las imágenes..."
        python extract_all_features.py
    fi
else
    echo "Extrayendo características de todas las imágenes..."
    python extract_all_features.py
fi

echo ""
echo "Extracción de características completada"
echo ""

# Paso 2: Construir índices FAISS
echo "======================================================================"
echo "CONSTRUYENDO ÍNDICES FAISS"
echo "======================================================================"
echo ""

if [ -d "faiss_indices" ] && [ "$(ls -A faiss_indices)" ]; then
    echo "El directorio faiss_indices/ ya contiene archivos"
    read -p "¿Deseas regenerar los índices? (s/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[SsYy]$ ]]; then
        echo "⏭Saltando construcción de índices"
    else
        echo "Eliminando índices anteriores..."
        rm -rf faiss_indices/*
        echo "Construyendo índices FAISS..."
        python build_faiss_indices.py
    fi
else
    echo "Construyendo índices FAISS..."
    python build_faiss_indices.py
fi

echo ""
echo "Construcción de índices completada"
echo ""

# Resumen final
echo "======================================================================"
echo "PROCESO COMPLETADO EXITOSAMENTE"
echo "======================================================================"
echo ""
echo "Archivos generados:"
echo "features/        - Características extraídas (.npy, .json)"
echo "faiss_indices/   - Índices FAISS (flat, ivf, ivfpq, hnsw)"
echo ""
echo "Para iniciar el servidor, ejecuta:"
echo "  ./cicd/run_server.sh"
echo ""
