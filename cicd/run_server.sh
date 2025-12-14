#!/bin/bash
# Script para Mac/Linux: Iniciar servidor FastAPI
# Uso: ./cicd/run_server.sh

set -e  # Detener en caso de error

echo "======================================================================"
echo "SERVIDOR"
echo "======================================================================"
echo ""

# Verificar/Crear entorno virtual
if [ ! -d ".venv" ]; then
    echo "Entorno virtual no encontrado. Creando .venv..."
    python3 -m venv .venv || python -m venv .venv
    echo "Entorno virtual creado"
fi

# Activar entorno virtual
echo "Activando entorno virtual..."
source .venv/bin/activate

# Verificar/Instalar dependencias
echo "Verificando dependencias..."
python -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "Instalando dependencias desde requirements.txt..."
    pip install -r requirements.txt
    echo "Dependencias instaladas"
}

echo "Dependencias verificadas"
echo ""

# Verificar que existe el frontend
if [ ! -f "frontend/package.json" ]; then
    echo "Error: No se encuentra el directorio frontend/"
    exit 1
fi

# Instalar dependencias del frontend si es necesario
echo "Verificando dependencias del frontend..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "Instalando dependencias del frontend..."
    npm install || {
        echo "Error al instalar dependencias del frontend"
        cd ..
        exit 1
    }
fi
cd ..

# Configuración
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
FRONTEND_PORT="5173"

echo "======================================================================"
echo " INICIANDO BACKEND Y FRONTEND"
echo "======================================================================"
echo ""
echo "   Backend API:"
echo "     - Host: $HOST"
echo "     - Port: $PORT"
echo "     - URL: http://localhost:$PORT"
echo "     - Docs: http://localhost:$PORT/docs"
echo ""
echo "   Frontend:"
echo "     - Port: $FRONTEND_PORT"
echo "     - URL: http://localhost:$FRONTEND_PORT"
echo ""
echo "   Abriendo navegador en http://localhost:$FRONTEND_PORT"
echo ""
echo "  Presiona Ctrl+C para detener ambos servidores"
echo ""
echo "======================================================================"
echo ""

# Función para limpiar procesos al salir
cleanup() {
    echo ""
    echo " Deteniendo servidores..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Iniciar backend en background
echo " Iniciando servidor backend..."
uvicorn api:app --host "$HOST" --port "$PORT" --reload &
BACKEND_PID=$!

# Esperar a que el backend inicie
sleep 3

# Abrir navegador (detectar OS)
open_browser() {
    sleep 3
    if command -v open >/dev/null 2>&1; then
        # macOS
        open "http://localhost:$FRONTEND_PORT"
    elif command -v xdg-open >/dev/null 2>&1; then
        # Linux
        xdg-open "http://localhost:$FRONTEND_PORT"
    fi
}

open_browser &

# Iniciar frontend
echo " Iniciando servidor frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Esperar a que los procesos terminen
wait $BACKEND_PID $FRONTEND_PID

echo ""
echo "✓ Servidores detenidos"
