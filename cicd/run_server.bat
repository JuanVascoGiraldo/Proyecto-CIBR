@echo off
REM Script para Windows: Iniciar servidor FastAPI
REM Uso: cicd\run_server.bat

setlocal

echo ======================================================================
echo SERVIDOR
echo ======================================================================
echo.

REM Verificar/Crear entorno virtual
if not exist ".venv\" (
    echo  Entorno virtual no encontrado. Creando .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo Error al crear entorno virtual
        exit /b 1
    )
    echo Entorno virtual creado
)

REM Activar entorno virtual
echo Activando entorno virtual...
call .venv\Scripts\activate.bat

REM Verificar/Instalar dependencias
echo Verificando dependencias...
python -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
    echo Instalando dependencias desde requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error al instalar dependencias
        exit /b 1
    )
    echo Dependencias instaladas
)

REM Verificar que existe el frontend
if not exist "frontend\package.json" (
    echo Error: No se encuentra el directorio frontend/
    exit /b 1
)

REM Instalar dependencias del frontend si es necesario
echo  Verificando dependencias del frontend...
cd frontend
if not exist "node_modules\" (
    echo  Instalando dependencias del frontend...
    call npm install
    if errorlevel 1 (
        echo Error al instalar dependencias del frontend
        cd ..
        exit /b 1
    )
)
cd ..

REM Configuración
if not defined HOST set HOST=0.0.0.0
if not defined PORT set PORT=8000
set FRONTEND_PORT=5173

echo ======================================================================
echo  INICIANDO BACKEND Y FRONTEND
echo ======================================================================
echo.
echo    Backend API:
echo      - Host: %HOST%
echo      - Port: %PORT%
echo      - URL: http://localhost:%PORT%
echo      - Docs: http://localhost:%PORT%/docs
echo.
echo    Frontend:
echo      - Port: %FRONTEND_PORT%
echo      - URL: http://localhost:%FRONTEND_PORT%
echo.
echo    Abriendo navegador en http://localhost:%FRONTEND_PORT%
echo.
echo   Presiona Ctrl+C para detener ambos servidores
echo.
echo ======================================================================
echo.

REM Iniciar backend en una nueva ventana
echo  Iniciando servidor backend...
start "Backend API" cmd /k "cd /d "%CD%" && call .venv\Scripts\activate.bat && uvicorn api:app --host %HOST% --port %PORT% --reload"

REM Esperar un momento para que el backend inicie
echo Esperando a 10 segundos a que el backend inicie...
timeout /t 10 /nobreak >nul

REM Iniciar frontend y abrir navegador
echo  Iniciando servidor frontend...
cd frontend

REM Abrir navegador después de 3 segundos
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:%FRONTEND_PORT%"

REM Ejecutar frontend (esto bloqueará hasta Ctrl+C)
call npm run dev

REM Al detener el frontend, volver al directorio raíz
cd ..

echo.
echo Servidor detenido

endlocal
