@echo off
REM Script para Windows: Extracción de características y construcción de índices FAISS
REM Uso: cicd\extract_and_build.bat

setlocal enabledelayedexpansion

echo ======================================================================
echo EXTRACCIÓN DE CARACTERÍSTICAS Y CONSTRUCCIÓN DE ÍNDICES FAISS
echo ======================================================================
echo.


REM Verificar/Crear entorno virtual
if not exist ".venv\" (
    echo Entorno virtual no encontrado. Creando .venv...
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
python -c "import tensorflow, faiss, cv2, skimage" 2>nul
if errorlevel 1 (
    echo Instalando dependencias desde requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error al instalar dependencias
        exit /b 1
    )
    echo  Dependencias instaladas
)

echo  Dependencias verificadas
echo.

REM Paso 1: Extraer características
echo ======================================================================
echo EXTRAYENDO CARACTERÍSTICAS
echo ======================================================================
echo.

if exist "features\" (
    dir /b "features\" | findstr "^" >nul 2>&1
    if not errorlevel 1 (
        echo El directorio features\ ya contiene archivos
        set /p "REPLY=¿Deseas regenerar las características? (s/N): "
        if /i "!REPLY!"=="s" (
            echo Eliminando características anteriores...
            del /q features\* 2>nul
            echo Extrayendo características de todas las imágenes...
            python extract_all_features.py
        ) else (
            echo Saltando extracción de características
        )
    ) else (
        echo Extrayendo características de todas las imágenes...
        python extract_all_features.py
    )
) else (
    echo Extrayendo características de todas las imágenes...
    python extract_all_features.py
)

echo.
echo Extracción de características completada
echo.

REM Paso 2: Construir índices FAISS
echo ======================================================================
echo CONSTRUYENDO ÍNDICES FAISS
echo ======================================================================
echo.

if exist "faiss_indices\" (
    dir /b /s "faiss_indices\*.index" >nul 2>&1
    if not errorlevel 1 (
        echo El directorio faiss_indices\ ya contiene archivos
        set /p "REPLY=¿Deseas regenerar los índices? (s/N): "
        if /i "!REPLY!"=="s" (
            echo Eliminando índices anteriores...
            for /d %%d in (faiss_indices\*) do (
                del /q "%%d\*" 2>nul
            )
            echo Construyendo índices FAISS...
            python build_faiss_indices.py
        ) else (
            echo Saltando construcción de índices
        )
    ) else (
        echo Construyendo índices FAISS...
        python build_faiss_indices.py
    )
) else (
    echo Construyendo índices FAISS...
    python build_faiss_indices.py
)

echo.
echo Construcción de índices completada
echo.

REM Resumen final
echo ======================================================================
echo PROCESO COMPLETADO EXITOSAMENTE
echo ======================================================================
echo.
echo Archivos generados:
echo   features\        - Características extraídas (.npy, .json)
echo   faiss_indices\   - Índices FAISS (flat, ivf, ivfpq, hnsw)
echo.
echo Para iniciar el servidor, ejecuta:
echo   cicd\run_server.bat
echo.

endlocal
