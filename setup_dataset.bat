@echo off
REM Script para configurar el dataset MABe en Windows
REM Dataset Setup Script for Windows

echo ========================================
echo    Configurador de Dataset QESN-MABe
echo ========================================
echo.

REM Verificar si Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no está instalado o no está en el PATH
    echo Por favor instala Python desde https://python.org
    pause
    exit /b 1
)

echo Python detectado correctamente
echo.

REM Verificar si 7-Zip está disponible
7z >nul 2>&1
if errorlevel 1 (
    echo ADVERTENCIA: 7-Zip no detectado
    echo Por favor instala 7-Zip desde https://www.7-zip.org/
    echo O WinRAR desde https://www.win-rar.com/
    echo.
    echo Continuando con configuración manual...
    echo.
) else (
    echo 7-Zip detectado correctamente
    echo.
)

REM Crear directorio de datos si no existe
if not exist "data" mkdir data
if not exist "data\downloads" mkdir data\downloads

echo Directorios creados:
echo   data\
echo   data\downloads\
echo.

REM Ejecutar script de configuración
echo Ejecutando configuración del dataset...
python setup_dataset.py

if errorlevel 1 (
    echo.
    echo ERROR: Falló la configuración automática
    echo.
    echo Opciones alternativas:
    echo 1. Descarga manual desde GitHub
    echo 2. Usa datos sintéticos: python create_demo_data.py
    echo 3. Verifica que tienes un extractor RAR instalado
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo    Configuración completada exitosamente
echo ========================================
echo.
echo El dataset está listo para usar.
echo Ejecuta: python notebooks\qesn_complete_classification_demo.py
echo.
pause
