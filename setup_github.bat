@echo off
REM Script para configurar GitHub automáticamente en Windows
REM GitHub Setup Script for Windows

echo ========================================
echo    Configurador GitHub para QESN-MABe
echo ========================================
echo.

REM Verificar si Git está instalado
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git no está instalado o no está en el PATH
    echo Por favor instala Git desde https://git-scm.com/
    pause
    exit /b 1
)

echo Git detectado correctamente
echo.

REM Verificar si estamos en un repositorio Git
if not exist ".git" (
    echo ERROR: No se encontró un repositorio Git
    echo Ejecuta 'git init' primero
    pause
    exit /b 1
)

echo Repositorio Git detectado
echo.

REM Ejecutar script de configuración
echo Ejecutando configuración de GitHub...
python setup_github.py

if errorlevel 1 (
    echo.
    echo ERROR: Falló la configuración de GitHub
    echo.
    echo Pasos manuales:
    echo 1. Ve a https://github.com/new
    echo 2. Crea un repositorio nuevo
    echo 3. Ejecuta: git remote add origin https://github.com/USERNAME/REPO.git
    echo 4. Ejecuta: git push -u origin master
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo    Configuración completada exitosamente
echo ========================================
echo.
echo El proyecto está ahora en GitHub.
echo Revisa github_config.txt para más información.
echo.
pause
