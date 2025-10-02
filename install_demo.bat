@echo off
echo 🚀 Instalando dependencias para QESN Demo...
echo.

REM Verificar Python
python --version
if errorlevel 1 (
    echo ❌ Python no encontrado. Por favor instala Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python encontrado
echo.

REM Instalar dependencias básicas
echo 📦 Instalando dependencias...
python -m pip install --user numpy pandas matplotlib seaborn plotly ipywidgets jupyter notebook

if errorlevel 1 (
    echo ❌ Error instalando dependencias
    pause
    exit /b 1
)

echo ✅ Dependencias instaladas correctamente
echo.

REM Verificar instalación
echo 🔍 Verificando instalación...
python -c "import numpy, pandas, matplotlib, seaborn, plotly; print('✅ Todas las librerías importadas correctamente')"

if errorlevel 1 (
    echo ❌ Error verificando instalación
    pause
    exit /b 1
)

echo.
echo 🎉 ¡Instalación completada!
echo.
echo 📚 Para ejecutar las demos:
echo    1. Demo rápido: python examples\quick_demo.py
echo    2. Notebook interactivo: jupyter notebook notebooks\QESN_Demo_Interactive.ipynb
echo    3. Jupyter Lab: jupyter lab notebooks\QESN_Demo_Interactive.ipynb
echo.
pause
