@echo off
echo ======================================================================
echo QESN-MABe V2: Instalador Automatico de Dependencias
echo ======================================================================
echo Autor: Francisco Angulo de Lafuente
echo GitHub: https://github.com/Agnuxo1
echo ======================================================================
echo.

REM Verificar Python
echo Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no encontrado
    echo Por favor instala Python 3.8+ desde: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python encontrado correctamente
echo.

REM Ejecutar instalador Python
echo Ejecutando instalador Python...
python install_dependencies.py

if errorlevel 1 (
    echo.
    echo ERROR: La instalacion fallo
    echo Intentando instalacion manual...
    echo.
    
    echo Instalando paquetes principales...
    python -m pip install --upgrade pip
    python -m pip install numpy pandas matplotlib seaborn plotly ipywidgets scipy tqdm
    
    echo.
    echo Instalando paquetes opcionales...
    python -m pip install jupyter notebook jupyterlab pyarrow h5py
    
    echo.
    echo Probando instalacion...
    python -c "import numpy, pandas, matplotlib, seaborn, plotly; print('Instalacion exitosa!')"
    
    if errorlevel 1 (
        echo ERROR: La instalacion manual tambien fallo
        echo Por favor revisa los errores anteriores
        pause
        exit /b 1
    )
)

echo.
echo ======================================================================
echo INSTALACION COMPLETADA
echo ======================================================================
echo.
echo Proximos pasos:
echo   1. Demo rapido: python examples\quick_demo.py
echo   2. Notebook: jupyter notebook notebooks\QESN_Demo_Interactive.ipynb
echo   3. Demo simple: python demo_simple_no_emoji.py
echo.
echo Para mas informacion: https://github.com/Agnuxo1/QESN-MABe-V2
echo.
pause
