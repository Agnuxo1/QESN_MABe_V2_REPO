#!/usr/bin/env python3
# ORCID: https://orcid.org/0009-0001-1634-7063
# Attribution recorded: 2026-11-06
"""
QESN-MABe V2: Instalador Automático de Dependencias
Author: Francisco Angulo de Lafuente
License: MIT

Este script instala automáticamente todas las dependencias necesarias para QESN.
"""

import subprocess
import sys
import importlib
import os

def print_header():
    """Imprimir encabezado del instalador"""
    print("=" * 70)
    print("🚀 QESN-MABe V2: Instalador Automático de Dependencias")
    print("=" * 70)
    print("Autor: Francisco Angulo de Lafuente")
    print("GitHub: https://github.com/Agnuxo1")
    print("=" * 70)
    print()

def check_python_version():
    """Verificar versión de Python"""
    print("🐍 Verificando versión de Python...")
    version = sys.version_info
    print(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Se requiere Python 3.8 o superior")
        print("   Por favor actualiza Python desde: https://www.python.org/downloads/")
        return False
    
    print("✅ Versión de Python compatible")
    return True

def install_package(package, upgrade=False):
    """Instalar un paquete usando pip"""
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {package} instalado correctamente")
            return True
        else:
            print(f"❌ Error instalando {package}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error instalando {package}: {e}")
        return False

def check_and_install_packages():
    """Verificar e instalar paquetes necesarios"""
    
    # Paquetes principales
    core_packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "ipywidgets>=7.6.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0"
    ]
    
    # Paquetes opcionales
    optional_packages = [
        "jupyter>=1.0.0",
        "notebook>=6.4.0",
        "jupyterlab>=3.0.0",
        "pyarrow>=8.0.0",
        "h5py>=3.1.0"
    ]
    
    print("🔍 Verificando dependencias principales...")
    missing_core = []
    
    for package in core_packages:
        package_name = package.split(">=")[0]
        try:
            importlib.import_module(package_name)
            print(f"✅ {package_name} ya está instalado")
        except ImportError:
            print(f"⚠️  {package_name} no encontrado")
            missing_core.append(package)
    
    if missing_core:
        print(f"\n📦 Instalando {len(missing_core)} paquetes principales...")
        for package in missing_core:
            install_package(package)
    
    print("\n🔍 Verificando paquetes opcionales...")
    missing_optional = []
    
    for package in optional_packages:
        package_name = package.split(">=")[0]
        try:
            importlib.import_module(package_name)
            print(f"✅ {package_name} ya está instalado")
        except ImportError:
            print(f"⚠️  {package_name} no encontrado")
            missing_optional.append(package)
    
    if missing_optional:
        print(f"\n📦 Instalando {len(missing_optional)} paquetes opcionales...")
        for package in missing_optional:
            install_package(package)
    
    return len(missing_core) == 0 and len(missing_optional) == 0

def test_imports():
    """Probar importaciones de todas las librerías"""
    print("\n🧪 Probando importaciones...")
    
    test_modules = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns"),
        ("plotly.graph_objects", "go"),
        ("plotly.express", "px"),
        ("plotly.subplots", "make_subplots"),
        ("ipywidgets", "widgets"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm")
    ]
    
    failed_imports = []
    
    for module_name, alias in test_modules:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name} importado correctamente")
            
            # Mostrar versión si está disponible
            if hasattr(module, '__version__'):
                print(f"   Versión: {module.__version__}")
                
        except ImportError as e:
            print(f"❌ Error importando {module_name}: {e}")
            failed_imports.append(module_name)
    
    return len(failed_imports) == 0

def create_requirements_file():
    """Crear archivo requirements.txt actualizado"""
    print("\n📝 Creando requirements.txt...")
    
    requirements_content = """# QESN-MABe V2 Python Dependencies
# Core scientific computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Data handling
pyarrow>=8.0.0
h5py>=3.1.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
ipywidgets>=7.6.0

# Machine learning utilities
scikit-learn>=1.0.0
tqdm>=4.62.0

# Jupyter notebook support
jupyter>=1.0.0
notebook>=6.4.0
ipykernel>=6.0.0

# Development dependencies
pytest>=6.2.0
pytest-cov>=2.12.0
black>=21.0.0
flake8>=3.9.0
mypy>=0.910
"""
    
    try:
        with open("requirements.txt", "w") as f:
            f.write(requirements_content)
        print("✅ requirements.txt creado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error creando requirements.txt: {e}")
        return False

def show_next_steps():
    """Mostrar próximos pasos"""
    print("\n" + "=" * 70)
    print("🎉 ¡INSTALACIÓN COMPLETADA!")
    print("=" * 70)
    print()
    print("📚 Próximos pasos:")
    print("   1. Ejecutar demo rápido:")
    print("      python examples/quick_demo.py")
    print()
    print("   2. Abrir notebook interactivo:")
    print("      jupyter notebook notebooks/QESN_Demo_Interactive.ipynb")
    print("      o")
    print("      jupyter lab notebooks/QESN_Demo_Interactive.ipynb")
    print()
    print("   3. Ejecutar demo simplificado:")
    print("      python demo_simple_no_emoji.py")
    print()
    print("📖 Para más información:")
    print("   - GitHub: https://github.com/Agnuxo1/QESN-MABe-V2")
    print("   - Documentación: docs/")
    print("   - README: README.md")
    print()
    print("🚀 ¡Disfruta explorando QESN!")

def main():
    """Función principal del instalador"""
    print_header()
    
    # Verificar Python
    if not check_python_version():
        return False
    
    # Instalar paquetes
    print("\n📦 Instalando dependencias...")
    if not check_and_install_packages():
        print("\n⚠️  Algunos paquetes no se pudieron instalar")
        print("   Intenta ejecutar manualmente:")
        print("   pip install numpy pandas matplotlib seaborn plotly ipywidgets")
        return False
    
    # Probar importaciones
    if not test_imports():
        print("\n⚠️  Algunas importaciones fallaron")
        print("   Reinicia tu terminal/IDE y vuelve a intentar")
        return False
    
    # Crear requirements.txt
    create_requirements_file()
    
    # Mostrar próximos pasos
    show_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ La instalación no se completó correctamente")
            print("   Por favor revisa los errores anteriores")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Instalación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)
