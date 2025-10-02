#!/usr/bin/env python3
"""
Script para configurar GitHub automáticamente y subir el proyecto completo.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Ejecuta un comando y maneja errores."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        print(f"   Salida: {e.stdout}")
        print(f"   Error: {e.stderr}")
        return False

def setup_github_repo():
    """Configura el repositorio de GitHub."""
    
    print("🚀 Configurador Automático de GitHub para QESN-MABe")
    print("=" * 60)
    
    # Verificar que estamos en un repositorio Git
    if not Path(".git").exists():
        print("❌ No se encontró un repositorio Git. Ejecuta 'git init' primero.")
        return False
    
    # Obtener información del usuario
    print("\n📝 Información del repositorio:")
    github_username = input("GitHub username: ").strip()
    repo_name = input("Nombre del repositorio (default: QESN_MABe_V2_REPO): ").strip()
    if not repo_name:
        repo_name = "QESN_MABe_V2_REPO"
    
    repo_url = f"https://github.com/{github_username}/{repo_name}.git"
    
    print(f"\n🔗 URL del repositorio: {repo_url}")
    confirm = input("¿Continuar? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Operación cancelada")
        return False
    
    # Configurar el repositorio remoto
    if not run_command(f"git remote add origin {repo_url}", "Configurando repositorio remoto"):
        # Si ya existe, actualizar la URL
        run_command(f"git remote set-url origin {repo_url}", "Actualizando URL del repositorio")
    
    # Verificar estado de Git
    run_command("git status", "Verificando estado del repositorio")
    
    # Hacer push inicial
    print(f"\n📤 Subiendo proyecto a GitHub...")
    print("⚠️  IMPORTANTE: Asegúrate de haber creado el repositorio en GitHub primero!")
    print("   1. Ve a https://github.com/new")
    print("   2. Crea un repositorio llamado:", repo_name)
    print("   3. NO inicialices con README, .gitignore o licencia")
    print("   4. Presiona Enter cuando esté listo...")
    input()
    
    # Push inicial
    if run_command("git push -u origin master", "Subiendo código inicial"):
        print(f"\n🎉 ¡Proyecto subido exitosamente!")
        print(f"   Repositorio: {repo_url}")
        print(f"   Puedes verlo en: https://github.com/{github_username}/{repo_name}")
        
        # Crear archivo de configuración local
        config_content = f"""# Configuración del repositorio GitHub
GITHUB_USERNAME = "{github_username}"
REPO_NAME = "{repo_name}"
REPO_URL = "{repo_url}"

# Comandos útiles:
# git add .
# git commit -m "Descripción del cambio"
# git push origin master

# Para clonar en otra máquina:
# git clone {repo_url}
"""
        
        with open("github_config.txt", "w", encoding="utf-8") as f:
            f.write(config_content)
        
        print(f"\n📄 Configuración guardada en: github_config.txt")
        return True
    else:
        print(f"\n❌ Error subiendo el proyecto")
        print("💡 Posibles soluciones:")
        print("   1. Verifica que el repositorio existe en GitHub")
        print("   2. Verifica tus credenciales de GitHub")
        print("   3. Ejecuta: git push -u origin master manualmente")
        return False

def show_next_steps():
    """Muestra los próximos pasos después de subir el proyecto."""
    print("\n🎯 Próximos Pasos:")
    print("=" * 30)
    print("1. 📊 Subir el dataset:")
    print("   - Comprime tu dataset en un archivo .rar")
    print("   - Ve a GitHub > Releases > Create a new release")
    print("   - Sube el archivo .rar como asset")
    print("   - Actualiza la URL en setup_dataset.py")
    
    print("\n2. 📚 Documentación:")
    print("   - Actualiza README.md con tu información")
    print("   - Agrega badges de estado")
    print("   - Crea issues y milestones")
    
    print("\n3. 🔧 Configuración:")
    print("   - Habilita GitHub Pages si quieres documentación web")
    print("   - Configura GitHub Actions para CI/CD")
    print("   - Agrega colaboradores si es necesario")
    
    print("\n4. 🚀 Despliegue:")
    print("   - Configura HuggingFace Spaces")
    print("   - Sube a Kaggle como dataset")
    print("   - Considera PyPI para distribución")

def main():
    """Función principal."""
    try:
        success = setup_github_repo()
        if success:
            show_next_steps()
        else:
            print("\n❌ Configuración fallida. Revisa los errores arriba.")
            return 1
    except KeyboardInterrupt:
        print("\n\n⏹️ Operación cancelada por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
