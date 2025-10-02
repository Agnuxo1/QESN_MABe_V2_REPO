@echo off
REM QESN-MABe V2: Build script adapted for existing installations
REM Author: Francisco Angulo de Lafuente
REM Using: VS2022 (E:\VS2022), Eigen (E:\eigen-3.4.0), CUDA 12

cd /d "%~dp0.."

echo ========================================
echo QESN-MABe V2 Build System
echo ========================================
echo.
echo Using existing installations:
echo   - Visual Studio 2022: E:\VS2022
echo   - Eigen 3.4.0: E:\eigen-3.4.0
echo   - CUDA 12: Detected automatically
echo.

REM Clean previous build
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)

echo Creating build directory...
mkdir build
cd build

echo.
echo Configuring CMake...
echo ----------------------------------------

REM Set Visual Studio path
set "VS_PATH=E:\VS2022\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

REM Check if VS CMake exists, otherwise use system cmake
if exist "%VS_PATH%" (
    echo Using Visual Studio CMake: %VS_PATH%
    "%VS_PATH%" .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
) else (
    echo Using system CMake
    cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
)

if errorlevel 1 (
    echo.
    echo [ERROR] CMake configuration failed!
    echo.
    echo Troubleshooting:
    echo   1. Verify Eigen is at E:\eigen-3.4.0
    echo   2. Ensure Visual Studio 2022 is properly installed
    echo   3. Check that CMake is in PATH
    echo.
    pause
    exit /b 1
)

echo.
echo Building project (Release)...
echo ----------------------------------------
cmake --build . --config Release -j 8

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    echo Check the compiler output above for errors.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build successful!
echo ========================================
echo.
echo Executable: build\Release\qesn_train.exe
echo.
echo Next steps:
echo   1. Preprocess data: python scripts\preprocess_parquet.py
echo   2. Train model: scripts\train.bat
echo   3. Or test: build\Release\qesn_train.exe --help
echo.
pause
