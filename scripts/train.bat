@echo off
REM QESN-MABe V2: Quantum Energy State Network for Mouse Behavior Classification
REM Author: Francisco Angulo de Lafuente
REM License: MIT
REM GitHub: https://github.com/Agnuxo1

setlocal

REM ========================================
REM Configuration
REM ========================================
REM Adjust these paths to match your MABe dataset location
set "METADATA=E:\QESN-MABe\train.csv"
set "TRACKING=E:\QESN-MABe\train_tracking"
set "ANNOTATION=E:\QESN-MABe\train_annotation"
set "CHECKPOINTS=checkpoints"
set "EXPORT=kaggle"

echo ========================================
echo QESN-MABe V2 Training
echo ========================================
echo.
echo Configuration:
echo   - Behavior classes:  37
echo   - Window size:       30 frames (FIXED from V1)
echo   - Grid resolution:   64x64
echo   - Training epochs:   30
echo   - Batch size:        32
echo   - Learning rate:     0.001
echo.
echo Dataset:
echo   - Metadata:     %METADATA%
echo   - Tracking:     %TRACKING%
echo   - Annotations:  %ANNOTATION%
echo.
echo Output:
echo   - Checkpoints:  %CHECKPOINTS%
echo   - Best model:   %CHECKPOINTS%\best_model.bin
echo   - Export:       %EXPORT%
echo.
echo ========================================
echo.

REM Check if executable exists
if not exist "build\Release\qesn_train.exe" (
    echo [ERROR] Executable not found!
    echo Please run scripts\build.bat first.
    echo.
    pause
    exit /b 1
)

REM Check if metadata exists
if not exist "%METADATA%" (
    echo [WARNING] Metadata file not found: %METADATA%
    echo Please update the METADATA path in this script.
    echo.
    pause
    exit /b 1
)

echo Starting training...
echo This will take approximately 12-15 hours.
echo.
echo Press Ctrl+C to cancel, or any key to continue...
pause > nul

echo.
echo ========================================
echo Training in progress...
echo ========================================
echo.

REM Run training
build\Release\qesn_train.exe ^
    --metadata "%METADATA%" ^
    --tracking "%TRACKING%" ^
    --annotation "%ANNOTATION%" ^
    --epochs 30 ^
    --window 30 ^
    --stride 15 ^
    --batch 32 ^
    --lr 0.001 ^
    --checkpoints "%CHECKPOINTS%" ^
    --best "%CHECKPOINTS%\best_model.bin" ^
    --export "%EXPORT%"

if errorlevel 1 (
    echo.
    echo ========================================
    echo [ERROR] Training failed!
    echo ========================================
    echo.
    echo Common issues:
    echo   1. Out of memory: Reduce --batch size
    echo   2. Dataset not found: Check paths above
    echo   3. NaN in loss: Check data loading
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training completed successfully!
echo ========================================
echo.
echo Outputs:
echo   - Training history:  %CHECKPOINTS%\training_history.csv
echo   - Best model:        %CHECKPOINTS%\best_model.bin
echo   - Inference weights: %EXPORT%\model_weights.bin
echo   - Inference config:  %EXPORT%\model_config.json
echo.
echo Expected metrics:
echo   - Training accuracy:   55-65%%
echo   - Validation accuracy: 50-60%%
echo   - F1-Score (macro):    0.40-0.50
echo.
echo Next steps:
echo   1. Review training_history.csv
echo   2. Test inference with python\qesn_inference.py
echo   3. Upload to Kaggle for submission
echo.
pause
