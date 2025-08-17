@echo off
REM =================================================================
REM == RealdataEXP Windows Unified Execution Script              ==
REM == (Device is now configured in the .yaml file)              ==
REM == (v2 - Patched for conda prefix activation)                ==
REM =================================================================

REM --- 1. Setup Environment Variables ---
set "PROJECT_DIR=E:\MyDocument\Codes_notnut\_notpad\IEDA\RealdataEXP"
set "CONDA_ENV_PATH=e:\MyDocument\Codes_notnut\_notpad\IEDA\.conda"

REM --- 2. Change to Project Directory ---
echo Changing directory to %PROJECT_DIR%
cd /d "%PROJECT_DIR%"
if %errorlevel% neq 0 (
    echo ERROR: Could not find the project directory. Please check the path.
    pause
    goto :eof
)

REM --- 3. Activate Conda Environment ---
echo.
echo Activating Conda environment from: %CONDA_ENV_PATH%

REM --- MODIFIED LINE BELOW ---
REM Use --prefix to explicitly tell conda this is a path, not a name.
call conda activate --prefix "%CONDA_ENV_PATH%"

if %errorlevel% neq 0 (
    echo ERROR: Failed to activate Conda environment.
    echo Please verify the path is correct and conda is initialized.
    echo You can list all environments with: conda info --envs
    pause
    goto :eof
)
echo Conda environment activated successfully.

REM --- 4. Set Python Path ---
set "PYTHONPATH=%PROJECT_DIR%"
echo PYTHONPATH set to: %PYTHONPATH%

REM --- 5. Run Experiment ---
echo.
echo [INFO] Running experiment with configuration from 'configs/experiment_optimized.yaml'.
echo [INFO] Hardware device selection is specified inside the YAML file.
echo.

python main.py --config configs/experiment_optimized.yaml

REM --- 6. Deactivate Environment and Exit ---
echo.
echo Experiment finished.
echo Deactivating Conda environment.
call conda deactivate

echo.
echo Script complete. Press any key to exit.
pause