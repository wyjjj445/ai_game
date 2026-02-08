@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

cd /d "%~dp0"

set "CONFIG=configs\dqn.yaml"
set "MODEL=outputs\checkpoints\latest.pt"

:menu
cls
echo ============================================
echo   AI Snake DQN Launcher
echo ============================================
echo [1] Setup (install deps)
echo [2] Train
echo [3] Evaluate latest model
echo [4] Demo (pygame)
echo [5] Run all (setup-train-eval-demo)
echo [0] Exit
echo ============================================
set /p choice=Select option:

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto train
if "%choice%"=="3" goto eval
if "%choice%"=="4" goto demo
if "%choice%"=="5" goto all
if "%choice%"=="0" goto end
goto menu

:check_python
python --version >nul 2>&1
if errorlevel 1 (
  echo [Error] Python not found. Install Python and add it to PATH.
  pause
  goto menu
)
goto :eof

:setup
call :check_python
echo [Step] Upgrade pip...
python -m pip install --upgrade pip
if errorlevel 1 goto failed
echo [Step] Install dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo [Done] Dependencies installed.
pause
goto menu

:train
call :check_python
echo [Step] Start training...
python train\train_dqn.py --config %CONFIG%
if errorlevel 1 goto failed
echo [Done] Training finished.
pause
goto menu

:eval
call :check_python
if not exist "%MODEL%" (
  echo [Error] Model not found: %MODEL%
  echo Please run training first (option 2 or 5).
  pause
  goto menu
)
echo [Step] Start evaluation...
python train\eval.py --config %CONFIG% --model %MODEL% --episodes 100
if errorlevel 1 goto failed
echo [Done] Evaluation finished.
pause
goto menu

:demo
call :check_python
if not exist "%MODEL%" (
  echo [Error] Model not found: %MODEL%
  echo Please run training first (option 2 or 5).
  pause
  goto menu
)
echo [Step] Start demo window...
python ui\play_demo.py --config %CONFIG% --model %MODEL% --fps 12
if errorlevel 1 goto failed
echo [Done] Demo finished.
pause
goto menu

:all
call :check_python
echo [Step] Start full pipeline...
python -m pip install --upgrade pip
if errorlevel 1 goto failed
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
python train\train_dqn.py --config %CONFIG%
if errorlevel 1 goto failed
python train\eval.py --config %CONFIG% --model %MODEL% --episodes 100
if errorlevel 1 goto failed
python ui\play_demo.py --config %CONFIG% --model %MODEL% --fps 12
if errorlevel 1 goto failed
echo [Done] Full pipeline finished.
pause
goto menu

:failed
echo [Error] Command failed. Check logs above.
pause
goto menu

:end
echo Exit.
endlocal
