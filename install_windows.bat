@echo off
setlocal ENABLEDELAYEDEXPANSION
REM Check if 'venv' directory exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if !ERRORLEVEL! neq 0 (
        echo Failed to create virtual environment.
        exit /b 1
    )
)

REM Activate the virtual environment if not already activated
if "%VIRTUAL_ENV%"=="" (
    echo Activating virtual environment...
    venv\Scripts\activate.bat
    if !ERRORLEVEL! neq 0 (
        echo Failed to activate virtual environment.
        exit /b 1
    )
)

REM Check if requirements.txt exists
if not exist requirements.txt (
    echo requirements.txt not found.
    exit /b 1
)

REM Upgrade pip to avoid warnings
echo Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to upgrade pip.
    exit /b 1
)

REM Check installed packages and install missing ones
echo Checking installed packages...
pip freeze > installed_packages.txt
if %errorlevel% neq 0 (
    echo Failed to list installed packages.
    exit /b 1
)

REM Compare installed packages with requirements.txt
python check_requirements.py
if %errorlevel% neq 0 (
    echo Installing missing packages...
    pip install -r requirements.txt
    
    if !ERRORLEVEL! neq 0 (
        echo Failed to install required packages.
        exit /b 1
    )
    echo Packages installed successfully.
) else (
    echo All required packages are already installed.
)

REM Clean up
del installed_packages.txt

echo Stop tracking changes to config.json so that it can be updated with local overrides
git update-index --assume-unchanged config.json
echo Setup complete.
endlocal
exit /b 0