@echo off
setlocal DisableDelayedExpansion
title NapariSBT
set "SBT_RUNTIME=%~dp0"

if not exist "%SBT_RUNTIME%python.exe" goto missing_runtime
if not exist "%SBT_RUNTIME%start_naparisbt.py" goto missing_runtime
if not exist "%SBT_RUNTIME%Scripts\activate.bat" goto missing_runtime

rem Isolate from an existing Python/Conda session. No system settings are changed.
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "QT_PLUGIN_PATH="
set "QML2_IMPORT_PATH="
set "CONDA_PREFIX="
set "CONDA_DEFAULT_ENV="
set "CONDA_SHLVL="
set "PATH=%SBT_RUNTIME%;%SBT_RUNTIME%Library\bin;%SBT_RUNTIME%Scripts;%PATH%"

rem Relocate before activation, so package activation hooks have current paths.
"%SBT_RUNTIME%python.exe" -E -s "%SBT_RUNTIME%start_naparisbt.py" --prepare-only
if errorlevel 1 goto failed
pushd "%SBT_RUNTIME%"
if errorlevel 1 goto failed
rem A relative CALL avoids expanding percent signs in the application path twice.
call Scripts\activate.bat
if errorlevel 1 goto failed_in_folder
"%SBT_RUNTIME%python.exe" -E -s "%SBT_RUNTIME%start_naparisbt.py"
set "SBT_EXIT_CODE=%ERRORLEVEL%"
popd
if not "%SBT_EXIT_CODE%"=="0" goto failed
exit /b 0

:failed_in_folder
popd
goto failed

:missing_runtime
echo Put this batch file and start_naparisbt.py beside python.exe in the extracted ZIP.
echo The Lib, Library and Scripts folders must also be present.
goto failed

:failed
echo.
echo NapariSBT could not start. Copy the details above when requesting help.
echo See README.html for command-prompt instructions and Windows security guidance.
pause
exit /b 1
