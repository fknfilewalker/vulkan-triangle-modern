@ECHO OFF
set PROJECT_DIR=_project
set INSTALL_DIR=_bin

if "%~1"=="" goto BLANK
if "%~1"=="install" goto install
if "%~1"=="clean" goto CLEAN
@ECHO ON

:BLANK
cmake -H. -B %PROJECT_DIR% -A "x64"
GOTO DONE

:INSTALL
cmake -H. -B %PROJECT_DIR% -A "x64"
cmake --build %PROJECT_DIR% --parallel 24 --config Debug --target install
cmake --build %PROJECT_DIR% --parallel 24 --config Release --target install
GOTO DONE

:CLEAN
rmdir /Q /S %PROJECT_DIR% 2>NUL
rmdir /Q /S %INSTALL_DIR% 2>NUL
GOTO DONE

:DONE
