@echo off
setlocal
pushd "%~dp0"
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
)
cl.exe /nologo /O2 /EHsc /std:c++17 /LD fast_brain3d_kernels.cpp /Fe:fast_brain3d_kernels.dll
set EXIT_CODE=%ERRORLEVEL%
popd
exit /b %EXIT_CODE%
