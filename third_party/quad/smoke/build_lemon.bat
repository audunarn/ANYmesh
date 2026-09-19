@echo off
setlocal enableextensions
rem === Deterministic MSVC + Windows SDK dev environment (no cmake/ninja) ===
set "REPO=C:\Github\ANYmesh\.worktrees\quad-first-v1"
set "VS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717"
set "SDK=C:\Program Files (x86)\Windows Kits\10"
set "VENDOR=%REPO%\third_party\quad\vendor"
set "OUTDIR=%REPO%\third_party\quad\smoke\out\lemon"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
set "INCVER=10.0.26100.0"
set "LIBVER=10.0.26100.0"
set "PATH=%VS%\bin\Hostx64\x64;%PATH%"
set "INCLUDE=%VS%\include;%SDK%\include\%INCVER%\ucrt;%SDK%\include\%INCVER%\um;%SDK%\include\%INCVER%\shared"
set "LIB=%VS%\lib\x64;%SDK%\lib\%LIBVER%\ucrt\x64;%SDK%\lib\%LIBVER%\um\x64"
echo === compile lemon_mcf_smoke ===
cl /nologo /std:c++17 /O2 /EHsc /MD ^
   /I"%VENDOR%" ^
   "%REPO%\third_party\quad\smoke\lemon_mcf_smoke.cc" ^
   /Fe:"%OUTDIR%\lemon_mcf_smoke.exe"
if errorlevel 1 ( echo COMPILE_FAILED & exit /b 1 )
echo === run lemon_mcf_smoke ===
"%OUTDIR%\lemon_mcf_smoke.exe"
endlocal
