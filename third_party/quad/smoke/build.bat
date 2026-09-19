@echo off
setlocal enableextensions
rem === Deterministic MSVC + Windows SDK dev environment (no cmake/ninja) ===
set "REPO=C:\Github\ANYmesh\.worktrees\quad-first-v1"
set "VS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717"
set "SDK=C:\Program Files (x86)\Windows Kits\10"
set "VENDOR=%REPO%\third_party\quad\vendor"
set "EIGEN=%VENDOR%\eigen"
set "OUTDIR=%REPO%\third_party\quad\smoke\out\tinyad"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

rem Windows SDK version dir (full, includes the trailing .0).
set INCVER=10.0.26100.0
set LIBVER=10.0.26100.0
set "PATH=%VS%\bin\Hostx64\x64;%PATH%"
set "INCLUDE=%VS%\include;%SDK%\include\%INCVER%\ucrt;%SDK%\include\%INCVER%\um;%SDK%\include\%INCVER%\shared"
set "LIB=%VS%\lib\x64;%SDK%\lib\%LIBVER%\ucrt\x64;%SDK%\lib\%LIBVER%\um\x64"

echo === toolchain ===
echo MSVC     : %VS%
echo SDK inc  : %INCVER%  lib: %LIBVER%
cl 2>nul | findstr /i /c:"version"

echo === compile tinyad_smoke ===
rem -DEIGEN_MPL2_ONLY is REQUIRED by policy (see ATTRIBUTION.md): it enforces that
rem only MPL2-compatible Eigen headers are reachable, keeping the chain MPL-2.0
rem compatible with ANYmesher.
cl /nologo /std:c++17 /O2 /EHsc /MD /DEIGEN_MPL2_ONLY ^
   /I"%VENDOR%\tinyad\include" ^
   /I"%EIGEN%" ^
   "%REPO%\third_party\quad\smoke\tinyad_smoke.cc" ^
   /Fe:"%OUTDIR%\tinyad_smoke.exe"
if errorlevel 1 ( echo COMPILE_FAILED & exit /b 1 )

echo === run tinyad_smoke ===
"%OUTDIR%\tinyad_smoke.exe"
endlocal
