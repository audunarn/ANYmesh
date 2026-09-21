@echo off
setlocal enableextensions
rem === Deterministic MSVC + Windows SDK dev environment (no cmake/ninja).
rem Compiles the Q5 TinyAD quad-patch local-optimization worker against the
rem pinned TinyAD + Eigen headers.
for %%I in ("%~dp0..\..\..") do set "REPO=%%~fI"
set "VS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717"
set "SDK=C:\Program Files (x86)\Windows Kits\10"
set "VENDOR=%REPO%\third_party\quad\vendor"
set "EIGEN=%VENDOR%\eigen"
set "OUTDIR=%REPO%\third_party\quad\worker\out\tinyad"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
set "INCVER=10.0.26100.0"
set "LIBVER=10.0.26100.0"
set "PATH=%VS%\bin\Hostx64\x64;%PATH%"
set "INCLUDE=%VS%\include;%SDK%\include\%INCVER%\ucrt;%SDK%\include\%INCVER%\um;%SDK%\include\%INCVER%\shared"
set "LIB=%VS%\lib\x64;%SDK%\lib\%LIBVER%\ucrt\x64;%SDK%\lib\%LIBVER%\um\x64"

echo === compile quad_tinyad_optimizer ===
rem -DEIGEN_MPL2_ONLY is REQUIRED by policy (see ATTRIBUTION.md): it enforces
rem that only MPL2-compatible Eigen headers are reachable, keeping the chain
rem MPL-2.0 compatible with ANYmesher.
cl /nologo /std:c++17 /O2 /EHsc /MD /W3 /DEIGEN_MPL2_ONLY ^
   /I"%VENDOR%\tinyad\include" ^
   /I"%EIGEN%" ^
   "%REPO%\third_party\quad\worker\quad_tinyad_optimizer.cc" ^
   /Fo:"%OUTDIR%\quad_tinyad_optimizer.obj" ^
   /Fe:"%OUTDIR%\quad_tinyad_optimizer.exe"
if errorlevel 1 ( echo COMPILE_FAILED & exit /b 1 )

echo === build OK; exe is at "%OUTDIR%\quad_tinyad_optimizer.exe" ===
endlocal
