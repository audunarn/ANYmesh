@echo off
rem Compatibility wrapper. Run from a Visual Studio developer shell so cl.exe
rem is on PATH; tools\build_quad_workers.py owns the flags and output names.
python "%~dp0..\..\..\tools\build_quad_workers.py" mcf
exit /b %errorlevel%
