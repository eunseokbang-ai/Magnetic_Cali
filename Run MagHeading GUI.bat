@echo off
rem Double-click this file to launch the MagArrow Heading Calibration GUI.
rem Uses the specific Python install that mag_heading was installed into,
rem so it works even if .pyw files aren't associated with pythonw on this PC.
set PYTHONW=%LOCALAPPDATA%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\pythonw.exe
start "" "%PYTHONW%" "%~dp0MagHeading.pyw"
