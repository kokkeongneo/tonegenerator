call set_python_path.bat

@echo off
for %%f in (*.csv) do (
    %python_path%\python tone_generator.py %%f
    )
pause 
