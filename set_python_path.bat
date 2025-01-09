@echo off

if exist "C:\Users\892nkeong\AppData\Local\Programs\Python\Python312" (SET python_path="C:\Users\892nkeong\AppData\Local\Programs\Python\Python312")

if not defined python_path (
    echo "Python path not found"
    echo "Please use "Python 37 version onwards"
    echo "Desperately trying to run it from your PATH environment ... this MAY work ... lets see. Press a key to try."
	SET python_path=""
    pause
    goto :eof
)

