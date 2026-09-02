@echo off
rem Regenerate gpu_peer.ptx from gpu_peer.cu. Requires the CUDA toolkit
rem (nvcc) and MSVC build tools; crate CONSUMERS never run this - the
rem checked-in PTX is driver-JIT'd at runtime.
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
nvcc -ptx -arch=compute_75 gpu_peer.cu -o gpu_peer.ptx
exit /b %errorlevel%
