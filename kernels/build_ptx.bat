@echo off
rem Regenerate the checked-in PTX (gpu_peer.ptx, linalg_f64.ptx) from
rem their .cu sources. Requires the CUDA toolkit (nvcc) and MSVC build
rem tools; crate CONSUMERS never run this - the checked-in PTX is
rem driver-JIT'd at runtime.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
nvcc -ptx -arch=compute_75 gpu_peer.cu -o gpu_peer.ptx || exit /b %errorlevel%
nvcc -ptx -arch=compute_75 -O3 -Werror all-warnings linalg_f64.cu -o linalg_f64.ptx || exit /b %errorlevel%
nvcc -ptx -arch=compute_75 -O3 -Werror all-warnings ozaki_f64.cu -o ozaki_f64.ptx || exit /b %errorlevel%
nvcc -ptx -arch=compute_75 -O3 -Werror all-warnings linalg_bisect_f64.cu -o linalg_bisect_f64.ptx || exit /b %errorlevel%
nvcc -ptx -arch=compute_75 -O3 -Werror all-warnings linalg_lu_f64.cu -o linalg_lu_f64.ptx
exit /b %errorlevel%
