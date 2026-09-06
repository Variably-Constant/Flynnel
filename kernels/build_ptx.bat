@echo off
rem Regenerate the checked-in PTX (gpu_peer, linalg_f64, ozaki_f64,
rem linalg_bisect_f64, linalg_lu_f64) from their .cu sources. Requires
rem the CUDA toolkit (nvcc) and MSVC build tools; crate CONSUMERS never
rem run this, because src/gpu_peer embeds each .ptx with include_str!
rem and the driver JITs that embedded text to SASS at load.
rem
rem So a .ptx edit reaches a consumer only through a REBUILD of the
rem crate they resolved. Replacing the file on a host changes nothing
rem for a binary already built, and a consumer pinned to a published
rem version does not see an unreleased regeneration at all.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
nvcc -ptx -arch=compute_75 gpu_peer.cu -o gpu_peer.ptx || exit /b %errorlevel%
nvcc -ptx -arch=compute_75 -O3 -Werror all-warnings linalg_f64.cu -o linalg_f64.ptx || exit /b %errorlevel%
nvcc -ptx -arch=compute_75 -O3 -Werror all-warnings ozaki_f64.cu -o ozaki_f64.ptx || exit /b %errorlevel%
nvcc -ptx -arch=compute_75 -O3 -Werror all-warnings linalg_bisect_f64.cu -o linalg_bisect_f64.ptx || exit /b %errorlevel%
nvcc -ptx -arch=compute_75 -O3 -Werror all-warnings linalg_lu_f64.cu -o linalg_lu_f64.ptx
exit /b %errorlevel%
