@echo off
rem 在纯英文路径下运行本脚本（MSVC 在中文路径下写 PDB 会报 LNK1201）
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build --config Release
