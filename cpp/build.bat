export VS2019INSTALLDIR="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"
"C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\bin\compilervars.bat" intel64 vs2019
icl.exe /Qmkl /O3 /QxHost /Qopenmp /DNDEBUG /Qparallel /EHsc mkl.cpp
