module load cmake/3.26.3
module load gcc/10.3.1-magic

export enable_mpi="${enable_mpi:=ON}"
export enable_openmp="${enable_openmp:=ON}"
export enable_python="${enable_python:=ON}"
export build_caliper="${build_caliper:=true}"
export build_pyvenv="${build_pyvenv:=true}"
export build_jobs="${build_jobs:=20}"
./build_ascent.sh


