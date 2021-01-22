#!/bin/bash

module purge
module load cuda-10.1.243-gcc-8.3.1-l23v7go
module load htop-2.2.0-gcc-8.3.1-du3m2qy
module load python-3.7.6-gcc-8.3.1-7zpobbl 
module load cmake-3.16.2-gcc-8.3.1-bvrtksd 
module load curl-7.68.0-gcc-8.3.1-pawp6gm
module load openblas-0.3.9-gcc-8.3.1-hoxypjr
module load hdf5-1.10.6-gcc-8.3.1-pj7hduf
module load openmpi-3.1.6-gcc-8.3.1-fytjofa
export IGUAZU_HDF5_ROOT="/data1/projects/spack/spack/opt/spack/linux-centos8-broadwell/gcc-8.3.1/hdf5-1.10.6-pj7hdufzcsom7rtfxsptoiaiqzz37csu"
export IGUAZU_BLAS_ROOT="/data1/projects/spack/spack/opt/spack/linux-centos8-broadwell/gcc-8.3.1/openblas-0.3.10-3fr25oz7i7ahayvwzizuzd4dsrcubkcq"
export OMPI_MPICC=clang
export OMPI_MPICXX=clang++
export OMPI_MPIF77=gfortran
export OMPI_MPIF90=gfortran
export CC=clang
export CXX=clang++
export F77=gfortran
export FC=gfortran
