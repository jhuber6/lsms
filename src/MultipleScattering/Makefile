OBJ = calculateTauMatrix.o \
      buildKKRMatrix_CPU.o linearSolvers_CPU.o \
      makegij_c.o setgij.o block_inverse_fortran.o zblock_lu.o wasinv.o zmar1.o wasinv_p.o \
      zuqmx.o zutfx.o zucpx.o zaxpby.o zrandn.o tau_inv_postproc.o trgtol.o green_function.o gf_local.o \
      int_zz_zj.o mdosms_c.o mgreen_c.o green_function_rel.o write_kkrmat.o relmtrx.o gfill.o gafill.o \
      magnet.o magnetic_dens.o new_dens.o block_inverse.o zblock_lu_cpp.o zblock_lu_cublas.o

ifdef CUDA_CXX
OBJ += linearSolvers_CUDA.o buildKKRMatrix_CUDA.o
endif

ifdef HIP_CXX
OBJ += linearSolvers_HIP.o buildKKRMatrix_HIP.o
endif

all: libMultipleScattering.a

clean:
	rm -f *.a *.o

%.o : %.cpp
	$(CXX) $(INC_PATH) -c -o $@ $<

%.o : %.f
	$(F77) -c -o $@ $<

%.o : %.F
	$(F77) -c -o $@ $<

%.o : %.f90
	$(F77) -c -o $@ $<

ifdef CUDA_CXX
%.o : %.cu
	$(CUDA_CXX) $(INC_PATH) -c -o $@ $<
endif

ifdef HIP_CXX
linearSolvers_HIP.o : linearSolvers_HIP.cpp
	$(HIP_CXX) $(INC_PATH) -c -o $@ $<

buildKKRMatrix_HIP.o: buildKKRMatrix_HIP.cpp
	$(HIP_CXX) $(INC_PATH) -c -o $@ $<
endif

libMultipleScattering.a : $(OBJ)
	$(AR) -r libMultipleScattering.a $(OBJ) 
	cp libMultipleScattering.a $(TOP_DIR)/lib
