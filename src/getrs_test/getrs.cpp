#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <complex>

#include <flens/flens.cxx>

using ZGeMatrix = flens::GeMatrix<flens::FullStorage<std::complex<double>> >;
using ZGeVector = flens::DenseVector<flens::Array<std::complex<double>> >;
using IGeMatrix = flens::GeMatrix<flens::FullStorage<int> >;
using IGeVector = flens::DenseVector<flens::Array<int> >;

double get_seconds() {
	struct timespec now;
	clock_gettime(CLOCK_REALTIME, &now);

	const double seconds = (double) now.tv_sec;
	const double nsec    = (double) now.tv_nsec;

	return seconds + (nsec * 1.0e-9);
}

int main(int argc, char* argv[]) {
	// ------------------------------------------------------- //
	// DO NOT CHANGE CODE BELOW
	// ------------------------------------------------------- //

	int M = 256;
	int N = 256;
	int seed = 1;
	int reps = 1;

	if(argc > 1) {
		if (argc == 2) {
			M = atoi(argv[1]);
			N = atoi(argv[1]);
		} else {
			M = atoi(argv[1]);
			N = atoi(argv[2]);
		}
		printf("Matrix size is A=%dx%d, B=%d\n", M, N, N);
		printf("Input Size = %d\n", M*N);

		if(argc > 3) {
			if (argc > 3) {
				reps = atoi(argv[3]);
			}
			if (argc > 4) {
				seed = atoi(argv[4]);
			}
		}
	} else {
		printf("Matrix size defaulted to A=%dx%d, B=%d\n", M, N, N);
	}

	printf("Seed    =    %d\n", seed);
	printf("Repeats =    %d\n", reps);

	if(M < 2 || N < 2) {
		printf("Error: the matrix is too small (%d,%d).\n", M, N);
		exit(1);
	}

	printf("Allocating Matrices...\n");

    std::complex<double>* matrixA = (std::complex<double>*) malloc(sizeof(std::complex<double>) * M * N);
    std::complex<double>* vectorB = (std::complex<double>*) malloc(sizeof(std::complex<double>) * N * 1);
    int* pivots = (int *)malloc(sizeof(int) * N * 1);

	printf("Allocation complete, populating with values...\n");

	int i, j, k, r;
	srand48(seed);

	for(i = 0; i < M; i++)
		for(j = 0; j < N; j++)
			matrixA[i*N + j] = std::complex<double>(drand48()*2.0 - 1.0);

	for(i = 0; i < N; i++)
        vectorB[i] = std::complex<double>(drand48()*2.0 - 1.0);

	for(i = 0; i < N; i++)
        pivots[i] = 0;

	const double start = get_seconds();

	// ------------------------------------------------------- //
	// VENDOR NOTIFICATION: START MODIFIABLE REGION
	//
	// Vendor is able to change the lines below to call optimized
	// DGEMM or other matrix multiplication routines. Do *NOT*
	// change any lines above this statement.
	// ------------------------------------------------------- //

#pragma omp target data \
    map(tofrom:matrixA[0 : M*N]) \
    map(tofrom:vectorB[0 : N])
  for (r = 0; r < reps; r++) {
    ZGeMatrix::View f_A = ZGeMatrix::EngineView(M, N, matrixA, N, 1);
    ZGeVector::View f_B = ZGeVector::EngineView(N, vectorB);
    IGeVector::View f_P = IGeVector::EngineView(N, pivots);

    flens::lapack::trf(f_A, f_P);

    flens::lapack::trs(flens::NoTrans, f_A, f_P, f_B);
  }

 
	// ------------------------------------------------------- //
	// VENDOR NOTIFICATION: END MODIFIABLE REGION
	// ------------------------------------------------------- //

	// ------------------------------------------------------- //
	// DO NOT CHANGE CODE BELOW
	// ------------------------------------------------------- //

	const double end = get_seconds();

	printf("Calculating matrix check...\n");

    std::complex<double>    final_sum = 0;
	long long int count     = 0;

	for(i = 0; i < N; i++) {
        final_sum += (double)pivots[i];
        count++;
	}
	for(i = 0; i < N; i++) {
        final_sum +=  vectorB[i];
        count++;
	}
	for(i = 0; i < M; i++) {
	    for(j = 0; j < N; j++) {
        final_sum += matrixA[i*N + j];
        count++;
        }
	}

	double M_dbl = (double) M;
	double N_dbl = (double) N;
	double matrix_memory = (M_dbl * N_dbl) * ((double) sizeof(double))
	                     + (N_dbl) * ((double) sizeof(double));

	printf("\n");
	printf("===============================================================\n");

	const double count_dbl = (double) count;
	const double scaled_result = std::abs(final_sum) / count_dbl;

	printf("Final Sum is:         %f\n", scaled_result);

	printf("Memory for Matrices:  %f MB\n",
		(matrix_memory / (1024 * 1024)));

	const double time_taken = (end - start);
	const double reps_dbl   = (double)reps;

	printf("Input Size:           %d\n", M*N);
	printf("Total time:           %f seconds\n", time_taken);
	printf("Average time:         %f seconds\n", time_taken / reps_dbl);

	const double flops_computed = ((M_dbl * N_dbl * 2.0) +
        	(M_dbl * N_dbl * 2.0))*reps;

	printf("FLOPs computed:       %f\n", flops_computed);
	printf("GFLOP/s rate:         %f GF/s\n", (flops_computed / time_taken) / 1000000000.0);

	printf("===============================================================\n");
	printf("\n");

	free(matrixA);
	free(vectorB);

	return 0;
}
