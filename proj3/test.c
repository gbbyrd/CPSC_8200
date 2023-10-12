#include <stdio.h>
#include <time.h>	
#include <stdlib.h>	
#include <math.h>	
#include <string.h>

#define SZ 4000		//Each matrix of entire A, B, and C is SZ by SZ. Set a small value for testing, and set a large value for collecting experimental data.


/**
*   Allocate space for a two-dimensional array
*/
double **alloc_2d_double(int n_rows, int n_cols) {
	double** array = (double**)malloc(n_rows*sizeof(double*));

	printf("got here");
	for (int i=0; i<n_rows; i++) {
		array[i] = (double*)malloc(n_cols * sizeof(double));
	}

	// int i;
	// double **array;
	// array = (double **)malloc(n_rows * sizeof (double *));
	// array[0] = (double *) malloc(n_rows * n_cols * sizeof(double));
	// for (i=1; i<n_rows; i++){
	// 		array[i] = array[0] + i * n_cols;
	// }
	return array;
}

/**
*	Initialize arrays A and B with random numbers, and array C with zeros. 
*	Each array is setup as a square block of blck_sz.
**/
void initialize(double **lA, double **lB, double **lC, int blck_sz){
	int i, j;
	double value;
	// Set random values...technically it is already random and this is redundant
	for (i=0; i<blck_sz; i++){
		for (j=0; j<blck_sz; j++){
			// lA[i][j] = (double)rand() / (double)RAND_MAX;
			// lB[i][j] = (double)rand() / (double)RAND_MAX;
            lA[i][j] = (double) 0;
			lB[i][j] = (double) 0;
			lC[i][j] = 0.0;
		}
	}
}

int main(int argc, char *argv[]) {
    printf("what the actual fuck is happening");
	printf("got here");
	int rank, num_proc;							//process rank and total number of processes
	double start_time, end_time, total_time;	// for timing
	int block_sz;								// Block size length for each processor to handle
	int proc_grid_sz;							// 'q' from the slides

	printf("got here");
	
	srand(time(NULL));							// Seed random numbers

/* insert MPI functions to 1) start process, 2) get total number of processors and 3) process rank*/

	// // initialize MPI
	// MPI_Init(&argc, &argv);

	// // get the total number of processors
	// MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

	// // get the processor rank
	// MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    num_proc = 1;
/* assign values to 1) proc_grid_sz and 2) block_sz*/
	printf("got here");
	proc_grid_sz = sqrt(num_proc);
	block_sz = SZ / proc_grid_sz;

	if (SZ % proc_grid_sz != 0){
		printf("Matrix size cannot be evenly split amongst resources!\n");
		printf("Quitting....\n");
		exit(-1);
	}

	// Create the local matrices on each process

	double **A, **B, **C;

	int test_size = 6;
	block_sz = test_size;
	// int test_size = block_sz;

	A = alloc_2d_double(block_sz, block_sz);
	B = alloc_2d_double(block_sz, block_sz);
	C = alloc_2d_double(block_sz, block_sz);

	// int test_size = 6;
	// block_sz = test_size;

	// A = (double **)malloc(test_size * sizeof (double *));
	// A[0] = (double *)malloc(test_size * test_size * sizeof(double));
	// B = (double **)malloc(test_size * sizeof (double *));
	// B[0] = (double *)malloc(test_size * test_size * sizeof(double));
	// C = (double **)malloc(test_size * sizeof (double *));
	// C[0] = (double *)malloc(test_size * test_size * sizeof(double));
	
	initialize(A, B, C, block_sz);

	printf("Printing A...\n\n");
	for (int i=0; i<test_size; i++) {
		for (int j=0; j<test_size; j++) {
			printf("%f ", A[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");

	for (int i=0; i<test_size; i++) {
		for (int j=0; j<test_size; j++) {
			// A[i][j] = 1;
			if (i == j || i-1==j) {
				printf("got here: %f -> ", C[i][j]);
				C[i][j] = (double) 1.0;
				printf("now: %f\n\n", C[i][j]);
			} else {
				C[i][j] = 1.0;
			}
		}
	}

	// // for (int i=0; i<test_size; i++) {
	// // 	for (int j=0; j<test_size; j++) {
	// // 		if (i == j || i+1==j) {
	// // 			B[i][j] = 1;
	// // 		} else {
	// // 			B[i][j] = 0;
	// // 		}
	// // 	}
	// // }

	// // print A
	// printf("Printing A...\n\n");
	// for (int i=0; i<test_size; i++) {
	// 	for (int j=0; j<test_size; j++) {
	// 		printf("%d ", A[i][j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n\n");

	// // print B
	// printf("Printing B...\n\n");
	// for (int i=0; i<test_size; i++) {
	// 	for (int j=0; j<test_size; j++) {
	// 		printf("%d ", B[i][j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n\n");

	// // Use MPI_Wtime to get the starting time
	// start_time = MPI_Wtime();


	// // Use SUMMA algorithm to calculate product C
	// matmul(rank, proc_grid_sz, block_sz, A, B, C);


	// // Use MPI_Wtime to get the finishing time
	// end_time = MPI_Wtime();


	// // Obtain the elapsed time and assign it to total_time
	// total_time = end_time - start_time;

	// // Insert statements for testing
	
	// // print C
	// printf("Printing C...\n\n");
	// for (int i=0; i<test_size; i++) {
	// 	for (int j=0; j<test_size; j++) {
	// 		printf("%d ", C[i][j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n\n");

	// if (rank == 0){
	// 	// Print in pseudo csv format for easier results compilation
	// 	printf("squareMatrixSideLength,%d,numMPICopies,%d,walltime,%lf\n",
	// 		SZ, num_proc, total_time);
	// }

	// Destroy MPI processes

	// MPI_Finalize();

	return 0;
}