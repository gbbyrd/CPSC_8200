/******************************************************************************************
*
*	Filename:	summa.c
*	Purpose:	A paritally implemented program for MSCS6060 HW. Students will complete 
*			the program by adding SUMMA implementation for matrix multiplication C = A * B.  
*	Assumptions:    A, B, and C are square matrices n by n; 
*			the total number of processors (np) is a square number (q^2).
*	To compile, use 
*	    mpicc -o summa summa.c
*       To run, use
*	    mpiexec -n $(NPROCS) ./summa
*********************************************************************************************/

#include <stdio.h>
#include <time.h>	
#include <stdlib.h>	
#include <math.h>	
#include <string.h>
#include "mpi.h"

#define min(a, b) ((a < b) ? a : b)
#define SZ 4000		//Each matrix of entire A, B, and C is SZ by SZ. Set a small value for testing, and set a large value for collecting experimental data.


/**
*   Allocate space for a two-dimensional array
*/
double **alloc_2d_double(int n_rows, int n_cols) {

	int i;
	double **array;
	array = (double **)malloc(n_rows * sizeof (double *));
	array[0] = (double *) malloc(n_rows * n_cols * sizeof(double));
	for (i=1; i<n_rows; i++){
			array[i] = array[0] + i * n_cols;
	}
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
			lA[i][j] = (double)rand() / (double)RAND_MAX;
			lB[i][j] = (double)rand() / (double)RAND_MAX;
			lC[i][j] = 0.0;
		}
	}
}

void print2dArray(double **array, int size, int rank, int before, char arrayName, int i) {
	char info[] = "before";
	if (before == 0)
		strcpy(info, "after");
	printf("Printing %c from %s rank: %d | it: %d\n", arrayName, info, rank, i);
	for (int i=0; i<size; i++) {
		for (int j=0; j<size; j++) {
			printf("%f ", array[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

void matmulAdd(double **C, double **A, double **B, int block_sz, int rank,
		char *info, int i) {

	for (int i=0; i<block_sz; i++) {
		for (int j=0; j<block_sz; j++) {
			for (int k=0; k<block_sz; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void print_process_info(double **A, double **B, double **C, int rank) {
	printf("Rank: %d\n", rank);
	
	printf("-----A-----\n");
	for (int i=0; i<2; i++) {
		for (int j=0; j<2; j++) {
			printf("%f ", A[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("-----B-----\n");
	for (int i=0; i<2; i++) {
		for (int j=0; j<2; j++) {
			printf("%f ", B[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("-----C-----\n");
	for (int i=0; i<2; i++) {
		for (int j=0; j<2; j++) {
			printf("%f ", C[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	return;
}

/**
*	Perform the SUMMA matrix multiplication. 
*       Follow the pseudo code in lecture slides.
*/
void matmul(int my_rank, int proc_grid_sz, int block_sz, double **my_A,
						double **my_B, double **my_C){

	//Add your implementation of SUMMA algorithm

	// define and allocate buffers
	double **buffA, **buffB;
	buffA = alloc_2d_double(block_sz, block_sz);
	buffB = alloc_2d_double(block_sz, block_sz);

	// define arrays
	int dimsizes[2];
	int free_coords[2];
	int wraparound[2];
	int coordinates[2];
	int p;
	int myrank;
	int q;
	int reorder=1;
	int my_grid_rank, grid_rank;

	// define grid, row, and col comms
	MPI_Comm grid_comm;
	MPI_Comm row_comm;
	MPI_Comm col_comm;

	// get global comm information
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	// define key values
	q = (int)sqrt((double)p);
	dimsizes[0] = dimsizes[1] = q;
	wraparound[0] = wraparound[1] = 1;
	int grid_size = proc_grid_sz * proc_grid_sz;

	// translate global comm to 2d cartesian coordinates comm and get relevant
	// variables
	MPI_Cart_create(MPI_COMM_WORLD, 2, dimsizes, wraparound, reorder, &grid_comm);
	MPI_Comm_rank(grid_comm, &my_grid_rank);
	MPI_Cart_coords(grid_comm, my_grid_rank, 2, coordinates);
	MPI_Cart_rank(grid_comm, coordinates, &grid_rank);

	// define sub communicators based on the row and column
	free_coords[0] = 0;
	free_coords[1] = 1;
	MPI_Cart_sub(grid_comm, free_coords, &row_comm);
	free_coords[1] = 0;
	free_coords[0] = 1;
	MPI_Cart_sub(grid_comm, free_coords, &col_comm);

	for (int i=0; i<q; i++) {
		if (coordinates[1] == i) {
			for (int j=0; j<block_sz; j++) {
				// copy my_A to buffer
				memcpy(buffA[j], my_A[j], block_sz * sizeof(double));
			}
		}
		// send the calling process's A matrix to all other processes in the row comm
		// recieve A matrix from each process that is in the row comm
		MPI_Bcast(*buffA, block_sz*block_sz, MPI_DOUBLE, i, row_comm);

		if (coordinates[0] == i) {
			for (int j=0; j<block_sz; j++) {
				// copy my_B to buffer
				memcpy(buffB[j], my_B[j], block_sz * sizeof(double));
			}
		}
		MPI_Bcast(*buffB, block_sz*block_sz, MPI_DOUBLE, i, col_comm);

		if (coordinates[0] == i && coordinates[1] == i) {
			matmulAdd(my_C, my_A, my_B, block_sz, myrank, "same", i);
		} else if (coordinates[1] == 1) {
			matmulAdd(my_C, buffA, my_B, block_sz, myrank, "buffA my B", i);
		} else if (coordinates[0] == 1) {
			matmulAdd(my_C, my_A, buffB, block_sz, myrank, "myA buffB", i);
		} else {
			matmulAdd(my_C, buffA, buffB, block_sz, myrank, "buffs", i);
		}
	}
	print_process_info(my_A, my_B, my_C, myrank);
}

int main(int argc, char *argv[]) {
	int rank, num_proc;							//process rank and total number of processes
	double start_time, end_time, total_time;	// for timing
	int block_sz;								// Block size length for each processor to handle
	int proc_grid_sz;							// 'q' from the slides
	
	srand(time(NULL));							// Seed random numbers

	// create a buffer that contains the global matrices
	double **A_global, **B_global, **C_global, **tempC;

	int test_size = 6;
	block_sz = test_size;
	// int test_size = block_sz;

	A_global = alloc_2d_double(block_sz, block_sz);
	B_global = alloc_2d_double(block_sz, block_sz);
	C_global = alloc_2d_double(block_sz, block_sz);
	tempC = alloc_2d_double(block_sz, block_sz);
	
	initialize(A_global, B_global, C_global, block_sz);

	for (int i=0; i<test_size; i++) {
		for (int j=0; j<test_size; j++) {
			if (i == j || i-1==j) {
				A_global[i][j] = 1.0;
			} else {
				A_global[i][j] = 0.0;
			}
		}
	}

	for (int i=0; i<test_size; i++) {
		for (int j=0; j<test_size; j++) {
			if (i == j || i+1==j) {
				B_global[i][j] = 1;
			} else {
				B_global[i][j] = 0;
			}
		}
	}

	/* insert MPI functions to 1) start process, 2) get total number of processors and 3) process rank*/

	// initialize MPI
	MPI_Init(&argc, &argv);

	// get the total number of processors
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

	// get the processor rank
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {

	
		// print A
		printf("Printing A global...\n\n");
		for (int i=0; i<test_size; i++) {
			for (int j=0; j<test_size; j++) {
				printf("%f ", A_global[i][j]);
			}
			printf("\n");
		}
		printf("\n\n");

		// print B
		printf("Printing B global...\n\n");
		for (int i=0; i<test_size; i++) {
			for (int j=0; j<test_size; j++) {
				printf("%f ", B_global[i][j]);
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// assign blocks of the initial matrix based on rank
	// printf("Num processes: %d\n", num_proc);
	/* assign values to 1) proc_grid_sz and 2) block_sz*/
	// proc_grid_sz = sqrt(num_proc);
	proc_grid_sz = 3;
	block_sz = 6 / proc_grid_sz;

	// Create the local matrices on each process
	double **A, **B, **C;

	// for testing.. assign a portion of the test matrix to each process based
	// on rank
	A = alloc_2d_double(2, 2);
	B = alloc_2d_double(2, 2);
	C = alloc_2d_double(2, 2);

	// set local A, B, and C values to 0
	A[0][0] = 0.0;
	A[0][1] = 0.0;
	A[1][0] = 0.0;
	A[1][1] = 0.0;

	B[0][0] = 0.0;
	B[0][1] = 0.0;
	B[1][0] = 0.0;
	B[1][1] = 0.0;

	C[0][0] = 0.0;
	C[0][1] = 0.0;
	C[1][0] = 0.0;
	C[1][1] = 0.0;

	// populate A and B matrices for each process based on rank
	int init_row = (rank / 3) * 2;
	int init_col = (rank % 3) * 2;

	A[0][0] = A_global[init_row][init_col];
	A[0][1] = A_global[init_row][init_col+1];
	A[1][0] = A_global[init_row+1][init_col];
	A[1][1] = A_global[init_row+1][init_col+1];

	B[0][0] = B_global[init_row][init_col];
	B[0][1] = B_global[init_row][init_col+1];
	B[1][0] = B_global[init_row+1][init_col];
	B[1][1] = B_global[init_row+1][init_col+1];

	// Use MPI_Wtime to get the starting time
	start_time = MPI_Wtime();

	// Use SUMMA algorithm to calculate product C
	matmul(rank, proc_grid_sz, block_sz, A, B, C);

	// Use MPI_Wtime to get the finishing time
	end_time = MPI_Wtime();

	// Obtain the elapsed time and assign it to total_time
	total_time = end_time - start_time;

	// Insert statements for testing

	if (rank == 0){
		// Print in pseudo csv format for easier results compilation
		printf("squareMatrixSideLength,%d,numMPICopies,%d,walltime,%lf\n",
			SZ, num_proc, total_time);
	}

	// Destroy MPI processes

	MPI_Finalize();

	return 0;
}