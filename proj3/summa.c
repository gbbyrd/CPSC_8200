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
#define SZ 3200		//Each matrix of entire A, B, and C is SZ by SZ. Set a small value for testing, and set a large value for collecting experimental data.

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

void print_process_info(double **A, double **B, double **C, int rank, int block_sz) {
	printf("Rank: %d\n", rank);
	
	printf("-----A-----\n");
	for (int i=0; i<block_sz; i++) {
		for (int j=0; j<block_sz; j++) {
			printf("%f ", A[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("-----B-----\n");
	for (int i=0; i<block_sz; i++) {
		for (int j=0; j<block_sz; j++) {
			printf("%f ", B[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("-----C-----\n");
	for (int i=0; i<block_sz; i++) {
		for (int j=0; j<block_sz; j++) {
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
}

int main(int argc, char *argv[]) {
	int rank, num_proc;							//process rank and total number of processes
	double start_time, end_time, total_time;	// for timing
	int block_sz;								// Block size length for each processor to handle
	int proc_grid_sz;							// 'q' from the slides
	
	srand(time(NULL));							// Seed random numbers

	/* insert MPI functions to 1) start process, 2) get total number of processors and 3) process rank*/

	// initialize MPI
	MPI_Init(&argc, &argv);

	// get the total number of processors
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

	// get the processor rank
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	/* assign values to 1) proc_grid_sz and 2) block_sz*/
	
	proc_grid_sz = sqrt(num_proc);
	block_sz = SZ / proc_grid_sz;

	if (SZ % proc_grid_sz != 0){
		printf("Matrix size cannot be evenly split amongst resources!\n");
		printf("Quitting....\n");
		exit(-1);
	}

	// Create the local matrices on each process

	double **A, **B, **C;
	A = alloc_2d_double(block_sz, block_sz);
	B = alloc_2d_double(block_sz, block_sz);
	C = alloc_2d_double(block_sz, block_sz);

	
	initialize(A, B, C, block_sz);

	// Use MPI_Wtime to get the starting time
	start_time = MPI_Wtime();

	// Use SUMMA algorithm to calculate product C
	matmul(rank, proc_grid_sz, block_sz, A, B, C);


	// Use MPI_Wtime to get the finishing time
	end_time = MPI_Wtime();


	// Obtain the elapsed time and assign it to total_time
	total_time = end_time - start_time;


	if (rank == 0){
		// Print in pseudo csv format for easier results compilation
		printf("squareMatrixSideLength,%d,numMPICopies,%d,walltime,%lf\n",
			SZ, num_proc, total_time);
	}

	// Destroy MPI processes

	MPI_Finalize();

	return 0;
}
