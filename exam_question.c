#include <stdio.h>
// add stdlib.h to avoid warning
#include<stdlib.h>

#include "mpi.h"
#define MAXSIZE 20

// add int before main below
int main(int argc, char *argv[])
{
    int myid, numprocs, localresult, result[MAXSIZE];
    int i, tag, final;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if ( numprocs > MAXSIZE )
    {
        if ( myid == 0 )
            printf("The number of processes must be less than %d\n", MAXSIZE);
        exit(0);
    }
    final = 0;
    for ( i = 0; i < numprocs; i++)
        result[i] = i;

    // MPI Scatter should have 1 instead of numprocs in the send count
    // MPI_Scatter(result, numprocs, MPI_INT, &localresult, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(result, 1, MPI_INT, &localresult, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // we do not need the below if statement
    if( myid )
    {
        localresult = localresult/myid;
        // the root value in MPI_Reduce below should be 0, not 1 and MPI_Reduce should
        // not be within the if statement
        // MPI_Reduce(&localresult, &final, 1, MPI_INT, MPI_SUM, 1, MPI_COMM_WORLD);
    }
    MPI_Reduce(&localresult, &final, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // the below code is redundent
    // if( final )
    // {
    //     tag = myid;
    //     MPI_Send(&final, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
    // }

    // if(myid == 0)
    //     MPI_Recv(&final, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    if(myid == 0)
        printf("The final result is: %d \n", final);
        
    MPI_Finalize();
    return (0);
}