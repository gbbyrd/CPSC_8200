// #include <pthread.h>
// #include <stdio.h>
// #define NUM_THREADS 5

// void *PrintHello(void *threadid)
// {
//     long tid;
//     tid = (long)threadid;
//     printf("Hello World! It's me, thread #%ld!\n", tid);
//     pthread_exit(NULL);
// }

// int main (int argc, char *argv[])
// {
//     // pthread_t threads[NUM_THREADS];
//     // int rc;
//     // long t;
//     // for (t=0; t < NUM_THREADS; t++) {
//     //     printf("In main: creating thread %ld\n", t);
//     //     rc = pthread_create(&threads[t], NULL, PrintHello, (void *)t);
//     //     if (rc) {
//     //         printf("ERROR; return code from pthread_create() is %d\n", rc);
//     //     }
//     // }

//     pthread_t threads[NUM_THREADS];
//     long t;
//     long safety[NUM_THREADS];
//     for (t=0; t < NUM_THREADS; t++)
//     {
//         safety[t] = t;
//         int rc = pthread_create(&threads[t], NULL, PrintHello, (void *)safety[t]);
//     }

//     /* Last thing that main should do */
//     pthread_exit(NULL);
// }

// #include <pthread.h>
// #include <stdio.h>

// #define NTHREADS 4
// #define N 1000
// #define MEGEXTRA 1000000

// pthread_attr_t attr;

// void *dowork(void *threadid)
// {
//    double A[N][N];
//    int i, j;
//    long tid;
//    size_t mystacksize;

//    tid = (long)threadid;
//    pthread_attr_getstacksize(&attr, &mystacksize);
//    printf("Thread %ld: stack size = %li bytes \n", tid, mystacksize);
//    for (i = 0; i < N; i++) {
//       for (j = 0; j < N; j++) {
//          A[i][j] = ((i * j) / 3.452) + (N - i);
//       }
//    }
//    pthread_exit(NULL);
// }

// int main(int argc, char *argv[])
// {
//    pthread_t threads[NTHREADS];
//    size_t stacksize;
//    int rc;
//    long t;

//    pthread_attr_init(&attr);
//    pthread_attr_getstacksize(&attr, &stacksize);
//    printf("Default stack size = %li\n", stacksize);

//    stacksize = sizeof(double)*N*N+MEGEXTRA;
//    printf("Amount of stack needed per thread = %li\n", stacksize);
//    pthread_attr_setstacksize (&attr, stacksize);

//    printf("Creating threads with stack size = %li bytes\n", stacksize);
//    for(t=0; t<NTHREADS; t++){
//       rc = pthread_create(&threads[t], &attr, dowork, (void *)t);
//       if (rc){
//          printf("ERROR; return code from pthread_create() is %d\n", rc);
//         //  exit(-1);
//       }
//    }
//    printf("Created %ld threads.\n", t);
//    pthread_exit(NULL);
// }

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_THREADS  3
#define TCOUNT 10
#define COUNT_LIMIT 12

int count = 0;
pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;

void *inc_count(void *t)
{
  int i;
  long my_id = (long)t;

  for (i = 0; i < TCOUNT; i++) {
    pthread_mutex_lock(&count_mutex);
    count++;

    /* Check the value of count and signal waiting thread when condition is
     * reached. Note that this occurs while mutex is locked.
     */
    if (count == COUNT_LIMIT) {
      printf("inc_count(): thread %ld, count = %d -- threshold reached.",
             my_id, count);
      pthread_cond_signal(&count_threshold_cv);
      printf("Just sent signal.\n");
    }
    printf("inc_count(): thread %ld, count = %d -- unlocking mutex\n",
           my_id, count);
    pthread_mutex_unlock(&count_mutex);

    /* Do some work so threads can alternate on mutex lock */
    sleep(1);
  }
  pthread_exit(NULL);
}

void *watch_count(void *t)
{
  long my_id = (long)t;

  printf("Starting watch_count(): thread %ld\n", my_id);

  /* Lock mutex and wait for signal. Note that the pthread_cond_wait routine
   * will automatically and atomically unlock mutex while it waits.
   * Also, note that if COUNT_LIMIT is reached before this routine is run by
   * the waiting thread, the loop will be skipped to prevent pthread_cond_wait
   * from never returning.
   */
  pthread_mutex_lock(&count_mutex);
  while (count < COUNT_LIMIT) {
    printf("watch_count(): thread %ld Count= %d. Going into wait...\n", my_id,count);
    pthread_cond_wait(&count_threshold_cv, &count_mutex);
    printf("watch_count(): thread %ld Condition signal received. Count= %d\n", my_id,count);
  }
  printf("watch_count(): thread %ld Updating the value of count...\n", my_id);
  count += 125;
  printf("watch_count(): thread %ld count now = %d.\n", my_id, count);
  printf("watch_count(): thread %ld Unlocking mutex.\n", my_id);
  pthread_mutex_unlock(&count_mutex);
  pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
  int i, rc;
  long t1 = 1,
       t2 = 2,
       t3 = 3;
  pthread_t threads[3];
  pthread_attr_t attr;

  /* Initialize mutex and condition variable objects */
  pthread_mutex_init(&count_mutex, NULL);
  pthread_cond_init (&count_threshold_cv, NULL);

  /* For portability, explicitly create threads in a joinable state */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_create(&threads[0], &attr, watch_count, (void *)t1);
  pthread_create(&threads[1], &attr, inc_count, (void *)t2);
  pthread_create(&threads[2], &attr, inc_count, (void *)t3);

  /* Wait for all threads to complete */
  for (i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
  printf ("Main(): Waited and joined with %d threads. Final value of count = %d. Done.\n",
          NUM_THREADS, count);

  /* Clean up and exit */
  pthread_attr_destroy(&attr);
  pthread_mutex_destroy(&count_mutex);
  pthread_cond_destroy(&count_threshold_cv);
  pthread_exit (NULL);

}