/*
Vector addition code for CS 4380 / CS 5351

Copyright (c) 2021 Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is *not* permitted. Use in source or binary form, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <cstdlib>
#include <cstdio>
#include <pthread.h>
#include <sys/time.h>

// shared variables
static long threads;
static int elems;
static int* a;
static int* b;
static int* c;

static void* vadd(void* arg)
{
  const long my_rank = (long)arg;

  // determine work for each thread
  const int beg = my_rank * (long)elems / threads;
  const int end = (my_rank + 1) * (long)elems / threads;

  // perform vector addition
  for (int i = beg; i < end; i++) {
    c[i] = a[i] + b[i];
  }

  return NULL;
}

int main(int argc, char* argv[])
{
  printf("Vector Addition v1.3\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s vector_elements threads\n", argv[0]); exit(-1);}
  elems = atoi(argv[1]);
  if (elems < 2) {fprintf(stderr, "ERROR: vector_elements must be at least 2\n"); exit(-1);}
  printf("vector elements: %d\n", elems);
  threads = atol(argv[2]);
  if (threads < 1) {fprintf(stderr, "ERROR: threads must be at least 1\n"); exit(-1);}
  printf("threads: %ld\n", threads);

  // allocate vectors
  a = new int [elems];
  b = new int [elems];
  c = new int [elems];

  // initialize vectors
  for (int i = 0; i < elems; i++) a[i] = i;
  for (int i = 0; i < elems; i++) b[i] = elems - i;
  for (int i = 0; i < elems; i++) c[i] = -1;

  // initialize pthread variables
  pthread_t* const handle = new pthread_t [threads - 1];

  // start time
  timeval beg, end;
  gettimeofday(&beg, NULL);

  // launch threads
  for (long thread = 0; thread < threads - 1; thread++) {
    pthread_create(&handle[thread], NULL, vadd, (void *)thread);
  }

  // work for master
  vadd((void*)(threads - 1));

  // join threads
  for (long thread = 0; thread < threads - 1; thread++) {
    pthread_join(handle[thread], NULL);
  }

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

  // verify result
  for (int i = 0; i < elems; i++) {
    if (c[i] != elems) {fprintf(stderr, "ERROR: incorrect result\n"); exit(-1);}
  }
  printf("verification passed\n");

  // clean up
  delete [] handle;
  delete [] a;
  delete [] b;
  delete [] c;
  return 0;
}
