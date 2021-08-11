/*
Collatz code for CS 4380 / CS 5351

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

#include <cstdio>
#include <algorithm>
#include <pthread.h>
#include <sys/time.h>

static long threads;
static long start;
static long bound;
static long step;
static int maxlen;

pthread_mutex_t mutex;

static void* collatz(void* arg)
{
  //typecast the arg to a long rank
  const long my_rank = (long)arg;

  // determine work for each thread

  // compute sequence lengths
  int my_len = 0; //the local solution
  int maxlen = 0; //the global solution
  for (long i = my_rank; i < bound; i += threads) {
    long val = i;
    int len = 1;
    while (val != 1) {
      len++;
      if ((val % 2) == 0) {
        val /= 2;  // even
      } else {
        val = 3 * val + 1;  // odd
      }
    }
    my_len = std::max(my_len, len);
  }

  //mutex lock on compiling the solutions
  pthread_mutex_lock(&mutex);
  maxlen = std::max(maxlen, my_len);
  pthread_mutex_unlock(&mutex);
}

int main(int argc, char *argv[])
{
  printf("Collatz v1.5\n");

  // check command line
  if (argc != 4) {fprintf(stderr, "USAGE: %s start bound step\n", argv[0]); exit(-1);}
  start = atol(argv[1]);
  bound = atol(argv[2]);
  step = atol(argv[3]);
  if (start < 1) {fprintf(stderr, "ERROR: start value must be at least 1\n"); exit(-1);}
  if (bound <= start) {fprintf(stderr, "ERROR: bound must be larger than start\n"); exit(-1);}
  if (step < 1) {fprintf(stderr, "ERROR: step size must be at least 1\n"); exit(-1);}
  printf("start value: %ld\n", start);
  printf("upper bound: %ld\n", bound);
  printf("step size: %ld\n", step);
  threads = atol(argv[4]);
  if (threads < 1) {fprintf(stderr, "ERROR: threads must be at least 1\n"); exit(-1);}
  printf("threads: %ld\n", threads);

  // initialize pthread variables
  pthread_t* const handle = new pthread_t [threads - 1];
  pthread_mutex_init(&mutex, NULL);

  // start time
  timeval beg, end;
  gettimeofday(&beg, NULL);

  // launch threads
  for (long thread = 0; thread < threads - 1; thread++) {
    pthread_create(&handle[thread], NULL, collatz, (void *)thread);
  }

  // work for master
  collatz((void*)(threads - 1));

  // join threads
  for (long thread = 0; thread < threads - 1; thread++) {
    pthread_join(handle[thread], NULL);
  }

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

  // print result
  printf("maximum sequence length: %d elements\n", maxlen);
  
  // clean up
  pthread_mutex_destroy(&mutex);
  delete [] handle;
  return 0;
}
