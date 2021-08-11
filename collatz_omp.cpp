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
#include <sys/time.h>

static int collatz(const long start, const long bound, const long step, const int threads)
{
  // compute sequence lengths
  int maxlen = 0;
  #pragma omp parallel for default(none) shared(start, bound, step) num_threads(threads) reduction(max:maxlen) SCHEDULE
  for (long i = start; i < bound; i += step) {
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
    maxlen = std::max(maxlen, len);
  }

  return maxlen;
}

int main(int argc, char *argv[])
{
  printf("Collatz v1.5\n");

  // check command line
  if (argc != 5) {fprintf(stderr, "USAGE: %s start bound step thread_count\n", argv[0]); exit(-1);}
  const long start = atol(argv[1]);
  const long bound = atol(argv[2]);
  const long step = atol(argv[3]);
  if (start < 1) {fprintf(stderr, "ERROR: start value must be at least 1\n"); exit(-1);}
  if (bound <= start) {fprintf(stderr, "ERROR: bound must be larger than start\n"); exit(-1);}
  if (step < 1) {fprintf(stderr, "ERROR: step size must be at least 1\n"); exit(-1);}
  printf("start value: %ld\n", start);
  printf("upper bound: %ld\n", bound);
  printf("step size: %ld\n", step);
  const int threads = atoi(argv[4]);
  if (threads < 1) {fprintf(stderr, "ERROR: thread_count must be at least 1\n"); exit(-1);}
  printf("thread count: %d\n", threads);


  // start time
  timeval beg, end;
  gettimeofday(&beg, NULL);

  // execute timed code
  const int maxlen = collatz(start, bound, step, threads);

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

  // print result
  printf("maximum sequence length: %d elements\n", maxlen);
  return 0;
}
