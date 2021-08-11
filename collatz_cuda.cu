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

Bryan Valenzuela
Connor Steed
*/

#include <cstdio>
#include <cuda.h>
#include <algorithm>
#include <sys/time.h>

static const int ThreadsPerBlock = 512;

static __global__ void collatz(const long start, const long bound, const long step, int* const maxlen)
{
  const long i = threadIdx.x + blockIdx.x * (long)blockDim.x + start;

  if( (i - start) % step == 0)
  if (i < bound){
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
      atomicMax(maxlen, len);
  }

}

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

int main(int argc, char *argv[])
{
  printf("Collatz v1.5\n");

  // check command line
  if (argc != 4) {fprintf(stderr, "USAGE: %s start bound step\n", argv[0]); exit(-1);}
  const long start = atol(argv[1]);
  const long bound = atol(argv[2]);
  const long step = atol(argv[3]);
  if (start < 1) {fprintf(stderr, "ERROR: start value must be at least 1\n"); exit(-1);}
  if (bound <= start) {fprintf(stderr, "ERROR: bound must be larger than start\n"); exit(-1);}
  if (step < 1) {fprintf(stderr, "ERROR: step size must be at least 1\n"); exit(-1);}
  printf("start value: %ld\n", start);
  printf("upper bound: %ld\n", bound);
  printf("step size: %ld\n", step);

  int maxlen = 0;
  int size = sizeof(int);

  int* d_maxlen;

  if (cudaSuccess != cudaMalloc((void **)&d_maxlen, size)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}

  if (cudaSuccess != cudaMemcpy(d_maxlen, &maxlen, sizeof(int), cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
  // start time
  timeval beg, end;
  gettimeofday(&beg, NULL);

  // execute timed code
  collatz<<<(bound + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(start, bound, step, d_maxlen);
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

   CheckCuda();
  if (cudaSuccess != cudaMemcpy(&maxlen, d_maxlen, sizeof(int), cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}


  // print result
  printf("maximum sequence length: %d elements\n", maxlen);

  cudaFree(d_maxlen);
  return 0;
}
