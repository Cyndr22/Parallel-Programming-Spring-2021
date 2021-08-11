/*
Maximal independent set code for CS 4380 / CS 5351

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

#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <sys/time.h>
#include "ECLgraph.h"

static const int ThreadsPerBlock = 512;

static const unsigned char undecided = 0;
static const unsigned char in = 1;
static const unsigned char out = 2;

// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static __device__ unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

static __global__ void init(const ECLgraph g, volatile unsigned char* const status, unsigned int* rndval) {
  // initialize arrays
  const int v = threadIdx.x + blockIdx.x * blockDim.x;
  if (v < g.nodes) status[v] = undecided;
  const int w = threadIdx.x + blockIdx.x * blockDim.x;
  if (w < g.nodes) rndval[w] = hash(w + 712453897);
}

static __global__ void mis(const ECLgraph g, volatile unsigned char* const status, const unsigned int* const rndval, volatile bool* const goagain)
{
  // go over all the nodes
  const int v = threadIdx.x + blockIdx.x * blockDim.x;
  if (v < g.nodes) {
    if (status[v] == undecided) {
      int i = g.nindex[v];
      // try to find a neighbor whose random number is lower
      while ((i < g.nindex[v + 1]) && ((status[g.nlist[i]] == out) || (rndval[v] < rndval[g.nlist[i]]) || ((rndval[v] == rndval[g.nlist[i]]) && (v < g.nlist[i])))) {
        i++;
      }
      if (i < g.nindex[v + 1]) {
        // found such a neighbor -> status still unknown
        *goagain = true;
      } else {
        // no such neighbor -> all neighbors are "out" and my status is "in"
        for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
          status[g.nlist[i]] = out;
        }
        status[v] = in;
      }
    }
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

int main(int argc, char* argv[])
{
  printf("Maximal Independent Set v1.6\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s input_file\n", argv[0]); exit(-1);}

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);

  // allocate arrays
  unsigned char* const status = new unsigned char [g.nodes];
  unsigned int* const rndval = new unsigned int [g.nodes];
  unsigned char* d_status;
  unsigned int* d_rndval;

  bool goagain;
  bool* d_goagain;

  ECLgraph d_g = g;
  cudaMalloc((void **) &d_g.nindex, sizeof(int) * (d_g.nodes + 1));
  cudaMalloc((void **) &d_g.nlist, sizeof(int) * d_g.edges);
  cudaMemcpy(d_g.nindex, g.nindex, sizeof(int) * (d_g.nodes + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g.nlist, g.nlist, sizeof(int) * d_g.edges, cudaMemcpyHostToDevice);

  if (cudaSuccess != cudaMalloc((void **)&d_status, sizeof(char) * d_g.nodes)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&d_rndval, sizeof(int) * d_g.nodes)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&d_goagain, sizeof(bool) * d_g.nodes)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}

  // start time
  timeval beg, end;
  gettimeofday(&beg, NULL);

  // execute timed code
  //initialize status and rndval
  init<<<(d_g.nodes * d_g.nodes - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g, d_status, d_rndval);
  // repeat until all nodes' status has been decided
  do {
    goagain = false;
    //copy goagain to gpu
    if (cudaSuccess != cudaMemcpy(d_goagain, &goagain, sizeof(bool), cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
    //launch mis threads
    mis<<<(d_g.nodes * d_g.nodes - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g, d_status, d_rndval, d_goagain);
    //copy goagain back to cpu
    if (cudaSuccess != cudaMemcpy(&goagain, d_goagain, sizeof(bool), cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
  } while (goagain);
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

  CheckCuda();
  if (cudaSuccess != cudaMemcpy(status, d_status, sizeof(char) * g.nodes, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}

  // determine and print set size
  int count = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (status[v] == in) {
      count++;
    }
  }
  printf("elements in set: %d (%.1f%%)\n", count, 100.0 * count / g.nodes);

  // verify result
  for (int v = 0; v < g.nodes; v++) {
    if ((status[v] != in) && (status[v] != out)) {fprintf(stderr, "ERROR: found unprocessed node\n"); exit(-1);}
    if (status[v] == in) {
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == in) {fprintf(stderr, "ERROR: found adjacent nodes in MIS\n"); exit(-1);}
      }
    } else {
      bool flag = true;
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == in) {
          flag = false;
          break;
        }
      }
      if (flag) {fprintf(stderr, "ERROR: set is not maximal\n"); exit(-1);}
    }
  }
  printf("verification passed\n");

  // clean up

  freeECLgraph(g);
  cudaFree(d_status);
  cudaFree(d_rndval);
  free(d_g.nindex);
  free(d_g.nlist);
  delete [] status;
  delete [] rndval;
  return 0;
}
