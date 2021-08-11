/*
Mystery code for CS 4380 / CS 5351

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
#include <climits>
#include <cuda.h>
#include <sys/time.h>
#include "ECLgraph.h"

static const int ThreadsPerBlock = 512;

static void mystery(const ECLgraph g, int* const mat)
{
  const int s = g.nodes;

  for (int i = 0; i < s; i++) {
    for (int j = 0; j < s; j++) {
      mat[i * s + j] = ((i == j) ? 0 : (INT_MAX / 2));
    }
  }

  for (int i = 0; i < s; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int n = g.nlist[j];
      mat[i * s + n] = g.eweight[j];
    }
  }

  for (int k = 0; k < s; k++) {
    for (int i = 0; i < s; i++) {
      for (int j = 0; j < s; j++) {
        const int sum = mat[i * s + k] + mat[k * s + j];
        if (mat[i * s + j] > sum) {
          mat[i * s + j] = sum;
        }
      }
    }
  }
}

static __global__ void mystery1(const int s, int* const mat)
{
  const int i = (threadIdx.x + blockIdx.x * blockDim.x) / s;
  const int j = (threadIdx.y + blockIdx.y * blockDim.y) % s;

  if (i < s) {
    if (j < s) {
      mat[i * s + j] = ((i == j) ? 0 : (INT_MAX / 2));
    }
  }
}

static __global__ void mystery2(const ECLgraph g, int* const mat)
{
  const int s = g.nodes;

  const int i = (threadIdx.x + blockIdx.x * blockDim.x) / s;

  if (i < s) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int n = g.nlist[j];
      mat[i * s + n] = g.eweight[j];
    }
  }
}

static __global__ void mystery3(const ECLgraph g, int* const mat, const int k)
{
  const int s = g.nodes;

  const int i = (threadIdx.x + blockIdx.x * blockDim.x) / s;
  const int j = (threadIdx.y + blockIdx.y * blockDim.y) % s;


  if (i < s) {
    if (j < s) {
      const int sum = mat[i * s + k] + mat[k * s + j];
      if (mat[i * s + j] > sum) {
        mat[i * s + j] = sum;
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

int main(int argc, char *argv[])
{
  printf("Mystery v1.0\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s input_file\n", argv[0]); exit(-1);}

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);

  // make all edge weights positive
  for (int j = 0; j < g.edges; j++) {
    if (g.eweight[j] < 0) g.eweight[j] = -g.eweight[j];
  }

  // allocate memory
  int* const mat1 = new int [g.nodes * g.nodes];
  int* const mat2 = new int [g.nodes * g.nodes];
  int* mat;

  ECLgraph d_g = g;
  cudaMalloc((void **) &d_g.nindex, sizeof(int) * (g.nodes + 1));
  cudaMalloc((void **) &d_g.nlist, sizeof(int) * g.edges);
  cudaMalloc((void **) &d_g.eweight, sizeof(int) * g.edges);
  cudaMemcpy(d_g.nindex, g.nindex, sizeof(int) * (g.nodes + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g.nlist, g.nlist, sizeof(int) * g.edges, cudaMemcpyHostToDevice);
  cudaMemcpy(d_g.eweight, g.eweight, sizeof(int) * g.edges, cudaMemcpyHostToDevice);

  //allocate mat on gpu
  if (cudaSuccess != cudaMalloc((void **)&mat, sizeof(int) *g.nodes *g.nodes )) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}


  // start time
  timeval beg, end;
  gettimeofday(&beg, NULL);

  // execute timed code
  mystery1<<<(g.nodes * g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g.nodes, mat);
  mystery2<<<(g.nodes * g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g, mat);
  for (int k = 0; k < g.nodes; k++) {
    mystery3<<<(g.nodes * g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g, mat, k);
  }
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

  // get result from GPU
  CheckCuda();
  if (cudaSuccess != cudaMemcpy(mat1, mat, sizeof(int) * g.nodes * g.nodes, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}

  // verify if problem size is small enough
  if (g.nodes < 2000) {
    // run serial code
    mystery(g, mat2);

    // compare results
    for (int i = 0; i < g.nodes; ++i) {
      for (int j = 0; j < g.nodes; ++j) {
        if (mat1[i * g.nodes + j] != mat2[i * g.nodes + j]) {fprintf(stderr, "ERROR: solutions differ\n"); exit(-1);}
      }
    }
    printf("verification passed\n");
  }

  // clean up

  freeECLgraph(g);
  free(d_g.nindex);
  free(d_g.nlist);
  free(d_g.eweight);
  delete [] mat1;
  delete [] mat2;
  return 0;
}
