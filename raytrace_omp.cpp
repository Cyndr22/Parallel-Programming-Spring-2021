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
*/

#include <cstdio>
#include <climits>
#include <sys/time.h>
#include "ECLgraph.h"

static void mystery(const ECLgraph g, int* const mat, const int threads)
{
  const int s = g.nodes;

  #pragma omp parallel default(none) shared(g, mat, s) num_threads(threads)
  {
    #pragma omp for
    for (int i = 0; i < s; i++) {
      for (int j = 0; j < s; j++) {
        mat[i * s + j] = ((i == j) ? 0 : (INT_MAX / 2));
      }
    }

    #pragma omp for
    for (int i = 0; i < s; i++) {
      for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
        const int n = g.nlist[j];
        mat[i * s + n] = g.eweight[j];
      }
    }

    for (int k = 0; k < s; k++) {
      #pragma omp for
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
}

int main(int argc, char *argv[])
{
  printf("Mystery v1.0\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s input_file thread_count\n", argv[0]); exit(-1);}
  const int threads = atoi(argv[2]);
  if (threads < 1) {fprintf(stderr, "ERROR: thread_count must be at least 1\n"); exit(-1);}
  printf("thread count: %d\n", threads);

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

  // start time
  timeval beg, end;
  gettimeofday(&beg, NULL);

  // run code
  mystery(g, mat1, threads);

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

  // verify if problem size is small enough
  if (g.nodes < 2000) {
    // run serial code with one thread
    mystery(g, mat2, 1);

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
  delete [] mat1;
  delete [] mat2;
  return 0;
}
