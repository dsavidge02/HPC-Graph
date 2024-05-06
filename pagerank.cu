/*
//@HEADER
// *****************************************************************************
//
//  HPCGraph: Graph Computation on High Performance Computing Systems
//              Copyright (2016) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?  Contact  George M. Slota   (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//                      Kamesh Madduri    (madduri@cse.psu.edu)
//
// *****************************************************************************
//@HEADER
*/


#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>

#include "dist_graph.h"
#include "comms.h"
#include "util.h"
#include "pagerank.h"

#define DAMPING_FACTOR 0.85

extern int procid, nprocs;
extern bool verbose, debug, verify;

__global__
void pagerank_init(dist_graph_t* g, double* pageranks, double* pageranks_next, 
                  double*sum_noouts, double* sum_noouts_next)
{
  uint64_t vert_index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (vert_index >= g->n_local) return;

  pageranks[vert_index] = (1.0 / (double)g->n);
  uint64_t out_degree = out_degree(g, vert_index);
  if(out_degree > 0)
    pageranks[vert_index] /= (double)out_degree;
  else{
    pageranks[vert_index] /= (double)g->n;
    atomicAdd(sum_noouts_next, pageranks[vert_index]);
  }
  return;
}

__global__
void update_pageranks(dist_graph_t* g, double* pageranks){
  uint64_t vert_index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (vert_index >= g->n_total || vert_index < g->n_local) return;

  pageranks[vert_index] = (1.0 / (double)g->n) / (double)g->n;
  return;
}

__global__
void update_pageranks_next(dist_graph_t* g, double* pageranks, double* pageranks_next){
  uint64_t vert_index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (vert_index >= g->n_total) return;

  pageranks[vert_index] = pageranks[vert_index];
  return;
}

__global__
void pagerank_iter_vert(dist_graph_t* g, double* pageranks, double* pageranks_next, double* sum_noouts, 
                        double* sum_noouts_next)
{
  uint64_t vert_index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (vert_index >= g->n_local) return;
  double vert_pagerank = *sum_noouts;
  
  uint64_t in_degree = in_degree(g, vert_index);
  uint64_t* ins = in_vertices(g, vert_index);
  for (uint64_t j = 0; j < in_degree; ++j){
    vert_pagerank += pageranks[ins[j]];
  }  

  vert_pagerank *= DAMPING_FACTOR;
  vert_pagerank += ((1.0 - DAMPING_FACTOR) / (double)g->n);

  uint64_t out_degree = out_degree(g, vert_index);
  if (out_degree > 0)
    vert_pagerank /= (double)out_degree;
  else
  {
    vert_pagerank /= (double)g->n;
    atomicAdd(sum_noouts_next, vert_pagerank);
  }

  pageranks_next[vert_index] = vert_pagerank;
  return;
}

__global__
void update_sendbuf_data_flt(mpi_data_t* comm, double* pageranks_next){
  uint64_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= comm->total_send) return;

  comm->sendbuf_data_flt[i] = pageranks_next[comm->sendbuf_vert[i]];
  return;
}

__global__
void update_pageranks_next_recv(mpi_data_t* comm, double* pageranks_next){
  uint64_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= comm->total_recv) return;
  
  pageranks_next[comm->recvbuf_vert[i]] = comm->recvbuf_data_flt[i];
  return;
}

__device__
uint64_t mult_hash_cuda(fast_map* map, uint64_t key)
{
  if (map->hashing)
    return (key*2654435761 % map->capacity);
  else
    return key;
}

__device__
uint64_t get_value_cuda(fast_map* map, uint64_t key){
  uint64_t cur_index = mult_hash_cuda(map, key)*2;
  int i = 1;
  while (map->arr[cur_index] != key && map->arr[cur_index] != NULL_KEY){
    cur_index = (cur_index + i * i) % (map->capacity*2);
    i++;
  }
  if (map->arr[cur_index] == NULL_KEY)
    return NULL_KEY;
  else
    return map->arr[cur_index+1];
}

__global__
void handle_update_recvbuf_data_cuda(dist_graph_t* g, mpi_data_t* comm, double* pageranks)
{
  uint64_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= comm->total_recv) return;

  uint64_t index = get_value_cuda(&g->map, comm->recvbuf_vert[i]);
  pageranks[index] = comm->recvbuf_data_flt[i];
  comm->recvbuf_vert[i] = index;
}

__global__
void handle_update_sendbuf_data_cuda(dist_graph_t* g, mpi_data_t* comm)
{
  uint64_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= comm->total_send) return;

  uint64_t index = get_value_cuda(&g->map, comm->sendbuf_vert[i]);
  comm->sendbuf_vert[i] = index;
}

int run_pagerank(dist_graph_t* g, mpi_data_t* comm, 
                 double*& pageranks, uint32_t num_iter)
{ 
  if (debug) { printf("Task %d run_pagerank() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  //SET TO TRUE FOR FUNCTION BY FUNCTION RUNTIME
  bool timing = false;
  int device = 0;
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  device = procid % num_devices;

  assert(cudaSetDevice(device)==cudaSuccess);
  double* pageranks_next = NULL;
  assert(cudaMallocManaged(&pageranks_next, g->n_total*sizeof(double))==cudaSuccess);
  double* sum_noouts = NULL;
  assert(cudaMallocManaged(&sum_noouts, sizeof(double))==cudaSuccess);
  *sum_noouts = 0.0;
  double* sum_noouts_next = NULL;
  assert(cudaMallocManaged(&sum_noouts_next, sizeof(double))==cudaSuccess);
  *sum_noouts_next = 0.0;
  int* nprocs_gpu = NULL;
  assert(cudaMallocManaged(&nprocs_gpu, sizeof(int))==cudaSuccess);
  cudaMemcpy(nprocs_gpu, &nprocs, sizeof(int), cudaMemcpyHostToDevice);

  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  comm->global_queue_size = 1;

#pragma omp parallel default(shared)
{
  thread_comm_t* tc;
  assert(cudaMallocManaged(&tc, sizeof(thread_comm_t))==cudaSuccess);
  init_thread_comm_flt(tc);

  int thread_blocks_n_local = g->n_local / BLOCK_SIZE + 1;

  
  double time_check = 0.0;
  double begin_init_time = omp_get_wtime();
  
#pragma omp single
{
  cudaMemPrefetchAsync(g, sizeof(dist_graph_t), device, NULL);
  cudaMemPrefetchAsync(g->in_edges, g->m*sizeof(uint64_t), device, NULL);
  cudaMemPrefetchAsync(g->in_degree_list, (g->n+1)*sizeof(uint64_t), device, NULL);
  cudaMemPrefetchAsync(g->out_edges, g->m*sizeof(uint64_t), device, NULL);
  cudaMemPrefetchAsync(g->out_degree_list, (g->n+1)*sizeof(uint64_t), device, NULL);
  cudaMemPrefetchAsync(pageranks, g->n*sizeof(double), device, NULL);
  cudaMemPrefetchAsync(pageranks_next, g->n*sizeof(double), device, NULL);
  cudaMemPrefetchAsync(sum_noouts, sizeof(double), device, NULL);
  cudaMemPrefetchAsync(sum_noouts_next, sizeof(double), device, NULL);
  cudaDeviceSynchronize();
  pagerank_init<<<thread_blocks_n_local, BLOCK_SIZE>>>(g, pageranks, pageranks_next, sum_noouts, sum_noouts_next);
  cudaDeviceSynchronize();
}
  if (timing){
    time_check = omp_get_wtime() - begin_init_time;
    printf("Task %d pagerank_init() time %9.6f (s)\n", procid, time_check);
  }

  int thread_blocks_n_total = g->n_total / BLOCK_SIZE + 1;

  
  double update_pagerank_time = omp_get_wtime();
  
#pragma omp single
{
  cudaMemPrefetchAsync(g, sizeof(dist_graph_t), device, NULL);
  cudaMemPrefetchAsync(pageranks, g->n*sizeof(double), device, NULL);
  cudaDeviceSynchronize();
  update_pageranks<<<thread_blocks_n_total, BLOCK_SIZE>>>(g, pageranks);
  cudaDeviceSynchronize();
}
  if (timing){
    time_check = omp_get_wtime() - update_pagerank_time;
    printf("Task %d update_pageranks() time %9.6f (s)\n", procid, time_check);
  }

  
  double update_pageranks_next_time = omp_get_wtime();
  
#pragma omp single
{
  cudaMemPrefetchAsync(g, sizeof(dist_graph_t), device, NULL);
  cudaMemPrefetchAsync(pageranks, g->n*sizeof(double), device, NULL);
  cudaMemPrefetchAsync(pageranks_next, g->n*sizeof(double), device, NULL);
  cudaDeviceSynchronize();
  update_pageranks_next<<<thread_blocks_n_total, BLOCK_SIZE>>>(g, pageranks, pageranks_next);
  cudaDeviceSynchronize();
}
  if (timing){
      time_check = omp_get_wtime() - update_pageranks_next_time;
      printf("Task %d update_pageranks_next() time %9.6f (s)\n", procid, time_check);
    }
   double update_sendcounts_time = omp_get_wtime();
  
#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    update_sendcounts_thread_out(g, tc, i);

  for (int32_t i = 0; i < nprocs; ++i)
  {
#pragma omp atomic
    comm->sendcounts_temp[i] += tc->sendcounts_thread[i];

    tc->sendcounts_thread[i] = 0;
  }
#pragma omp barrier
  if (timing){
    time_check = omp_get_wtime() - update_sendcounts_time;
    printf("Task %d update_sendcounts() time %9.6f (s)\n", procid, time_check);
  }

#pragma omp single
{
  init_sendbuf_vid_data_flt(comm);
  init_recvbuf_vid_data_flt(comm);
}

  double update_vid_queues_time = omp_get_wtime();
  
#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    update_vid_data_queues_out(g, tc, comm, i, pageranks[i]);

  empty_vid_data_flt(tc, comm);
#pragma omp barrier
  if (timing){
    time_check = omp_get_wtime() - update_vid_queues_time;
    printf("Task %d update_vid_queues() time %9.6f (s)\n", procid, time_check);
  }

#pragma omp single
{
  exchange_verts(comm);
  exchange_data_flt(comm);

  *sum_noouts = 0.0;
  MPI_Allreduce(sum_noouts_next, sum_noouts, 1, 
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  *sum_noouts_next = 0.0;
}

  int thread_blocks_total_recv = comm->total_recv / BLOCK_SIZE + 1;

  double get_value_1_loops_time = omp_get_wtime();

#pragma omp single
{
  cudaMemPrefetchAsync(g, sizeof(dist_graph_t), device, NULL);
  cudaMemPrefetchAsync(comm, sizeof(mpi_data_t), device, NULL);
  cudaMemPrefetchAsync(pageranks, g->n*sizeof(double), device, NULL);
  cudaDeviceSynchronize();
  handle_update_recvbuf_data_cuda<<<thread_blocks_total_recv, BLOCK_SIZE>>>(g, comm, pageranks);
  cudaDeviceSynchronize();
}

  if (timing){
    time_check = omp_get_wtime() - get_value_1_loops_time;
    printf("Task %d get_value_1_loops() time %9.6f (s)\n", procid, time_check);
  }

  double get_value_2_loops_time = omp_get_wtime();
  int thread_blocks_total_send = comm->total_send / BLOCK_SIZE + 1;
#pragma omp single
{
  cudaMemPrefetchAsync(g, sizeof(dist_graph_t), device, NULL);
  cudaMemPrefetchAsync(comm, sizeof(mpi_data_t), device, NULL);
  cudaDeviceSynchronize();
  handle_update_sendbuf_data_cuda<<<thread_blocks_total_send, BLOCK_SIZE>>>(g, comm);
  cudaDeviceSynchronize();
}

  if (timing){
    time_check = omp_get_wtime() - get_value_2_loops_time;
    printf("Task %d get_value_2_loops() time %9.6f (s)\n", procid, time_check);
  }

  for (uint32_t iter = 0; iter < num_iter; ++iter)
  {
    if (debug && tc->tid == 0) {
      printf("Task %d Iter %u run_pagerank() sink contribution sum %e\n", procid, iter, *sum_noouts); 
    }

  
  double pagerank_iter_time = omp_get_wtime();
  
#pragma omp single
{   
  cudaMemPrefetchAsync(g, sizeof(dist_graph_t), device, NULL);
  cudaMemPrefetchAsync(g->in_edges, g->m*sizeof(uint64_t), device, NULL);
  cudaMemPrefetchAsync(g->in_degree_list, (g->n+1)*sizeof(uint64_t), device, NULL);
  cudaMemPrefetchAsync(g->out_edges, g->m*sizeof(uint64_t), device, NULL);
  cudaMemPrefetchAsync(g->out_degree_list, (g->n+1)*sizeof(uint64_t), device, NULL);
  cudaMemPrefetchAsync(pageranks, g->n*sizeof(double), device, NULL);
  cudaMemPrefetchAsync(pageranks_next, g->n*sizeof(double), device, NULL);
  cudaMemPrefetchAsync(sum_noouts, sizeof(double),device, NULL);
  cudaMemPrefetchAsync(sum_noouts_next, sizeof(double), device, NULL);
  cudaDeviceSynchronize();
  pagerank_iter_vert<<<thread_blocks_n_local, BLOCK_SIZE>>>(g, pageranks, pageranks_next, sum_noouts, sum_noouts_next);
  cudaDeviceSynchronize();
}
  if (timing){
    time_check = omp_get_wtime() - pagerank_iter_time;
    printf("Task %d pagerank_iter() time %9.6f (s)\n", procid, time_check);
  }

  double update_sendbuf_data_time = omp_get_wtime();
  
#pragma omp single
{
  cudaMemPrefetchAsync(comm, sizeof(comm), device, NULL);
  cudaMemPrefetchAsync(pageranks_next, g->n*sizeof(double), device, NULL);
  cudaDeviceSynchronize();
  update_sendbuf_data_flt<<<thread_blocks_total_send, BLOCK_SIZE>>>(comm, pageranks_next);
  cudaDeviceSynchronize();
}
  if (timing){
    time_check = omp_get_wtime() - update_sendbuf_data_time;
    printf("Task %d update_sendbuf_data() time %9.6f (s)\n", procid, time_check);
  }

#pragma omp single
{
    exchange_data_flt(comm);
    *sum_noouts = 0.0;
    MPI_Allreduce(sum_noouts_next, sum_noouts, 1, 
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    *sum_noouts_next = 0.0;
}
  
  double update_pageranks_next_recv_time = omp_get_wtime();
  
#pragma omp single
{
  cudaMemPrefetchAsync(comm, sizeof(comm), device, NULL);
  cudaMemPrefetchAsync(pageranks_next, g->n*sizeof(double), device, NULL);
  cudaDeviceSynchronize();
  int thread_blocks_total_recv = comm->total_recv / BLOCK_SIZE + 1;
  update_pageranks_next_recv<<<thread_blocks_total_recv, BLOCK_SIZE>>>(comm, pageranks_next);
  cudaDeviceSynchronize();
}
  if (timing){
    time_check = omp_get_wtime() - update_pageranks_next_recv_time;
    printf("Task %d update_pageranks_next_recv() time %9.6f (s)\n", procid, time_check);
  }

#pragma omp single
{
    double* temp = pageranks;
    pageranks = pageranks_next;
    pageranks_next = temp;
}
  } // end for loop

  clear_thread_comm(tc);
} // end parallel

  clear_allbuf_vid_data(comm);
  cudaFree(pageranks_next);
  cudaFree(sum_noouts);
  cudaFree(sum_noouts_next);

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d, run_pagerank() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d run_pagerank() success\n", procid); }

  return 0;
}


int pagerank_output(dist_graph_t* g, double* pageranks, char* output_file)
{
  if (debug) printf("Task %d pageranks to %s\n", procid, output_file); 

  double* global_pageranks = (double*)malloc(g->n*sizeof(double));
  
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_pageranks[i] = -1.0;

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t out_degree = out_degree(g, i);
    assert(g->local_unmap[i] < g->n);
    if (out_degree > 0)
      global_pageranks[g->local_unmap[i]] = pageranks[i] * (double)out_degree;
    else
      global_pageranks[g->local_unmap[i]] = pageranks[i] * (double)g->n;
  }

  if (procid == 0)
    MPI_Reduce(MPI_IN_PLACE, global_pageranks, (int32_t)g->n,
      MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  else
    MPI_Reduce(global_pageranks, global_pageranks, (int32_t)g->n,
      MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (procid == 0)
  {
    if (debug)
      for (uint64_t i = 0; i < g->n; ++i)
        if (global_pageranks[i] == -1.0)
        {
          printf("Pagerank error: %lu not assigned\n", i);
          global_pageranks[i] = 0.0;
        }
        
    std::ofstream outfile;
    outfile.open(output_file);

    for (uint64_t i = 0; i < g->n; ++i)
      outfile << global_pageranks[i] << std::endl;

    outfile.close();
  }

  free(global_pageranks);

  if (debug) printf("Task %d done writing pageranks\n", procid); 

  return 0;
}


int pagerank_verify(dist_graph_t* g, double* pageranks)
{
  if (debug) { printf("Task %d pagerank_verify() start\n", procid); }

  double* global_pageranks = (double*)malloc(g->n*sizeof(double));
  
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_pageranks[i] = -1.0;
   
#pragma omp parallel for 
  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t out_degree = out_degree(g, i);
    if (out_degree > 0)
      global_pageranks[g->local_unmap[i]] = pageranks[i] * (double)out_degree;
    else
      global_pageranks[g->local_unmap[i]] = pageranks[i] * (double)g->n;
  }

  if (procid == 0)
    MPI_Reduce(MPI_IN_PLACE, global_pageranks, (int32_t)g->n,
      MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  else
    MPI_Reduce(global_pageranks, global_pageranks, (int32_t)g->n,
      MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (procid == 0)
  {
    double pr_sum = 0.0;
    for (uint64_t i = 0; i < g->n; ++i)
      pr_sum += global_pageranks[i];

    printf("PageRanks sum (should be 1.0): %9.6lf\n", pr_sum);
  }

  free(global_pageranks);

  if (debug) { printf("Task %d pagerank_verify() success\n", procid); }

  return 0;
}

int pagerank_dist(dist_graph_t *g, mpi_data_t* comm, 
                  uint32_t num_iter, char* output_file)
{  
  if (debug) { printf("Task %d pagerank_dist() start\n", procid); }

  MPI_Barrier(MPI_COMM_WORLD); 
  double elt = omp_get_wtime();

  double* pageranks = NULL;
  assert(cudaMallocManaged(&pageranks, g->n_total*sizeof(double))==cudaSuccess);
  run_pagerank(g, comm, pageranks, num_iter);

  MPI_Barrier(MPI_COMM_WORLD); 
  elt = omp_get_wtime() - elt;
  if (procid == 0) printf("PageRank time %9.6f (s)\n", elt);

  if (output) {
    pagerank_output(g, pageranks, output_file);
  }

  if (verify) { 
    pagerank_verify(g, pageranks);
  }

  cudaFree(pageranks);

  if (debug)  printf("Task %d pagerank_dist() success\n", procid); 
  return 0;
}

