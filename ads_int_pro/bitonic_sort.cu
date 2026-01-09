#include "bitonic_sort.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>

#define MAX_DEGREE_SHARED 2048
#define THREADS_PER_BLOCK 512
#define WARP_SIZE 32

__device__ __forceinline__ bool edge_less_than(edge_data_type w1, index_type d1, 
                                                edge_data_type w2, index_type d2) {
    return (w1 < w2) || (w1 == w2 && d1 < d2);
}

// Kernel for very small degree nodes - one warp per node
__global__ void bitonic_sort_warp_kernel(index_type *row_start, index_type *edge_dst, 
                                          edge_data_type *edge_data, index_type nnodes) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    
    if (warp_id >= nnodes) return;
    
    index_type start = row_start[warp_id];
    index_type end = row_start[warp_id + 1];
    int degree = end - start;
    
    if (degree <= 1 || degree > WARP_SIZE) return;
    
    edge_data_type weight = (lane < degree) ? edge_data[start + lane] : INT_MAX;
    index_type dst = (lane < degree) ? edge_dst[start + lane] : UINT32_MAX;
    
    int n = 1;
    while (n < degree) n <<= 1;
    
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            edge_data_type partner_w = __shfl_xor_sync(0xFFFFFFFF, weight, j);
            index_type partner_d = __shfl_xor_sync(0xFFFFFFFF, dst, j);
            
            unsigned int partner_lane = lane ^ j;
            bool ascending = ((lane & k) == 0);
            
            bool my_less = edge_less_than(weight, dst, partner_w, partner_d);
            bool should_swap;
            
            if (lane < partner_lane) {
                should_swap = ascending ? !my_less : my_less;
            } else {
                should_swap = ascending ? my_less : !my_less;
            }
            
            if (should_swap) {
                weight = partner_w;
                dst = partner_d;
            }
        }
    }
    
    if (lane < degree) {
        edge_data[start + lane] = weight;
        edge_dst[start + lane] = dst;
    }
}


__global__ void bitonic_sort_optimized_kernel(index_type *row_start, index_type *edge_dst, 
                                               edge_data_type *edge_data, index_type nnodes) {
    __shared__ index_type s_dst[MAX_DEGREE_SHARED];
    __shared__ edge_data_type s_data[MAX_DEGREE_SHARED];

    for (index_type u = blockIdx.x; u < nnodes; u += gridDim.x) {
        index_type start = row_start[u];
        index_type end = row_start[u+1];
        index_type degree = end - start;

        if (degree <= WARP_SIZE || degree <= 1) continue;
        
        if (degree <= MAX_DEGREE_SHARED) {
            for (int i = threadIdx.x; i < degree; i += blockDim.x) {
                s_dst[i] = edge_dst[start + i];
                s_data[i] = edge_data[start + i];
            }
            __syncthreads();
            
            int n = 1;
            while (n < degree) n <<= 1;

            for (int k = 2; k <= n; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    for (int tid = threadIdx.x; tid < (n >> 1); tid += blockDim.x) {
                        int idx = 2 * tid - (tid & (j - 1));
                        int partner = idx + j;
                        
                        if (partner >= degree) continue;
                        
                        edge_data_type w1 = s_data[idx];
                        edge_data_type w2 = s_data[partner];
                        index_type dst1 = s_dst[idx];
                        index_type dst2 = s_dst[partner];
                        
                        bool ascending = ((idx & k) == 0);
                        bool should_swap = ascending ? 
                            !edge_less_than(w1, dst1, w2, dst2) : 
                            edge_less_than(w1, dst1, w2, dst2);
                        
                        if (should_swap) {
                            s_data[idx] = w2;
                            s_data[partner] = w1;
                            s_dst[idx] = dst2;
                            s_dst[partner] = dst1;
                        }
                    }
                    __syncthreads();
                }
            }

            for (int i = threadIdx.x; i < degree; i += blockDim.x) {
                edge_dst[start + i] = s_dst[i];
                edge_data[start + i] = s_data[i];
            }
            __syncthreads();
        }
    }
}

#define SHARED_J_THRESHOLD 512

// Kernel for large j values - each thread handles one comparison in global memory
// Kernel for global memory bitonic sort step
__global__ void bitonic_step_global_kernel(edge_data_type *d_keys, index_type *d_values, 
                                            int degree, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    unsigned int ixj = i ^ j;
    
    if (i >= degree || ixj >= degree) return;
    
    if (ixj > i) {
        bool ascending = ((i & k) == 0);
        
        edge_data_type w_i = d_keys[i];
        edge_data_type w_ixj = d_keys[ixj];
        index_type dst_i = d_values[i];
        index_type dst_ixj = d_values[ixj];
        
        bool i_less = edge_less_than(w_i, dst_i, w_ixj, dst_ixj);
        bool should_swap = ascending ? !i_less : i_less;
        
        if (should_swap) {
            d_keys[i] = w_ixj;
            d_keys[ixj] = w_i;
            d_values[i] = dst_ixj;
            d_values[ixj] = dst_i;
        }
    }
}

// Optimized kernel: merge multiple small j steps using shared memory
__global__ void bitonic_merge_local_kernel(edge_data_type *d_keys, index_type *d_values,
                                            int degree, int k, int start_j) {
    extern __shared__ char shared_mem[];
    edge_data_type *s_keys = (edge_data_type*)shared_mem;
    index_type *s_values = (index_type*)(shared_mem + blockDim.x * sizeof(edge_data_type));
    
    int block_start = blockIdx.x * blockDim.x;
    int local_idx = threadIdx.x;
    int global_idx = block_start + local_idx;
    
    if (global_idx < degree) {
        s_keys[local_idx] = d_keys[global_idx];
        s_values[local_idx] = d_values[global_idx];
    }
    __syncthreads();
    
    for (int j = start_j; j > 0; j >>= 1) {
        if (j < blockDim.x) {
            int partner_local = local_idx ^ j;
            if (partner_local >= 0 && partner_local < blockDim.x) {
                int my_global = block_start + local_idx;
                int partner_global = block_start + partner_local;
                
                if (local_idx < partner_local && my_global < degree && partner_global < degree) {
                    bool ascending = ((my_global & k) == 0);
                    
                    edge_data_type w1 = s_keys[local_idx];
                    edge_data_type w2 = s_keys[partner_local];
                    index_type d1 = s_values[local_idx];
                    index_type d2 = s_values[partner_local];
                    
                    bool less = edge_less_than(w1, d1, w2, d2);
                    bool should_swap = ascending ? !less : less;
                    
                    if (should_swap) {
                        s_keys[local_idx] = w2;
                        s_keys[partner_local] = w1;
                        s_values[local_idx] = d2;
                        s_values[partner_local] = d1;
                    }
                }
            }
        }
        __syncthreads();
    }
    
    if (global_idx < degree) {
        d_keys[global_idx] = s_keys[local_idx];
        d_values[global_idx] = s_values[local_idx];
    }
}

static inline int next_power_of_2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Sort a single large-degree node's edges using optimized global memory bitonic sort
static void sort_large_node_gpu_optimized(edge_data_type *h_keys, index_type *h_values, int degree,
                                           edge_data_type *d_keys, index_type *d_values) {
    int N = next_power_of_2(degree);
    
    cudaMemcpy(d_keys, h_keys, degree * sizeof(edge_data_type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, degree * sizeof(index_type), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * (sizeof(edge_data_type) + sizeof(index_type));
    
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; ) {
            if (j >= threadsPerBlock) {
                bitonic_step_global_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_keys, d_values, degree, j, k);
                j >>= 1;
            } else {
                bitonic_merge_local_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
                    d_keys, d_values, degree, k, j);
                break;
            }
        }
    }
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_keys, d_keys, degree * sizeof(edge_data_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values, d_values, degree * sizeof(index_type), cudaMemcpyDeviceToHost);
}


void bitonic_edge_sort_gpu(CSRGraph &g) {
    printf("Applying Optimized Bitonic Sort on GPU (3-tier hybrid)...\n");

    index_type *d_row_start, *d_edge_dst;
    edge_data_type *d_edge_data;
    
    cudaMalloc(&d_row_start, (g.nnodes + 1) * sizeof(index_type));
    cudaMalloc(&d_edge_dst, g.nedges * sizeof(index_type));
    cudaMalloc(&d_edge_data, g.nedges * sizeof(edge_data_type));
    
    cudaMemcpy(d_row_start, g.row_start, (g.nnodes + 1) * sizeof(index_type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_dst, g.edge_dst, g.nedges * sizeof(index_type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_data, g.edge_data, g.nedges * sizeof(edge_data_type), cudaMemcpyHostToDevice);
    
    // Step 1: Sort very small degree nodes (<=32) using warp shuffle - VERY FAST
    // Each warp handles one node, so we need (nnodes * 32) threads
    {
        int warps_needed = g.nnodes;
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / WARP_SIZE;
        int blocks = (warps_needed + warps_per_block - 1) / warps_per_block;
        blocks = min(blocks, 65535);
        
        bitonic_sort_warp_kernel<<<blocks, threads_per_block>>>(
            d_row_start, d_edge_dst, d_edge_data, g.nnodes);
    }
    
    // Step 2: Sort medium degree nodes (32 < degree <= 2048) using shared memory
    {
        int num_blocks = min((int)g.nnodes, 65535);
        
        bitonic_sort_optimized_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_row_start, d_edge_dst, d_edge_data, g.nnodes);
    }
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(g.edge_dst, d_edge_dst, g.nedges * sizeof(index_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(g.edge_data, d_edge_data, g.nedges * sizeof(edge_data_type), cudaMemcpyDeviceToHost);
    
    cudaFree(d_row_start);
    cudaFree(d_edge_dst);
    cudaFree(d_edge_data);
    
    // Step 3: Find and sort large-degree nodes (> 2048)
    index_type max_large_degree = 0;
    int large_nodes = 0;
    for (index_type u = 0; u < g.nnodes; ++u) {
        index_type degree = g.row_start[u + 1] - g.row_start[u];
        if (degree > MAX_DEGREE_SHARED) {
            large_nodes++;
            if (degree > max_large_degree) max_large_degree = degree;
        }
    }
    
    if (large_nodes > 0) {
        int max_N = next_power_of_2(max_large_degree);
        
        edge_data_type *d_keys;
        index_type *d_values;
        cudaMalloc(&d_keys, max_N * sizeof(edge_data_type));
        cudaMalloc(&d_values, max_N * sizeof(index_type));
        
        for (index_type u = 0; u < g.nnodes; ++u) {
            index_type start = g.row_start[u];
            index_type end = g.row_start[u + 1];
            index_type degree = end - start;
            
            if (degree > MAX_DEGREE_SHARED) {
                std::vector<edge_data_type> keys(degree);
                std::vector<index_type> values(degree);
                for (index_type i = 0; i < degree; ++i) {
                    keys[i] = g.edge_data[start + i];
                    values[i] = g.edge_dst[start + i];
                }
                
                sort_large_node_gpu_optimized(keys.data(), values.data(), degree, d_keys, d_values);
                
                for (index_type i = 0; i < degree; ++i) {
                    g.edge_data[start + i] = keys[i];
                    g.edge_dst[start + i] = values[i];
                }
            }
        }
        
        cudaFree(d_keys);
        cudaFree(d_values);
        
        printf("Sorted %d large-degree nodes (degree > %d) using global memory bitonic sort.\n", 
               large_nodes, MAX_DEGREE_SHARED);
    }
    
    printf("Optimized Bitonic Sort on GPU completed.\n");
}

// ============================================================================
// CPU Sort Version (for comparison)
// ============================================================================

struct EdgePair {
    edge_data_type weight;
    index_type dst;
    
    bool operator<(const EdgePair &other) const {
        if (weight == other.weight) return dst < other.dst;
        return weight < other.weight;
    }
};

void bitonic_edge_sort_cpu(CSRGraph &g) {
    printf("Applying std::sort on CPU...\n");
    
    for (index_type u = 0; u < g.nnodes; ++u) {
        index_type start = g.row_start[u];
        index_type end = g.row_start[u + 1];
        index_type degree = end - start;
        
        if (degree > 1) {
            std::vector<EdgePair> edges(degree);
            for (index_type i = 0; i < degree; ++i) {
                edges[i].weight = g.edge_data[start + i];
                edges[i].dst = g.edge_dst[start + i];
            }
            
            std::sort(edges.begin(), edges.end());
            
            for (index_type i = 0; i < degree; ++i) {
                g.edge_data[start + i] = edges[i].weight;
                g.edge_dst[start + i] = edges[i].dst;
            }
        }
    }
    
    printf("CPU sort completed.\n");
}

// ============================================================================
// Thrust Sort Version (using NVIDIA Thrust library)
// ============================================================================

struct EdgeComparator {
    __host__ __device__
    bool operator()(const thrust::tuple<edge_data_type, index_type>& a,
                    const thrust::tuple<edge_data_type, index_type>& b) const {
        edge_data_type w_a = thrust::get<0>(a);
        edge_data_type w_b = thrust::get<0>(b);
        if (w_a != w_b) return w_a < w_b;
        return thrust::get<1>(a) < thrust::get<1>(b);
    }
};

struct SegmentedEdgeComparator {
    __host__ __device__
    bool operator()(const thrust::tuple<index_type, edge_data_type, index_type>& a,
                    const thrust::tuple<index_type, edge_data_type, index_type>& b) const {
        if (thrust::get<0>(a) != thrust::get<0>(b)) 
            return thrust::get<0>(a) < thrust::get<0>(b);
        if (thrust::get<1>(a) != thrust::get<1>(b)) 
            return thrust::get<1>(a) < thrust::get<1>(b);
        return thrust::get<2>(a) < thrust::get<2>(b);
    }
};

void thrust_edge_sort_gpu(CSRGraph &g) {
    printf("Applying Thrust Sort on GPU...\n");
    
    thrust::device_vector<edge_data_type> d_edge_data(g.edge_data, g.edge_data + g.nedges);
    thrust::device_vector<index_type> d_edge_dst(g.edge_dst, g.edge_dst + g.nedges);
    thrust::device_vector<index_type> d_row_start(g.row_start, g.row_start + g.nnodes + 1);
    
    for (index_type u = 0; u < g.nnodes; ++u) {
        index_type start = g.row_start[u];
        index_type end = g.row_start[u + 1];
        index_type degree = end - start;
        
        if (degree > 1) {
            auto keys_begin = thrust::make_zip_iterator(
                thrust::make_tuple(d_edge_data.begin() + start, d_edge_dst.begin() + start));
            auto keys_end = thrust::make_zip_iterator(
                thrust::make_tuple(d_edge_data.begin() + end, d_edge_dst.begin() + end));
            
            thrust::sort(keys_begin, keys_end, EdgeComparator());
        }
    }
    
    thrust::copy(d_edge_data.begin(), d_edge_data.end(), g.edge_data);
    thrust::copy(d_edge_dst.begin(), d_edge_dst.end(), g.edge_dst);
    
    printf("Thrust Sort on GPU completed.\n");
}

// Batch version: sort all nodes in parallel using segmented sort
void thrust_edge_sort_gpu_batch(CSRGraph &g) {
    printf("Applying Thrust Segmented Sort on GPU...\n");
    
    thrust::device_vector<edge_data_type> d_edge_data(g.edge_data, g.edge_data + g.nedges);
    thrust::device_vector<index_type> d_edge_dst(g.edge_dst, g.edge_dst + g.nedges);
    
    thrust::device_vector<index_type> d_segment_keys(g.nedges);
    
    std::vector<index_type> h_segment_keys(g.nedges);
    for (index_type u = 0; u < g.nnodes; ++u) {
        index_type start = g.row_start[u];
        index_type end = g.row_start[u + 1];
        for (index_type i = start; i < end; ++i) {
            h_segment_keys[i] = u;
        }
    }
    thrust::copy(h_segment_keys.begin(), h_segment_keys.end(), d_segment_keys.begin());
    
    typedef thrust::tuple<index_type, edge_data_type, index_type> SortKey;
    
    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(d_segment_keys.begin(), d_edge_data.begin(), d_edge_dst.begin()));
    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(d_segment_keys.end(), d_edge_data.end(), d_edge_dst.end()));
    
    thrust::sort(zip_begin, zip_end, SegmentedEdgeComparator());
    
    thrust::copy(d_edge_data.begin(), d_edge_data.end(), g.edge_data);
    thrust::copy(d_edge_dst.begin(), d_edge_dst.end(), g.edge_dst);
    
    printf("Thrust Segmented Sort on GPU completed.\n");
}
