/*
 csr_graph.cu

 Implements CSR Graph. Part of the GGC source code.

 Copyright (C) 2014--2016, The University of Texas at Austin

 See LICENSE.TXT for copyright license.

 Author: Sreepathi Pai <sreepai@ices.utexas.edu>
 */

/* -*- mode: c++ -*- */


#include "csr_graph.h"
#include "bitonic_sort.h"
#include <boost/mpl/if.hpp>
#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdint.h>
#include <vector>
#include <random>
#include <sstream>
#include <fcntl.h>
#include <cstdlib>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <utility> // for std::pair
#include <chrono>

static void check_cuda_error(const cudaError_t e, const char *file, const int line)
{
  if (e != cudaSuccess) {
    fprintf(stderr, "%s:%d: %s (%d)\n", file, line, cudaGetErrorString(e), e);
    exit(1);
  }
}

template <typename T>
static void check_retval(const T retval, const T expected, const char *file, const int line) {
  if(retval != expected) {
    fprintf(stderr, "%s:%d: Got %d, expected %d\n", file, line, retval, expected);
    exit(1);
  }
}

#define check_cuda(x) check_cuda_error(x, __FILE__, __LINE__)
#define check_rv(r, x) check_retval(r, x, __FILE__, __LINE__)


unsigned CSRGraph::init() {
	row_start = edge_dst = NULL;
	edge_data = NULL;
	node_data = NULL;
	old_to_new_mapping = NULL;
	nnodes = nedges = 0;
	device_graph = false;

	return 0;
}

unsigned CSRGraph::allocOnHost() {
	assert(nnodes > 0);
	assert(!device_graph);

	if (row_start != NULL) // already allocated
		return true;

	row_start = (index_type *) calloc(nnodes + 1, sizeof(index_type));
	edge_dst = (index_type *) calloc(nedges, sizeof(index_type));
	edge_data = (edge_data_type *) calloc(nedges, sizeof(edge_data_type));
	node_data = (node_data_type *) calloc(nnodes, sizeof(node_data_type));

	size_t mem_usage = ((nnodes + 1) + nedges) * sizeof(index_type)
			+ (nedges) * sizeof(edge_data_type)
			+ (nnodes) * sizeof(node_data_type);

	printf("Host memory for graph: %3u MB\n", mem_usage / 1048756);

	return (edge_data && row_start && edge_dst && node_data);
}

unsigned CSRGraph::allocOnDevice() {
	if (edge_dst != NULL)  // already allocated
		return true;

	assert(edge_dst == NULL); // make sure not already allocated

	check_cuda(cudaMalloc((void ** ) &edge_dst, nedges * sizeof(index_type)));
	check_cuda(
			cudaMalloc((void ** ) &row_start,
					(nnodes + 1) * sizeof(index_type)));

	check_cuda(
			cudaMalloc((void ** ) &edge_data, nedges * sizeof(edge_data_type)));
	check_cuda(
			cudaMalloc((void ** ) &node_data, nnodes * sizeof(node_data_type)));

	device_graph = true;

	return (edge_dst && edge_data && row_start && node_data);
}

void CSRGraphTex::copy_to_gpu(struct CSRGraphTex &copygraph) {
	copygraph.nnodes = nnodes;
	copygraph.nedges = nedges;

	assert(copygraph.allocOnDevice());

	check_cuda(
			cudaMemcpy(copygraph.edge_dst, edge_dst,
					nedges * sizeof(index_type), cudaMemcpyHostToDevice));
	check_cuda(
			cudaMemcpy(copygraph.edge_data, edge_data,
					nedges * sizeof(edge_data_type), cudaMemcpyHostToDevice));
	check_cuda(
			cudaMemcpy(copygraph.node_data, node_data,
					nnodes * sizeof(edge_data_type), cudaMemcpyHostToDevice));

	check_cuda(
			cudaMemcpy(copygraph.row_start, row_start,
					(nnodes + 1) * sizeof(index_type), cudaMemcpyHostToDevice));
}

unsigned CSRGraphTex::allocOnDevice() {
	if (CSRGraph::allocOnDevice()) {
		assert(sizeof(index_type) <= 4); // 32-bit only!
		assert(sizeof(node_data_type) <= 4); // 32-bit only!

		cudaResourceDesc resDesc;

		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
		resDesc.res.linear.desc.x = 32; // bits per channel

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;

		resDesc.res.linear.devPtr = edge_dst;
		resDesc.res.linear.sizeInBytes = nedges * sizeof(index_type);
		//check_cuda(cudaCreateTextureObject(&edge_dst_tx, &resDesc, &texDesc, NULL));

		resDesc.res.linear.devPtr = row_start;
		resDesc.res.linear.sizeInBytes = (nnodes + 1) * sizeof(index_type);
		//check_cuda(cudaCreateTextureObject(&row_start_tx, &resDesc, &texDesc, NULL));

		resDesc.res.linear.devPtr = node_data;
		resDesc.res.linear.sizeInBytes = (nnodes) * sizeof(node_data_type);
		check_cuda(
				cudaCreateTextureObject(&node_data_tx, &resDesc, &texDesc, NULL));

		return 1;
	}

	return 0;
}

unsigned CSRGraph::deallocOnHost() {
	if (!device_graph) {
		free(row_start);
		free(edge_dst);
		free(edge_data);
		free(node_data);
		if (old_to_new_mapping) free(old_to_new_mapping);
	}

	return 0;
}
unsigned CSRGraph::deallocOnDevice() {
	if (device_graph) {
		cudaFree(edge_dst);
		cudaFree(edge_data);
		cudaFree(row_start);
		cudaFree(node_data);
	}

	return 0;
}

static void verify_sorted_edges(const CSRGraph &g) {
    printf("Verifying edge sort...\n");
    bool sorted = true;
    for (index_type i = 0; i < g.nnodes; ++i) {
        index_type start = g.row_start[i];
        index_type end = g.row_start[i+1];
        for (index_type j = start; j < end - 1; ++j) {
            if (g.edge_data[j] > g.edge_data[j+1]) {
                printf("Error: Edges for node %u not sorted at index %u (%d > %d)\n", 
                       i, j, g.edge_data[j], g.edge_data[j+1]);
                sorted = false;
                break; 
            }
        }
        if (!sorted) break;
    }
    if (sorted) {
        printf("Verification passed: All edges sorted by weight.\n");
    } else {
        printf("Verification failed!\n");
    }
}

void CSRGraph::apply_property_driven_reordering(int &start_node_ref, bool do_vertex_reorder) {
    printf("Applying Property-driven Reordering (PRO)...\n");

	// Debug: Check if graph is already sorted
    printf("First 10 nodes degrees before reordering:\n");
    for(int i=0; i<10 && i<nnodes; ++i) {
        printf("Node %d: %d\n", i, row_start[i+1]-row_start[i]);
    }

    // --- Step 1: Vertex Reordering (Descending Degree) ---

    // 1. Store pairs of (degree, old_id)
    std::vector<std::pair<index_type, index_type>> node_degrees(nnodes);
    for (index_type i = 0; i < nnodes; ++i) {
        index_type deg = row_start[i + 1] - row_start[i];
        node_degrees[i] = std::make_pair(deg, i);
    }

    // 2. Sort by degree descending
    std::sort(node_degrees.begin(), node_degrees.end(),
              [](const std::pair<index_type, index_type> &a, const std::pair<index_type, index_type> &b) {
                  return a.first > b.first;
              });

    // 3. Create mapping tables
    old_to_new_mapping = (index_type *)malloc(nnodes * sizeof(index_type));
    std::vector<index_type> new_to_old(nnodes);
    if (do_vertex_reorder) {
        for (index_type new_id = 0; new_id < nnodes; ++new_id) {
            index_type old_id = node_degrees[new_id].second;
            new_to_old[new_id] = old_id;
            old_to_new_mapping[old_id] = new_id;
        }
    } else {
        // Identity mapping: no vertex reordering
        for (index_type i = 0; i < nnodes; ++i) {
            new_to_old[i] = i;
            old_to_new_mapping[i] = i;
        }
    }

    // 4. Update the start_node to its new ID
    printf("  Remapping start_node %d -> %d\n", start_node_ref, old_to_new_mapping[start_node_ref]);
    start_node_ref = old_to_new_mapping[start_node_ref];

    // 5. Allocate new CSR arrays
    index_type *new_row_start = (index_type *)calloc(nnodes + 1, sizeof(index_type));
    index_type *new_edge_dst = (index_type *)calloc(nedges, sizeof(index_type));
    edge_data_type *new_edge_data = (edge_data_type *)calloc(nedges, sizeof(edge_data_type));

    // 6. Rebuild Graph
    new_row_start[0] = 0;
    for (index_type i = 0; i < nnodes; ++i) {
        index_type u_old = new_to_old[i];
        // index_type degree = node_degrees[i].first;
        index_type degree = row_start[u_old + 1] - row_start[u_old];
        
        // Set row pointer
        new_row_start[i + 1] = new_row_start[i] + degree;

        // Copy edges
        index_type old_edge_start = row_start[u_old];
        index_type new_edge_start = new_row_start[i];

        for (index_type j = 0; j < degree; ++j) {
            index_type v_old = edge_dst[old_edge_start + j];
            edge_data_type w = edge_data[old_edge_start + j];

            // Translate destination to new ID
            new_edge_dst[new_edge_start + j] = old_to_new_mapping[v_old];
            new_edge_data[new_edge_start + j] = w;
        }
    }

    // 7. Replace old arrays
    free(row_start); row_start = new_row_start;
    free(edge_dst);  edge_dst = new_edge_dst;
    free(edge_data); edge_data = new_edge_data;
    
    // Re-init node_data (it's just scratch space, but needs to be clean)
    free(node_data); 
    node_data = (node_data_type *)calloc(nnodes, sizeof(node_data_type));


    // --- Step 2: Edge Reordering (Ascending Weight) ---

    auto edge_sort_start = std::chrono::high_resolution_clock::now();
    bitonic_edge_sort_gpu(*this);
    auto edge_sort_end = std::chrono::high_resolution_clock::now();
    double edge_sort_time_ms = std::chrono::duration<double, std::milli>(edge_sort_end - edge_sort_start).count();
    printf("Edge reordering time: %.3f ms\n", edge_sort_time_ms);

    verify_sorted_edges(*this);

    // index_type max_degree = 0;
    // for(index_type i = 0; i < nnodes; ++i) {
    //     index_type deg = row_start[i+1] - row_start[i];
    //     if(deg > max_degree) max_degree = deg;
    // }
    // std::vector<std::pair<edge_data_type, index_type>> edge_buffer(max_degree);

    // for (index_type i = 0; i < nnodes; ++i) {
    //     index_type start = row_start[i];
    //     index_type end = row_start[i + 1];
    //     index_type degree = end - start;

    //     if (degree > 1) {
    //         // Copy to buffer
    //         for (index_type j = 0; j < degree; ++j) {
    //             edge_buffer[j] = std::make_pair(edge_data[start + j], edge_dst[start + j]);
    //         }

    //         // Sort by weight (first element of pair)
    //         std::sort(edge_buffer.begin(), edge_buffer.begin() + degree);

    //         // Write back sorted edges
    //         for (index_type j = 0; j < degree; ++j) {
    //             edge_data[start + j] = edge_buffer[j].first;
    //             edge_dst[start + j] = edge_buffer[j].second;
    //         }
    //     }
    // }

    printf("PRO completed.\n");
}

CSRGraph::CSRGraph() {
	init();
}

void CSRGraph::progressPrint(unsigned maxii, unsigned ii) {
	const unsigned nsteps = 10;
	unsigned ineachstep = (maxii / nsteps);
	if (ineachstep == 0)
		ineachstep = 1;
	/*if (ii == maxii) {
	 printf("\t100%%\n");
	 } else*/if (ii % ineachstep == 0) {
		int progress = ((size_t) ii * 100) / maxii + 1;

		printf("\t%3d%%\r", progress);
		fflush(stdout);
	}
}

unsigned CSRGraph::readFromGR(char file[]) {
    std::ifstream cfile;
    cfile.open(file);

    int masterFD = open(file, O_RDONLY);
    if (masterFD == -1) {
        printf("FileGraph::structureFromFile: unable to open %s.\n", file);
        return 1;
    }

    struct stat buf;
    if (fstat(masterFD, &buf) == -1) {
        printf("FileGraph::structureFromFile: unable to stat %s.\n", file);
        close(masterFD);
        return 1;
    }
    size_t masterLength = buf.st_size;

    void* m = mmap(0, masterLength, PROT_READ, MAP_PRIVATE, masterFD, 0);
    if (m == MAP_FAILED) {
        printf("FileGraph::structureFromFile: mmap failed.\n");
        close(masterFD);
        return 1;
    }

    // 解析 Header
    uint64_t* fptr = (uint64_t*) m;
    uint64_t version = le64toh(*fptr++);
    assert(version == 1);
    uint64_t sizeEdgeTy = le64toh(*fptr++);
    uint64_t numNodes = le64toh(*fptr++);
    uint64_t numEdges = le64toh(*fptr++);

    // 打印元数据 (保留)
    printf("\n==== Graph Meta Data ====\n");
    printf("Nodes: %llu, Edges: %llu, EdgeWeightSize: %llu\n", numNodes, numEdges, sizeEdgeTy);

    // 节点偏移数组起始
    uint64_t *outIdx = fptr;
    fptr += numNodes;

    // 目标节点数组起始 (uint32)
    uint32_t *outs = (uint32_t*) fptr;
    
    // 权重数组起始位置计算：跳过目标节点数组
    // Galois 格式规定：如果 numEdges 是奇数，会填充 4 字节以对齐 8 字节
    uint32_t *fptr_wt = outs + numEdges;
    if (numEdges % 2 != 0) {
        fptr_wt += 1; 
    }
    void *edgeDataRaw = (void *) fptr_wt;

    // 分配主机内存
    nnodes = (unsigned)numNodes;
    nedges = (unsigned)numEdges;
    allocOnHost();

    row_start[0] = 0;

    // 加载 CSR 结构
    for (unsigned ii = 0; ii < nnodes; ++ii) {
        row_start[ii + 1] = (index_type)le64toh(outIdx[ii]);
        index_type degree = row_start[ii + 1] - row_start[ii];

        for (unsigned jj = 0; jj < degree; ++jj) {
            unsigned edgeindex = row_start[ii] + jj;

            // 读取目标节点
            edge_dst[edgeindex] = le32toh(outs[edgeindex]);

            // 根据 sizeEdgeTy 的实际宽度读取权重，防止 1,0 交替错误
            if (sizeEdgeTy == 8) {
                uint64_t *edgeData64 = (uint64_t *) edgeDataRaw;
                edge_data[edgeindex] = (edge_data_type)le64toh(edgeData64[edgeindex]);
            } else if (sizeEdgeTy == 4) {
                uint32_t *edgeData32 = (uint32_t *) edgeDataRaw;
                edge_data[edgeindex] = (edge_data_type)le32toh(edgeData32[edgeindex]);
            } else {
                // 如果没有权重定义，默认为 1
                edge_data[edgeindex] = 1;
            }
        }
    }

    // 提取文件名
    {
        char path_copy[1024];
        strncpy(path_copy, file, sizeof(path_copy));
        char* last_slash = strrchr(path_copy, '/');
        if (last_slash) {
            strcpy(file_name, last_slash + 1);
        } else {
            strcpy(file_name, path_copy);
        }
    }

    printf("Successfully read %u nodes and %u edges.\n\n", nnodes, nedges);

    // 清理资源
    munmap(m, masterLength);
    close(masterFD);
    cfile.close();

    return 0;
}

unsigned CSRGraph::read(char file[]) {
	return readFromGR(file);
}

void CSRGraph::dealloc() {
	if (device_graph)
		deallocOnDevice();
	else
		deallocOnHost();
}

void CSRGraph::copy_to_gpu(struct CSRGraph &copygraph) {
	copygraph.nnodes = nnodes;
	copygraph.nedges = nedges;

	assert(copygraph.allocOnDevice());

	check_cuda(
			cudaMemcpy(copygraph.edge_dst, edge_dst,
					nedges * sizeof(index_type), cudaMemcpyHostToDevice));
	check_cuda(
			cudaMemcpy(copygraph.edge_data, edge_data,
					nedges * sizeof(edge_data_type), cudaMemcpyHostToDevice));
	check_cuda(
			cudaMemcpy(copygraph.node_data, node_data,
					nnodes * sizeof(edge_data_type), cudaMemcpyHostToDevice));

	check_cuda(
			cudaMemcpy(copygraph.row_start, row_start,
					(nnodes + 1) * sizeof(index_type), cudaMemcpyHostToDevice));
}

void CSRGraph::copy_to_cpu(struct CSRGraph &copygraph) {
	assert(device_graph);

	// cpu graph is not allocated
	assert(copygraph.nnodes = nnodes);
	assert(copygraph.nedges = nedges);

	check_cuda(
			cudaMemcpy(copygraph.edge_dst, edge_dst,
					nedges * sizeof(index_type), cudaMemcpyDeviceToHost));
	check_cuda(
			cudaMemcpy(copygraph.edge_data, edge_data,
					nedges * sizeof(edge_data_type), cudaMemcpyDeviceToHost));
	check_cuda(
			cudaMemcpy(copygraph.node_data, node_data,
					nnodes * sizeof(edge_data_type), cudaMemcpyDeviceToHost));

	check_cuda(
			cudaMemcpy(copygraph.row_start, row_start,
					(nnodes + 1) * sizeof(index_type), cudaMemcpyDeviceToHost));
}

struct EdgeIterator {
	CSRGraph *g;
	index_type node;
	index_type s;

	__device__ EdgeIterator(CSRGraph& g, index_type node) {
		this->g = &g;
		this->node = node;
	}

	__device__
	index_type size() const {
		return g->row_start[node + 1] - g->row_start[node];
	}

	__device__
	index_type start() {
		s = g->row_start[node];
		return s;
	}

	__device__
	index_type end() const {
		return g->row_start[node + 1];
	}

	__device__
	void next() {
		s++;
	}

	__device__
	index_type dst() const {
		return g->edge_dst[s];
	}

	__device__
	edge_data_type data() const {
		return g->edge_data[s];
	}
};

