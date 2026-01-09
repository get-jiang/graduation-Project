/*  -*- mode: c++ -*-  */
#include <cuda.h>
#include <inttypes.h>
#include <stdio.h>
#include "common.h"
#include "csr_graph.h"
#include "support.h"
#include "cpu_verify.h"

#define TB_SIZE 512
#define WARP_SIZE 32
int CUDA_DEVICE = 0;
int start_node = 0;
char *INPUT, *OUTPUT;

// 1. 初始化距离和访问标志
__global__ void init_kernel(CSRGraph graph, int src) {
    unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < graph.nnodes) {
        // 关键点：显式强制设置，确保除了 src 外全是 INF
        if (tid == (unsigned)src) {
            graph.node_data[tid] = 0;
        } else {
            graph.node_data[tid] = INF; // 使用具体的数值 10^9 避免宏定义冲突
        }
    }
}

// 2. 核心 SSSP Kernel：Warp 协作版
__global__ void sssp_warp_kernel(
    CSRGraph graph, 
    index_type* in_queue, uint32_t in_size, 
    index_type* out_queue, uint32_t* out_size,
    int* visited) 
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_id = tid / WARP_SIZE;
    uint32_t lane_id = tid % WARP_SIZE;

    // 每个 Warp 处理队列中的一个活跃节点
    if (warp_id < in_size) {
        index_type u = in_queue[warp_id];
        node_data_type u_dist = graph.node_data[u];
        
        index_type start = graph.row_start[u];
        index_type end = graph.row_start[u + 1];

        // Warp 内 32 个线程共同分担邻居遍历
        for (index_type i = start + lane_id; i < end; i += WARP_SIZE) {
            index_type v = graph.edge_dst[i];
            edge_data_type wt = graph.edge_data[i];
            node_data_type new_dist = u_dist + wt;

            // 使用原子操作尝试更新最短路径
            if (atomicMin(&(graph.node_data[v]), new_dist) > new_dist) {
                // 成功更新后，如果该点本轮未入队，则将其加入下一轮队列
                if (atomicExch(&visited[v], 1) == 0) {
                    uint32_t pos = atomicAdd(out_size, 1);
                    out_queue[pos] = v;
                }
            }
        }
    }
}

// 3. 重置访问位
__global__ void clear_visited_kernel(index_type* out_queue, uint32_t out_size, int* visited) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < out_size) {
        visited[out_queue[tid]] = 0;
    }
}

/* 修正后的 gg_main */
void gg_main(CSRGraph& hg, CSRGraph& gg) {
    // 1. 处理起始节点索引
    // 假设命令行传入的是 1，而程序内部需要的是 0
    int internal_src = start_node; 
    if (internal_src >= 1) internal_src -= 1; // 如果传入 1-based，转为 0-based
    
    printf("Running SSSP from internal source index: %d (original input: %d)\n", internal_src, start_node);

    // 2. 分配辅助队列
    index_type *d_q1, *d_q2;
    uint32_t *d_out_size;
    int *d_visited;
    cudaMalloc(&d_q1, hg.nnodes * sizeof(index_type));
    cudaMalloc(&d_q2, hg.nnodes * sizeof(index_type));
    cudaMalloc(&d_out_size, sizeof(uint32_t));
    cudaMalloc(&d_visited, hg.nnodes * sizeof(int));

    // 3. 多轮运行（对应你的 RUN_LOOP）
    for (int loop = 0; loop < 1; loop++) { // 先跑 1 轮测试正确性
        // 显式初始化显存为 0
        cudaMemset(d_visited, 0, hg.nnodes * sizeof(int));
        
        // 核心：初始化距离数组
        init_kernel<<<(hg.nnodes + 255) / 256, 256>>>(gg, internal_src);
        cudaDeviceSynchronize();

        // 将校准后的起始点放入队列
        index_type h_src = (index_type)internal_src;
        cudaMemcpy(d_q1, &h_src, sizeof(index_type), cudaMemcpyHostToDevice);
        uint32_t h_in_size = 1;

        float elapsed_time;
		cudaEvent_t start_event, stop_event;
		cudaEventCreate(&start_event);
		cudaEventCreate(&stop_event);
		cudaEventRecord(start_event, 0);

        int iter = 0;
        while (h_in_size > 0) {
            cudaMemset(d_out_size, 0, sizeof(uint32_t));

            uint32_t threads = 256;
            uint32_t blocks = (h_in_size * 32 + threads - 1) / threads;

            // 调用上一个回答中的 sssp_warp_kernel
            sssp_warp_kernel<<<blocks, threads>>>(gg, d_q1, h_in_size, d_q2, d_out_size, d_visited);
            
            cudaMemcpy(&h_in_size, d_out_size, sizeof(uint32_t), cudaMemcpyDeviceToHost);

            if (h_in_size > 0) {
                clear_visited_kernel<<<(h_in_size + 255) / 256, 256>>>(d_q2, h_in_size, d_visited);
            }

            // 交换指针
            index_type* temp = d_q1; d_q1 = d_q2; d_q2 = temp;
            
            iter++;
            if (iter > hg.nnodes) break; 
        }
        printf("SSSP completed in %d iterations.\n", iter);

        cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
		printf("Measured time for sample = %.3fs\n", elapsed_time / 1000.0f);
    }

    // 最终同步并释放
    cudaDeviceSynchronize();
    cudaFree(d_q1); cudaFree(d_q2); cudaFree(d_out_size); cudaFree(d_visited);
}

int main(int argc, char *argv[]) {
	if (argc == 1) {
		usage(argc, argv);
		exit(1);
	}
	parse_args(argc, argv);
	cudaSetDevice(CUDA_DEVICE);
	CSRGraphTy g, gg;
	g.read(INPUT);
	g.copy_to_gpu(gg);
	gg_main(g, gg);
	gg.copy_to_cpu(g);

	node_data_type* cpu_reference = (node_data_type*)malloc(g.nnodes * sizeof(node_data_type));
    
    // 2. 运行 CPU 参考算法 (注意 start_node 的索引偏移)
    int internal_src = start_node;
    if (internal_src >= 1) internal_src -= 1; 
    
    cpu_sssp_reference(g, internal_src, cpu_reference);
    
    // 3. 对比 GPU 结果 (g.node_data) 和 CPU 结果 (cpu_reference)
    verify_results(g.node_data, cpu_reference, g.nnodes);
    
    // 4. 清理并保存
    free(cpu_reference);
    output(g, OUTPUT);
	return 0;
}
