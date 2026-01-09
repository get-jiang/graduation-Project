#include <queue>
#include <vector>
#include <limits>
// 必须包含你项目原有的头文件
#include "common.h"
#include "csr_graph.h"
#include "support.h"
// 定义一个用于优先队列的结构
struct Node {
    index_type id;
    node_data_type dist;

    // 优先队列默认是最大堆，我们需要最小堆，所以重载大于号
    bool operator>(const Node& other) const {
        return dist > other.dist;
    }
};



// 如果 INF 没有在 common.h 定义，这里补一个
#ifndef INF
#define INF 1000000000
#endif

/**
 * @brief CPU 版本的 Dijkstra 算法，用于产生参考答案
 * @param hg    主机端的 CSRGraph 引用
 * @param src   起始节点索引 (0-based)
 * @param cpu_dist 输出数组，长度需为 hg.nnodes
 */
void cpu_sssp_reference(CSRGraph& hg, int src, node_data_type* cpu_dist) {
    printf("Computing CPU reference SSSP (Dijkstra)... ");
    fflush(stdout);

    // 1. 初始化距离为 INF
    for (unsigned i = 0; i < hg.nnodes; i++) {
        cpu_dist[i] = INF;
    }
    cpu_dist[src] = 0;

    // 2. 定义最小堆优先队列
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    pq.push({(index_type)src, 0});

    while (!pq.empty()) {
        Node current = pq.top();
        pq.pop();

        index_type u = current.id;
        node_data_type d = current.dist;

        // 如果弹出的距离已经大于当前记录的距离，跳过（过期路径）
        if (d > cpu_dist[u]) continue;

        // 遍历邻居
        index_type start = hg.row_start[u];
        index_type end = hg.row_start[u + 1];

        for (index_type i = start; i < end; i++) {
            index_type v = hg.edge_dst[i];
            edge_data_type weight = hg.edge_data[i];
            node_data_type new_dist = cpu_dist[u] + weight;

            // 如果找到更短路径
            if (new_dist < cpu_dist[v]) {
                cpu_dist[v] = new_dist;
                pq.push({v, new_dist});
            }
        }
    }
    printf("Done.\n");
}

/**
 * @brief 校验 GPU 和 CPU 的结果
 */
void verify_results(node_data_type* gpu_results, node_data_type* cpu_results, unsigned n) {
    unsigned errors = 0;
    for (unsigned i = 0; i < n; i++) {
        if (gpu_results[i] != cpu_results[i]) {
            if (errors < 10) { // 只打印前 10 个错误
                printf("Mismatch at node %u: GPU=%u, CPU=%u\n", i, gpu_results[i], cpu_results[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("VERIFICATION PASSED! (All %u nodes match)\n", n);
    } else {
        printf("VERIFICATION FAILED! Total mismatches: %u / %u\n", errors, n);
    }
}