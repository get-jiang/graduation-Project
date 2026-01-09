#ifndef CPU_VERIFY_H_
#define CPU_VERIFY_H_

void cpu_sssp_reference(CSRGraph& hg, int src, node_data_type* cpu_dist);
void verify_results(node_data_type* gpu_results, node_data_type* cpu_results, unsigned n);

#endif /* CPU_VERIFY_H_ */
