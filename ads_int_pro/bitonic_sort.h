#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

#include "csr_graph.h"

void bitonic_edge_sort_gpu(CSRGraph &g);
void bitonic_edge_sort_cpu(CSRGraph &g);
void thrust_edge_sort_gpu(CSRGraph &g);
void thrust_edge_sort_gpu_batch(CSRGraph &g);

#endif
