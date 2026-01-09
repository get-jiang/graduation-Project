
all: sssp  

sssp: kernel.cu support.cu csr_graph.cu bitonic_sort.cu
	nvcc $^ -O3 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70  -rdc=true -lcudadevrt -Xptxas -O3  -Xptxas -v -lcuda -lcudart -o $@

clean:
	rm -f sssp *.o


