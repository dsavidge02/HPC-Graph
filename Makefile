CXX = mpicxx
CXXFLAGS =  -fopenmp -Wextra -O0 -g -std=c++11
# LINKFLAGS = -L/usr/local/cuda/lib64 -lcudart
CUDA = nvcc
CUDAFLAGS = -g -G -O0 -arch=sm_70 -lpthread -lgomp -lmpi_cxx -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
TARGET = hpcgraph
TOCOMPILE = util.o fast_map.o dist_graph.o comms.o io_pp.o wcc.o scc.o labelprop.o harmonic.o kcore.o main.o pagerank.o


all: $(TOCOMPILE)
	$(CUDA) $(CUDAFLAGS) -o $(TARGET) $(TOCOMPILE)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $*.cpp

main.o: main.cu
	$(CUDA) $(CUDAFLAGS) -c main.cu

comms.o: comms.cu
	$(CUDA) $(CUDAFLAGS) -c comms.cu

pagerank.o: pagerank.cu
	$(CUDA) $(CUDAFLAGS) -c pagerank.cu

dist_graph.o: dist_graph.cu
	$(CUDA) $(CUDAFLAGS) -c dist_graph.cu

clean:
	rm -f *.o $(TARGET)


