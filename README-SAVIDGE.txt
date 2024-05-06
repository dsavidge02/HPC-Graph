Here are some of the code updates and changes I made file by file:

main.cu:
Originally main.cpp, changed to a cuda file so that the graph (dist_graph_t) and communication structure (comm) could have
their memory allocated in the shared memory space.

dist_graph.cu:
Originally dist_graph.cpp, changed to a cuda file so that all internal graph information could be allocated in the shared
memory space.

comms.cu:
Originally comms.cpp, changed to a cuda file so that all communication structures could be allocated in the shared memory
space. When allocating send and recv buffers, added a check to see if total_send or total_comm was 0 so there wouldn't
be an error in the memory allocation.

fast_map.cu:
Originally fast_map.cpp, changed to a cuda file so that all communication structures could be allocated in the shared
memory space. Another change was made to the get_value function that was written in the header file, this change was
implemented in the pagerank.cu file for simplicity.

pagerank.cu:
Originally pagerank.cpp, changed to a cuda file so that the main operations could be performed on GPU. The main pagerank
algorithms occur in pagerank_init and pagerank_iter_vert. The functions update_pageranks and update_pageranks_next come
from original #pragma omp for loops written in the .cpp version. The same goes for the two functions update_sendbuf_data_flt
and update_pageranks_next_recv. The device functions, mult_hash_cuda and get_value_cuda were both originally written in
the fast_map.h file, but were slightly changed for use by the GPU. The get_value_cuda function was also changed so 
that it implemented quadratic probing, rather than linear probing. The handle_update_recvbuf_data_cuda and
handle_update_sendbuf_data_cuda are both for loops rewritten for GPU use.