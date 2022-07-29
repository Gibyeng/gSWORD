#pragma once
#include "../graph/graph.h"
#include  <string.h>
#include <cassert>
#include "computesetintersection.h"
#include "computesetintersection.cpp"
#include "../cub/cub/cub.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cstring>
#include <omp.h>
#define INVALID_VERTEX_ID 100000000

// parse parameters
void ParseInputPara (int opt, int argc, char * argv, char *optstring,std::string& input_data_graph_file, std::string& input_query_graph_file,  int& method, ui& step, ui&sample_time, ui& inter_count,std::string& output_file, ui& threadnum, ui& orderid){
	switch(opt){
		case 'd':{
			input_data_graph_file = std::string (optarg);
			std::cout << "input_data_graph_file " << input_data_graph_file<<std::endl;
			break;
		}
		case 'q':{
			input_query_graph_file = std::string (optarg);
			std::cout << "input_query_graph_file " << input_query_graph_file<<std::endl;
			break;
		}
		/*m is the method to be run*/
		case 'm':{
			method = atoi(optarg);
			std::cout << "method " << method<<std::endl;
			break;
		}
		case 's':{
			step = atoi(optarg);
			std::cout << "step " << step<<std::endl;
			break;
		}
		case 't': {
			sample_time = atoi(optarg);
			std::cout << "sample_time: " << sample_time<<std::endl;
			break;
		}
		case 'i':{
			inter_count = atoi(optarg);
			std::cout << "inter_count:  " << inter_count<<std::endl;
			break;
		}
		case 'o':{
			output_file = std::string(optarg);
			std::cout << "output_file: " << output_file<<std::endl;
			break;
		}
	
		case 'c':{
			threadnum = atoi(optarg);
			std::cout << "threadnum: " <<threadnum <<std::endl;
			break;
		}
		case 'e':{
			orderid = atoi(optarg);
			std::cout << "using matching order id:" << orderid << std::endl;
		}

	}
}

//  parse matching order
//  0:QSI 1:GQL 2:TSO 3:CFL 4:DPiso 5:CECI 6:RI 7:VF2PP 8:Spectrum
SelectQueryPlan(Graph* data_graph, Graph* query_graph,Edges ***edge_matrix,	ui* matching_order, ui* pivots, TreeNode* tso_tree,ui* tso_order,TreeNode* dpiso_tree, ui* dpiso_order,TreeNode*cfl_tree, ui* cfl_order ,std::vector<std::vector<ui>>& spectrum,ui* candidates_count,TreeNode* ceci_tree,ui* ceci_order,ui** weight_array){
	if (orderid == 0) {
		std::cout << "use QSI query plan..." << std::endl;
		generateQSIQueryPlan(data_graph, query_graph, edge_matrix, matching_order, pivots);
	} else if (orderid == 1) {
		std::cout << "use GQL query plan..." << std::endl;
		generateGQLQueryPlan(data_graph, query_graph, candidates_count, matching_order, pivots);
	} else if (orderid == 2) {
		if (tso_tree == NULL) {
			generateTSOFilterPlan(data_graph, query_graph, tso_tree, tso_order);
		}
		std::cout << "use TSO query plan..." << std::endl;
		generateTSOQueryPlan(query_graph, edge_matrix, matching_order, pivots, tso_tree, tso_order);
	} else if (orderid == 3){
		if (cfl_tree == NULL) {
			int level_count;
			ui* level_offset;
			generateCFLFilterPlan(data_graph, query_graph, cfl_tree, cfl_order, level_count, level_offset);
			delete[] level_offset;
		}
		std::cout << "use CFL query plan..." << std::endl;
		generateCFLQueryPlan(data_graph, query_graph, edge_matrix, matching_order, pivots, cfl_tree, cfl_order, candidates_count);
	} else if (orderid == 4) {
		if (dpiso_tree == NULL) {
			generateDPisoFilterPlan(data_graph, query_graph, dpiso_tree, dpiso_order);
		}
		std::cout << "use DPiso query plan..." << std::endl;
		generateDSPisoQueryPlan(query_graph, edge_matrix, matching_order, pivots, dpiso_tree, dpiso_order,
													candidates_count, weight_array);
	}
	else if (orderid == 5) {
		std::cout << "use CECI query plan..." << std::endl;
		generateCECIQueryPlan(query_graph, ceci_tree, ceci_order, matching_order, pivots);
	}
	else if (orderid == 6) {
		std::cout << "use RI query plan..." << std::endl;
		generateRIQueryPlan(data_graph, query_graph, matching_order, pivots);
	}
	else if (orderid == 7) {
		std::cout << "use VF2 query plan..." << std::endl;
		generateVF2PPQueryPlan(data_graph, query_graph, matching_order, pivots);
	}
	else if (orderid == 8) {
		std::cout << "use Spectrum query plan..." << std::endl;
		generateOrderSpectrum(query_graph, spectrum, order_num);
	}
	else {
		std::cout << "The specified order id " << orderid << "' is not supported." << std::endl;
	}
	// ordering vertices
	generateGQLQueryPlan(data_graph, query_graph, candidates_count, matching_order, pivots);

}

struct CindexScore{
    ui candidateIndex;
    ui score;
};

struct Comparator {
    bool operator () (CindexScore const &i,CindexScore const &j) {

        return i.score > j.score;
    }
}comparator;

void errorCheck(std::string message){
	auto err = cudaGetLastError();
	if ( cudaSuccess != err ){
		printf("Error! %s : %s\n",message.c_str(),cudaGetErrorString(err));
	}
}
bool Search(ui* array, ui x, ui low, ui high) {

	for (int i = low ; i <= high ; ++i){
		if(x == array[i]){
			return true;
		}
	}
	return false;
}

bool binarySearch(ui* array, ui x, ui low, ui high) {
	// Repeat until the pointers low and high meet each other
	  high = high + 1;
	  while (low < high) {
	    ui mid = low + (high - low ) / 2;
	    if (x == array [mid]){
	    	 return true;
	    }
	    if (x > array [mid]){
	    	low = mid + 1;
	    }else{
	    	high = mid;
	    }
	  }
	  return false;
}

bool allocateGPU1D( ui* &dst  ,ui* &src, ui byte ){
	cudaMalloc(&dst, byte);
	cudaMemcpy(dst,src, byte, cudaMemcpyHostToDevice);
}

bool allocateGPU1D( double* &dst  ,double* &src, ui byte ){
	cudaMalloc(&dst, byte);
	cudaMemcpy(dst,src, byte, cudaMemcpyHostToDevice);
}

bool allocateGPU2D(ui* &dst,ui** &src, ui divX, ui divY){
	ui* src_flattern = new ui [divX* divY];
	for (int i = 0; i< divX; ++i ){
		memcpy(src_flattern+i*divY,src[i],divY* sizeof(ui));
	}
	cudaMalloc(&dst,divX*divY*sizeof(ui));
	cudaMemcpy(dst,src_flattern,divX*divY*sizeof(ui),cudaMemcpyHostToDevice );
	//free flattern
	delete [] src_flattern;
}

bool allocateGPU2DPitch(ui* &dst,ui** &src, ui divX, ui divY, size_t &pitch){
	ui* src_flattern = new ui [divX* divY];
	for (int i = 0; i< divX; ++i ){
		memcpy(src_flattern+i*divY,src[i],divY* sizeof(ui));
	}
	cudaMallocPitch(&dst, &pitch,divY*sizeof(ui), divX);
	cudaMemcpy2D(dst,pitch,src_flattern,divY*sizeof(ui), divY*sizeof(ui),divX,cudaMemcpyHostToDevice );
	//free flattern
	delete [] src_flattern;
}

bool allocateMemoryPerThread(ui* &d_dist ,ui byte, ui number){
	unsigned long long total_bytes = (unsigned long long) byte *( unsigned long long) number;
	printf("allocate each bytes: %ld, numbers %ld, total: %ld \n", byte, number, total_bytes);
	cudaMalloc(&d_dist,total_bytes);
}

bool allocateMemoryPerThread(double* &d_dist ,ui byte, ui number){
	unsigned long long total_bytes = (unsigned long long) byte *( unsigned long long) number;
	printf("allocate each bytes: %ld, numbers %ld, total: %ld \n", byte, number, total_bytes);
	cudaMalloc(&d_dist, byte*number);
}

bool allocateGPUEdges(ui* &d_offset_index,ui*&  d_offsets,  ui*& d_edge_index,ui*& d_edges, Edges*** edge_matrix, ui query_vertices_num, ui* candidates_count, unsigned long long & alloc_bytes){
	ui* offset_index = new ui [query_vertices_num* query_vertices_num + 1];
	ui* edge_index = new ui [query_vertices_num* query_vertices_num + 1];
	ui offsets_length = 0;
	ui edges_length = 0;
	for (int i = 0; i < query_vertices_num; ++i){
		for (int j = 0; j < query_vertices_num; ++j){
			Edges* cur_edge = edge_matrix[i][j];
			// offsets num = vertex_count plus 1
			if(cur_edge != NULL){
				ui vertex_count = candidates_count[i];
				offset_index[i* query_vertices_num  + j] = offsets_length;
				edge_index[i* query_vertices_num  + j] = edges_length;
				offsets_length += vertex_count+1;
				edges_length += cur_edge->offset_[vertex_count];

			}else{
				offset_index[i* query_vertices_num  + j] = offsets_length;
				edge_index[i* query_vertices_num  + j] = edges_length;
			}
		}
	}
	offset_index[query_vertices_num* query_vertices_num] = offsets_length;
	edge_index[query_vertices_num* query_vertices_num] = edges_length;
//	std::cout << "edges_length: " << edges_length << " offsets_length: " << offsets_length <<std::endl;
	ui* fattern_edges = new ui [edges_length];
	ui* fattern_offsets = new ui [offsets_length];

	for (int i = 0; i < query_vertices_num; ++i){
		for (int j = 0; j < query_vertices_num; ++j){
			Edges* cur_edge = edge_matrix[i][j];
			ui offset_len = offset_index[i* query_vertices_num  + j + 1 ] - offset_index[i* query_vertices_num  + j] ;
			if(offset_len > 0 ){
				memcpy(fattern_offsets + offset_index[i* query_vertices_num  + j] , cur_edge->offset_, (offset_len)* sizeof(ui));
			}
			ui edgenum = edge_index[i* query_vertices_num  + j + 1 ] - edge_index[i* query_vertices_num  + j] ;
			if(edgenum > 0 ){
				memcpy(fattern_edges + edge_index[i* query_vertices_num  + j] , cur_edge->edge_, (edgenum)* sizeof(ui));
			}

		}
	}

	// copy arr to GPU;
	cudaMalloc(&d_offset_index,(query_vertices_num* query_vertices_num + 1)*sizeof(ui));
	cudaMalloc(&d_edge_index,(query_vertices_num* query_vertices_num + 1)*sizeof(ui));
	cudaMalloc(&d_offsets,(offsets_length)*sizeof(ui));
	cudaMalloc(&d_edges,(edges_length)*sizeof(ui));
	alloc_bytes += (query_vertices_num* query_vertices_num + 1)*sizeof(ui);
	alloc_bytes += (query_vertices_num* query_vertices_num + 1)*sizeof(ui);
	alloc_bytes += (offsets_length)*sizeof(ui);
	alloc_bytes += (edges_length)*sizeof(ui);
	cudaMemcpy(d_offset_index,offset_index,(query_vertices_num* query_vertices_num + 1)*sizeof(ui), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_index,edge_index,(query_vertices_num* query_vertices_num + 1)*sizeof(ui), cudaMemcpyHostToDevice);
	cudaMemcpy(d_offsets,fattern_offsets,(offsets_length)*sizeof(ui), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edges,fattern_edges,(edges_length)*sizeof(ui), cudaMemcpyHostToDevice);

	delete [] offset_index;
	delete [] fattern_offsets;
	delete [] fattern_edges;
	delete [] edge_index;

}


bool allocateGPUmatrix(Edges*** dst, Edges*** src, ui divX, ui divY){
	cudaMalloc(&dst, sizeof (Edges **)* divX);
	for (ui i = 0; i < divX; ++i) {
		cudaMalloc(&dst[i], sizeof (Edges *)* divX);
		for(ui j = 0; j < divY; ++j){
			cudaMalloc(&dst[i][j], sizeof (Edges *)* divX);
			Edges* d_edge;
			cudaMalloc(&d_edge, sizeof(Edges) );
			cudaMemcpy(d_edge,src[i][j], sizeof (Edges), cudaMemcpyHostToDevice);
			dst[i][j] = d_edge;
		}
	}
}



//return an int from 0 to length -1. according to the weight p.
ui selectroot(double* p, ui length){
	double sum = 0;
	double prefix = 0;
	// rand [0,1]
	double rand_num = rand()/double(RAND_MAX);
	for (int i = 0; i< length; ++i){
		sum += p[i];
	}
	for (int i = 0; i< length; ++i){
		prefix += p[i];
		if(prefix/sum >= rand_num){
			return i;
		}
	}
	return length -1;
}

//return the possibilty of choose i
double getP(double* p, ui length, ui root_id){
	double sum = 0;
	for (int i = 0; i< length; ++i){
		sum += p[i];
	}
	return p[root_id]/ sum;
}


//CPU memory
/*allocate candidate*/
void allocateBuffer(const Graph *data_graph, const Graph *query_graph, ui **&candidates,
                                    ui *&candidates_count) {
    ui query_vertex_num = query_graph->getVerticesCount();
    ui candidates_max_num = data_graph->getGraphMaxLabelFrequency();

    candidates_count = new ui[query_vertex_num];
    memset(candidates_count, 0, sizeof(ui) * query_vertex_num);

    candidates = new ui*[query_vertex_num];

    for (ui i = 0; i < query_vertex_num; ++i) {
        candidates[i] = new ui[candidates_max_num];
    }
}
/*allocate embedings*/
void
allocateBuffer(const Graph *data_graph, const Graph *query_graph, ui *candidates_count, ui *&idx,
                              ui *&idx_count, ui *&embedding, ui *&idx_embedding, ui *&temp_buffer,
                              ui **&valid_candidate_idx, bool *&visited_vertices) {
    ui query_vertices_num = query_graph->getVerticesCount();
    ui data_vertices_num = data_graph->getVerticesCount();
    ui max_candidates_num = candidates_count[0];

    for (ui i = 1; i < query_vertices_num; ++i) {
        VertexID cur_vertex = i;
        ui cur_candidate_num = candidates_count[cur_vertex];

        if (cur_candidate_num > max_candidates_num) {
            max_candidates_num = cur_candidate_num;
        }
    }

    idx = new ui[query_vertices_num];
    idx_count = new ui[query_vertices_num];
    embedding = new ui[query_vertices_num];
    idx_embedding = new ui[query_vertices_num];
    visited_vertices = new bool[data_vertices_num];
    temp_buffer = new ui[max_candidates_num];
    valid_candidate_idx = new ui *[query_vertices_num];
    for (ui i = 0; i < query_vertices_num; ++i) {
        valid_candidate_idx[i] = new ui[max_candidates_num];
    }

    std::fill(visited_vertices, visited_vertices + data_vertices_num, false);
}

/*allocate embedings*/
void
allocateBuffer(const Graph *data_graph, const Graph *query_graph, ui *candidates_count, ui *&idx,
                              double *&idx_count, ui *&embedding, ui *&idx_embedding, ui *&temp_buffer,
                              ui **&valid_candidate_idx, bool *&visited_vertices) {
    ui query_vertices_num = query_graph->getVerticesCount();
    ui data_vertices_num = data_graph->getVerticesCount();
    ui max_candidates_num = candidates_count[0];

    for (ui i = 1; i < query_vertices_num; ++i) {
        VertexID cur_vertex = i;
        ui cur_candidate_num = candidates_count[cur_vertex];

        if (cur_candidate_num > max_candidates_num) {
            max_candidates_num = cur_candidate_num;
        }
    }

    idx = new ui[query_vertices_num];
    idx_count = new double[query_vertices_num];
    embedding = new ui[query_vertices_num];
    idx_embedding = new ui[query_vertices_num];
    visited_vertices = new bool[data_vertices_num];
    temp_buffer = new ui[max_candidates_num];
    valid_candidate_idx = new ui *[query_vertices_num];
    for (ui i = 0; i < query_vertices_num; ++i) {
        valid_candidate_idx[i] = new ui[max_candidates_num];
    }

    std::fill(visited_vertices, visited_vertices + data_vertices_num, false);
}

void releaseBuffer(ui query_vertices_num, ui *idx, ui *idx_count, ui *embedding, ui *idx_embedding,
                                  ui *temp_buffer, ui **valid_candidate_idx, bool *visited_vertices, ui **bn,
                                  ui *bn_count) {
    delete[] idx;
    delete[] idx_count;
    delete[] embedding;
    delete[] idx_embedding;
    delete[] visited_vertices;
    delete[] bn_count;
    delete[] temp_buffer;
    for (ui i = 0; i < query_vertices_num; ++i) {
        delete[] valid_candidate_idx[i];
        delete[] bn[i];
    }

    delete[] valid_candidate_idx;
    delete[] bn;
}

void releaseBn (ui query_vertices_num,ui **bn ){
	for (ui i = 0; i < query_vertices_num; ++i) {
		delete[] bn[i];
	}
}
//Fitering functions

bool isCandidateSetValid(ui **&candidates, ui *&candidates_count, ui query_vertex_num) {
    for (ui i = 0; i < query_vertex_num; ++i) {
        if (candidates_count[i] == 0)
            return false;
    }
    return true;
}

void
computeCandidateWithNLF(const Graph *data_graph, const Graph *query_graph, VertexID query_vertex,
                                               ui &count, ui *buffer) {
    LabelID label = query_graph->getVertexLabel(query_vertex);
    ui degree = query_graph->getVertexDegree(query_vertex);
#if OPTIMIZED_LABELED_GRAPH == 1
    const std::unordered_map<LabelID, ui>* query_vertex_nlf = query_graph->getVertexNLF(query_vertex);
#endif
    ui data_vertex_num;
    const ui* data_vertices = data_graph->getVerticesByLabel(label, data_vertex_num);
    count = 0;
    for (ui j = 0; j < data_vertex_num; ++j) {
        ui data_vertex = data_vertices[j];
        if (data_graph->getVertexDegree(data_vertex) >= degree) {

            // NFL check
#if OPTIMIZED_LABELED_GRAPH == 1
            const std::unordered_map<LabelID, ui>* data_vertex_nlf = data_graph->getVertexNLF(data_vertex);

            if (data_vertex_nlf->size() >= query_vertex_nlf->size()) {
                bool is_valid = true;

                for (auto element : *query_vertex_nlf) {
                    auto iter = data_vertex_nlf->find(element.first);
                    if (iter == data_vertex_nlf->end() || iter->second < element.second) {
                        is_valid = false;
                        break;
                    }
                }

                if (is_valid) {
                    if (buffer != NULL) {
                        buffer[count] = data_vertex;
                    }
                    count += 1;
                }
            }
#else
            if (buffer != NULL) {
                buffer[count] = data_vertex;
            }
            count += 1;
#endif
        }
    }

}

bool
verifyExactTwigIso(const Graph *data_graph, const Graph *query_graph, ui data_vertex, ui query_vertex,
                                   bool **valid_candidates, int *left_to_right_offset, int *left_to_right_edges,
                                   int *left_to_right_match, int *right_to_left_match, int* match_visited,
                                   int* match_queue, int* match_previous) {
    // Construct the bipartite graph between N(query_vertex) and N(data_vertex)
    ui left_partition_size;
    ui right_partition_size;
    const VertexID* query_vertex_neighbors = query_graph->getVertexNeighbors(query_vertex, left_partition_size);
    const VertexID* data_vertex_neighbors = data_graph->getVertexNeighbors(data_vertex, right_partition_size);

    ui edge_count = 0;
    for (int i = 0; i < left_partition_size; ++i) {
        VertexID query_vertex_neighbor = query_vertex_neighbors[i];
        left_to_right_offset[i] = edge_count;

        for (int j = 0; j < right_partition_size; ++j) {
            VertexID data_vertex_neighbor = data_vertex_neighbors[j];

            if (valid_candidates[query_vertex_neighbor][data_vertex_neighbor]) {
                left_to_right_edges[edge_count++] = j;
            }
        }
    }
    left_to_right_offset[left_partition_size] = edge_count;

    memset(left_to_right_match, -1, left_partition_size * sizeof(int));
    memset(right_to_left_match, -1, right_partition_size * sizeof(int));

    GraphOperations::match_bfs(left_to_right_offset, left_to_right_edges, left_to_right_match, right_to_left_match,
                               match_visited, match_queue, match_previous, left_partition_size, right_partition_size);
    for (int i = 0; i < left_partition_size; ++i) {
        if (left_to_right_match[i] == -1)
            return false;
    }

    return true;
}

void compactCandidates(ui **&candidates, ui *&candidates_count, ui query_vertex_num) {
    for (ui i = 0; i < query_vertex_num; ++i) {
        VertexID query_vertex = i;
        ui next_position = 0;
        for (ui j = 0; j < candidates_count[query_vertex]; ++j) {
            VertexID data_vertex = candidates[query_vertex][j];

            if (data_vertex != INVALID_VERTEX_ID) {
                candidates[query_vertex][next_position++] = data_vertex;
            }
        }

        candidates_count[query_vertex] = next_position;
    }
}

bool NLFFilter(const Graph *data_graph, const Graph *query_graph, ui **&candidates, ui *&candidates_count) {
    allocateBuffer(data_graph, query_graph, candidates, candidates_count);

    for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
        VertexID query_vertex = i;
        computeCandidateWithNLF(data_graph, query_graph, query_vertex, candidates_count[query_vertex], candidates[query_vertex]);

        if (candidates_count[query_vertex] == 0) {
            return false;
        }
    }

    return true;
}

bool GQLFilter(const Graph *data_graph, const Graph *query_graph, ui **&candidates, ui *&candidates_count) {
    // Local refinement.
    if (!NLFFilter(data_graph, query_graph, candidates, candidates_count))
        return false;

    // Allocate buffer.
    ui query_vertex_num = query_graph->getVerticesCount();
    ui data_vertex_num = data_graph->getVerticesCount();

    bool** valid_candidates = new bool*[query_vertex_num];
    for (ui i = 0; i < query_vertex_num; ++i) {
        valid_candidates[i] = new bool[data_vertex_num];
        memset(valid_candidates[i], 0, sizeof(bool) * data_vertex_num);
    }

    ui query_graph_max_degree = query_graph->getGraphMaxDegree();
    ui data_graph_max_degree = data_graph->getGraphMaxDegree();

    int* left_to_right_offset = new int[query_graph_max_degree + 1];
    int* left_to_right_edges = new int[query_graph_max_degree * data_graph_max_degree];
    int* left_to_right_match = new int[query_graph_max_degree];
    int* right_to_left_match = new int[data_graph_max_degree];
    int* match_visited = new int[data_graph_max_degree + 1];
    int* match_queue = new int[query_vertex_num];
    int* match_previous = new int[data_graph_max_degree + 1];

    // Record valid candidate vertices for each query vertex.
    for (ui i = 0; i < query_vertex_num; ++i) {
        VertexID query_vertex = i;
        for (ui j = 0; j < candidates_count[query_vertex]; ++j) {
            VertexID data_vertex = candidates[query_vertex][j];
            valid_candidates[query_vertex][data_vertex] = true;
        }
    }

    // Global refinement.
    for (ui l = 0; l < 2; ++l) {
        for (ui i = 0; i < query_vertex_num; ++i) {
            VertexID query_vertex = i;
            for (ui j = 0; j < candidates_count[query_vertex]; ++j) {
                VertexID data_vertex = candidates[query_vertex][j];

                if (data_vertex == INVALID_VERTEX_ID)
                    continue;

                if (!verifyExactTwigIso(data_graph, query_graph, data_vertex, query_vertex, valid_candidates,
                                        left_to_right_offset, left_to_right_edges, left_to_right_match,
                                        right_to_left_match, match_visited, match_queue, match_previous)) {
                    candidates[query_vertex][j] = INVALID_VERTEX_ID;
                    valid_candidates[query_vertex][data_vertex] = false;
                }
            }
        }
    }

    // Compact candidates.
    compactCandidates(candidates, candidates_count, query_vertex_num);

    // Release memory.
    for (ui i = 0; i < query_vertex_num; ++i) {
        delete[] valid_candidates[i];
    }
    delete[] valid_candidates;
    delete[] left_to_right_offset;
    delete[] left_to_right_edges;
    delete[] left_to_right_match;
    delete[] right_to_left_match;
    delete[] match_visited;
    delete[] match_queue;
    delete[] match_previous;

    return isCandidateSetValid(candidates, candidates_count, query_vertex_num);
}

// sort candidate
void sortCandidates(ui **candidates, ui *candidates_count, ui num) {
    for (ui i = 0; i < num; ++i) {
        std::sort(candidates[i], candidates[i] + candidates_count[i]);
    }
}

VertexID selectGQLStartVertex(const Graph *query_graph, ui *candidates_count) {
    /**
     * Select the vertex with the minimum number of candidates as the start vertex.
     * Tie Handling:
     *  1. degree
     *  2. label id
     */

     ui start_vertex = 0;

     for (ui i = 1; i < query_graph->getVerticesCount(); ++i) {
          VertexID cur_vertex = i;

          if (candidates_count[cur_vertex] < candidates_count[start_vertex]) {
               start_vertex = cur_vertex;
          }
          else if (candidates_count[cur_vertex] == candidates_count[start_vertex]
                   && query_graph->getVertexDegree(cur_vertex) > query_graph->getVertexDegree(start_vertex)) {
               start_vertex = cur_vertex;
          }
     }

     return start_vertex;
}


void updateValidVertices(const Graph *query_graph, VertexID query_vertex, std::vector<bool> &visited,
                                            std::vector<bool> &adjacent) {
     visited[query_vertex] = true;
     ui nbr_cnt;
     const ui* nbrs = query_graph->getVertexNeighbors(query_vertex, nbr_cnt);

     for (ui i = 0; i < nbr_cnt; ++i) {
          ui nbr = nbrs[i];
          adjacent[nbr] = true;
     }
}

void generateGQLQueryPlan(const Graph *data_graph, const Graph *query_graph, ui *candidates_count,
                                             ui *&order, ui *&pivot) {
     /**
      * Select the vertex v such that (1) v is adjacent to the selected vertices; and (2) v has the minimum number of candidates.
      */
     std::vector<bool> visited_vertices(query_graph->getVerticesCount(), false);
     std::vector<bool> adjacent_vertices(query_graph->getVerticesCount(), false);
     order = new ui[query_graph->getVerticesCount()];
     pivot = new ui[query_graph->getVerticesCount()];

     VertexID start_vertex = selectGQLStartVertex(query_graph, candidates_count);
     order[0] = start_vertex;
     updateValidVertices(query_graph, start_vertex, visited_vertices, adjacent_vertices);

     for (ui i = 1; i < query_graph->getVerticesCount(); ++i) {
          VertexID next_vertex;
          ui min_value = data_graph->getVerticesCount() + 1;
          for (ui j = 0; j < query_graph->getVerticesCount(); ++j) {
               VertexID cur_vertex = j;

               if (!visited_vertices[cur_vertex] && adjacent_vertices[cur_vertex]) {
                    if (candidates_count[cur_vertex] < min_value) {
                         min_value = candidates_count[cur_vertex];
                         next_vertex = cur_vertex;
                    }
                    else if (candidates_count[cur_vertex] == min_value && query_graph->getVertexDegree(cur_vertex) > query_graph->getVertexDegree(next_vertex)) {
                         next_vertex = cur_vertex;
                    }
               }
          }
          updateValidVertices(query_graph, next_vertex, visited_vertices, adjacent_vertices);
          order[i] = next_vertex;
     }

     // Pick a pivot randomly.
     for (ui i = 1; i < query_graph->getVerticesCount(); ++i) {
         VertexID u = order[i];
         for (ui j = 0; j < i; ++j) {
             VertexID cur_vertex = order[j];
             if (query_graph->checkEdgeExistence(u, cur_vertex)) {
                 pivot[i] = cur_vertex;
                 break;
             }
         }
     }
}

void printSimplifiedQueryPlan(const Graph *query_graph, ui *order) {
    ui query_vertices_num = query_graph->getVerticesCount();
    printf("Query Plan: ");
    for (ui i = 0; i < query_vertices_num; ++i) {
        printf("%u ", order[i]);
    }
    printf("\n");
}

void checkQueryPlanCorrectness(const Graph *query_graph, ui *order, ui *pivot) {
    ui query_vertices_num = query_graph->getVerticesCount();
    std::vector<bool> visited_vertices(query_vertices_num, false);
    // Check whether each query vertex is in the order.
    for (ui i = 0; i < query_vertices_num; ++i) {
        VertexID vertex = order[i];
        assert(vertex < query_vertices_num && vertex >= 0);

        visited_vertices[vertex] = true;
    }

    for (ui i = 0; i < query_vertices_num; ++i) {
        VertexID vertex = i;
        assert(visited_vertices[vertex]);
    }

    // Check whether the order is connected.

    std::fill(visited_vertices.begin(), visited_vertices.end(), false);
    visited_vertices[order[0]] = true;
    for (ui i = 1; i < query_vertices_num; ++i) {
        VertexID vertex = order[i];
        VertexID pivot_vertex = pivot[i];
        assert(query_graph->checkEdgeExistence(vertex, pivot_vertex));
        assert(visited_vertices[pivot_vertex]);
        visited_vertices[vertex] = true;
    }
}

/*CPU matching */
/*bn: [order][index] return a neighbor need to be joined*/
void generateBN(const Graph *query_graph, ui *order, ui **&bn, ui *&bn_count) {
    ui query_vertices_num = query_graph->getVerticesCount();
    bn_count = new ui[query_vertices_num];
    std::fill(bn_count, bn_count + query_vertices_num, 0);
    bn = new ui *[query_vertices_num];
    for (ui i = 0; i < query_vertices_num; ++i) {
        bn[i] = new ui[query_vertices_num];
    }

    std::vector<bool> visited_vertices(query_vertices_num, false);
    visited_vertices[order[0]] = true;
    for (ui i = 1; i < query_vertices_num; ++i) {
        VertexID vertex = order[i];

        ui nbrs_cnt;
        const ui *nbrs = query_graph->getVertexNeighbors(vertex, nbrs_cnt);
        for (ui j = 0; j < nbrs_cnt; ++j) {
            VertexID nbr = nbrs[j];

            if (visited_vertices[nbr]) {
                bn[i][bn_count[i]++] = nbr;
            }
        }

        visited_vertices[vertex] = true;
    }
}


void reassignBN(const Graph *query_graph, ui *order, ui **&bn, ui *&bn_count) {
    ui query_vertices_num = query_graph->getVerticesCount();
    std::fill(bn_count, bn_count + query_vertices_num, 0);
    std::vector<bool> visited_vertices(query_vertices_num, false);
    visited_vertices[order[0]] = true;
    for (ui i = 1; i < query_vertices_num; ++i) {
        VertexID vertex = order[i];

        ui nbrs_cnt;
        const ui *nbrs = query_graph->getVertexNeighbors(vertex, nbrs_cnt);
        for (ui j = 0; j < nbrs_cnt; ++j) {
            VertexID nbr = nbrs[j];

            if (visited_vertices[nbr]) {
                bn[i][bn_count[i]++] = nbr;
            }
        }

        visited_vertices[vertex] = true;
    }
}



template <const ui blocksize>
ui WJ (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record ){
	//
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	record. cand_alloc_time = 0;
	ui fixednum = record.fixednum;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer in CPU
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
    double* score;
    ui* score_count;
    ui* denominator;
    bool *visited_vertices;
    ui* random_list;
    allocateBuffer(data_graph, query_graph, candidates_count, idx, idx_count, embedding, idx_embedding,
                   temp_buffer, valid_candidate_idx, visited_vertices);
    size_t embedding_cnt = 0;
    int cur_depth = 0;
    int max_depth = query_graph->getVerticesCount();
    VertexID start_vertex = order[0];

    idx[cur_depth] = 0;
    idx_count[cur_depth] = candidates_count[start_vertex];
    unsigned long long GPU_bytes = 0;

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    denominator = new ui [1];
    score[0] = 0;
    score_count[0] = 0;
    denominator[0] = 0;

    auto GPU_alloc_start = std::chrono::high_resolution_clock::now();
    // allocate GPU mmeory;
    ui query_vertices_num = query_graph->getVerticesCount();
    ui data_vertices_num = data_graph->getVerticesCount();
    ui max_candidates_num = candidates_count[0];
	for (ui i = 1; i < query_vertices_num; ++i) {
		VertexID cur_vertex = i;
		ui cur_candidate_num = candidates_count[cur_vertex];

		if (cur_candidate_num > max_candidates_num) {
			max_candidates_num = cur_candidate_num;
		}
	}
    // 1-d array only read
    ui* d_bn;
    ui* d_bn_count;

    ui* d_candidates_count;
    ui* d_order;
//    ui* d_sampling_visited_vertices;
    double* d_score;
    ui* d_score_count;
    ui* d_denominator;
    // 1-d array write by thread
    ui* d_idx;
	ui* d_idx_count;
	ui* d_embedding;
	ui* d_idx_embedding;
	ui* d_temp;
	ui* d_temp_size;
	ui* d_range;
	ui* d_intersection;
    // 2d array
//    ui* d_valid_candidate_idx;
    ui* d_candidates;

    // 3d array
    ui* d_offset_index;
    ui* d_offsets;
    ui* d_edge_index;
    ui* d_edges;
    cudaDeviceSynchronize();

    /*  allocate memory structure for GPU computation*/
    std::cout << "assign GPU memory..." <<std::endl;
    allocateGPU1D( d_bn_count ,bn_count, query_vertices_num* sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);
//    allocateGPU1D( d_idx ,idx,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_count ,idx_count,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_embedding ,embedding,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_embedding ,idx_embedding,query_vertices_num * sizeof(ui));
    allocateGPU1D( d_order, order, query_vertices_num * sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);
//    allocateGPU1D( d_temp_buffer ,temp_buffer, max_candidates_num * sizeof(ui));
    allocateGPU1D( d_score ,score, 1* sizeof(double));
    allocateGPU1D( d_score_count ,score_count, 1* sizeof(double));
    allocateGPU1D( d_denominator ,denominator, 1* sizeof(ui));
    allocateGPU1D( d_candidates_count ,candidates_count, query_vertices_num* sizeof(ui));
    GPU_bytes += sizeof(double)*2 +  query_vertices_num* sizeof(ui) ;
//    size_t valid_candidate_idx_pitch;
//    size_t candidates_pitch;
//    allocateGPU2DUI(d_valid_candidate_idx,valid_candidate_idx, query_vertices_num, max_candidates_num,valid_candidate_idx_pitch);
    allocateGPU2D(d_candidates,candidates,query_vertices_num,max_candidates_num);
    allocateGPU2D(d_bn,bn,query_vertices_num,query_vertices_num);
    GPU_bytes += query_vertices_num* query_vertices_num + sizeof(ui) + query_vertices_num* max_candidates_num * sizeof (ui);
//    allocateGPU2DPitch(d_candidates,candidates,query_vertices_num,query_vertices_num, candidates_pitch);
    allocateGPUEdges(d_offset_index,d_offsets, d_edge_index, d_edges, edge_matrix, query_vertices_num, candidates_count,GPU_bytes );
    // test correctness of GPU memory
    // allocate global memory for each thread
    ui threadnum = record.threadnum;


	ui numBlocks = (threadnum-1) / blocksize + 1;
	ui taskPerRound = numBlocks* record. taskPerBlock;

	std::cout << "readonly GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
	std::cout << "threadsPerBlock: "<< blocksize << " numBlocks: "<< numBlocks << " total threads: " << blocksize*numBlocks << " max_candidates_num " << max_candidates_num<<std::endl;

	// for each thread we assign its own global memoory.
    allocateMemoryPerThread(d_idx ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_range ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_count ,query_vertices_num * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui)) * threadnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
		record. successrun = false;
		std::cout <<"An error ocurrs when allocate memory!"<<std::endl;
	}else{
		std::cout <<"Pass memory assignment test!"<<std::endl;
	}
	// compute total bytes allocated.


	// test candidate

    auto GPU_alloc_end = std::chrono::high_resolution_clock::now();
    record. cand_alloc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(GPU_alloc_end - GPU_alloc_start).count();

//    while (true) {
//
//      	while (idx[cur_depth] < idx_count[cur_depth]) {
          	// sampling part
          	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
          		auto sampling_start = std::chrono::high_resolution_clock::now();
  				ui sample_time = record. sample_time;
  				// record the possibility weight to sample in the current（first） layer
  				ui round = (sample_time - 1)/ taskPerRound + 1;
  				double aver_score = 0;
  				ui h_score_count = 0;
  				// round is 1 in fact
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					HelpWJ<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock,d_denominator);
					cudaDeviceSynchronize();
					cudaMemcpy( &aver_score, d_score, sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy( denominator, d_denominator, sizeof(ui), cudaMemcpyDeviceToHost);

					auto err = cudaGetLastError();
					if (err != cudaSuccess){
						std::cout <<"An error ocurrs when sampling!"<<std::endl;
					}else{
						std::cout <<"Sampling end!"<<std::endl;
					}
  				}
				// beacuse 1st only run once, so * fixednum
  				printf("denominator: %d \n",denominator[0]);
//  				record.est_path = aver_score/denominator[0]* fixednum;
  				record.est_path = aver_score/sample_time* fixednum;
  				auto sampling_end = std::chrono::high_resolution_clock::now();
				record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
          	}
          	return 0;
}


template <const ui blocksize>
ui AL (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record ){
	//initalize record.
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	record. cand_alloc_time = 0;
	ui fixednum = record.fixednum;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate candidate graph.
    ui **bn;
    ui *bn_count;
    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer in CPU
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
    double* score;
    ui* score_count;
    bool *visited_vertices;
    ui* random_list;
    allocateBuffer(data_graph, query_graph, candidates_count, idx, idx_count, embedding, idx_embedding,
                   temp_buffer, valid_candidate_idx, visited_vertices);
    size_t embedding_cnt = 0;
    int cur_depth = 0;
    int max_depth = query_graph->getVerticesCount();
    VertexID start_vertex = order[0];

    idx[cur_depth] = 0;
    idx_count[cur_depth] = candidates_count[start_vertex];
    unsigned long long GPU_bytes = 0;

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

    // allocate GPU mmeory;
    ui query_vertices_num = query_graph->getVerticesCount();
    ui data_vertices_num = data_graph->getVerticesCount();
    ui max_candidates_num = candidates_count[0];
	for (ui i = 1; i < query_vertices_num; ++i) {
		VertexID cur_vertex = i;
		ui cur_candidate_num = candidates_count[cur_vertex];

		if (cur_candidate_num > max_candidates_num) {
			max_candidates_num = cur_candidate_num;
		}
	}
    /***
    Define variables
    ***/
    // 1-d array only read
    ui* d_bn;
    ui* d_bn_count;

    ui* d_candidates_count;
    ui* d_order;
//    ui* d_sampling_visited_vertices;
    double* d_score;
    ui* d_score_count;
    // 1-d array write by thread
    ui* d_idx;
	ui* d_idx_count;
	ui* d_embedding;
	ui* d_idx_embedding;
	ui* d_temp;
	ui* d_temp_size;
	ui* d_range;
	ui* d_intersection;
    // 2d array
//    ui* d_valid_candidate_idx;
    ui* d_candidates;

    // 3d array
    ui* d_offset_index;
    ui* d_offsets;
    ui* d_edge_index;
    ui* d_edges;
    cudaDeviceSynchronize();
    auto GPU_alloc_start = std::chrono::high_resolution_clock::now();
    /*  allocate memory structure for GPU computation*/
    std::cout << "assign GPU memory..." <<std::endl;
    allocateGPU1D( d_bn_count ,bn_count, query_vertices_num* sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);

    allocateGPU1D( d_order, order, query_vertices_num * sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);

    allocateGPU1D( d_score ,score, 1* sizeof(double));
    allocateGPU1D( d_score_count ,score_count, 1* sizeof(double));
    allocateGPU1D( d_candidates_count ,candidates_count, query_vertices_num* sizeof(ui));
    GPU_bytes += sizeof(double)*2 +  query_vertices_num* sizeof(ui) ;

    allocateGPU2D(d_candidates,candidates,query_vertices_num,max_candidates_num);
    allocateGPU2D(d_bn,bn,query_vertices_num,query_vertices_num);
    GPU_bytes += query_vertices_num* query_vertices_num + sizeof(ui) + query_vertices_num* max_candidates_num * sizeof (ui);
//    allocateGPU2DPitch(d_candidates,candidates,query_vertices_num,query_vertices_num, candidates_pitch);
    allocateGPUEdges(d_offset_index,d_offsets, d_edge_index, d_edges, edge_matrix, query_vertices_num, candidates_count,GPU_bytes );
    // test correctness of GPU memory
    // allocate global memory for each thread
    ui threadnum = record.threadnum;

    auto GPU_alloc_end = std::chrono::high_resolution_clock::now();
    record. cand_alloc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(GPU_alloc_end - GPU_alloc_start).count();
    std::cout<< "alloc memory: "<< record.cand_alloc_time /1000000000<< std::endl;

	ui numBlocks = (threadnum-1) / blocksize + 1;
	ui taskPerRound = numBlocks* record. taskPerBlock;

	std::cout << "readonly GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
	std::cout << "threadsPerBlock: "<< blocksize << " numBlocks: "<< numBlocks << " total threads: " << blocksize*numBlocks << " max_candidates_num " << max_candidates_num<<std::endl;

	// for each thread we assign its own global memoory.
    allocateMemoryPerThread(d_idx ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_range ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_count ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui)) * threadnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;


    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
		record. successrun = false;
		std::cout <<"An error ocurrs when allocate memory!"<<std::endl;
	}else{
		std::cout <<"Pass memory assignment test!"<<std::endl;
	}
	// compute total bytes allocated.
	// ui* d_test;
	// auto fast_alloc_begin = std::chrono::high_resolution_clock::now();
    // cudaMalloc(&d_test,GPU_bytes);
    // auto fast_alloc_end = std::chrono::high_resolution_clock::now();
    // printf("fast alloc memory: %f s", (double)std::chrono::duration_cast<std::chrono::nanoseconds>(fast_alloc_end - fast_alloc_begin).count()/1000000000 );
//    cudaFree(&d_test);
	// test candidate

//    while (true) {
//
//      	while (idx[cur_depth] < idx_count[cur_depth]) {
          	// sampling part
          	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
          		auto sampling_start = std::chrono::high_resolution_clock::now();
  				ui sample_time = record. sample_time;
  				// record the possibility weight to sample in the current（first） layer
  				ui round = (sample_time - 1)/ taskPerRound + 1;
  				double aver_score = 0;
  				ui h_score_count = 0;
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					ggecoal2<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
					cudaDeviceSynchronize();
					cudaMemcpy( &aver_score, d_score, sizeof(double), cudaMemcpyDeviceToHost);
	//				cudaMemcpy( &h_score_count, d_score_count, sizeof(ui), cudaMemcpyDeviceToHost);
	//				std::cout << "total_score: " << aver_score << "path count " << h_score_count <<std::endl;
					auto err = cudaGetLastError();
					if (err != cudaSuccess){
						std::cout <<"An error ocurrs when sampling!"<<std::endl;
					}else{
						std::cout <<"Sampling end!"<<std::endl;
					}
  				}
				// beacuse 1st only run once, so * fixednum
  				record.est_path = aver_score/sample_time * fixednum;
  				auto sampling_end = std::chrono::high_resolution_clock::now();
				record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
          	}

          	return 0;
}


template <const ui blocksize>
ui PR (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record ){
	//
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	record. cand_alloc_time = 0;
	ui fixednum = record.fixednum;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer in CPU
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
    double* score;
    ui* score_count;
    bool *visited_vertices;
    ui* random_list;
    allocateBuffer(data_graph, query_graph, candidates_count, idx, idx_count, embedding, idx_embedding,
                   temp_buffer, valid_candidate_idx, visited_vertices);
    size_t embedding_cnt = 0;
    int cur_depth = 0;
    int max_depth = query_graph->getVerticesCount();
    VertexID start_vertex = order[0];

    idx[cur_depth] = 0;
    idx_count[cur_depth] = candidates_count[start_vertex];
    unsigned long long GPU_bytes = 0;

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

    // allocate GPU mmeory;
    ui query_vertices_num = query_graph->getVerticesCount();
    ui data_vertices_num = data_graph->getVerticesCount();
    ui max_candidates_num = candidates_count[0];
	for (ui i = 1; i < query_vertices_num; ++i) {
		VertexID cur_vertex = i;
		ui cur_candidate_num = candidates_count[cur_vertex];

		if (cur_candidate_num > max_candidates_num) {
			max_candidates_num = cur_candidate_num;
		}
	}
    // 1-d array only read
    ui* d_bn;
    ui* d_bn_count;

    ui* d_candidates_count;
    ui* d_order;
//    ui* d_sampling_visited_vertices;
    double* d_score;
    ui* d_score_count;
    // 1-d array write by thread
    ui* d_idx;
	ui* d_idx_count;
	ui* d_embedding;
	ui* d_idx_embedding;
	ui* d_temp;
	ui* d_temp_size;
	ui* d_range;
	ui* d_intersection;
    // 2d array
//    ui* d_valid_candidate_idx;
    ui* d_candidates;

    // 3d array
    ui* d_offset_index;
    ui* d_offsets;
    ui* d_edge_index;
    ui* d_edges;
    cudaDeviceSynchronize();
    auto GPU_alloc_start = std::chrono::high_resolution_clock::now();
    /*  allocate memory structure for GPU computation*/
    std::cout << "assign GPU memory..." <<std::endl;
    allocateGPU1D( d_bn_count ,bn_count, query_vertices_num* sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);

    allocateGPU1D( d_order, order, query_vertices_num * sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);

    allocateGPU1D( d_score ,score, 1* sizeof(double));
    allocateGPU1D( d_score_count ,score_count, 1* sizeof(double));
    allocateGPU1D( d_candidates_count ,candidates_count, query_vertices_num* sizeof(ui));
    GPU_bytes += sizeof(double)*2 +  query_vertices_num* sizeof(ui) ;

    allocateGPU2D(d_candidates,candidates,query_vertices_num,max_candidates_num);
    allocateGPU2D(d_bn,bn,query_vertices_num,query_vertices_num);
    GPU_bytes += query_vertices_num* query_vertices_num + sizeof(ui) + query_vertices_num* max_candidates_num * sizeof (ui);

    allocateGPUEdges(d_offset_index,d_offsets, d_edge_index, d_edges, edge_matrix, query_vertices_num, candidates_count,GPU_bytes );
    // test correctness of GPU memory
    // allocate global memory for each thread
    ui threadnum = record.threadnum;

    auto GPU_alloc_end = std::chrono::high_resolution_clock::now();
    record. cand_alloc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(GPU_alloc_end - GPU_alloc_start).count();
    std::cout<< "alloc memory: "<< record.cand_alloc_time /1000000000<< std::endl;

	ui numBlocks = (threadnum-1) / blocksize + 1;
	ui taskPerRound = numBlocks* record. taskPerBlock;

	std::cout << "readonly GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
	std::cout << "threadsPerBlock: "<< blocksize << " numBlocks: "<< numBlocks << " total threads: " << blocksize*numBlocks << " max_candidates_num " << max_candidates_num<<std::endl;

	// for each thread we assign its own global memoory.
    allocateMemoryPerThread(d_idx ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_range ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_count ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui)) * threadnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;


    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
		record. successrun = false;
		std::cout <<"An error ocurrs when allocate memory!"<<std::endl;
	}else{
		std::cout <<"Pass memory assignment test!"<<std::endl;
	}
	// compute total bytes allocated.
	ui* d_test;
	auto fast_alloc_begin = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_test,GPU_bytes);
    auto fast_alloc_end = std::chrono::high_resolution_clock::now();
    printf("fast alloc memory: %f s", (double)std::chrono::duration_cast<std::chrono::nanoseconds>(fast_alloc_end - fast_alloc_begin).count()/1000000000 );
//    cudaFree(&d_test);
	// test candidate

//    while (true) {
//
//      	while (idx[cur_depth] < idx_count[cur_depth]) {
          	// sampling part
          	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
          		auto sampling_start = std::chrono::high_resolution_clock::now();
  				ui sample_time = record. sample_time;
  				// record the possibility weight to sample in the current（first） layer
  				ui round = (sample_time - 1)/ taskPerRound + 1;
  				double aver_score = 0;
  				ui h_score_count = 0;
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					ggecopr<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
					cudaDeviceSynchronize();
					cudaMemcpy( &aver_score, d_score, sizeof(double), cudaMemcpyDeviceToHost);
	//				cudaMemcpy( &h_score_count, d_score_count, sizeof(ui), cudaMemcpyDeviceToHost);
	//				std::cout << "total_score: " << aver_score << "path count " << h_score_count <<std::endl;
					auto err = cudaGetLastError();
					if (err != cudaSuccess){
						std::cout <<"An error ocurrs when sampling!"<<std::endl;
					}else{
						std::cout <<"Sampling end!"<<std::endl;
					}
  				}
				// beacuse 1st only run once, so * fixednum
  				record.est_path = aver_score/sample_time * fixednum;
  				auto sampling_end = std::chrono::high_resolution_clock::now();
				record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
          	}

          	return 0;
}


template <const ui blocksize>
ui UD (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record ){
	//
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	record. cand_alloc_time = 0;
	ui fixednum = record.fixednum;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer in CPU
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
    double* score;
    ui* score_count;
    ui* denominator;
    bool *visited_vertices;
    ui* random_list;
    allocateBuffer(data_graph, query_graph, candidates_count, idx, idx_count, embedding, idx_embedding,
                   temp_buffer, valid_candidate_idx, visited_vertices);
    size_t embedding_cnt = 0;
    int cur_depth = 0;
    int max_depth = query_graph->getVerticesCount();
    VertexID start_vertex = order[0];

    idx[cur_depth] = 0;
    idx_count[cur_depth] = candidates_count[start_vertex];
    unsigned long long GPU_bytes = 0;

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    denominator = new ui [1];
    score[0] = 0;
    score_count[0] = 0;
    denominator[0] = 0;

    auto GPU_alloc_start = std::chrono::high_resolution_clock::now();
    // allocate GPU mmeory;
    ui query_vertices_num = query_graph->getVerticesCount();
    ui data_vertices_num = data_graph->getVerticesCount();
    ui max_candidates_num = candidates_count[0];
	for (ui i = 1; i < query_vertices_num; ++i) {
		VertexID cur_vertex = i;
		ui cur_candidate_num = candidates_count[cur_vertex];

		if (cur_candidate_num > max_candidates_num) {
			max_candidates_num = cur_candidate_num;
		}
	}
    // 1-d array only read
    ui* d_bn;
    ui* d_bn_count;

    ui* d_candidates_count;
    ui* d_order;
//    ui* d_sampling_visited_vertices;
    double* d_score;
    ui* d_score_count;
    ui* d_denominator;
    // 1-d array write by thread
    ui* d_idx;
	ui* d_idx_count;
	ui* d_embedding;
	ui* d_idx_embedding;
	ui* d_temp;
	ui* d_temp_size;
	ui* d_range;
	ui* d_intersection;
    // 2d array
//    ui* d_valid_candidate_idx;
    ui* d_candidates;

    // 3d array
    ui* d_offset_index;
    ui* d_offsets;
    ui* d_edge_index;
    ui* d_edges;
    cudaDeviceSynchronize();

    /*  allocate memory structure for GPU computation*/
    std::cout << "assign GPU memory..." <<std::endl;
    allocateGPU1D( d_bn_count ,bn_count, query_vertices_num* sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);
//    allocateGPU1D( d_idx ,idx,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_count ,idx_count,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_embedding ,embedding,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_embedding ,idx_embedding,query_vertices_num * sizeof(ui));
    allocateGPU1D( d_order, order, query_vertices_num * sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);
//    allocateGPU1D( d_temp_buffer ,temp_buffer, max_candidates_num * sizeof(ui));
    allocateGPU1D( d_score ,score, 1* sizeof(double));
    allocateGPU1D( d_score_count ,score_count, 1* sizeof(double));
    allocateGPU1D( d_denominator ,denominator, 1* sizeof(ui));
    allocateGPU1D( d_candidates_count ,candidates_count, query_vertices_num* sizeof(ui));
    GPU_bytes += sizeof(double)*2 +  query_vertices_num* sizeof(ui) ;
//    size_t valid_candidate_idx_pitch;
//    size_t candidates_pitch;
//    allocateGPU2DUI(d_valid_candidate_idx,valid_candidate_idx, query_vertices_num, max_candidates_num,valid_candidate_idx_pitch);
    allocateGPU2D(d_candidates,candidates,query_vertices_num,max_candidates_num);
    allocateGPU2D(d_bn,bn,query_vertices_num,query_vertices_num);
    GPU_bytes += query_vertices_num* query_vertices_num + sizeof(ui) + query_vertices_num* max_candidates_num * sizeof (ui);
//    allocateGPU2DPitch(d_candidates,candidates,query_vertices_num,query_vertices_num, candidates_pitch);
    allocateGPUEdges(d_offset_index,d_offsets, d_edge_index, d_edges, edge_matrix, query_vertices_num, candidates_count,GPU_bytes );
    // test correctness of GPU memory
    // allocate global memory for each thread
    ui threadnum = record.threadnum;


	ui numBlocks = (threadnum-1) / blocksize + 1;
	ui taskPerRound = numBlocks* record. taskPerBlock;

	std::cout << "readonly GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
	std::cout << "threadsPerBlock: "<< blocksize << " numBlocks: "<< numBlocks << " total threads: " << blocksize*numBlocks << " max_candidates_num " << max_candidates_num<<std::endl;

	// for each thread we assign its own global memoory.
    allocateMemoryPerThread(d_idx ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_range ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_count ,query_vertices_num * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui)) * threadnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
		record. successrun = false;
		std::cout <<"An error ocurrs when allocate memory!"<<std::endl;
	}else{
		std::cout <<"Pass memory assignment test!"<<std::endl;
	}
	// compute total bytes allocated.


	// test candidate

    auto GPU_alloc_end = std::chrono::high_resolution_clock::now();
    record. cand_alloc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(GPU_alloc_end - GPU_alloc_start).count();

//    while (true) {
//
//      	while (idx[cur_depth] < idx_count[cur_depth]) {
          	// sampling part
          	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
          		auto sampling_start = std::chrono::high_resolution_clock::now();
  				ui sample_time = record. sample_time;
  				// record the possibility weight to sample in the current（first） layer
  				ui round = (sample_time - 1)/ taskPerRound + 1;
  				double aver_score = 0;
  				ui h_score_count = 0;
  				// round is 1 in fact
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					UserDefine<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock,d_denominator);
					cudaDeviceSynchronize();
					cudaMemcpy( &aver_score, d_score, sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy( denominator, d_denominator, sizeof(ui), cudaMemcpyDeviceToHost);

					auto err = cudaGetLastError();
					if (err != cudaSuccess){
						std::cout <<"An error ocurrs when sampling!"<<std::endl;
					}else{
						std::cout <<"Sampling end!"<<std::endl;
					}
  				}
				// beacuse 1st only run once, so * fixednum
  				printf("denominator: %d \n",denominator[0]);
//  				record.est_path = aver_score/denominator[0]* fixednum;
  				record.est_path = aver_score/sample_time* fixednum;
  				auto sampling_end = std::chrono::high_resolution_clock::now();
				record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
          	}
          	return 0;
}