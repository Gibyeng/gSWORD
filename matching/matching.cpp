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


void generateValidCandidateIndex(ui depth, ui *idx_embedding, ui *idx_count, ui **valid_candidate_index,
                                                Edges ***edge_matrix, ui **bn, ui *bn_cnt, ui *order,
                                                ui *&temp_buffer) {
    VertexID u = order[depth];
    VertexID previous_bn = bn[depth][0];
    ui previous_index_id = idx_embedding[previous_bn];
    ui valid_candidates_count = 0;

#if ENABLE_QFLITER == 1
    BSRGraph &bsr_graph = *qfliter_bsr_graph_[previous_bn][u];
    BSRSet &bsr_set = bsr_graph.bsrs[previous_index_id];

    if (bsr_set.size_ != 0){
        offline_bsr_trans_uint(bsr_set.base_, bsr_set.states_, bsr_set.size_,
                               (int *) valid_candidate_index[depth]);
        // represent bsr size
        valid_candidates_count = bsr_set.size_;
    }

    if (bn_cnt[depth] > 0) {
        if (temp_bsr_base1_ == nullptr) { temp_bsr_base1_ = new int[1024 * 1024]; }
        if (temp_bsr_state1_ == nullptr) { temp_bsr_state1_ = new int[1024 * 1024]; }
        if (temp_bsr_base2_ == nullptr) { temp_bsr_base2_ = new int[1024 * 1024]; }
        if (temp_bsr_state2_ == nullptr) { temp_bsr_state2_ = new int[1024 * 1024]; }
        int *res_base_ = temp_bsr_base1_;
        int *res_state_ = temp_bsr_state1_;
        int *input_base_ = temp_bsr_base2_;
        int *input_state_ = temp_bsr_state2_;

        memcpy(input_base_, bsr_set.base_, sizeof(int) * bsr_set.size_);
        memcpy(input_state_, bsr_set.states_, sizeof(int) * bsr_set.size_);

        for (ui i = 1; i < bn_cnt[depth]; ++i) {
            VertexID current_bn = bn[depth][i];
            ui current_index_id = idx_embedding[current_bn];
            BSRGraph &cur_bsr_graph = *qfliter_bsr_graph_[current_bn][u];
            BSRSet &cur_bsr_set = cur_bsr_graph.bsrs[current_index_id];

            if (valid_candidates_count == 0 || cur_bsr_set.size_ == 0) {
                valid_candidates_count = 0;
                break;
            }

            valid_candidates_count = intersect_qfilter_bsr_b4_v2(cur_bsr_set.base_, cur_bsr_set.states_,
                                                                 cur_bsr_set.size_,
                                                                 input_base_, input_state_, valid_candidates_count,
                                                                 res_base_, res_state_);

            swap(res_base_, input_base_);
            swap(res_state_, input_state_);
        }

        if (valid_candidates_count != 0) {
            valid_candidates_count = offline_bsr_trans_uint(input_base_, input_state_, valid_candidates_count,
                                                            (int *) valid_candidate_index[depth]);
        }
    }

    idx_count[depth] = valid_candidates_count;

    // Debugging.
#ifdef YCHE_DEBUG
    Edges &previous_edge = *edge_matrix[previous_bn][u];

    auto gt_valid_candidates_count =
            previous_edge.offset_[previous_index_id + 1] - previous_edge.offset_[previous_index_id];
    ui *previous_candidates = previous_edge.edge_ + previous_edge.offset_[previous_index_id];
    ui *gt_valid_candidate_index = new ui[1024 * 1024];
    memcpy(gt_valid_candidate_index, previous_candidates, gt_valid_candidates_count * sizeof(ui));

    ui temp_count;
    for (ui i = 1; i < bn_cnt[depth]; ++i) {
        VertexID current_bn = bn[depth][i];
        Edges &current_edge = *edge_matrix[current_bn][u];
        ui current_index_id = idx_embedding[current_bn];

        ui current_candidates_count =
                current_edge.offset_[current_index_id + 1] - current_edge.offset_[current_index_id];
        ui *current_candidates = current_edge.edge_ + current_edge.offset_[current_index_id];

        ComputeSetIntersection::ComputeCandidates(current_candidates, current_candidates_count,
                                                  gt_valid_candidate_index, gt_valid_candidates_count, temp_buffer,
                                                  temp_count);

        std::swap(temp_buffer, gt_valid_candidate_index);
        gt_valid_candidates_count = temp_count;
    }
    assert(valid_candidates_count == gt_valid_candidates_count);

    cout << "Ret, Level:" << bn_cnt[depth] << ", BSR:"
         << pretty_print_array(valid_candidate_index[depth], valid_candidates_count)
         << "; GT: " << pretty_print_array(gt_valid_candidate_index, gt_valid_candidates_count) << "\n";

    for (auto i = 0; i < valid_candidates_count; i++) {
        assert(gt_valid_candidate_index[i] == valid_candidate_index[depth][i]);
    }
    delete[] gt_valid_candidate_index;
#endif
#else
    Edges& previous_edge = *edge_matrix[previous_bn][u];

    valid_candidates_count = previous_edge.offset_[previous_index_id + 1] - previous_edge.offset_[previous_index_id];
    ui* previous_candidates = previous_edge.edge_ + previous_edge.offset_[previous_index_id];

    memcpy(valid_candidate_index[depth], previous_candidates, valid_candidates_count * sizeof(ui));

    ui temp_count;
    for (ui i = 1; i < bn_cnt[depth]; ++i) {
        VertexID current_bn = bn[depth][i];
        Edges& current_edge = *edge_matrix[current_bn][u];
        ui current_index_id = idx_embedding[current_bn];

        ui current_candidates_count = current_edge.offset_[current_index_id + 1] - current_edge.offset_[current_index_id];
        ui* current_candidates = current_edge.edge_ + current_edge.offset_[current_index_id];

        ComputeSetIntersection::ComputeCandidates(current_candidates, current_candidates_count, valid_candidate_index[depth], valid_candidates_count,
                        temp_buffer, temp_count);

        std::swap(temp_buffer, valid_candidate_index[depth]);
        valid_candidates_count = temp_count;
    }

    idx_count[depth] = valid_candidates_count;

#endif
}

void generateValidCandidateIndex2(ui depth, ui *idx_embedding, ui *idx_count, ui **valid_candidate_index,
                                                Edges ***edge_matrix, ui **bn, ui *bn_cnt, ui *order,
                                                ui *&temp_buffer,unsigned long long &intersection_count, unsigned long long &compare_count) {
    VertexID u = order[depth];
    VertexID previous_bn = bn[depth][0];
    ui previous_index_id = idx_embedding[previous_bn];
    ui valid_candidates_count = 0;

    Edges& previous_edge = *edge_matrix[previous_bn][u];

    valid_candidates_count = previous_edge.offset_[previous_index_id + 1] - previous_edge.offset_[previous_index_id];
    ui* previous_candidates = previous_edge.edge_ + previous_edge.offset_[previous_index_id];



    memcpy(valid_candidate_index[depth], previous_candidates, valid_candidates_count * sizeof(ui));

    ui temp_count;
    for (ui i = 1; i < bn_cnt[depth]; ++i) {
        VertexID current_bn = bn[depth][i];
        Edges& current_edge = *edge_matrix[current_bn][u];
        ui current_index_id = idx_embedding[current_bn];

        ui current_candidates_count = current_edge.offset_[current_index_id + 1] - current_edge.offset_[current_index_id];
        ui* current_candidates = current_edge.edge_ + current_edge.offset_[current_index_id];



        ComputeSetIntersection::ComputeCandidates(current_candidates, current_candidates_count, valid_candidate_index[depth], valid_candidates_count,
                        temp_buffer, temp_count, compare_count);
        std::swap(temp_buffer, valid_candidate_index[depth]);
        valid_candidates_count = temp_count;
        intersection_count ++;


    }

    idx_count[depth] = valid_candidates_count;

}

void generateValidCandidateIndex_BS(ui depth, ui *idx_embedding, ui *idx_count, ui **valid_candidate_index,
                                                Edges ***edge_matrix, ui **bn, ui *bn_cnt, ui *order,
                                                ui *&temp_buffer) {
    VertexID u = order[depth];
    VertexID previous_bn = bn[depth][0];
    ui previous_index_id = idx_embedding[previous_bn];
    ui valid_candidates_count = 0;

    Edges& previous_edge = *edge_matrix[previous_bn][u];

    valid_candidates_count = previous_edge.offset_[previous_index_id + 1] - previous_edge.offset_[previous_index_id];
    ui* previous_candidates = previous_edge.edge_ + previous_edge.offset_[previous_index_id];

    memcpy(valid_candidate_index[depth], previous_candidates, valid_candidates_count * sizeof(ui));
    ui success = 0;
    for (ui j = 0; j < valid_candidates_count; ++j ){
		ui candidate_node = previous_candidates[j];
		ui find = 1;
		for (ui i = 1; i < bn_cnt[depth]; ++i) {
			VertexID current_bn = bn[depth][i];
			Edges& current_edge = *edge_matrix[current_bn][u];
			ui current_index_id = idx_embedding[current_bn];

			ui current_candidates_count = current_edge.offset_[current_index_id + 1] - current_edge.offset_[current_index_id];
			ui* current_candidates = current_edge.edge_ + current_edge.offset_[current_index_id];

			if (binarySearch(current_candidates, candidate_node, 0, current_candidates_count - 1 )){

			}else{
				find = 0;
				break;
			}

		}
		success += find;
		if (find ){
			valid_candidate_index[depth][success - 1] = candidate_node;
		}
	 }
    idx_count[depth] = success;
}

void generateValidCandidateIndex_partially(ui depth, ui *idx_embedding, double *idx_count, ui **valid_candidate_index,
                                                Edges ***edge_matrix, ui **bn, ui *bn_cnt, ui *order,
                                                ui *&temp_buffer) {

    VertexID u = order[depth];
    VertexID previous_bn = bn[depth][0];
    ui previous_index_id = idx_embedding[previous_bn];
    ui valid_candidates_count = 0;

    Edges& previous_edge = *edge_matrix[previous_bn][u];

    valid_candidates_count = previous_edge.offset_[previous_index_id + 1] - previous_edge.offset_[previous_index_id];
    ui* previous_candidates = previous_edge.edge_ + previous_edge.offset_[previous_index_id];

    memcpy(valid_candidate_index[depth], previous_candidates, valid_candidates_count * sizeof(ui));
    ui success = 0;
    for (ui j = 0; j < valid_candidates_count*0.1; ++j ){
		ui candidate_node = previous_candidates[j];
		ui find = 1;
		for (ui i = 1; i < bn_cnt[depth]; ++i) {
			VertexID current_bn = bn[depth][i];
			Edges& current_edge = *edge_matrix[current_bn][u];
			ui current_index_id = idx_embedding[current_bn];

			ui current_candidates_count = current_edge.offset_[current_index_id + 1] - current_edge.offset_[current_index_id];
			ui* current_candidates = current_edge.edge_ + current_edge.offset_[current_index_id];

			if (binarySearch(current_candidates, candidate_node, 0, current_candidates_count - 1 )){

			}else{
				find = 0;
				break;
			}

		}
		success += find;
		if (find ){
			valid_candidate_index[depth][success - 1] = candidate_node;
		}
	 }
    idx_count[depth] = success;
}


size_t LFTJ(const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates,
                    ui *candidates_count,
                    ui *order, size_t output_limit_num, size_t &call_count, timer &record) {

	auto start = std::chrono::high_resolution_clock::now();
	record. enumerating_time = 0;

#ifdef DISTRIBUTION
    distribution_count_ = new size_t[data_graph->getVerticesCount()];
    memset(distribution_count_, 0, data_graph->getVerticesCount() * sizeof(size_t));
    size_t* begin_count = new size_t[query_graph->getVerticesCount()];
    memset(begin_count, 0, query_graph->getVerticesCount() * sizeof(size_t));
#endif

    // Generate bn.
    ui **bn;
    ui *bn_count;
    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
    bool *visited_vertices;
    allocateBuffer(data_graph, query_graph, candidates_count, idx, idx_count, embedding, idx_embedding,
                   temp_buffer, valid_candidate_idx, visited_vertices);

    size_t embedding_cnt = 0;
    int cur_depth = 0;
    int max_depth = query_graph->getVerticesCount();
    VertexID start_vertex = order[0];

    idx[cur_depth] = 0;
    idx_count[cur_depth] = candidates_count[start_vertex];

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

#ifdef ENABLE_FAILING_SET
    std::vector<std::bitset<MAXIMUM_QUERY_GRAPH_SIZE>> ancestors;
    computeAncestor(query_graph, bn, bn_count, order, ancestors);
    std::vector<std::bitset<MAXIMUM_QUERY_GRAPH_SIZE>> vec_failing_set(max_depth);
    std::unordered_map<VertexID, VertexID> reverse_embedding;
    reverse_embedding.reserve(MAXIMUM_QUERY_GRAPH_SIZE * 2);
#endif

#ifdef SPECTRUM
    exit_ = false;
#endif

    while (true) {
        while (idx[cur_depth] < idx_count[cur_depth]) {
            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;
#ifdef ENABLE_FAILING_SET
                vec_failing_set[cur_depth] = ancestors[u];
                vec_failing_set[cur_depth] |= ancestors[reverse_embedding[v]];
                vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
#endif
                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;

#ifdef DISTRIBUTION
            begin_count[cur_depth] = embedding_cnt;
            // printf("Cur Depth: %d, v: %u, begin: %zu\n", cur_depth, v, embedding_cnt);
#endif

#ifdef ENABLE_FAILING_SET
            reverse_embedding[v] = u;
#endif

            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                visited_vertices[v] = false;

#ifdef DISTRIBUTION
                distribution_count_[v] += 1;
#endif

#ifdef ENABLE_FAILING_SET
                reverse_embedding.erase(embedding[u]);
                vec_failing_set[cur_depth].set();
                vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
#endif
//                if (embedding_cnt >= output_limit_num) {
//                    goto EXIT;
//                }
            } else {
                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer);

#ifdef ENABLE_FAILING_SET
                if (idx_count[cur_depth] == 0) {
                    vec_failing_set[cur_depth - 1] = ancestors[order[cur_depth]];
                } else {
                    vec_failing_set[cur_depth - 1].reset();
                }
#endif
            }
        }

#ifdef SPECTRUM
        if (exit_) {
            goto EXIT;
        }
#endif

        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];
#ifdef ENABLE_FAILING_SET
            reverse_embedding.erase(embedding[u]);
            if (cur_depth != 0) {
                if (!vec_failing_set[cur_depth].test(u)) {
                    vec_failing_set[cur_depth - 1] = vec_failing_set[cur_depth];
                    idx[cur_depth] = idx_count[cur_depth];
                } else {
                    vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
                }
            }
#endif
            visited_vertices[embedding[u]] = false;

#ifdef DISTRIBUTION
            distribution_count_[embedding[u]] += embedding_cnt - begin_count[cur_depth];
            // printf("Cur Depth: %d, v: %u, begin: %zu, end: %zu\n", cur_depth, embedding[u], begin_count[cur_depth], embedding_cnt);
#endif
        }
    }

    // Release the buffer.

#ifdef DISTRIBUTION
    if (embedding_cnt >= output_limit_num) {
        for (int i = 0; i < max_depth - 1; ++i) {
            ui v = embedding[order[i]];
            distribution_count_[v] += embedding_cnt - begin_count[i];
        }
    }
    delete[] begin_count;
#endif

    EXIT:
    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
                  visited_vertices,
                  bn, bn_count);

#if ENABLE_QFLITER == 1
    delete[] temp_bsr_base1_;
    delete[] temp_bsr_base2_;
    delete[] temp_bsr_state1_;
    delete[] temp_bsr_state2_;

    for (ui i = 0; i < max_depth; ++i) {
        for (ui j = 0; j < max_depth; ++j) {
//            delete qfliter_bsr_graph_[i][j];
        }
        delete[] qfliter_bsr_graph_[i];
    }
    delete[] qfliter_bsr_graph_;
#endif

    auto end = std::chrono::high_resolution_clock::now();
	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count();
    return embedding_cnt;
}

/*GPU intersection*/

void generateValidCandidateIndexByGPU(ui depth, ui *idx_embedding, ui *idx_count, ui **valid_candidate_index,
                                                Edges ***edge_matrix, ui **bn, ui *bn_cnt, ui *order,
                                                ui *&temp_buffer) {
	VertexID u = order[depth];
    VertexID previous_bn = bn[depth][0];
    ui previous_index_id = idx_embedding[previous_bn];
    ui valid_candidates_count = 0;
    Edges& previous_edge = *edge_matrix[previous_bn][u];

    valid_candidates_count = previous_edge.offset_[previous_index_id + 1] - previous_edge.offset_[previous_index_id];
    ui old_candidates_count = valid_candidates_count;
    ui* previous_candidates = previous_edge.edge_ + previous_edge.offset_[previous_index_id];
    //copy candidates from graph
    memcpy(valid_candidate_index[depth], previous_candidates, valid_candidates_count * sizeof(ui));

    ui temp_count;
    //transform to GPU
    // candidate set that to be joined in GPU
    ui* d_joinlists;
    // number of candidate in each list , size = bn_cnt[depth] +1
    ui* d_listOffsets;
    ui* h_listOffsets;
    ui* d_results;
    ui* d_results_count;
    ui* d_suffix_sum;

    cudaMalloc(&d_results_count,      sizeof(ui) * valid_candidates_count);
    cudaMalloc(&d_suffix_sum,      sizeof(ui) * valid_candidates_count);
    //compute the size of all edges to be joined
    cudaMalloc(&d_listOffsets,         sizeof(ui) * (bn_cnt[depth]+1 ));

    h_listOffsets = new ui [ bn_cnt[depth] +1];
    h_listOffsets[0] = 0;
    h_listOffsets[1] = valid_candidates_count;
    for (ui i = 1; i < bn_cnt[depth]; ++i) {
    	VertexID current_bn = bn[depth][i];
		Edges& current_edge = *edge_matrix[current_bn][u];
		ui current_index_id = idx_embedding[current_bn];
		ui current_candidates_count = current_edge.offset_[current_index_id + 1] - current_edge.offset_[current_index_id];
		ui* current_candidates = current_edge.edge_ + current_edge.offset_[current_index_id];
		//transfer memory from host to GPU
		h_listOffsets[i+1] = current_candidates_count + h_listOffsets[i];
    }
    cudaMemcpy(d_listOffsets, h_listOffsets, sizeof(ui) * (bn_cnt[depth]+1 ), cudaMemcpyHostToDevice);
    size_t sizeoflist = h_listOffsets[bn_cnt[depth]];

    cudaMalloc(&d_joinlists,      sizeof(ui) * sizeoflist);

    // assign joinlists
    // the first list
    cudaMemcpy(d_joinlists, valid_candidate_index[depth], sizeof(ui) * valid_candidates_count, cudaMemcpyHostToDevice);

    for (ui i = 1; i < bn_cnt[depth]; ++i) {
       	VertexID current_bn = bn[depth][i];
   		Edges& current_edge = *edge_matrix[current_bn][u];
   		ui current_index_id = idx_embedding[current_bn];
   		ui* current_candidates = current_edge.edge_ + current_edge.offset_[current_index_id];
   		ui current_candidates_count = current_edge.offset_[current_index_id + 1] - current_edge.offset_[current_index_id];
   		ui begin = h_listOffsets[i];
   		cudaMemcpy(d_joinlists+begin, current_candidates, sizeof(ui) * current_candidates_count, cudaMemcpyHostToDevice);
    }
    // assign threads ( binary search on one array and find results in anothers)
    /* we should assign warpnumber = valid_candidates_count* /
     *
     */
    ui blocknum = divup( valid_candidates_count, 128);
    bool flag = false;
    if (blocknum > 0 ){
		if(bn_cnt[depth] > 0){
			GPUsetIntersection_count <<<blocknum,128,0,0>>>(d_joinlists, d_listOffsets,  bn_cnt[depth], d_results_count);
			/* compute suffix sum
			 */
			void *d_temp_storage = NULL;
			size_t temp_storage_bytes = 0;
			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,d_results_count,d_suffix_sum, valid_candidates_count);
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,d_results_count,d_suffix_sum, valid_candidates_count);
			cudaFree (d_temp_storage);
			/*compute how many elements are needed, the last element in suffix sum*/
			ui total_element = 0;
			cudaMemcpy(&total_element, d_suffix_sum + valid_candidates_count-1, sizeof(ui) , cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			if(total_element >0){
				flag = true;
			}
			cudaMalloc(&d_results, total_element*sizeof(ui));
			/* compute results*/
			GPUsetIntersection<<<blocknum , 128,0,0>>>(d_joinlists,d_suffix_sum , d_results, valid_candidates_count);
			/*copy results to CPU*/
			cudaMemcpy( valid_candidate_index[depth], d_results, sizeof(ui) * total_element, cudaMemcpyDeviceToHost);
			valid_candidates_count = total_element;
		}
    }

    idx_count[depth] = valid_candidates_count;
    //free memory
    if (old_candidates_count !=0){
    	 cudaFree (d_results_count);
    	 cudaFree (d_suffix_sum);
    	 cudaFree (d_joinlists);
    	 cudaFree (d_listOffsets);
    }
    if(flag){
    	cudaFree (d_results);
    }


}

void generateValidCandidateWJ(ui depth, ui *idx_embedding, ui *idx_count, ui **valid_candidate_index,
                                                Edges ***edge_matrix, ui **bn, ui *bn_cnt, ui *order,
                                                ui *&temp_buffer) {
	VertexID u = order[depth];
    VertexID previous_bn = bn[depth][0];
    ui previous_index_id = idx_embedding[previous_bn];
    ui valid_candidates_count = 0;

    Edges& previous_edge = *edge_matrix[previous_bn][u];

    valid_candidates_count = previous_edge.offset_[previous_index_id + 1] - previous_edge.offset_[previous_index_id];
    ui* previous_candidates = previous_edge.edge_ + previous_edge.offset_[previous_index_id];
    //copy candidates from graph
    memcpy(valid_candidate_index[depth], previous_candidates, valid_candidates_count * sizeof(ui));
    idx_count[depth] = valid_candidates_count;
}

size_t LFTJ_subtreesize(const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates,
                    ui *candidates_count,
                    ui *order, size_t output_limit_num, size_t &call_count, size_t * subtree_size) {


    // Generate bn.
    ui **bn;
    ui *bn_count;
    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
    bool *visited_vertices;
    allocateBuffer(data_graph, query_graph, candidates_count, idx, idx_count, embedding, idx_embedding,
                   temp_buffer, valid_candidate_idx, visited_vertices);

    size_t embedding_cnt = 0;
    int cur_depth = 0;
    int max_depth = query_graph->getVerticesCount();
    VertexID start_vertex = order[0];

    idx[cur_depth] = 0;
    idx_count[cur_depth] = candidates_count[start_vertex];

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

#ifdef ENABLE_FAILING_SET
    std::vector<std::bitset<MAXIMUM_QUERY_GRAPH_SIZE>> ancestors;
    computeAncestor(query_graph, bn, bn_count, order, ancestors);
    std::vector<std::bitset<MAXIMUM_QUERY_GRAPH_SIZE>> vec_failing_set(max_depth);
    std::unordered_map<VertexID, VertexID> reverse_embedding;
    reverse_embedding.reserve(MAXIMUM_QUERY_GRAPH_SIZE * 2);
#endif

#ifdef SPECTRUM
    exit_ = false;
#endif

    while (true) {
        while (idx[cur_depth] < idx_count[cur_depth]) {
            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;
#ifdef ENABLE_FAILING_SET
                vec_failing_set[cur_depth] = ancestors[u];
                vec_failing_set[cur_depth] |= ancestors[reverse_embedding[v]];
                vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
#endif
                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;

#ifdef DISTRIBUTION
            begin_count[cur_depth] = embedding_cnt;
            // printf("Cur Depth: %d, v: %u, begin: %zu\n", cur_depth, v, embedding_cnt);
#endif

#ifdef ENABLE_FAILING_SET
            reverse_embedding[v] = u;
#endif

            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                subtree_size[idx_embedding[order[0]]] += 1;
                visited_vertices[v] = false;

#ifdef DISTRIBUTION
                distribution_count_[v] += 1;
#endif

#ifdef ENABLE_FAILING_SET
                reverse_embedding.erase(embedding[u]);
                vec_failing_set[cur_depth].set();
                vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
#endif
                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {
                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;

                generateValidCandidateIndex(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer);

#ifdef ENABLE_FAILING_SET
                if (idx_count[cur_depth] == 0) {
                    vec_failing_set[cur_depth - 1] = ancestors[order[cur_depth]];
                } else {
                    vec_failing_set[cur_depth - 1].reset();
                }
#endif
            }
        }

#ifdef SPECTRUM
        if (exit_) {
            goto EXIT;
        }
#endif

        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];
#ifdef ENABLE_FAILING_SET
            reverse_embedding.erase(embedding[u]);
            if (cur_depth != 0) {
                if (!vec_failing_set[cur_depth].test(u)) {
                    vec_failing_set[cur_depth - 1] = vec_failing_set[cur_depth];
                    idx[cur_depth] = idx_count[cur_depth];
                } else {
                    vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
                }
            }
#endif
            visited_vertices[embedding[u]] = false;

#ifdef DISTRIBUTION
            distribution_count_[embedding[u]] += embedding_cnt - begin_count[cur_depth];
            // printf("Cur Depth: %d, v: %u, begin: %zu, end: %zu\n", cur_depth, embedding[u], begin_count[cur_depth], embedding_cnt);
#endif
        }
    }

    // Release the buffer.

#ifdef DISTRIBUTION
    if (embedding_cnt >= output_limit_num) {
        for (int i = 0; i < max_depth - 1; ++i) {
            ui v = embedding[order[i]];
            distribution_count_[v] += embedding_cnt - begin_count[i];
        }
    }
    delete[] begin_count;
#endif

    EXIT:
    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
                  visited_vertices,
                  bn, bn_count);
    // release GPU memory

#if ENABLE_QFLITER == 1
    delete[] temp_bsr_base1_;
    delete[] temp_bsr_base2_;
    delete[] temp_bsr_state1_;
    delete[] temp_bsr_state2_;

    for (ui i = 0; i < max_depth; ++i) {
        for (ui j = 0; j < max_depth; ++j) {
//            delete qfliter_bsr_graph_[i][j];
        }
        delete[] qfliter_bsr_graph_[i];
    }
    delete[] qfliter_bsr_graph_;
#endif

    return embedding_cnt;
}


size_t LFTJ_spanningtreesize(const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates,
                    ui *candidates_count,
                    ui *order, size_t output_limit_num, size_t &call_count, size_t * subtree_size) {


    // Generate bn.
    ui **bn;
    ui *bn_count;
    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
    bool *visited_vertices;
    allocateBuffer(data_graph, query_graph, candidates_count, idx, idx_count, embedding, idx_embedding,
                   temp_buffer, valid_candidate_idx, visited_vertices);

    size_t embedding_cnt = 0;
    int cur_depth = 0;
    int max_depth = query_graph->getVerticesCount();
    VertexID start_vertex = order[0];

    idx[cur_depth] = 0;
    idx_count[cur_depth] = candidates_count[start_vertex];

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }



    while (true) {
        while (idx[cur_depth] < idx_count[cur_depth]) {
            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;

                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;


            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                subtree_size[idx_embedding[order[0]]] += 1;
                visited_vertices[v] = false;


                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {
                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;

                generateValidCandidateWJ(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer);


            }
        }



        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];

            visited_vertices[embedding[u]] = false;


        }
    }

    // Release the buffer.


    EXIT:
    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
                  visited_vertices,
                  bn, bn_count);
    // release GPU memory

    return embedding_cnt;
}

size_t LFTJ_GPU(const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates,
                    ui *candidates_count,
                    ui *order, size_t output_limit_num, size_t &call_count) {


    // Generate bn.
    ui **bn;
    ui *bn_count;
    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
    bool *visited_vertices;
    allocateBuffer(data_graph, query_graph, candidates_count, idx, idx_count, embedding, idx_embedding,
                   temp_buffer, valid_candidate_idx, visited_vertices);

    size_t embedding_cnt = 0;
    int cur_depth = 0;
    int max_depth = query_graph->getVerticesCount();
    VertexID start_vertex = order[0];

    idx[cur_depth] = 0;
    idx_count[cur_depth] = candidates_count[start_vertex];

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

#ifdef ENABLE_FAILING_SET
    std::vector<std::bitset<MAXIMUM_QUERY_GRAPH_SIZE>> ancestors;
    computeAncestor(query_graph, bn, bn_count, order, ancestors);
    std::vector<std::bitset<MAXIMUM_QUERY_GRAPH_SIZE>> vec_failing_set(max_depth);
    std::unordered_map<VertexID, VertexID> reverse_embedding;
    reverse_embedding.reserve(MAXIMUM_QUERY_GRAPH_SIZE * 2);
#endif

#ifdef SPECTRUM
    exit_ = false;
#endif

    while (true) {
        while (idx[cur_depth] < idx_count[cur_depth]) {
            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;
#ifdef ENABLE_FAILING_SET
                vec_failing_set[cur_depth] = ancestors[u];
                vec_failing_set[cur_depth] |= ancestors[reverse_embedding[v]];
                vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
#endif
                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;

#ifdef DISTRIBUTION
            begin_count[cur_depth] = embedding_cnt;
            // printf("Cur Depth: %d, v: %u, begin: %zu\n", cur_depth, v, embedding_cnt);
#endif

#ifdef ENABLE_FAILING_SET
            reverse_embedding[v] = u;
#endif

            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                visited_vertices[v] = false;

#ifdef DISTRIBUTION
                distribution_count_[v] += 1;
#endif

#ifdef ENABLE_FAILING_SET
                reverse_embedding.erase(embedding[u]);
                vec_failing_set[cur_depth].set();
                vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
#endif
                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {
                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;

                generateValidCandidateIndexByGPU(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer);

#ifdef ENABLE_FAILING_SET
                if (idx_count[cur_depth] == 0) {
                    vec_failing_set[cur_depth - 1] = ancestors[order[cur_depth]];
                } else {
                    vec_failing_set[cur_depth - 1].reset();
                }
#endif
            }
        }

#ifdef SPECTRUM
        if (exit_) {
            goto EXIT;
        }
#endif

        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];
#ifdef ENABLE_FAILING_SET
            reverse_embedding.erase(embedding[u]);
            if (cur_depth != 0) {
                if (!vec_failing_set[cur_depth].test(u)) {
                    vec_failing_set[cur_depth - 1] = vec_failing_set[cur_depth];
                    idx[cur_depth] = idx_count[cur_depth];
                } else {
                    vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
                }
            }
#endif
            visited_vertices[embedding[u]] = false;

#ifdef DISTRIBUTION
            distribution_count_[embedding[u]] += embedding_cnt - begin_count[cur_depth];
            // printf("Cur Depth: %d, v: %u, begin: %zu, end: %zu\n", cur_depth, embedding[u], begin_count[cur_depth], embedding_cnt);
#endif
        }
    }

    // Release the buffer.

#ifdef DISTRIBUTION
    if (embedding_cnt >= output_limit_num) {
        for (int i = 0; i < max_depth - 1; ++i) {
            ui v = embedding[order[i]];
            distribution_count_[v] += embedding_cnt - begin_count[i];
        }
    }
    delete[] begin_count;
#endif

    EXIT:
    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
                  visited_vertices,
                  bn, bn_count);
    // release GPU memory

#if ENABLE_QFLITER == 1
    delete[] temp_bsr_base1_;
    delete[] temp_bsr_base2_;
    delete[] temp_bsr_state1_;
    delete[] temp_bsr_state2_;

    for (ui i = 0; i < max_depth; ++i) {
        for (ui j = 0; j < max_depth; ++j) {
//            delete qfliter_bsr_graph_[i][j];
        }
        delete[] qfliter_bsr_graph_[i];
    }
    delete[] qfliter_bsr_graph_;
#endif

    return embedding_cnt;
}

/* allocate visited_vertices memory for sampling*/
void initMemorySampling (const Graph *data_graph, bool* &sampling_visited_vertices, bool* visited_vertices, ui cur_cand_num){
	ui data_vertices_num = data_graph->getVerticesCount();
	sampling_visited_vertices = new bool [data_vertices_num];
	memcpy(sampling_visited_vertices,visited_vertices,sizeof(bool)* data_vertices_num);
}

void resetMemorySampling(const Graph *data_graph, bool* &sampling_visited_vertices, bool* visited_vertices, ui cur_cand_num){
	ui data_vertices_num = data_graph->getVerticesCount();
//	memcpy(sampling_visited_vertices,visited_vertices,sizeof(bool)* data_vertices_num);
	memset(sampling_visited_vertices,0,sizeof(bool)* data_vertices_num);
}

/* sampling one path from current layer, return the product of each layer's candidate size */
ui samplingByGPU(Edges ***edge_matrix, ui **valid_candidate_idx, ui **candidates, ui *candidates_count, ui* idx_count, ui *order, ui **bn, ui *bn_count, bool* &sampling_visited_vertices, ui * embedding,ui *idx_embedding, ui *temp_buffer,ui root_idx,ui cur_depth ,ui max_depth,ui step, double &workloads)
{
	ui sample_depth = cur_depth;
	/* sampling one path rooted at candidate index "root_idx" then compute a score */
	ui score = 1;
	workloads = 1;
	/* sample index, we sample one path from rooted at v*/
	ui valid_idx = valid_candidate_idx[cur_depth][root_idx];
	VertexID u = order[cur_depth];
	VertexID v = candidates[u][valid_idx];
	if( v== 100000000){
		return 0;
	}
	ui resample = 0;
	while (sample_depth < cur_depth+ step  && sample_depth < max_depth){
		if(resample > 0){
			// ealry exit, return score equals to 0
			return 0;
		}
		/* at non-cur_depth we sample a new v*/
		if(sample_depth != cur_depth){
			ui idx_range = idx_count[sample_depth];
			// compute score
			if(resample == 0){
				score *= idx_range;
				workloads*= idx_range;
			}

			if (idx_range == 0){
				return 0;
			}
			// sample a candidate
			ui rand_num = rand();
			ui sample_idx = rand_num%idx_range;
			valid_idx = valid_candidate_idx[sample_depth][sample_idx];
			u = order[sample_depth];
			v = candidates[u][valid_idx];
			if( v== 100000000){
				return 0;
			}
		}

		if (sampling_visited_vertices[v]) {
			resample ++ ;

			continue;
		}

		/*reset resample*/
		resample = 0;
		embedding[u] = v;
		idx_embedding[u] = valid_idx;
		sampling_visited_vertices[v] = true;

		if (sample_depth ==  cur_depth+ step -1 || sample_depth == max_depth-1) {
			sample_depth += 1;
			sampling_visited_vertices[v] = false;

		} else {
			sample_depth += 1;

			generateValidCandidateIndexByGPU(sample_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
													bn_count, order, temp_buffer);

		}
	}
//	std::cout << "root_idx: " << root_idx << " score: "<< score <<std::endl;
	return score;
}

ui samplingByCPU(Edges ***edge_matrix, ui **valid_candidate_idx, ui **candidates, ui *candidates_count, ui* idx_count, ui *order, ui **bn, ui *bn_count, bool* &sampling_visited_vertices, ui * embedding,ui *idx_embedding, ui *temp_buffer,ui root_idx,ui cur_depth ,ui max_depth,ui step, double &workloads)
{
	ui sample_depth = cur_depth;
	/* sampling one path rooted at candidate index "root_idx" then compute a score */
	ui score = 1;
	workloads = 1;
	/* sample index, we sample one path from rooted at v*/
	ui valid_idx = valid_candidate_idx[cur_depth][root_idx];
	VertexID u = order[cur_depth];
	VertexID v = candidates[u][valid_idx];
	if( v== 100000000){
		return 0;
	}
	ui resample = 0;
	while (sample_depth < cur_depth+ step  && sample_depth < max_depth){
		if(resample > 0){
			// ealry exit, return score equals to 0
			return 0;
		}
		/* at non-cur_depth we sample a new v*/
		if(sample_depth != cur_depth){
			ui idx_range = idx_count[sample_depth];
			// compute score
			if(resample == 0){
				score *= idx_range;
				workloads*= idx_range;
			}

			if (idx_range == 0){
				return 0;
			}
			// sample a candidate
			ui rand_num = rand();
			ui sample_idx = rand_num%idx_range;
			valid_idx = valid_candidate_idx[sample_depth][sample_idx];
			u = order[sample_depth];
			v = candidates[u][valid_idx];
			if( v== 100000000){
				return 0;
			}
		}

		if (sampling_visited_vertices[v]) {
			resample ++ ;

			continue;
		}

		/*reset resample*/
		resample = 0;
		embedding[u] = v;
		idx_embedding[u] = valid_idx;
		sampling_visited_vertices[v] = true;

		if (sample_depth ==  cur_depth+ step -1 || sample_depth == max_depth-1) {
			sample_depth += 1;
			sampling_visited_vertices[v] = false;

		} else {
			sample_depth += 1;

			generateValidCandidateIndex(sample_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
													bn_count, order, temp_buffer);

		}
	}
//	std::cout << "root_idx: " << root_idx << " score: "<< score <<std::endl;
	return score;
}

ui samplingByCPU_2(Edges ***edge_matrix, ui **valid_candidate_idx, ui **candidates, ui *candidates_count, ui* idx_count, ui *order, ui **bn, ui *bn_count, bool* &sampling_visited_vertices, ui * embedding,ui *idx_embedding, ui *temp_buffer,ui root_idx,ui cur_depth ,ui max_depth,ui step, double &workloads)
{
	ui sample_depth = cur_depth;
	/* sampling one path rooted at candidate index "root_idx" then compute a score */
	ui score = 1;
	workloads = 1;
	/* sample index, we sample one path from rooted at v*/
	ui valid_idx = valid_candidate_idx[cur_depth][root_idx];
	VertexID u = order[cur_depth];
	VertexID v = candidates[u][valid_idx];
	if( v== 100000000){
		return 0;
	}
	ui resample = 0;
	while (sample_depth < cur_depth+ step  && sample_depth < max_depth){
		if(resample > 0){
			// ealry exit, return score equals to 0
			return 0;
		}
		/* at non-cur_depth we sample a new v*/
		if(sample_depth != cur_depth){
			ui idx_range = idx_count[sample_depth];
			// compute score
			if(resample == 0){
				score *= idx_range;
				workloads*= idx_range;
			}

			if (idx_range == 0){
				return 0;
			}
			// sample a candidate
			ui rand_num = rand();
			ui sample_idx = rand_num%idx_range;
			valid_idx = valid_candidate_idx[sample_depth][sample_idx];
			u = order[sample_depth];
			v = candidates[u][valid_idx];
			if( v== 100000000){
				return 0;
			}
		}

		if (sampling_visited_vertices[v]) {
			resample ++ ;

			continue;
		}

		/*reset resample*/
		resample = 0;
		embedding[u] = v;
		idx_embedding[u] = valid_idx;
		sampling_visited_vertices[v] = true;

		if (sample_depth ==  cur_depth+ step -1 || sample_depth == max_depth-1) {
			sample_depth += 1;
			sampling_visited_vertices[v] = false;

		} else {
			sample_depth += 1;

			generateValidCandidateIndex_BS(sample_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
													bn_count, order, temp_buffer);

		}
	}
//	std::cout << "root_idx: " << root_idx << " score: "<< score <<std::endl;
	return score;
}

double samplingByCPU_partially(Edges ***edge_matrix, ui **valid_candidate_idx, ui **candidates, ui *candidates_count, double* idx_count, ui *order, ui **bn, ui *bn_count, bool* &sampling_visited_vertices, ui * embedding,ui *idx_embedding, ui *temp_buffer,ui root_idx,ui cur_depth ,ui max_depth,ui step, double &workloads)
{
	ui sample_depth = cur_depth;
	/* sampling one path rooted at candidate index "root_idx" then compute a score */
	double score = 1;
	workloads = 1;
	/* sample index, we sample one path from rooted at v*/
	ui valid_idx = valid_candidate_idx[cur_depth][root_idx];
	VertexID u = order[cur_depth];
	VertexID v = candidates[u][valid_idx];
	if( v== 100000000){
		return 0;
	}
	ui resample = 0;
	while (sample_depth < cur_depth+ step  && sample_depth < max_depth){
		if(resample > 0){
			// ealry exit, return score equals to 0
			return 0;
		}
		/* at non-cur_depth we sample a new v*/
		if(sample_depth != cur_depth){
			double idx_range = idx_count[sample_depth];
			// compute score
			if(resample == 0){
				score *= idx_range;
				workloads*= idx_range;
			}

			if (idx_range == 0.0){
				return 0;
			}
			// sample a candidate
//			ui rand_idx = rand()%(ui)idx_range;
			ui rand_idx = 0;
			valid_idx = valid_candidate_idx[sample_depth][rand_idx];
			u = order[sample_depth];
			v = candidates[u][valid_idx];
			if( v== 100000000){
				return 0;
			}
		}

		if (sampling_visited_vertices[v]) {
			resample ++ ;

			continue;
		}

		/*reset resample*/
		resample = 0;
		embedding[u] = v;
		idx_embedding[u] = valid_idx;
		sampling_visited_vertices[v] = true;

		if (sample_depth ==  cur_depth+ step -1 || sample_depth == max_depth-1) {
			sample_depth += 1;
			sampling_visited_vertices[v] = false;

		} else {

			sample_depth += 1;

			generateValidCandidateIndex_partially(sample_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
													bn_count, order, temp_buffer);
		}
	}
//	std::cout << "root_idx: " << root_idx << " score: "<< score <<std::endl;
	return score;
}

/* sampling one path from current layer, return the product of each layer's candidate size */
ui samplingByGPU_branch(Edges ***edge_matrix, ui **valid_candidate_idx, ui **candidates, ui *candidates_count, ui* idx_count, ui *order, ui **bn, ui *bn_count, bool* &sampling_visited_vertices, ui * embedding,ui *idx_embedding, ui *temp_buffer,ui root_idx,ui cur_depth ,ui max_depth,ui step, double &workloads, double &b, ui &branch_sample_count)
{
	ui sample_depth = cur_depth;
	/* sampling one path rooted at candidate index "root_idx" then compute a score */
	ui score = 0;

	/* sample index, we sample one path from rooted at v*/
	ui valid_idx = valid_candidate_idx[cur_depth][root_idx];
	VertexID u = order[cur_depth];
	VertexID v = candidates[u][valid_idx];
	if( v== 100000000){
		branch_sample_count += 1;
		return 0;
	}
	ui resample = 0;
	// allocate
	ui* cur_b_idx = new ui [max_depth];
	for(int i = 0; i< max_depth; ++i){
		cur_b_idx[i] = 0;
	}

	if(idx_count[sample_depth] > 0){
		while (true) {

			while (cur_b_idx[sample_depth] <= ceil(b*(double)idx_count[sample_depth])){

				/* at non-cur_depth we sample a new v*/
				if(sample_depth != cur_depth){
					ui idx_range = idx_count[sample_depth];
					if (idx_range == 0){
						cur_b_idx[sample_depth] +=1;
						branch_sample_count += 1;
						continue;
					}

					ui rand_num = rand();
					ui sample_idx = rand_num%idx_range;

					valid_idx = valid_candidate_idx[sample_depth][sample_idx];
					u = order[sample_depth];
					v = candidates[u][valid_idx];
					if( v== 100000000){
						cur_b_idx[sample_depth] +=1;
						branch_sample_count += 1;
						continue;
					}
				}

				if (sampling_visited_vertices[v]) {
					// if we should add 1?
					branch_sample_count += 1;
					cur_b_idx[sample_depth] +=1;
					continue;
				}

				/*reset resample*/
				resample = 0;
				embedding[u] = v;
				idx_embedding[u] = valid_idx;
				sampling_visited_vertices[v] = true;
				cur_b_idx[sample_depth]+=1;

				if (sample_depth ==  cur_depth+ step -1 || sample_depth == max_depth-1) {
					sample_depth += 1;
					sampling_visited_vertices[v] = false;
					score += 1;
					branch_sample_count += 1;
				} else {
					sample_depth += 1;
					cur_b_idx[sample_depth] = 0;
					generateValidCandidateIndexByGPU(sample_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
															bn_count, order, temp_buffer);
				}
			}
			sample_depth -= 1;
			if (sample_depth == 0)
				break;
			else {
				VertexID u = order[sample_depth];
				sampling_visited_vertices[embedding[u]] = false;
			}
		}
	}else{
		branch_sample_count += 1;
	}
	delete [] cur_b_idx;

	return score;
}

/* sampling one subtree from current layer, return */
double samplingByGPU_branchV2(Edges ***edge_matrix, ui **valid_candidate_idx, ui **candidates, ui *candidates_count, ui* idx_count, ui *order, ui **bn, ui *bn_count, bool* &sampling_visited_vertices, ui * embedding,ui *idx_embedding, ui *temp_buffer,ui root_idx,ui cur_depth ,ui max_depth,ui step, double &workloads, double &b, unsigned long long &branch_sample_count)
{
	ui sample_depth = cur_depth;
	/* sampling one path rooted at candidate index "root_idx" then compute a score */
	double score = 0;
//	std::cout << "fixedNum: " << fixedNum <<std::endl;
	/* sample index, we sample one path from rooted at v*/
	ui valid_idx = valid_candidate_idx[cur_depth][root_idx];
	VertexID u = order[cur_depth];
	VertexID v = candidates[u][valid_idx];
	if( v== 100000000){
		branch_sample_count += 1;
		return 0;
	}
	ui resample = 0;
	// allocate
	ui* cur_b_idx = new ui [max_depth];
	ui* cur_b_range = new ui [max_depth];
	for(int i = 0; i< max_depth; ++i){
		cur_b_idx[i] = 0;
		cur_b_range[i] = 1;
	}
	cur_b_range[sample_depth] = idx_count[sample_depth];
	if(idx_count[sample_depth] > 0){
		while (true) {
			while (cur_b_idx[sample_depth] <  ceil(b* cur_b_range[sample_depth]) && cur_b_idx[sample_depth] < cur_b_range[sample_depth] ){
				cur_b_range[sample_depth] = idx_count[sample_depth];
				/* at non-cur_depth we sample a new v*/
				if(sample_depth != cur_depth){
					ui idx_range = idx_count[sample_depth];
					if (idx_range == 0){
						cur_b_idx[sample_depth] +=1;
						branch_sample_count += 1;
						continue;
					}
					ui rand_num = rand();
					ui sample_idx = (rand_num  )%idx_range;

					valid_idx = valid_candidate_idx[sample_depth][sample_idx];
					u = order[sample_depth];
					v = candidates[u][valid_idx];
					if( v== 100000000){
						cur_b_idx[sample_depth] +=1;
						branch_sample_count += 1;
						continue;
					}

				}

				if (sampling_visited_vertices[v]) {
					// if we should add 1?
					branch_sample_count += 1;
//					std::cout << "revisit "<<std::endl;
					cur_b_idx[sample_depth] +=1;
					continue;
				}

				/*reset resample*/
				resample = 0;
				embedding[u] = v;
				idx_embedding[u] = valid_idx;
				sampling_visited_vertices[v] = true;
				cur_b_idx[sample_depth]+=1;

				if (sample_depth ==  cur_depth+ step -1 || sample_depth == max_depth-1) {
					sample_depth += 1;
					sampling_visited_vertices[v] = false;
					double  accum = 1;
					for(int i = 1; i< max_depth; ++i){
						ui flexNum =  ceil(b* cur_b_range[i]);
						accum *= ((double)cur_b_range[i] / flexNum);
					}

					score += accum;
					branch_sample_count += 1;
				} else {
					sample_depth += 1;
					cur_b_idx[sample_depth] = 0;
					generateValidCandidateIndexByGPU(sample_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
															bn_count, order, temp_buffer);
				}
			}
			sample_depth -= 1;
			if (sample_depth == 0)
				break;
			else {
				VertexID u = order[sample_depth];
				sampling_visited_vertices[embedding[u]] = false;
			}
		}
	}else{
		branch_sample_count += 1;
	}
	delete [] cur_b_idx;
	delete [] cur_b_range;
	return score;
}


/* sampling one subtree from current layer, return */
double samplingByGPU_fixedbranch(Edges ***edge_matrix, ui **valid_candidate_idx, ui **candidates, ui *candidates_count, ui* idx_count, ui *order, ui **bn, ui *bn_count, bool* &sampling_visited_vertices, ui * embedding,ui *idx_embedding, ui *temp_buffer,ui root_idx,ui cur_depth ,ui max_depth,ui step, double &workloads, double &b, unsigned long long &branch_sample_count, ui fixedNum)
{
	ui sample_depth = cur_depth;
	/* sampling one path rooted at candidate index "root_idx" then compute a score */
	double score = 0;
//	std::cout << "fixedNum: " << fixedNum <<std::endl;
	/* sample index, we sample one path from rooted at v*/
	ui valid_idx = valid_candidate_idx[cur_depth][root_idx];
	VertexID u = order[cur_depth];
	VertexID v = candidates[u][valid_idx];
	if( v== 100000000){
		branch_sample_count += 1;
		return 0;
	}
	ui resample = 0;
	// allocate
	ui* cur_b_idx = new ui [max_depth];
	ui* cur_b_range = new ui [max_depth];
	for(int i = 0; i< max_depth; ++i){
		cur_b_idx[i] = 0;
		cur_b_range[i] = 1;
	}

	if(idx_count[sample_depth] > 0){
		while (true) {

			while (cur_b_idx[sample_depth] < fixedNum && cur_b_idx[sample_depth] < cur_b_range[sample_depth] ){

				cur_b_range[sample_depth] = idx_count[sample_depth];
				/* at non-cur_depth we sample a new v*/
				if(sample_depth != cur_depth){
					ui idx_range = idx_count[sample_depth];
					if (idx_range == 0){

						cur_b_idx[sample_depth] +=1;
						branch_sample_count += 1;
						continue;
					}
					ui rand_num = rand();
					ui sample_idx = (rand_num  )%idx_range;

					valid_idx = valid_candidate_idx[sample_depth][sample_idx];
					u = order[sample_depth];
					v = candidates[u][valid_idx];
					if( v== 100000000){
						cur_b_idx[sample_depth] +=1;
						branch_sample_count += 1;
						continue;
					}

				}

				if (sampling_visited_vertices[v]) {
					// if we should add 1?
					branch_sample_count += 1;
//					std::cout << "revisit "<<std::endl;
					cur_b_idx[sample_depth] +=1;
					continue;
				}

				/*reset resample*/
				resample = 0;
				embedding[u] = v;
				idx_embedding[u] = valid_idx;
				sampling_visited_vertices[v] = true;
				cur_b_idx[sample_depth]+=1;

				if (sample_depth ==  cur_depth+ step -1 || sample_depth == max_depth-1) {
					sample_depth += 1;
					sampling_visited_vertices[v] = false;
					double  accum = 1;
					for(int i = 1; i< max_depth; ++i){
						if(cur_b_range[i] > fixedNum){
							accum *= ((double)cur_b_range[i] / fixedNum);
						}
					}

					score += accum;
					branch_sample_count += 1;
				} else {
					sample_depth += 1;
					cur_b_idx[sample_depth] = 0;
					generateValidCandidateIndexByGPU(sample_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
															bn_count, order, temp_buffer);
				}
			}
			sample_depth -= 1;
			if (sample_depth == 0)
				break;
			else {
				VertexID u = order[sample_depth];
				sampling_visited_vertices[embedding[u]] = false;
			}
		}
	}else{
		branch_sample_count += 1;
	}
	delete [] cur_b_idx;
	delete [] cur_b_range;
	return score;
}

ui samplingallinGPU(Edges ***edge_matrix, ui **valid_candidate_idx, ui **candidates, ui *candidates_count, ui* idx_count, ui *order, ui **bn, ui *bn_count, bool* &sampling_visited_vertices, ui * embedding,ui *idx_embedding, ui *temp_buffer,ui root_idx,ui cur_depth ,ui max_depth,ui step, double &workloads)
{
	ui sample_depth = cur_depth;
	/* sampling one path rooted at candidate index "root_idx" then compute a score */
	ui score = 1;
	workloads = 1;
	/* sample index, we sample one path from rooted at v*/
	ui valid_idx = valid_candidate_idx[cur_depth][root_idx];
	VertexID u = order[cur_depth];
	VertexID v = candidates[u][valid_idx];
	if( v== 100000000){
		return 0;
	}
	ui resample = 0;
	while (sample_depth < cur_depth+ step  && sample_depth < max_depth){
		if(resample > 0){
			// ealry exit, return score equals to 0
			return 0;
		}
		/* at non-cur_depth we sample a new v*/
		if(sample_depth != cur_depth){
			ui idx_range = idx_count[sample_depth];
			// compute score
			if(resample == 0){
				score *= idx_range;
				workloads*= idx_range;
			}

			if (idx_range == 0){
				return 0;
			}
			// sample a candidate
			ui rand_num = rand();
			ui sample_idx = rand_num%idx_range;
			valid_idx = valid_candidate_idx[sample_depth][sample_idx];
			u = order[sample_depth];
			v = candidates[u][valid_idx];
			if( v== 100000000){
				return 0;
			}
		}

		if (sampling_visited_vertices[v]) {
			resample ++ ;

			continue;
		}

		/*reset resample*/
		resample = 0;
		embedding[u] = v;
		idx_embedding[u] = valid_idx;
		sampling_visited_vertices[v] = true;

		if (sample_depth ==  cur_depth+ step -1 || sample_depth == max_depth-1) {
			sample_depth += 1;
			sampling_visited_vertices[v] = false;

		} else {
			sample_depth += 1;

			generateValidCandidateIndexByGPU(sample_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
													bn_count, order, temp_buffer);

		}
	}
//	std::cout << "root_idx: " << root_idx << " score: "<< score <<std::endl;
	return score;
}


// each 1st layer candidate sample k times
// use GPU sample to lead CPU
ui LFTJ_GuideSearch (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record ){
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;
    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
    bool *visited_vertices;
    allocateBuffer(data_graph, query_graph, candidates_count, idx, idx_count, embedding, idx_embedding,
                   temp_buffer, valid_candidate_idx, visited_vertices);

    size_t embedding_cnt = 0;
    int cur_depth = 0;
    int max_depth = query_graph->getVerticesCount();
    VertexID start_vertex = order[0];

    idx[cur_depth] = 0;
    idx_count[cur_depth] = candidates_count[start_vertex];

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    while (true) {
//    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

    	while (idx[cur_depth] < idx_count[cur_depth]) {
        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();
        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
        		record.total_sample_count = record. sample_time* idx_count[0];
        		std::cout << "root num: " << idx_count[cur_depth];
        		for (int root_index = 0; root_index < idx_count[cur_depth];++root_index){
					aver_score = 0;
					double total_workloads = 0;
					double workloads= 0;
					ui sample_time = record. sample_time;

					for(ui iter = 0; iter < sample_time ; ++iter){
						resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
						double score = (double) samplingByGPU(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);
						aver_score +=score;

						total_workloads += workloads;
					}
					aver_score /= sample_time;
					total_workloads /= sample_time;
					record.est_path += aver_score;
					record.est_workload += total_workloads;
	//					auto reorder_start = std::chrono::high_resolution_clock::now();
	//					CindexScore scorepair;
	//					scorepair.candidateIndex = valid_candidate_idx[cur_depth][j];
	//					scorepair.score = aver_score;
	//					sorted_arr.push_back(scorepair);
	//					auto reorder_end = std::chrono::high_resolution_clock::now();
	//					time_record.reorder_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(reorder_end - reorder_start).count();

	//        		auto reorder_start = std::chrono::high_resolution_clock::now();
	//        		//sort intersected valid_candidate_idx arr by score
	//        		std::sort(sorted_arr.begin(), sorted_arr.end(), comparator);
	//        		// copy data back to valid_candidate_idx
	//        		for (ui j = 0; j < idx_count[cur_depth];j++){
	//        			valid_candidate_idx[cur_depth][j] = sorted_arr[j].candidateIndex;
	//        		}
	//        		auto reorder_end = std::chrono::high_resolution_clock::now();
	//        		time_record.reorder_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(reorder_end - reorder_start).count();
        		}
        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;

                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;


            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                record. real_workload +=1;
                visited_vertices[v] = false;

                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {


                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

            	if(idx_count[cur_depth] == 0){
            		record. real_workload +=1;
            	}
            }
        }


        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];

            visited_vertices[embedding[u]] = false;


        }
    }



    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;

}

// random pick then sample
ui LFTJ_GuideSearch_v2 (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;
    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    while (true) {
//    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

    	while (idx[cur_depth] < idx_count[cur_depth]) {
        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();
        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {

        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
        		record.total_sample_count = record. sample_time* idx_count[0];
        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth];
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;
				ui sample_time = record. sample_time;

				for(ui iter = 0; iter < sample_time ; ++iter){
					ui root_index = rand()%idx_count[cur_depth];
					if(iter == 0 ){
						std::cout <<"random pick:"<< root_index<<"~~"<<std::endl;
					}
					resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
					double score = (double) samplingByGPU(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);
					aver_score += score;

					total_workloads += workloads;
				}
				aver_score /= sample_time;
				total_workloads /= sample_time;
				record.est_path = aver_score*idx_count[cur_depth];
				record.est_workload = total_workloads*idx_count[cur_depth];
//					auto reorder_start = std::chrono::high_resolution_clock::now();
//					CindexScore scorepair;
//					scorepair.candidateIndex = valid_candidate_idx[cur_depth][j];
//					scorepair.score = aver_score;
//					sorted_arr.push_back(scorepair);
//					auto reorder_end = std::chrono::high_resolution_clock::now();
//					time_record.reorder_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(reorder_end - reorder_start).count();

//        		auto reorder_start = std::chrono::high_resolution_clock::now();
//        		//sort intersected valid_candidate_idx arr by score
//        		std::sort(sorted_arr.begin(), sorted_arr.end(), comparator);
//        		// copy data back to valid_candidate_idx
//        		for (ui j = 0; j < idx_count[cur_depth];j++){
//        			valid_candidate_idx[cur_depth][j] = sorted_arr[j].candidateIndex;
//        		}
//        		auto reorder_end = std::chrono::high_resolution_clock::now();
//        		time_record.reorder_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(reorder_end - reorder_start).count();
        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;

                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;


            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                record. real_workload +=1;
                visited_vertices[v] = false;

                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {


                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

            	if(idx_count[cur_depth] == 0){
            		record. real_workload +=1;
            	}
            }
        }


        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];

            visited_vertices[embedding[u]] = false;


        }
    }



    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;

}

ui LFTJ_WJ (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;
    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }
//    while (true) {
//    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

//    	while (idx[cur_depth] < idx_count[cur_depth]) {
        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();
        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {

        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
        		record.total_sample_count = record. sample_time* idx_count[0];
        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth] <<std::endl;

				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;
				ui sample_time = record. sample_time;
				srand(time(0));
//				#pragma omp parallel num_threads(32)
				{
//					#pragma omp for
					for(ui iter = 0; iter < sample_time ; ++iter){
//						int ID = omp_get_thread_num();
//						printf("Hello (%d)\n", ID);
						ui root_index = rand()%idx_count[cur_depth];
//						ui root_index = 0;
						resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
						auto sampling_start = std::chrono::high_resolution_clock::now();
//						#pragma omp critical
						aver_score += (double) WJ(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);
						auto sampling_end = std::chrono::high_resolution_clock::now();
						record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
						total_workloads += workloads;
					}
				}
				aver_score /= sample_time;
				total_workloads /= sample_time;
				record.est_path = aver_score*idx_count[cur_depth];
				record.est_workload = total_workloads*idx_count[cur_depth];

        	}


//            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//            VertexID u = order[cur_depth];
//            VertexID v = candidates[u][valid_idx];
//
//            if (visited_vertices[v]) {
//                idx[cur_depth] += 1;
//
//                continue;
//            }
//
//            embedding[u] = v;
//            idx_embedding[u] = valid_idx;
//            visited_vertices[v] = true;
//            idx[cur_depth] += 1;
//
//
//            if (cur_depth == max_depth - 1) {
//                embedding_cnt += 1;
//                record. real_workload +=1;
//                visited_vertices[v] = false;
//
////                if (embedding_cnt >= output_limit_num) {
////                    goto EXIT;
////                }
//            } else {
//
//
//                call_count += 1;
//                cur_depth += 1;
//
//                idx[cur_depth] = 0;
//                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//            	if(idx_count[cur_depth] == 0){
//            		record. real_workload +=1;
//            	}
//            }
//        }


//        cur_depth -= 1;
//        if (cur_depth < 0)
//            break;
//        else {
//            VertexID u = order[cur_depth];
//
//            visited_vertices[embedding[u]] = false;
//
//
//        }
//    }



//    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

//	auto end = std::chrono::high_resolution_clock::now();

//	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
//    return embedding_cnt;
	return 0;
}



ui LFTJ_WJ_roundrubin (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record ){
    //
	ui rubin_time = 2000;
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;
    // bn for sampling
//    ui **sampling_bn;
//    ui *sampling_bn_count;
    // order for sampling
    ui *sampling_order;
    sampling_order = new ui[query_graph->getVerticesCount()];
    memcpy(sampling_order, order , query_graph->getVerticesCount()*sizeof(ui));
    generateBN(query_graph, order, bn, bn_count);
//    generateBN(query_graph, sampling_order, sampling_bn, sampling_bn_count);
    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    while (true) {
//    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

    	while (idx[cur_depth] < idx_count[cur_depth]) {
        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();
        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {

        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;

        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
        		record.total_sample_count = record. sample_time* idx_count[0];
        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth];
				aver_score = 0;
				double total_workloads = 0;
				double inter_wl = 0;
				ui sample_time = record. sample_time;

				for(ui iter = 0; iter < sample_time/rubin_time ; ++iter){
					double inter_score = 0;
					double workloads= 0;
					inter_wl = 0;
					int inter_count = 0;

					for (ui inter_iter = 0; inter_iter <  rubin_time; ++ inter_iter){
						ui root_index = rand()%idx_count[cur_depth];
						resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
						// check the whether this order is a good order, otherwise we use another order.
						inter_score += (double) WJ(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, sampling_order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);
						inter_wl += workloads;
					}
					if (inter_score == 0){
						// round rubin
						for (int i = 1; i < query_graph->getVerticesCount()-1; ++i){
							if(inter_score != 0){
								break;
							}
							for (int j = i+1; j< query_graph->getVerticesCount()-1; ++ j ){
								if(inter_score != 0){
									break;
								}

								inter_wl= 0;
								//select a new order
								memcpy(sampling_order, order , query_graph->getVerticesCount()*sizeof(ui));
								std::swap(sampling_order[1], sampling_order[2]);
								//generate a new bn for the order

								reassignBN(query_graph, sampling_order, bn, bn_count);
								for (ui inter_iter = 0; inter_iter <  rubin_time; ++ inter_iter){

									ui root_index = rand()%idx_count[cur_depth];
									resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
									// check the whether this order is a good order, otherwise we use another order.

									inter_score += (double) WJ(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, sampling_order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);

									inter_wl += workloads;
								}
								inter_count ++;
							}

						}
					}

					aver_score += inter_score;
					total_workloads += inter_wl;
				}
				aver_score /= sample_time;
//				total_workloads /= sample_time;
				record.est_path = aver_score*idx_count[cur_depth];
//				record.est_workload = total_workloads*idx_count[cur_depth];
				reassignBN(query_graph, order, bn, bn_count);

        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;

                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;


            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                record. real_workload +=1;
                visited_vertices[v] = false;

                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {


                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

            	if(idx_count[cur_depth] == 0){
            		record. real_workload +=1;
            	}
            }
        }


        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];

            visited_vertices[embedding[u]] = false;


        }
    }



    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;

}

// importance sampling with
ui LFTJ_uniform (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }
    srand(time(0));
//    while (true) {
////    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;
//
//    	while (idx[cur_depth] < idx_count[cur_depth]) {
        	// sampling part
        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
        	    //  possiblity weight of current layer
//        	    double * p = new double [idx_count[cur_depth]];
//        	    // sum of embs of current layer.
//        	    double * ac_emb = new double [idx_count[cur_depth]];
        	    int * sampled_count = new int [idx_count[cur_depth]];
//        	    int * sampled_count_all = new int [idx_count[cur_depth]];
//        	    for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
////        	    	ac_emb[i] = 0;
//        	    	p[i] = 1.0;
//        	    	sampled_count_all[i] = 0;
//        	    }
        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth];
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;

				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
				ui round =  sample_time;
//				for(ui iter = 0; iter < round ; ++iter){
					// every inter_iter time update p array once.
//					double sum_interScore = 0;
					// update p and p_temp
//					memcpy (p,p_temp, idx_count[cur_depth]*sizeof(double));
//					memcpy (sampled_count,sampled_count_temp, idx_count[cur_depth]*sizeof(double));
//					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
////						if(i < 10)
////						std::cout << "p: " << p[i]<< " p_count: " << sampled_count[i]<<std::endl;
//					}
					//reset
//					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
//						ac_emb[i] = 0;
//						sampled_count[i] = 0;
//					}
//					std::cout << "p 1:" << p[1];
					double score = 0;
					for (ui inter_iter = 0; inter_iter < round; inter_iter++){
						ui root_index = rand()%idx_count[cur_depth];
//						if(inter_iter == 0 ){
//							std::cout <<"random pick:"<< root_index<<"~~"<<std::endl;
//						}
						resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
			    		auto sampling_start = std::chrono::high_resolution_clock::now();
						score  =  samplingByCPU_2(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);
						auto sampling_end = std::chrono::high_resolution_clock::now();
					    record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
						score *= idx_count[cur_depth];
						aver_score += score;

//						sampled_count[root_index]++;
//						sampled_count_all[root_index]++;
//						ac_emb[root_index] += score;
//						total_workloads += workloads;
						// for each iteration return a estimation for the current p.
//						double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
//						if(est != 0)
//						sum_interScore += est;
//						std::cout << "est: " << score / round;
					}

//				}
				aver_score = aver_score / round;
				total_workloads /= sample_time;
				record.est_path = aver_score;

				record.est_workload = total_workloads;
        	}


    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;

}

ui LFTJ_partialSampling (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    double *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }
    srand(time(0));
//    while (true) {
////    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;
//
//    	while (idx[cur_depth] < idx_count[cur_depth]) {
        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();
        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
        	    //  possiblity weight of current layer
//        	    double * p = new double [idx_count[cur_depth]];
//        	    // sum of embs of current layer.
//        	    double * ac_emb = new double [idx_count[cur_depth]];
//        	    int * sampled_count = new int [idx_count[cur_depth]];
//        	    int * sampled_count_all = new int [idx_count[cur_depth]];
//        	    for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
////        	    	ac_emb[i] = 0;
//        	    	p[i] = 1.0;
//        	    	sampled_count_all[i] = 0;
//        	    }
        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth];
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;

				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
				ui round =  sample_time;
//				for(ui iter = 0; iter < round ; ++iter){
					// every inter_iter time update p array once.
//					double sum_interScore = 0;
					// update p and p_temp
//					memcpy (p,p_temp, idx_count[cur_depth]*sizeof(double));
//					memcpy (sampled_count,sampled_count_temp, idx_count[cur_depth]*sizeof(double));
//					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
////						if(i < 10)
////						std::cout << "p: " << p[i]<< " p_count: " << sampled_count[i]<<std::endl;
//					}
					//reset
//					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
//						ac_emb[i] = 0;
//						sampled_count[i] = 0;
//					}
//					std::cout << "p 1:" << p[1];
					double score = 0;
					for (ui inter_iter = 0; inter_iter < round; inter_iter++){
						ui root_index = rand()%(ui)idx_count[cur_depth];
//						if(inter_iter == 0 ){
//							std::cout <<"random pick:"<< root_index<<"~~"<<std::endl;
//						}
						resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
						score  =  samplingByCPU_partially(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);
						score *= idx_count[cur_depth];

						aver_score += score;
//						std::cout << aver_score <<std::endl;
//						sampled_count[root_index]++;
//						sampled_count_all[root_index]++;
//						ac_emb[root_index] += score;
//						total_workloads += workloads;
						// for each iteration return a estimation for the current p.
//						double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
//						if(est != 0)
//						sum_interScore += est;
//						std::cout << "est: " << score / round;
					}

//				}
				aver_score = aver_score / round;
				total_workloads /= sample_time;
				record.est_path = aver_score;

				record.est_workload = total_workloads;
        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;

}

// importance sampling
ui LFTJ_adaptive (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    while (true) {
//    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

    	while (idx[cur_depth] < idx_count[cur_depth]) {
        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();
        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
        	    //  possiblity weight of current layer
        	    double * p = new double [idx_count[cur_depth]];
        	    double * p_temp = new double [idx_count[cur_depth]];
        	    // sum of embs of current layer.
        	    double * ac_emb = new double [idx_count[cur_depth]];
        	    int * sampled_count = new int [idx_count[cur_depth]];
        	    int * sampled_count_all = new int [idx_count[cur_depth]];
        	    for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
//        	    	ac_emb[i] = 0;
        	    	p[i] = 1.0;
        	    	sampled_count_all[i] = 0;
        	    }
        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth];
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;

				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
				ui round = ceil((double) sample_time/It_count);
				for(ui iter = 0; iter < round ; ++iter){
					// every inter_iter time update p array once.
					double sum_interScore = 0;
					// update p and p_temp
//					memcpy (p,p_temp, idx_count[cur_depth]*sizeof(double));
//					memcpy (sampled_count,sampled_count_temp, idx_count[cur_depth]*sizeof(double));
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
//						if(i < 10)
//						std::cout << "p: " << p[i]<< " p_count: " << sampled_count[i]<<std::endl;
					}
					//reset
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						ac_emb[i] = 0;
						sampled_count[i] = 0;
					}
//					std::cout << "p 1:" << p[1];
					for (ui inter_iter = 0; inter_iter < It_count; inter_iter++){
						ui root_index = selectroot(p,idx_count[cur_depth]);
						if(inter_iter == 0 ){
							std::cout <<"random pick:"<< root_index<<"~~"<<std::endl;
						}
						resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
						double score  =  samplingByGPU(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);

						sampled_count[root_index]++;
						sampled_count_all[root_index]++;
						ac_emb[root_index] += score;
						total_workloads += workloads;
						// for each iteration return a estimation for the current p.
						double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
						if(est != 0)
						sum_interScore += est;
					}
//					std::cout << "est: " << sum_interScore / It_count;
					//update p for next run
					std::memcpy(p_temp, p , sizeof(double)* idx_count[cur_depth] );
					updateByEmb  (p,ac_emb ,sampled_count,sampled_count_all,idx_count[cur_depth],It_count,p_temp);
					aver_score += sum_interScore/ It_count;
				}
				aver_score /= round;
				total_workloads /= sample_time;
				record.est_path = aver_score;

				record.est_workload = total_workloads;
				/*write the sampling possibity first 20 to a file */
				std::ofstream out;
				out.open("./p_m5.txt", std::ios_base::app);
				int cand_len = idx_count[cur_depth];
				double sum = 0;
				for (int i = 0; i< cand_len; i++){
					sum += p[i];
				}
				for (int i = 0; i< 20; i++){
					int space = (cand_len-1) / 20 + 1;
					double poss = 0;
					for (int j = 0 ; j< space; j++){
						int pos = i*space + j;
						poss +=  p[pos];
					}
					out << i << " " << poss/sum << "\n";
				}
				out <<  "~~~~~" << "\n";
				out.close();
        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;

                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;


            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                record. real_workload +=1;
                visited_vertices[v] = false;

                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {


                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

            	if(idx_count[cur_depth] == 0){
            		record. real_workload +=1;
            	}
            }
        }


        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];

            visited_vertices[embedding[u]] = false;


        }
    }



    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;

}

// importance sampling
ui LFTJ_adaptive_v2 (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    while (true) {
//    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

    	while (idx[cur_depth] < idx_count[cur_depth]) {
        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();
        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
        	    //  possiblity weight of current layer
        	    double * p = new double [idx_count[cur_depth]];
        	    double * p_temp = new double [idx_count[cur_depth]];
        	    // sum of embs of current layer.
        	    double * ac_emb = new double [idx_count[cur_depth]];
        	    int * sampled_count = new int [idx_count[cur_depth]];
        	    int * sampled_count_all = new int [idx_count[cur_depth]];
        	    for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
//        	    	ac_emb[i] = 0;
        	    	p[i] = 1.0;
        	    	sampled_count_all[i] = 0;
        	    }
        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth];
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;
				double memory_p = 1;
				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
				ui round = ceil((double) sample_time/It_count);
				for(ui iter = 0; iter < round ; ++iter){
					// every inter_iter time update p array once.
					double sum_interScore = 0;
					// update p and p_temp
//					memcpy (p,p_temp, idx_count[cur_depth]*sizeof(double));
//					memcpy (sampled_count,sampled_count_temp, idx_count[cur_depth]*sizeof(double));

					//reset
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						ac_emb[i] = 0;
						sampled_count[i] = 0;
					}
//					std::cout << "p 1:" << p[1];
					for (ui inter_iter = 0; inter_iter < It_count; inter_iter++){
						ui root_index = selectroot(p,idx_count[cur_depth]);
						if(inter_iter == 0 ){
							std::cout <<"random pick:"<< root_index<<"~~"<<std::endl;
						}
						resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
						double score  =  samplingByGPU(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);

						sampled_count[root_index]++;
						sampled_count_all[root_index]++;
						ac_emb[root_index] += score;
						total_workloads += workloads;
						// for each iteration return a estimation for the current p.
						double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
						if(est != 0)
						sum_interScore += est;

					}
//					std::cout << "est: " << sum_interScore / It_count;
					//update p for next run
					std::memcpy (p_temp,p, sizeof(double) *idx_count[cur_depth] );
					updateByEmb_random (p,ac_emb ,sampled_count,sampled_count_all,idx_count[cur_depth],It_count, memory_p, p_temp);
					aver_score += sum_interScore/ It_count;
				}
				aver_score /= round;
				total_workloads /= sample_time;
				record.est_path = aver_score;

				record.est_workload = total_workloads;
				/*write the sampling possibity first 20 to a file */
				std::ofstream out;
				out.open("./p_m6.txt", std::ios_base::app);
				int cand_len = idx_count[cur_depth];
				double sum = 0;
				for (int i = 0; i< cand_len; i++){
					sum += p[i];
				}
				for (int i = 0; i< 20; i++){
					int space = (cand_len-1) / 20 + 1;
					double poss = 0;
					for (int j = 0 ; j< space; j++){
						int pos = i*space + j;
						poss +=  p[pos];
					}
					out << i << " " << poss/sum << "\n";
				}
				out <<  "~~~~~" << "\n";
				out.close();
        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;

                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;


            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                record. real_workload +=1;
                visited_vertices[v] = false;

                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {


                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

            	if(idx_count[cur_depth] == 0){
            		record. real_workload +=1;
            	}
            }
        }


        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];

            visited_vertices[embedding[u]] = false;


        }
    }



    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;

}

// importance sampling focus sample at least 50 times using average method.
ui LFTJ_adaptive_multi_v1 (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record ){
    //
	ui trust_time = 50;
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    while (true) {
//    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

    	while (idx[cur_depth] < idx_count[cur_depth]) {
        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();
        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
        	    //  possiblity weight of current layer
        	    double * p = new double [idx_count[cur_depth]];
        	    double * p_temp = new double [idx_count[cur_depth]];
        	    // sum of embs of current layer.
        	    double * ac_emb = new double [idx_count[cur_depth]];
        	    int * sampled_count = new int [idx_count[cur_depth]];
        	    int * sampled_count_all = new int [idx_count[cur_depth]];
        	    for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
//        	    	ac_emb[i] = 0;
        	    	p[i] = 1.0;
        	    	sampled_count_all[i] = 0;
        	    }
        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth];
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;

				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
				ui round = ceil((double) sample_time/It_count);
				for(ui iter = 0; iter < round ; ++iter){
					// every inter_iter time update p array once.
					double sum_interScore = 0;
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						if(i < 10)
						std::cout << "p: " << p[i]<< " p_count: " << sampled_count[i]<<std::endl;
					}
					//reset
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						ac_emb[i] = 0;
						sampled_count[i] = 0;
					}
//					std::cout << "p 1:" << p[1];
					for (ui inter_iter = 0; inter_iter < It_count/trust_time; inter_iter++){
						ui root_index = selectroot(p,idx_count[cur_depth]);
						if(inter_iter == 0 ){
							std::cout <<"random pick:"<< root_index<<"~~"<<std::endl;
						}
						for (ui focus_iter = 0 ;focus_iter < trust_time;  focus_iter++){
							resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
							double score  =  samplingByGPU(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);

							sampled_count[root_index]++;
							sampled_count_all[root_index]++;
							ac_emb[root_index] += score;
							total_workloads += workloads;
							// for each iteration return a estimation for the current p.
							double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
							if(est != 0)
							sum_interScore += est;
						}
					}
//					std::cout << "est: " << sum_interScore / It_count;
					//update p for next run
					std::memcpy(p_temp, p , sizeof(double)* idx_count[cur_depth] );
					updateByEmb (p,ac_emb ,sampled_count,sampled_count_all,idx_count[cur_depth],It_count,p_temp);
					aver_score += sum_interScore/ It_count;
				}
				aver_score /= round;
				total_workloads /= sample_time;
				record.est_path = aver_score;

				record.est_workload = total_workloads;
				/*write the sampling possibity first 20 to a file */
				std::ofstream out;
				out.open("./p_multi_AVE.txt", std::ios_base::app);
				int cand_len = idx_count[cur_depth];
				double sum = 0;
				for (int i = 0; i< cand_len; i++){
					sum += p[i];
				}
				for (int i = 0; i< 20; i++){
					int space = (cand_len-1) / 20 + 1;
					double poss = 0;
					for (int j = 0 ; j< space; j++){
						int pos = i*space + j;
						poss +=  p[pos];
					}
					out << i << " " << poss/sum << "\n";
				}
				out <<  "~~~~~" << "\n";
				out.close();
        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;

                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;


            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                record. real_workload +=1;
                visited_vertices[v] = false;

                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {


                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

            	if(idx_count[cur_depth] == 0){
            		record. real_workload +=1;
            	}
            }
        }


        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];

            visited_vertices[embedding[u]] = false;


        }
    }



    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;

}


ui LFTJ_adaptive_multi_v2 (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record ){
    //
	ui trust_time = 50;
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    while (true) {
//    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

    	while (idx[cur_depth] < idx_count[cur_depth]) {
        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();
        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
        	    //  possiblity weight of current layer
        	    double * p = new double [idx_count[cur_depth]];
        	    double * p_temp = new double [idx_count[cur_depth]];
        	    // sum of embs of current layer.
        	    double * ac_emb = new double [idx_count[cur_depth]];
        	    int * sampled_count = new int [idx_count[cur_depth]];
        	    int * sampled_count_all = new int [idx_count[cur_depth]];
        	    for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
//        	    	ac_emb[i] = 0;
        	    	p[i] = 1.0;
        	    	sampled_count_all[i] = 0;
        	    }
        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth];
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;
				double memory_p = 1;
				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
				ui round = ceil((double) sample_time/It_count);
				for(ui iter = 0; iter < round ; ++iter){
					// every inter_iter time update p array once.
					double sum_interScore = 0;
					// update p and p_temp
//					memcpy (p,p_temp, idx_count[cur_depth]*sizeof(double));
//					memcpy (sampled_count,sampled_count_temp, idx_count[cur_depth]*sizeof(double));

					//reset
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						ac_emb[i] = 0;
						sampled_count[i] = 0;
					}
//					std::cout << "p 1:" << p[1];
					for (ui inter_iter = 0; inter_iter < It_count/trust_time; inter_iter++){
						ui root_index = selectroot(p,idx_count[cur_depth]);
						if(inter_iter == 0 ){
							std::cout <<"random pick:"<< root_index<<"~~"<<std::endl;
						}
						for (ui focus_iter = 0 ;focus_iter < trust_time;  focus_iter++){
							resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
							double score  =  samplingByGPU(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);

							sampled_count[root_index]++;
							sampled_count_all[root_index]++;
							ac_emb[root_index] += score;
							total_workloads += workloads;
							// for each iteration return a estimation for the current p.
							double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
							if(est != 0)
							sum_interScore += est;
						}
					}
//					std::cout << "est: " << sum_interScore / It_count;
					//update p for next run
					std::memcpy (p_temp,p, sizeof(double) *idx_count[cur_depth] );
					updateByEmb_random_trust (p,ac_emb ,sampled_count,sampled_count_all,idx_count[cur_depth],It_count, memory_p, p_temp);
					aver_score += sum_interScore/ It_count;
				}
				aver_score /= round;
				total_workloads /= sample_time;
				record.est_path = aver_score;

				record.est_workload = total_workloads;
				/*write the sampling possibity first 20 to a file */
				std::ofstream out;
				out.open("./p_multi_rand.txt", std::ios_base::app);
				int cand_len = idx_count[cur_depth];
				double sum = 0;
				for (int i = 0; i< cand_len; i++){
					sum += p[i];
				}
				for (int i = 0; i< 20; i++){
					int space = (cand_len-1) / 20 + 1;
					double poss = 0;
					for (int j = 0 ; j< space; j++){
						int pos = i*space + j;
						poss +=  p[pos];
					}
					out << i << " " << poss/sum << "\n";
				}
				out <<  "~~~~~" << "\n";
				out.close();
        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;

                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;


            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                record. real_workload +=1;
                visited_vertices[v] = false;

                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {


                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

            	if(idx_count[cur_depth] == 0){
            		record. real_workload +=1;
            	}
            }
        }


        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];

            visited_vertices[embedding[u]] = false;


        }
    }



    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;

}

ui LFTJ_adaptive_alpha (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record ){
    //
	ui trust_time = 50;
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	double alpha = record.alpha;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    while (true) {
//    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

    	while (idx[cur_depth] < idx_count[cur_depth]) {
        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();
        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
        	    //  possiblity weight of current layer
        	    double * p = new double [idx_count[cur_depth]];
        	    double * p_temp = new double [idx_count[cur_depth]];
        	    // sum of embs of current layer.
        	    double * ac_emb = new double [idx_count[cur_depth]];
        	    int * sampled_count = new int [idx_count[cur_depth]];
        	    int * sampled_count_all = new int [idx_count[cur_depth]];
        	    for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
//        	    	ac_emb[i] = 0;
        	    	p[i] = 1.0;
        	    	sampled_count_all[i] = 0;
        	    }
        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth];
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;
				double memory_p = 1;
				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
				ui round = ceil((double) sample_time/It_count);
				for(ui iter = 0; iter < round ; ++iter){
					// every inter_iter time update p array once.
					double sum_interScore = 0;
					// update p and p_temp
//					memcpy (p,p_temp, idx_count[cur_depth]*sizeof(double));
//					memcpy (sampled_count,sampled_count_temp, idx_count[cur_depth]*sizeof(double));

					//reset
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						ac_emb[i] = 0;
						sampled_count[i] = 0;
					}
//					std::cout << "p 1:" << p[1];
					for (ui inter_iter = 0; inter_iter < It_count/trust_time; inter_iter++){
						ui root_index = selectroot(p,idx_count[cur_depth]);
						if(inter_iter == 0 ){
							std::cout <<"random pick:"<< root_index<<"~~"<<std::endl;
						}
						for (ui focus_iter = 0 ;focus_iter < trust_time;  focus_iter++){
							resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
							double score  =  samplingByGPU(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);

							sampled_count[root_index]++;
							sampled_count_all[root_index]++;
							ac_emb[root_index] += score;
							total_workloads += workloads;
							// for each iteration return a estimation for the current p.
							double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
							if(est != 0)
							sum_interScore += est;
						}
					}
//					std::cout << "est: " << sum_interScore / It_count;
					//update p for next run
					std::memcpy (p_temp,p, sizeof(double) *idx_count[cur_depth] );
					updateByEmb_balance (p,ac_emb ,sampled_count,sampled_count_all,idx_count[cur_depth],It_count, memory_p, p_temp,alpha);
					aver_score += sum_interScore/ It_count;
				}
				aver_score /= round;
				total_workloads /= sample_time;
				record.est_path = aver_score;

				record.est_workload = total_workloads;
				/*write the sampling possibity first 20 to a file */
				std::ofstream out;
				out.open("./p_multi_balance.txt", std::ios_base::app);
				int cand_len = idx_count[cur_depth];
				double sum = 0;
				for (int i = 0; i< cand_len; i++){
					sum += p[i];
				}
				for (int i = 0; i< 20; i++){
					int space = (cand_len-1) / 20 + 1;
					double poss = 0;
					for (int j = 0 ; j< space; j++){
						int pos = i*space + j;
						poss +=  p[pos];
					}
					out << i << " " << poss/sum << "\n";
				}
				out <<  "~~~~~" << "\n";
				out.close();
        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;

                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;


            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                record. real_workload +=1;
                visited_vertices[v] = false;

                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {


                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

            	if(idx_count[cur_depth] == 0){
            		record. real_workload +=1;
            	}
            }
        }


        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];

            visited_vertices[embedding[u]] = false;


        }
    }



    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;

}

ui LFTJ_adaptive_rank (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record ){
    //
	ui trust_time = 50;
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	double alpha = record.alpha;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    while (true) {
//    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

    	while (idx[cur_depth] < idx_count[cur_depth]) {
        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();
        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
        	    //  possiblity weight of current layer
        	    double * p = new double [idx_count[cur_depth]];
        	    double * p_temp = new double [idx_count[cur_depth]];
        	    // sum of embs of current layer.
        	    double * ac_emb = new double [idx_count[cur_depth]];
        	    int * sampled_count = new int [idx_count[cur_depth]];
        	    int * sampled_count_all = new int [idx_count[cur_depth]];
        	    for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
//        	    	ac_emb[i] = 0;
        	    	p[i] = 1.0;
        	    	sampled_count_all[i] = 0;
        	    }
        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth];
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;
				double memory_p = 1;
				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
				ui round = ceil((double) sample_time/It_count);
				for(ui iter = 0; iter < round ; ++iter){
					// every inter_iter time update p array once.
					double sum_interScore = 0;
					// update p and p_temp
//					memcpy (p,p_temp, idx_count[cur_depth]*sizeof(double));
//					memcpy (sampled_count,sampled_count_temp, idx_count[cur_depth]*sizeof(double));

					//reset
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						ac_emb[i] = 0;
						sampled_count[i] = 0;
					}
//					std::cout << "p 1:" << p[1];
					for (ui inter_iter = 0; inter_iter < It_count/trust_time; inter_iter++){
						ui root_index = selectroot(p,idx_count[cur_depth]);
						if(inter_iter == 0 ){
							std::cout <<"random pick:"<< root_index<<"~~"<<std::endl;
						}
						for (ui focus_iter = 0 ;focus_iter < trust_time;  focus_iter++){
							resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
							double score  =  samplingByGPU(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);

							sampled_count[root_index]++;
							sampled_count_all[root_index]++;
							ac_emb[root_index] += score;
							total_workloads += workloads;
							// for each iteration return a estimation for the current p.
							double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
							if(est != 0)
							sum_interScore += est;
						}
					}
//					std::cout << "est: " << sum_interScore / It_count;
					//update p for next run
					std::memcpy (p_temp,p, sizeof(double) *idx_count[cur_depth] );
					updateByEmb_rank(p,ac_emb ,sampled_count,sampled_count_all,idx_count[cur_depth],It_count, memory_p, p_temp,alpha);
					aver_score += sum_interScore/ It_count;
				}
				aver_score /= round;
				total_workloads /= sample_time;
				record.est_path = aver_score;

				record.est_workload = total_workloads;
				/*write the sampling possibity first 20 to a file */
				std::ofstream out;
				out.open("./p_multi_balance.txt", std::ios_base::app);
				int cand_len = idx_count[cur_depth];
				double sum = 0;
				for (int i = 0; i< cand_len; i++){
					sum += p[i];
				}
				for (int i = 0; i< 20; i++){
					int space = (cand_len-1) / 20 + 1;
					double poss = 0;
					for (int j = 0 ; j< space; j++){
						int pos = i*space + j;
						poss +=  p[pos];
					}
					out << i << " " << poss/sum << "\n";
				}
				out <<  "~~~~~" << "\n";
				out.close();
        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;

                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;


            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                record. real_workload +=1;
                visited_vertices[v] = false;

                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {


                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

            	if(idx_count[cur_depth] == 0){
            		record. real_workload +=1;
            	}
            }
        }


        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];

            visited_vertices[embedding[u]] = false;


        }
    }



    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;

}


ui LFTJ_perfectP (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record, size_t* subtree){
    //
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    while (true) {
//    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

    	while (idx[cur_depth] < idx_count[cur_depth]) {
        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();
        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
        	    //  possiblity weight of current layer
        	    double * p = new double [idx_count[cur_depth]];
        	    // sum of embs of current layer.
        	    double * ac_emb = new double [idx_count[cur_depth]];
        	    int * sampled_count = new int [idx_count[cur_depth]];
        	    int * sampled_count_all = new int [idx_count[cur_depth]];
        	    for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
//        	    	ac_emb[i] = 0;
        	    	p[i] = subtree[i];
        	    	sampled_count_all[i] = 0;
        	    	sampled_count[i] = 0;
        	    }
        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth];
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;
				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
				ui round = ceil((double) sample_time/It_count);
				for(ui iter = 0; iter < round ; ++iter){
					// every inter_iter time update p array once.
					double sum_interScore = 0;
					// update p and p_temp
//					memcpy (p,p_temp, idx_count[cur_depth]*sizeof(double));
//					memcpy (sampled_count,sampled_count_temp, idx_count[cur_depth]*sizeof(double));
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						if(i < 10)
						std::cout << "p: " << p[i]<< " p_count: " << sampled_count[i]<<std::endl;
					}
					//reset
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						ac_emb[i] = 0;
						sampled_count[i] = 0;
					}
					std::cout << "p 1:" << p[1];
					for (ui inter_iter = 0; inter_iter < It_count; inter_iter++){
						ui root_index = selectroot(p,idx_count[cur_depth]);
						if(inter_iter == 0 ){
							std::cout <<"~`"<< root_index<<"~~"<<std::endl;
						}
						resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
						double score  =  samplingByGPU(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);
						sampled_count[root_index]++;
						sampled_count_all[root_index]++;
						ac_emb[root_index] += score;
						total_workloads += workloads;
						// for each iteration return a estimation for the current p.
						double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
						if(est != 0)
						sum_interScore += est;
					}
//					std::cout << "est: " << sum_interScore / It_count <<std::endl;
					//update p for next run
//					updateByEmb  (p,ac_emb ,sampled_count,sampled_count_all,idx_count[cur_depth],It_count);

					aver_score += sum_interScore / It_count;
//					aver_score += sum_interScore;
					std::cout << "total: "<< aver_score <<"round "<< iter<<std::endl;
				}

				aver_score /= round;

//				aver_score /= sample_time;
				total_workloads /= sample_time;
				record.est_path = aver_score;
				record.est_workload = total_workloads;
        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;

                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;


            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                record. real_workload +=1;
                visited_vertices[v] = false;

                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {


                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

            	if(idx_count[cur_depth] == 0){
            		record. real_workload +=1;
            	}
            }
        }


        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];

            visited_vertices[embedding[u]] = false;


        }
    }



    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;

}

// importance sampling focus sample at least 50 times using average method.
ui LFTJ_branch (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
	double b = record. b;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

//    while (true) {
////    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;
//
//    	while (idx[cur_depth] < idx_count[cur_depth]) {

        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();

        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {

        	    //  possiblity weight of current layer
        	    double * p = new double [idx_count[cur_depth]];
        	    // sum of embs of current layer.
        	    double * ac_emb = new double [idx_count[cur_depth]];
        	    int * sampled_count = new int [idx_count[cur_depth]];
        	    int * sampled_count_all = new int [idx_count[cur_depth]];
        	    for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
//        	    	ac_emb[i] = 0;
        	    	p[i] = 1.0;
        	    	sampled_count_all[i] = 0;
        	    }
        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);

        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth]<<std::endl;
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;
				double total_path = 0;
				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
				double round = 0;
				for(ui iter = 0; iter < sample_time ; ++iter){
					// every inter_iter time update p array once.
					double sum_interScore = 0;
					round ++;
					//reset
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						ac_emb[i] = 0;
						sampled_count[i] = 0;
					}
//					std::cout << "p 1:" << p[1];
					ui root_index = selectroot(p,idx_count[cur_depth]);
					resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
					unsigned long long branch_sample_count = 0;
					double score = samplingByGPU_branchV2(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads, b, branch_sample_count);
//					std::cout << "root_index " << root_index  <<" score "<< score <<std::endl;
//					std::cout << "branch_sample_count: " << branch_sample_count<<std::endl;
					total_path += branch_sample_count;
//						score /= branch_sample_count;
					// one iteration count many paths
					//inter_iter+=branch_sample_count;
					sampled_count[root_index]+= branch_sample_count;
					sampled_count_all[root_index]+= branch_sample_count;
					ac_emb[root_index] += score;
					total_workloads += workloads;
					// for each iteration return a estimation for the current p.
//						double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
					total_path += branch_sample_count;
					aver_score += score*idx_count[cur_depth] ;
					std::cout <<"est_score " << aver_score/(iter + 1)<<std::endl;
				}
				aver_score /= sample_time;
				total_workloads /= sample_time;
				record.est_path = aver_score;
				record.est_workload = total_workloads;
				record.total_path = total_path;
				// write to file.
				std::ofstream out;
				out.open("./p_balance.txt", std::ios_base::app);
				int cand_len = idx_count[cur_depth];
				double sum = 0;
				for (int i = 0; i< cand_len; i++){
					sum += p[i];
				}
				for (int i = 0; i< 20; i++){
					int space = (cand_len-1) / 20 + 1;
					double poss = 0;
					for (int j = 0 ; j< space; j++){
						int pos = i*space + j;
						poss +=  p[pos];
					}
					out << i << " " << poss/sum << "\n";
				}
				out <<  "~~~~~" << "\n";
				out.close();
        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

//            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//            VertexID u = order[cur_depth];
//            VertexID v = candidates[u][valid_idx];
//
//            if (visited_vertices[v]) {
//                idx[cur_depth] += 1;
//
//                continue;
//            }
//
//            embedding[u] = v;
//            idx_embedding[u] = valid_idx;
//            visited_vertices[v] = true;
//            idx[cur_depth] += 1;
//
//
//            if (cur_depth == max_depth - 1) {
//                embedding_cnt += 1;
//                record. real_workload +=1;
//                visited_vertices[v] = false;
//
//                if (embedding_cnt >= output_limit_num) {
//                    goto EXIT;
//                }
//            } else {
//
//
//                call_count += 1;
//                cur_depth += 1;
//
//                idx[cur_depth] = 0;
//                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//            	if(idx_count[cur_depth] == 0){
//            		record. real_workload +=1;
//            	}
//            }
//        }


//        cur_depth -= 1;
//        if (cur_depth < 0)
//            break;
//        else {
//            VertexID u = order[cur_depth];
//
//            visited_vertices[embedding[u]] = false;
//
//
//        }
//    }



    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;
}

ui LFTJ_combine (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record ){
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	double alpha = record.alpha;
	double b = record. b;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    while (true) {
//    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

    	while (idx[cur_depth] < idx_count[cur_depth]) {

        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();

        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {

        	    //  possiblity weight of current layer
        	    double * p = new double [idx_count[cur_depth]];
        	    double * p_temp = new double [idx_count[cur_depth]];
        	    // sum of embs of current layer.
        	    double * ac_emb = new double [idx_count[cur_depth]];
        	    int * sampled_count = new int [idx_count[cur_depth]];
        	    int * sampled_count_all = new int [idx_count[cur_depth]];
        	    for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
//        	    	ac_emb[i] = 0;
        	    	p[i] = 1.0;
        	    	sampled_count_all[i] = 0;
        	    }
        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);

        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth]<<std::endl;
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;
				double memory_p = 1;

				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
				ui round = ceil((double) sample_time/It_count);
				std::cout << "round: " << round<<std::endl;
				for(ui iter = 0; iter < round ; ++iter){
					// every inter_iter time update p array once.
					double sum_interScore = 0;
					// update p and p_temp
//					memcpy (p,p_temp, idx_count[cur_depth]*sizeof(double));
//					memcpy (sampled_count,sampled_count_temp, idx_count[cur_depth]*sizeof(double));
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
//						if(i < 10)
//						std::cout << "p: " << p[i]<< " p_count: " << sampled_count[i]<<std::endl;
					}
					//reset
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						ac_emb[i] = 0;
						sampled_count[i] = 0;
					}
//					std::cout << "p 1:" << p[1];
					ui it = 0;
					for (ui inter_iter = 0; inter_iter < It_count;){
//						std::cout << "it: " << it<< " inter_iter: "<<inter_iter<<" It_count: "<<It_count<<std::endl;
						it++;
						ui root_index = selectroot(p,idx_count[cur_depth]);
//						if(inter_iter == 0 ){
//							std::cout <<"random pick:"<< root_index<<"~~"<<std::endl;
//						}
						resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
						ui branch_sample_count = 0;
						double score = samplingByGPU_branch(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads, b, branch_sample_count);
						for (int k= 0 ; k< max_depth -2 ; ++k){
							score = score / (b);
						}
						inter_iter+=branch_sample_count;
						sampled_count[root_index]+= branch_sample_count;
						sampled_count_all[root_index]+= branch_sample_count;
						ac_emb[root_index] += score;
						total_workloads += workloads;
						// for each iteration return a estimation for the current p.
						double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
						if(est != 0)
						sum_interScore += est;
					}
//					std::cout << "est: " << sum_interScore / It_count;
					std::memcpy (p_temp,p, sizeof(double) *idx_count[cur_depth] );
					updateByEmb_balance (p,ac_emb ,sampled_count,sampled_count_all,idx_count[cur_depth],It_count, memory_p, p_temp,alpha);
					aver_score += sum_interScore/ it;
				}
				aver_score /= round;
				total_workloads /= sample_time;
				record.est_path = aver_score;

				record.est_workload = total_workloads;
        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;

                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;


            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                record. real_workload +=1;
                visited_vertices[v] = false;

                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {


                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

            	if(idx_count[cur_depth] == 0){
            		record. real_workload +=1;
            	}
            }
        }


        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];

            visited_vertices[embedding[u]] = false;


        }
    }



    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;
}


ui LFTJ_fixedbranch_wrong (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record ){
	ui fixednum = record.fixednum;
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	double b = record. b;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
	ui **bn;
	ui *bn_count;

	generateBN(query_graph, order, bn, bn_count);

	// Allocate the memory buffer.
	ui *idx;
	ui *idx_count;
	ui *embedding;
	ui *idx_embedding;
	ui *temp_buffer;
	ui **valid_candidate_idx;
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

	for (ui i = 0; i < idx_count[cur_depth]; ++i) {
		valid_candidate_idx[cur_depth][i] = i;
	}

	while (true) {

		while (idx[cur_depth] < idx_count[cur_depth]) {

			// sampling part
			auto sampling_start = std::chrono::high_resolution_clock::now();

			if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {

				//  possiblity weight of current layer
				double * p = new double [idx_count[cur_depth]];
				// sum of embs of current layer.
				double * ac_emb = new double [idx_count[cur_depth]];
				unsigned long long  * sampled_count = new unsigned long long [idx_count[cur_depth]];
				unsigned long long  * sampled_count_all = new unsigned long long [idx_count[cur_depth]];
				for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
					p[i] = 1.0;
					sampled_count_all[i] = 0;
				}
				//judge whether do a sampling for current node if true
				//perform a sampling at current node
				bool* sampling_visited_vertices;
				//use std library to sort
				double aver_score;
				initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);

				// random select one neighbor
				std::cout << "root num: " << idx_count[cur_depth]<<std::endl;
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;

				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
//				ui round = ceil((double) sample_time/It_count);
//				std::cout << "round: " << round<<std::endl;
				ui round = 0;
				for(ui iter = 0; iter < sample_time ; ++iter){
					// every inter_iter time update p array once.
					double sum_interScore = 0;
					round++;
					//reset
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						ac_emb[i] = 0;
						sampled_count[i] = 0;
					}
					ui it = 0;
					for (ui inter_iter = 0; inter_iter < idx_count[cur_depth]; ++ inter_iter){
						it++;
//						ui root_index = selectroot(p,idx_count[cur_depth]);
						ui root_index =inter_iter;
						resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
						unsigned long long branch_sample_count = 0;
						double score = samplingByGPU_fixedbranch(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads, b, branch_sample_count,fixednum);
//						std::cout << "branch_sample_count: " << branch_sample_count<<std::endl;
						score /= branch_sample_count;
						// one iteration count many paths
						//inter_iter+=branch_sample_count;
						sampled_count[root_index]+= branch_sample_count;
						sampled_count_all[root_index]+= branch_sample_count;
						ac_emb[root_index] += score;
						total_workloads += workloads;
						// for each iteration return a estimation for the current p.
						double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
						if(est != 0)
						sum_interScore += est;
					}
					std::cout <<"iter "<< iter << "it: "<< it << " est: " << sum_interScore / it<<std::endl;
					aver_score += sum_interScore/ it;
//					std::cout <<"aver_score " << aver_score<<std::endl;
				}
				std::cout <<"round " << round<<" total_score "<< aver_score<<std::endl;
				aver_score /= round;
				total_workloads /= sample_time;
				record.est_path = aver_score;

				record.est_workload = total_workloads;
			}
			auto sampling_end = std::chrono::high_resolution_clock::now();
			record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

			ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
			VertexID u = order[cur_depth];
			VertexID v = candidates[u][valid_idx];

			if (visited_vertices[v]) {
				idx[cur_depth] += 1;

				continue;
			}

			embedding[u] = v;
			idx_embedding[u] = valid_idx;
			visited_vertices[v] = true;
			idx[cur_depth] += 1;


			if (cur_depth == max_depth - 1) {
				embedding_cnt += 1;
				record. real_workload +=1;
				visited_vertices[v] = false;

				if (embedding_cnt >= output_limit_num) {
					goto EXIT;
				}
			} else {


				call_count += 1;
				cur_depth += 1;

				idx[cur_depth] = 0;
				generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
											bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

				if(idx_count[cur_depth] == 0){
					record. real_workload +=1;
				}
			}
		}


		cur_depth -= 1;
		if (cur_depth < 0)
			break;
		else {
			VertexID u = order[cur_depth];

			visited_vertices[embedding[u]] = false;


		}
	}



	EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
	return embedding_cnt;
}

ui LFTJ_fixedbranch (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
        ui *order, size_t output_limit_num, size_t &call_count, ui step, timer &record ){
	ui fixednum = record.fixednum;
	record. sampling_time = 0;
	record. enumerating_time = 0;
	record. reorder_time = 0;
	record. est_path = 0;
	record. est_workload  = 0;
	record. real_workload = 0;
	record. set_intersection_count = 0;
	record. total_compare = 0;
	double b = record. b;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
	ui **bn;
	ui *bn_count;

	generateBN(query_graph, order, bn, bn_count);

	// Allocate the memory buffer.
	ui *idx;
	ui *idx_count;
	ui *embedding;
	ui *idx_embedding;
	ui *temp_buffer;
	ui **valid_candidate_idx;
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

	for (ui i = 0; i < idx_count[cur_depth]; ++i) {
		valid_candidate_idx[cur_depth][i] = i;
	}

	while (true) {

		while (idx[cur_depth] < idx_count[cur_depth]) {

			// sampling part
			auto sampling_start = std::chrono::high_resolution_clock::now();

			if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {

				//  possiblity weight of current layer
				double * p = new double [idx_count[cur_depth]];
				// sum of embs of current layer.
				double * ac_emb = new double [idx_count[cur_depth]];
				unsigned long long  * sampled_count = new unsigned long long [idx_count[cur_depth]];
				unsigned long long  * sampled_count_all = new unsigned long long [idx_count[cur_depth]];
				for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
					p[i] = 1.0;
					sampled_count_all[i] = 0;
				}
				//judge whether do a sampling for current node if true
				//perform a sampling at current node
				bool* sampling_visited_vertices;
				//use std library to sort
				double aver_score = 0.0;
				initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);

				// random select one neighbor
				std::cout << "root num: " << idx_count[cur_depth]<<std::endl;
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;

				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
//				ui round = ceil((double) sample_time/It_count);
//				std::cout << "round: " << round<<std::endl;
				ui round = 0;
				double total_path = 0.0;
				for(ui iter = 0; iter < sample_time ; ++iter){
					// every inter_iter time update p array once.

					//reset
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						ac_emb[i] = 0;
						sampled_count[i] = 0;
					}

						ui root_index = selectroot(p,idx_count[cur_depth]);
						resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
						unsigned long long branch_sample_count = 0;
						double score = samplingByGPU_fixedbranch(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads, b, branch_sample_count,fixednum);
						std::cout << "root_index " << root_index  <<" score "<< score <<std::endl;
						std::cout << "branch_sample_count: " << branch_sample_count<<std::endl;
//						score /= branch_sample_count;
						// one iteration count many paths
						//inter_iter+=branch_sample_count;
						sampled_count[root_index]+= branch_sample_count;
						sampled_count_all[root_index]+= branch_sample_count;
						ac_emb[root_index] += score;
						total_workloads += workloads;
						// for each iteration return a estimation for the current p.
//						double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
						total_path += branch_sample_count;
					aver_score += score*idx_count[cur_depth] ;
					std::cout <<"est_score " << aver_score/(iter + 1)<<std::endl;
				}
				std::cout<<" total_score "<< aver_score<<std::endl;
				aver_score /= sample_time;
				total_workloads /= sample_time;
				record.est_path = aver_score;
				record.total_path = total_path;
				record.est_workload = total_workloads;
			}
			auto sampling_end = std::chrono::high_resolution_clock::now();
			record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

			ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
			VertexID u = order[cur_depth];
			VertexID v = candidates[u][valid_idx];

			if (visited_vertices[v]) {
				idx[cur_depth] += 1;

				continue;
			}

			embedding[u] = v;
			idx_embedding[u] = valid_idx;
			visited_vertices[v] = true;
			idx[cur_depth] += 1;


			if (cur_depth == max_depth - 1) {
				embedding_cnt += 1;
				record. real_workload +=1;
				visited_vertices[v] = false;

				if (embedding_cnt >= output_limit_num) {
					goto EXIT;
				}
			} else {


				call_count += 1;
				cur_depth += 1;

				idx[cur_depth] = 0;
				generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
											bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

				if(idx_count[cur_depth] == 0){
					record. real_workload +=1;
				}
			}
		}


		cur_depth -= 1;
		if (cur_depth < 0)
			break;
		else {
			VertexID u = order[cur_depth];

			visited_vertices[embedding[u]] = false;


		}
	}



	EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
	return embedding_cnt;
}

// importance sampling focus sample at least 50 times using average method.
ui LFTJ_half(const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
	double b = record. b;
	ui It_count = record.inter_count;
	auto start = std::chrono::high_resolution_clock::now();
	// Generate bn.
    ui **bn;
    ui *bn_count;

    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
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

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

    while (true) {
//    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

    	while (idx[cur_depth] < idx_count[cur_depth]) {

        	// sampling part
    		auto sampling_start = std::chrono::high_resolution_clock::now();

        	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {

        	    //  possiblity weight of current layer
        	    double * p = new double [idx_count[cur_depth]];
        	    // sum of embs of current layer.
        	    double * ac_emb = new double [idx_count[cur_depth]];
        	    double * ac_emb_all = new double [idx_count[cur_depth]];
        	    int * sampled_count = new int [idx_count[cur_depth]];
        	    int * sampled_count_all = new int [idx_count[cur_depth]];
        	    for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
        	    	ac_emb_all[i] = 0;
        	    	p[i] = 1.0;
        	    	sampled_count_all[i] = 0;
        	    }
        		//judge whether do a sampling for current node if true
				//perform a sampling at current node
        		bool* sampling_visited_vertices;
        		//use std library to sort
//        		std::vector<CindexScore> sorted_arr;
        		double aver_score;
        		initMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);

        		// random select one neighbor
        		std::cout << "root num: " << idx_count[cur_depth]<<std::endl;
				aver_score = 0;
				double total_workloads = 0;
				double workloads= 0;
				bool updated = false;
				ui sample_time = record. sample_time;
				// record the possibility weight to sample in the current（first） layer
				double round = 0;
				for(ui iter = 0; iter < sample_time ; ++iter){
					// every inter_iter time update p array once.
					double sum_interScore = 0;
					round ++;
					//reset
					for(int i = 0 ; i < idx_count[cur_depth] ; ++i){
						ac_emb[i] = 0;
						sampled_count[i] = 0;
					}
//					std::cout << "p 1:" << p[1];
					if(updated == false){
						for (ui inter_iter = 0; inter_iter < idx_count[cur_depth]; ++ inter_iter){
	//						std::cout << "it: " << it<< " inter_iter: "<<inter_iter<<" It_count: "<<It_count<<std::endl;
							ui root_index = inter_iter;

							resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
							ui branch_sample_count = 0;
							double score  =  samplingByGPU(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);
							branch_sample_count ++;
							iter += branch_sample_count;
							sampled_count[root_index]+= branch_sample_count;
							sampled_count_all[root_index]+= branch_sample_count;
							ac_emb[root_index] += score;
							ac_emb_all[root_index] +=  score;
							total_workloads += workloads;
							// for each iteration return a estimation for the current p.
							double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
	//						std::cout << "per est: " << est << std::endl;
							if(est != 0)
							sum_interScore += est;
						}
					}else{
						ui root_index = selectroot(p,idx_count[cur_depth]);
						resetMemorySampling(data_graph,sampling_visited_vertices, visited_vertices, idx_count[cur_depth]);
						double score  =  samplingByGPU(edge_matrix, valid_candidate_idx, candidates, candidates_count, idx_count, order, bn, bn_count, sampling_visited_vertices,  embedding,idx_embedding, temp_buffer,root_index, cur_depth , max_depth,step, workloads);
						iter++;
						sampled_count[root_index]+= 1;
						sampled_count_all[root_index]+= 1;
						ac_emb[root_index] += score;
						ac_emb_all[root_index] +=  score;
						total_workloads += workloads;
						// for each iteration return a estimation for the current p.
						double est = estimateEmb(p, idx_count[cur_depth],root_index,score);
						if(est != 0)
						sum_interScore += est;
					}
					std::cout << "est: " << sum_interScore/ idx_count[cur_depth];
					aver_score += sum_interScore/ idx_count[cur_depth];
					if(updated == false && iter > It_count){
						adjustPoss (p,ac_emb_all ,sampled_count_all,idx_count[cur_depth]);
						updated = true;
					}
				}
				aver_score /= round;
				total_workloads /= sample_time;
				record.est_path = aver_score;
				record.est_workload = total_workloads;

				// write to file.
				std::ofstream out;
				out.open("./p_half.txt", std::ios_base::app);
				int cand_len = idx_count[cur_depth];
				double sum = 0;
				for (int i = 0; i< cand_len; i++){
					sum += p[i];
				}
				for (int i = 0; i< 20; i++){
					int space = (cand_len-1) / 20 + 1;
					double poss = 0;
					for (int j = 0 ; j< space; j++){
						int pos = i*space + j;
						poss +=  p[pos];
					}
					out << i << " " << poss/sum << "\n";
				}
				out <<  "~~~~~" << "\n";
				out.close();
        	}
        	auto sampling_end = std::chrono::high_resolution_clock::now();
        	record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();

            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;

                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;


            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                record. real_workload +=1;
                visited_vertices[v] = false;

                if (embedding_cnt >= output_limit_num) {
                    goto EXIT;
                }
            } else {


                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

            	if(idx_count[cur_depth] == 0){
            		record. real_workload +=1;
            	}
            }
        }


        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];

            visited_vertices[embedding[u]] = false;


        }
    }



    EXIT:
//    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//                  visited_vertices,
//                  bn, bn_count);

	auto end = std::chrono::high_resolution_clock::now();

	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time;
    return embedding_cnt;
}

ui LFTJ_GPU_all (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
    ui threadnum = record.sample_time;
    ui max_thread = threadnum;
    if(threadnum > max_thread){
    	threadnum = max_thread;
    }

	const ui threadsPerBlock = 32;
	ui numBlocks = (threadnum-1) / threadsPerBlock + 1;
	std::cout << "readonly GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
	std::cout << "threadsPerBlock: "<< threadsPerBlock << " numBlocks: "<< numBlocks << " total threads: " << threadsPerBlock*numBlocks << " max_candidates_num " << max_candidates_num<<std::endl;

	// for each thread we assign its own global memoory.
    allocateMemoryPerThread(d_idx ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_range ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_count ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui) + max_candidates_num * sizeof(ui)) * threadnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
		std::cout <<"An error ocurrs when allocate memory!"<<std::endl;
	}else{
		std::cout <<"Pass memory assignment test!"<<std::endl;
	}
	// compute total bytes allocated.


	// test candidate

//    std::cout << "pitch : " << valid_candidate_idx_pitch << " ";
    auto GPU_alloc_end = std::chrono::high_resolution_clock::now();
    record. cand_alloc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(GPU_alloc_end - GPU_alloc_start).count();

    while (true) {
  //    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

      	while (idx[cur_depth] < idx_count[cur_depth]) {
          	// sampling part
          	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
          		auto sampling_start = std::chrono::high_resolution_clock::now();
  				ui sample_time = record. sample_time;
  				// record the possibility weight to sample in the current（first） layer
  				ui round = (sample_time - 1)/ max_thread + 1;
  				double aver_score = 0;
  				ui h_score_count = 0;
  				for (ui k = 0; k< round; ++k){
  					ui iter_sample_time = sample_time - max_thread*k;
					//one thread one path
					samplingByGPUThread_v2<threadsPerBlock><<<numBlocks,threadsPerBlock>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count);
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
              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
              VertexID u = order[cur_depth];
              VertexID v = candidates[u][valid_idx];

              if (visited_vertices[v]) {
                  idx[cur_depth] += 1;

                  continue;
              }

              embedding[u] = v;
              idx_embedding[u] = valid_idx;
              visited_vertices[v] = true;
              idx[cur_depth] += 1;


              if (cur_depth == max_depth - 1) {
                  embedding_cnt += 1;
                  record. real_workload +=1;
                  visited_vertices[v] = false;
                  //print a path
//                  for (int i = 0; i<= cur_depth; i++){
//                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
//                  }

                  if (embedding_cnt >= output_limit_num) {
                      goto EXIT;
                  }
              } else {


                  call_count += 1;
                  cur_depth += 1;

                  idx[cur_depth] = 0;
                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

              	if(idx_count[cur_depth] == 0){
              		record. real_workload +=1;
              	}
              }
          }


          cur_depth -= 1;
          if (cur_depth < 0)
              break;
          else {
              VertexID u = order[cur_depth];

              visited_vertices[embedding[u]] = false;


          }
      }



      EXIT:
  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
  //                  visited_vertices,
  //                  bn, bn_count);

  	auto end = std::chrono::high_resolution_clock::now();

  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
    return embedding_cnt;
}

ui LFTJ_GPU_warp (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
    ui warpSize = 32;
    ui warpnum = record.sample_time;
    ui threadnum = warpnum* warpSize;
	const ui threadsPerBlock = 32;
	ui numBlocks = (threadnum-1) / threadsPerBlock + 1;
	std::cout << "readonly GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
	std::cout << "threadsPerBlock: "<< threadsPerBlock << " numBlocks: "<< numBlocks << " total threads: " << threadsPerBlock*numBlocks << " max_candidates_num " << max_candidates_num<<std::endl;

	// for each thread we assign its own global memoory.
    allocateMemoryPerThread(d_idx ,query_vertices_num * sizeof(ui), warpnum);
    allocateMemoryPerThread(d_embedding ,query_vertices_num * sizeof(ui), warpnum);
    allocateMemoryPerThread(d_idx_embedding ,query_vertices_num * sizeof(ui), warpnum);
    allocateMemoryPerThread(d_range ,query_vertices_num * sizeof(ui), warpnum);
    allocateMemoryPerThread(d_idx_count ,query_vertices_num * sizeof(ui), warpnum);
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), warpnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), warpnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui) + max_candidates_num * sizeof(ui)) * warpnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
		std::cout <<"An error ocurrs when allocate memory!"<<std::endl;
	}else{
		std::cout <<"Pass memory assignment test!"<<std::endl;
	}
	// compute total bytes allocated.


	// test candidate

//    std::cout << "pitch : " << valid_candidate_idx_pitch << " ";
    auto GPU_alloc_end = std::chrono::high_resolution_clock::now();
    record. cand_alloc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(GPU_alloc_end - GPU_alloc_start).count();

    while (true) {
  //    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

      	while (idx[cur_depth] < idx_count[cur_depth]) {
          	// sampling part

          	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
          		auto sampling_start = std::chrono::high_resolution_clock::now();
  				ui sample_time = record. sample_time;
  				// record the possibility weight to sample in the current（first） layer
  				ui round =  1;
  				double aver_score = 0;
  				ui h_score_count = 0;
  				for (ui k = 0; k< round; ++k){
  					ui iter_sample_time = sample_time;
					//one thread one path
					samplingByGPUWarp<threadsPerBlock><<<numBlocks,threadsPerBlock>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count);
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


              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
              VertexID u = order[cur_depth];
              VertexID v = candidates[u][valid_idx];

              if (visited_vertices[v]) {
                  idx[cur_depth] += 1;

                  continue;
              }

              embedding[u] = v;
              idx_embedding[u] = valid_idx;
              visited_vertices[v] = true;
              idx[cur_depth] += 1;


              if (cur_depth == max_depth - 1) {
                  embedding_cnt += 1;
                  record. real_workload +=1;
                  visited_vertices[v] = false;
                  //print a path
//                  for (int i = 0; i<= cur_depth; i++){
//                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
//                  }

                  if (embedding_cnt >= output_limit_num) {
                      goto EXIT;
                  }
              } else {


                  call_count += 1;
                  cur_depth += 1;

                  idx[cur_depth] = 0;
                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

              	if(idx_count[cur_depth] == 0){
              		record. real_workload +=1;
              	}
              }
          }


          cur_depth -= 1;
          if (cur_depth < 0)
              break;
          else {
              VertexID u = order[cur_depth];

              visited_vertices[embedding[u]] = false;


          }
      }



      EXIT:
  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
  //                  visited_vertices,
  //                  bn, bn_count);

  	auto end = std::chrono::high_resolution_clock::now();

  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time- record. cand_alloc_time;
    return embedding_cnt;
}

ui LFTJ_GPU_warp_lessmem (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
    ui warpSize = 32;
    ui warpnum = record.sample_time;
    ui threadnum = warpnum* warpSize;
	const ui threadsPerBlock = 32;
	ui numBlocks = (threadnum-1) / threadsPerBlock + 1;
	std::cout << "readonly GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
	std::cout << "threadsPerBlock: "<< threadsPerBlock << " numBlocks: "<< numBlocks << " total threads: " << threadsPerBlock*numBlocks << " max_candidates_num " << max_candidates_num<<std::endl;

	// for each thread we assign its own global memoory.
    allocateMemoryPerThread(d_idx ,query_vertices_num * sizeof(ui), warpnum);
    allocateMemoryPerThread(d_embedding ,query_vertices_num * sizeof(ui), warpnum);
    allocateMemoryPerThread(d_idx_embedding ,query_vertices_num * sizeof(ui), warpnum);
    allocateMemoryPerThread(d_range ,query_vertices_num * sizeof(ui), warpnum);
    allocateMemoryPerThread(d_idx_count ,query_vertices_num * sizeof(ui), warpnum);
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), warpnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), warpnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui) + max_candidates_num * sizeof(ui)) * warpnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
		std::cout <<"An error ocurrs when allocate memory!"<<std::endl;
	}else{
		std::cout <<"Pass memory assignment test!"<<std::endl;
	}
	// compute total bytes allocated.


	// test candidate

//    std::cout << "pitch : " << valid_candidate_idx_pitch << " ";
    auto GPU_alloc_end = std::chrono::high_resolution_clock::now();
    record. cand_alloc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(GPU_alloc_end - GPU_alloc_start).count();

    while (true) {
  //    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

      	while (idx[cur_depth] < idx_count[cur_depth]) {
          	// sampling part

          	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
          		auto sampling_start = std::chrono::high_resolution_clock::now();
  				ui sample_time = record. sample_time;
  				// record the possibility weight to sample in the current（first） layer
  				ui round =  1;
  				double aver_score = 0;
  				ui h_score_count = 0;
  				for (ui k = 0; k< round; ++k){
  					ui iter_sample_time = sample_time;
					//one thread one path
					samplingByGPUWarpLessmem<threadsPerBlock><<<numBlocks,threadsPerBlock>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count);
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


              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
              VertexID u = order[cur_depth];
              VertexID v = candidates[u][valid_idx];

              if (visited_vertices[v]) {
                  idx[cur_depth] += 1;

                  continue;
              }

              embedding[u] = v;
              idx_embedding[u] = valid_idx;
              visited_vertices[v] = true;
              idx[cur_depth] += 1;


              if (cur_depth == max_depth - 1) {
                  embedding_cnt += 1;
                  record. real_workload +=1;
                  visited_vertices[v] = false;
                  //print a path
//                  for (int i = 0; i<= cur_depth; i++){
//                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
//                  }

                  if (embedding_cnt >= output_limit_num) {
                      goto EXIT;
                  }
              } else {


                  call_count += 1;
                  cur_depth += 1;

                  idx[cur_depth] = 0;
                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);

              	if(idx_count[cur_depth] == 0){
              		record. real_workload +=1;
              	}
              }
          }


          cur_depth -= 1;
          if (cur_depth < 0)
              break;
          else {
              VertexID u = order[cur_depth];

              visited_vertices[embedding[u]] = false;


          }
      }



      EXIT:
  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
  //                  visited_vertices,
  //                  bn, bn_count);

  	auto end = std::chrono::high_resolution_clock::now();

  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time- record. cand_alloc_time;
    return embedding_cnt;
}
//
template <const ui blocksize>
ui blockPathBalance (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui) + max_candidates_num * sizeof(ui)) * threadnum;
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
    std::cout << "alloc and memory transfer: " << record. cand_alloc_time / 1000000000<<std::endl;
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
  					BlockPathBalance<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
ui blockPathBalanceLessmem (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
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
//    allocateGPU1D( d_idx ,idx,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_count ,idx_count,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_embedding ,embedding,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_embedding ,idx_embedding,query_vertices_num * sizeof(ui));
    allocateGPU1D( d_order, order, query_vertices_num * sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);
//    allocateGPU1D( d_temp_buffer ,temp_buffer, max_candidates_num * sizeof(ui));
    allocateGPU1D( d_score ,score, 1* sizeof(double));
    allocateGPU1D( d_score_count ,score_count, 1* sizeof(double));
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
  					BlockPathBalanceLessmem<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}


template <const ui blocksize>
ui blockPathBalanceLessmemV2 (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					BlockPathBalanceLessmemV2<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}


template <const ui blocksize>
ui blockPathBalanceLessmemV3 (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					BlockPathBalanceLessmemV3<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}


template <const ui blocksize>
ui blockPathBalanceLessmemV4 (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					BlockPathBalanceLessmemV4<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}

template <const ui blocksize>
ui blockLayerBalance (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
	std::cout << "blocksize: "<< blocksize << " numBlocks: "<< numBlocks << " total threads: " << blocksize*numBlocks << " max_candidates_num " << max_candidates_num<<std::endl;

	// for each thread we assign its own global memoory.
    allocateMemoryPerThread(d_idx ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_range ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_count ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui) + max_candidates_num * sizeof(ui)) * threadnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
		std::cout <<"An error ocurrs when allocate memory!"<<std::endl;
	}else{
		std::cout <<"Pass memory assignment test!"<<std::endl;
	}
	// compute total bytes allocated.


	// test candidate

//    std::cout << "pitch : " << valid_candidate_idx_pitch << " ";
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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					BlockLayerBalance< blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }



//      EXIT:
  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
  //                  visited_vertices,
  //                  bn, bn_count);

//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}

template <const ui blocksize>
ui warpPathBalance (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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

	ui warpPerBlock = blocksize/32;
	ui taskPerBlock = warpPerBlock* record.taskPerWarp;
	ui numBlocks = (threadnum-1) / blocksize + 1;
	ui taskPerRound = numBlocks*  taskPerBlock;

	std::cout << "readonly GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
	std::cout << "blocksize: "<< blocksize << " numBlocks: "<< numBlocks << " total threads: " << blocksize*numBlocks << " max_candidates_num " << max_candidates_num<<std::endl;

	// for each thread we assign its own global memoory.
    allocateMemoryPerThread(d_idx ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_range ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_count ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui) + max_candidates_num * sizeof(ui)) * threadnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
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
  					WarpPathBalance<blocksize><<<numBlocks,blocksize,blocksize/32*sizeof(int) >>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerWarp);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }



//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
    return 0;
}

template <const ui blocksize>
ui warpLayerBalance (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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

	ui warpPerBlock = blocksize/32;
	ui taskPerBlock = warpPerBlock* record.taskPerWarp;
	ui numBlocks = (threadnum-1) / blocksize + 1;
	ui taskPerRound = numBlocks*  taskPerBlock;

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
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui) + max_candidates_num * sizeof(ui)) * threadnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
		std::cout <<"An error ocurrs when allocate memory!"<<std::endl;
	}else{
		std::cout <<"Pass memory assignment test!"<<std::endl;
	}
	// compute total bytes allocated.


	// test candidate

//    std::cout << "pitch : " << valid_candidate_idx_pitch << " ";
    auto GPU_alloc_end = std::chrono::high_resolution_clock::now();
    record. cand_alloc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(GPU_alloc_end - GPU_alloc_start).count();

//    while (true) {
//
//
//      	while (idx[cur_depth] < idx_count[cur_depth]) {
//          	// sampling part
          	if(idx[cur_depth] == 0 && if_sampling(cur_depth, step)) {
          		auto sampling_start = std::chrono::high_resolution_clock::now();
  				ui sample_time = record. sample_time;
  				// record the possibility weight to sample in the current（first） layer
  				ui round = (sample_time - 1)/ taskPerRound + 1;
  				double aver_score = 0;
  				ui h_score_count = 0;
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					WarpLayerBalance_v2<blocksize><<<numBlocks,blocksize,blocksize/32*sizeof(int)>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerWarp);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}

template <const ui threadsPerBlock>
ui test_intersection_count (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
	record. arr_range_count  = new ui [5];
	for (int i = 0; i< 5; ++i){
		record.arr_range_count[i] = 0;
	}
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
    ui* d_arr_range_count;
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
    allocateGPU1D( d_candidates_count ,candidates_count, query_vertices_num* sizeof(ui));
    allocateGPU1D( d_arr_range_count ,record. arr_range_count, 5* sizeof(ui));

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


	ui numBlocks = (threadnum-1) / threadsPerBlock + 1;
	ui taskPerRound = numBlocks* record. taskPerBlock;

	std::cout << "readonly GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
	std::cout << "threadsPerBlock: "<< threadsPerBlock << " numBlocks: "<< numBlocks << " total threads: " << threadsPerBlock*numBlocks << " max_candidates_num " << max_candidates_num<<std::endl;

	// for each thread we assign its own global memoory.
    allocateMemoryPerThread(d_idx ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_range ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_count ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui) + max_candidates_num * sizeof(ui)) * threadnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
		std::cout <<"An error ocurrs when allocate memory!"<<std::endl;
	}else{
		std::cout <<"Pass memory assignment test!"<<std::endl;
	}
	// compute total bytes allocated.


	// test candidate

//    std::cout << "pitch : " << valid_candidate_idx_pitch << " ";
    auto GPU_alloc_end = std::chrono::high_resolution_clock::now();
    record. cand_alloc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(GPU_alloc_end - GPU_alloc_start).count();

//    while (true) {
//      //    	std::cout << "cur_depth: " << cur_depth <<" idx[cur_depth]: " << idx[cur_depth]<<" idx_count[cur_depth]: " << idx_count[cur_depth]<<std::endl;

//          	while (idx[cur_depth] < idx_count[cur_depth]) {
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
						test_intersection<threadsPerBlock><<<numBlocks,threadsPerBlock>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock,d_arr_range_count);
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
					cudaMemcpy( record.arr_range_count, d_arr_range_count, sizeof(ui)*5, cudaMemcpyDeviceToHost);
					std::cout <<"< 32 : "<< record.arr_range_count[0] << std::endl;
					std::cout <<"< 128 : " << record.arr_range_count[1] << std::endl;
					std::cout <<"< 512 : "<< record.arr_range_count[2] << std::endl;
					std::cout <<"< 2048 : "<< record.arr_range_count[3] <<std::endl;
					std::cout <<"> 2048 : "<< record.arr_range_count[4] << std::endl;
					// beacuse 1st only run once, so * fixednum
					record.est_path = aver_score/sample_time * fixednum;
					auto sampling_end = std::chrono::high_resolution_clock::now();
					record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
				}
//                  ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//                  VertexID u = order[cur_depth];
//                  VertexID v = candidates[u][valid_idx];
//
//                  if (visited_vertices[v]) {
//                      idx[cur_depth] += 1;
//
//                      continue;
//                  }
//
//                  embedding[u] = v;
//                  idx_embedding[u] = valid_idx;
//                  visited_vertices[v] = true;
//                  idx[cur_depth] += 1;
//
//
//                  if (cur_depth == max_depth - 1) {
//                      embedding_cnt += 1;
//                      record. real_workload +=1;
//                      visited_vertices[v] = false;
//                      //print a path
//    //                  for (int i = 0; i<= cur_depth; i++){
//    //                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
//    //                  }
//
//                      if (embedding_cnt >= output_limit_num) {
//                          goto EXIT;
//                      }
//                  } else {
//
//
//                      call_count += 1;
//                      cur_depth += 1;
//
//                      idx[cur_depth] = 0;
//                      generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                                  bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//                  	if(idx_count[cur_depth] == 0){
//                  		record. real_workload +=1;
//                  	}
//                  }
//              }
//
//
//              cur_depth -= 1;
//              if (cur_depth < 0)
//                  break;
//              else {
//                  VertexID u = order[cur_depth];
//
//                  visited_vertices[embedding[u]] = false;
//
//
//              }
//          }
//
//
//
//          EXIT:
//      //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//      //                  visited_vertices,
//      //                  bn, bn_count);
//
//      	auto end = std::chrono::high_resolution_clock::now();
//
//      	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//        return embedding_cnt;
          		return 0;
}


template <const ui blocksize>
ui help (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					Help<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}



template <const ui blocksize>
ui helpplus (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					HelpPlus<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}



template <const ui blocksize>
ui helpplusres (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					HelpPlusRes<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}

template <const ui blocksize>
ui help_Baseline_pathsync (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					baselinePath<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}

template <const ui blocksize>
ui help_Baseline_layersync (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					baselineLayer<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
					cudaDeviceSynchronize();
					cudaMemcpy( &aver_score, d_score, sizeof(double), cudaMemcpyDeviceToHost);
	//				cudaMemcpy( &h_score_count, d_score_count, sizeof(ui), cudaMemcpyDeviceToHost);
	//				std::cout << "total_score: " << aver_score << "path count " << h_score_count <<std::endl;
					auto err = cudaGetLastError();
					if (err != cudaSuccess){
						record. successrun = false;
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}

//

template <const ui blocksize>
ui GPUWandorJoin (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					wanderJoin<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
					cudaDeviceSynchronize();
					cudaMemcpy( &aver_score, d_score, sizeof(double), cudaMemcpyDeviceToHost);
//					cudaMemcpy( &h_score_count, d_score_count, sizeof(ui), cudaMemcpyDeviceToHost);
					std::cout << "total_score: " << aver_score << "path count " << h_score_count <<std::endl;
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}

template <const ui blocksize>
ui branchJoin (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
    // 1-d array write by thread
    ui* d_idx;
	ui* d_idx_count;
	ui* d_embedding;
	ui* d_idx_embedding;
	ui* d_temp;
	ui* d_temp_size;
	double* d_range;
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
    allocateMemoryPerThread(d_range ,query_vertices_num * sizeof(double), threadnum);
    allocateMemoryPerThread(d_idx_count ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui) + max_candidates_num * sizeof(ui)) * threadnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					BranchJoin<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
ui helpIndependent (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
  					HelpIndependent<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock,d_denominator);
					cudaDeviceSynchronize();
					cudaMemcpy( &aver_score, d_score, sizeof(double), cudaMemcpyDeviceToHost);
//					cudaMemcpy( denominator, d_denominator, sizeof(ui), cudaMemcpyDeviceToHost);

					auto err = cudaGetLastError();
					if (err != cudaSuccess){
						std::cout <<"An error ocurrs when sampling!"<<std::endl;
					}else{
						std::cout <<"Sampling end!"<<std::endl;
					}
  				}
				// beacuse 1st only run once, so * fixednum
//  				printf("denominator: %d \n",denominator[0]);
//  				record.est_path = aver_score/denominator[0]* fixednum;
  				record.est_path = aver_score/sample_time* fixednum;
  				auto sampling_end = std::chrono::high_resolution_clock::now();
				record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
          	}
          	return 0;
}

template <const ui blocksize>
ui helpIndependentplus (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
  					HelpIndependentPlus<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock,d_denominator);
					cudaDeviceSynchronize();
					cudaMemcpy( &aver_score, d_score, sizeof(double), cudaMemcpyDeviceToHost);
	//				cudaMemcpy( denominator, d_denominator, sizeof(ui), cudaMemcpyDeviceToHost);
	//				cudaMemcpy( &h_score_count, d_score_count, sizeof(ui), cudaMemcpyDeviceToHost);
	//				std::cout << "total_score: " << aver_score << "path count " << h_score_count <<std::endl;
					auto err = cudaGetLastError();
					if (err != cudaSuccess){
						std::cout <<"An error ocurrs when sampling!"<<std::endl;
					}else{
						std::cout <<"Sampling end!"<<std::endl;
					}
  				}
//  				printf("aver_score : %f denominator: %f, result: %f \n",aver_score,(double)denominator[0],aver_score/denominator[0]);
				// beacuse 1st only run once, so * fixednum
//  				record.est_path = aver_score/denominator[0]* fixednum;
  				record.est_path = aver_score/sample_time* fixednum;
  				auto sampling_end = std::chrono::high_resolution_clock::now();
				record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
          	}
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
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
ui helpIndependentplusres (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
  					HelpIndependentPlusRes<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock,d_denominator);
					cudaDeviceSynchronize();
					cudaMemcpy( &aver_score, d_score, sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy( denominator, d_denominator, sizeof(ui), cudaMemcpyDeviceToHost);
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
  				record.est_path = aver_score/denominator[0]* fixednum;
  				auto sampling_end = std::chrono::high_resolution_clock::now();
				record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
          	}
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}

// will end when waiting time > 20min
size_t LFTJ_waitingtime (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates,
                    ui *candidates_count,
                    ui *order, size_t output_limit_num, size_t &call_count, timer &record) {

	auto start = std::chrono::high_resolution_clock::now();
	record. enumerating_time = 0;

#ifdef DISTRIBUTION
    distribution_count_ = new size_t[data_graph->getVerticesCount()];
    memset(distribution_count_, 0, data_graph->getVerticesCount() * sizeof(size_t));
    size_t* begin_count = new size_t[query_graph->getVerticesCount()];
    memset(begin_count, 0, query_graph->getVerticesCount() * sizeof(size_t));
#endif

    // Generate bn.
    ui **bn;
    ui *bn_count;
    generateBN(query_graph, order, bn, bn_count);

    // Allocate the memory buffer.
    ui *idx;
    ui *idx_count;
    ui *embedding;
    ui *idx_embedding;
    ui *temp_buffer;
    ui **valid_candidate_idx;
    bool *visited_vertices;
    allocateBuffer(data_graph, query_graph, candidates_count, idx, idx_count, embedding, idx_embedding,
                   temp_buffer, valid_candidate_idx, visited_vertices);

    size_t embedding_cnt = 0;
    int cur_depth = 0;
    int max_depth = query_graph->getVerticesCount();
    VertexID start_vertex = order[0];

    idx[cur_depth] = 0;
    idx_count[cur_depth] = candidates_count[start_vertex];

    for (ui i = 0; i < idx_count[cur_depth]; ++i) {
        valid_candidate_idx[cur_depth][i] = i;
    }

#ifdef ENABLE_FAILING_SET
    std::vector<std::bitset<MAXIMUM_QUERY_GRAPH_SIZE>> ancestors;
    computeAncestor(query_graph, bn, bn_count, order, ancestors);
    std::vector<std::bitset<MAXIMUM_QUERY_GRAPH_SIZE>> vec_failing_set(max_depth);
    std::unordered_map<VertexID, VertexID> reverse_embedding;
    reverse_embedding.reserve(MAXIMUM_QUERY_GRAPH_SIZE * 2);
#endif

#ifdef SPECTRUM
    exit_ = false;
#endif

    while (true) {
        while (idx[cur_depth] < idx_count[cur_depth]) {
            ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
            VertexID u = order[cur_depth];
            VertexID v = candidates[u][valid_idx];

            if (visited_vertices[v]) {
                idx[cur_depth] += 1;
#ifdef ENABLE_FAILING_SET
                vec_failing_set[cur_depth] = ancestors[u];
                vec_failing_set[cur_depth] |= ancestors[reverse_embedding[v]];
                vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
#endif
                continue;
            }

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_vertices[v] = true;
            idx[cur_depth] += 1;

#ifdef DISTRIBUTION
            begin_count[cur_depth] = embedding_cnt;
            // printf("Cur Depth: %d, v: %u, begin: %zu\n", cur_depth, v, embedding_cnt);
#endif

#ifdef ENABLE_FAILING_SET
            reverse_embedding[v] = u;
#endif

            if (cur_depth == max_depth - 1) {
                embedding_cnt += 1;
                visited_vertices[v] = false;

#ifdef DISTRIBUTION
                distribution_count_[v] += 1;
#endif

#ifdef ENABLE_FAILING_SET
                reverse_embedding.erase(embedding[u]);
                vec_failing_set[cur_depth].set();
                vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
#endif
                if (embedding_cnt % 1000000000 == 0) {
                	auto checkpoint = std::chrono::high_resolution_clock::now();
                	auto passing_time = std::chrono::duration_cast<std::chrono::nanoseconds>(checkpoint -start).count();
                	// stop if > 20 mins (1200000000000)
                	if(passing_time > 600000000000){
                		record.successrun = false;
                		goto EXIT;
                	}

                }
            } else {
                call_count += 1;
                cur_depth += 1;

                idx[cur_depth] = 0;
                generateValidCandidateIndex(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
                                            bn_count, order, temp_buffer);

#ifdef ENABLE_FAILING_SET
                if (idx_count[cur_depth] == 0) {
                    vec_failing_set[cur_depth - 1] = ancestors[order[cur_depth]];
                } else {
                    vec_failing_set[cur_depth - 1].reset();
                }
#endif
            }
        }

#ifdef SPECTRUM
        if (exit_) {
            goto EXIT;
        }
#endif

        cur_depth -= 1;
        if (cur_depth < 0)
            break;
        else {
            VertexID u = order[cur_depth];
#ifdef ENABLE_FAILING_SET
            reverse_embedding.erase(embedding[u]);
            if (cur_depth != 0) {
                if (!vec_failing_set[cur_depth].test(u)) {
                    vec_failing_set[cur_depth - 1] = vec_failing_set[cur_depth];
                    idx[cur_depth] = idx_count[cur_depth];
                } else {
                    vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
                }
            }
#endif
            visited_vertices[embedding[u]] = false;

#ifdef DISTRIBUTION
            distribution_count_[embedding[u]] += embedding_cnt - begin_count[cur_depth];
            // printf("Cur Depth: %d, v: %u, begin: %zu, end: %zu\n", cur_depth, embedding[u], begin_count[cur_depth], embedding_cnt);
#endif
        }
    }

    // Release the buffer.

#ifdef DISTRIBUTION
    if (embedding_cnt >= output_limit_num) {
        for (int i = 0; i < max_depth - 1; ++i) {
            ui v = embedding[order[i]];
            distribution_count_[v] += embedding_cnt - begin_count[i];
        }
    }
    delete[] begin_count;
#endif

    EXIT:
    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
                  visited_vertices,
                  bn, bn_count);

#if ENABLE_QFLITER == 1
    delete[] temp_bsr_base1_;
    delete[] temp_bsr_base2_;
    delete[] temp_bsr_state1_;
    delete[] temp_bsr_state2_;

    for (ui i = 0; i < max_depth; ++i) {
        for (ui j = 0; j < max_depth; ++j) {
//            delete qfliter_bsr_graph_[i][j];
        }
        delete[] qfliter_bsr_graph_[i];
    }
    delete[] qfliter_bsr_graph_;
#endif

    auto end = std::chrono::high_resolution_clock::now();
	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count();
    return embedding_cnt;
}

//first
template <const ui blocksize>
ui helpIndepentauto (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    double * sample_ratio;
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
    sample_ratio = new double [step];
    score[0] = 0;
    score_count[0] = 0;
    denominator[0] = 0;
    for (auto i = 0; i<step; ++i){
    	sample_ratio[i] = 0.4;
    }

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
    double* d_sample_ratio;
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
    allocateGPU1D(d_sample_ratio,sample_ratio,sizeof(double)*step);
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
  				double pass_ratio = 0;
  				ui h_score_count = 0;
  				// run some iterations to deside the proper ratio.
  				for (ui turingdepth = 0; turingdepth < max_depth;++turingdepth){
  					printf("test layer: %d, set ratio %f \n",turingdepth, sample_ratio[turingdepth]);
  					decideRatio<blocksize><<<10,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,blocksize,d_denominator, d_sample_ratio,turingdepth);
  					cudaDeviceSynchronize();
  					// use  h_score_count to store the number of success samples
  					cudaMemcpy( &h_score_count, d_score_count, sizeof(ui), cudaMemcpyDeviceToHost);
  					pass_ratio = (double) h_score_count/ (10*blocksize);
  					printf("success : %d, total: %d, ratio: %f \n", h_score_count, 10*blocksize, pass_ratio);
  					h_score_count = 0;
					cudaMemcpy( d_score_count, &h_score_count, sizeof(ui), cudaMemcpyHostToDevice);
					double default_ratio = pass_ratio/2;
  					// tuning ratio.
					while (pass_ratio > default_ratio &&  sample_ratio[turingdepth] > 0.002 ){
						sample_ratio[turingdepth] = sample_ratio[turingdepth] /2;
						cudaMemcpy( d_sample_ratio, sample_ratio, sizeof(ui)*max_depth, cudaMemcpyHostToDevice);
						printf("test layer: %d, set ratio %f \n",turingdepth, sample_ratio[turingdepth]);
						decideRatio<blocksize><<<10,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,blocksize,d_denominator, d_sample_ratio,turingdepth);
						cudaDeviceSynchronize();
						// use  h_score_count to store the number of success samples
						cudaMemcpy( &h_score_count, d_score_count, sizeof(ui), cudaMemcpyDeviceToHost);
						pass_ratio = (double) h_score_count/ (10*blocksize);
						printf("success : %d, total: %d, ratio: %f \n", h_score_count, 10*blocksize, pass_ratio);
						h_score_count = 0;
						cudaMemcpy( d_score_count, &h_score_count, sizeof(ui), cudaMemcpyHostToDevice);

					}
  				}
  				//clear GPU data
				h_score_count = 0;
				cudaMemcpy( d_score_count, &h_score_count, sizeof(ui), cudaMemcpyHostToDevice);
  				// round is 1 in fact
  				for (ui k = 0; k< round; ++k){
					//one thread one path
//  					HelpIndependentPlus<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock,d_denominator);
  					HelpIndependentauto<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock,d_denominator,d_sample_ratio);
					cudaDeviceSynchronize();
					cudaMemcpy( &aver_score, d_score, sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy( denominator, d_denominator, sizeof(ui), cudaMemcpyDeviceToHost);
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
  				record.est_path = aver_score/denominator[0]* fixednum;
  				auto sampling_end = std::chrono::high_resolution_clock::now();
				record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
          	}

          	return 0;
}

//
template <const ui blocksize>
ui helpplusadaptratio (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    double * sample_ratio;
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
    sample_ratio = new double [1];
    score[0] = 0;
    score_count[0] = 0;
    denominator[0] = 0;
    sample_ratio[0] = 1.0;


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
    double* d_sample_ratio;
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
    allocateGPU1D(d_sample_ratio,sample_ratio,sizeof(double));
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
  				double pass_ratio = 0;
  				ui h_score_count = 0;
  				// run some iterations to deside the proper ratio.
				printf("set ratio %f \n", sample_ratio[0]);
				decideRatio<blocksize><<<10,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,blocksize,d_denominator, d_sample_ratio);
				cudaDeviceSynchronize();
				// use  h_score_count to store the number of success samples
				cudaMemcpy( &h_score_count, d_score_count, sizeof(ui), cudaMemcpyDeviceToHost);
				pass_ratio = (double) h_score_count/ (10*blocksize);
				printf("success : %d, total: %d, ratio: %f \n", h_score_count, 10*blocksize, pass_ratio);
				h_score_count = 0;
				cudaMemcpy( d_score_count, &h_score_count, sizeof(ui), cudaMemcpyHostToDevice);
				//100% success ratio
				record.full_ratio = pass_ratio;
				printf("pass_ratio for 100% is %f \n", pass_ratio);
				sample_ratio[0] = 0.8;
				// tuning ratio.
				record.adapt_ratio = pass_ratio;
				record.sample_ratio = 0.1;
				double record_ratio = pass_ratio;
				while (sample_ratio[0] >= 0.05 ){
					cudaMemcpy( d_sample_ratio, sample_ratio, sizeof(double), cudaMemcpyHostToDevice);
					printf(" set ratio %f \n", sample_ratio[0]);
					decideRatio<blocksize><<<10,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,blocksize,d_denominator, d_sample_ratio);
					cudaDeviceSynchronize();
					// use  h_score_count to store the number of success samples
					cudaMemcpy( &h_score_count, d_score_count, sizeof(ui), cudaMemcpyDeviceToHost);
					pass_ratio = (double) h_score_count/ (10*blocksize);
					printf("success : %d, total: %d, ratio: %f \n", h_score_count, 10*blocksize, pass_ratio);
					h_score_count = 0;
					cudaMemcpy( d_score_count, &h_score_count, sizeof(double), cudaMemcpyHostToDevice);
					if(pass_ratio >= record.full_ratio *0.8 && pass_ratio <= record.full_ratio *1.2 ){
						// adapt successs ratio
						record.adapt_ratio = pass_ratio;
						record.sample_ratio = sample_ratio[0];
					}
					if(sample_ratio[0] == 0.4 ){
						//40% successs ratio
						record.base_ratio = pass_ratio;
						printf("if sample ratio is 0,4 then success ratio is %f",pass_ratio );
					}
					sample_ratio[0] = sample_ratio[0]/2;
				}
				cudaMemcpy( d_sample_ratio, &record.sample_ratio, sizeof(double), cudaMemcpyHostToDevice);
				printf("the running sample intersection ratio is %f \n", record.sample_ratio);

  				//clear GPU data
				h_score_count = 0;
				cudaMemcpy( d_score_count, &h_score_count, sizeof(ui), cudaMemcpyHostToDevice);
  				// round is 1 in fact
  				for (ui k = 0; k< round; ++k){
					//one thread one path
//  					HelpIndependentPlus<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock,d_denominator);
  					HelpIndependentadapt<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock,d_denominator,d_sample_ratio);
					cudaDeviceSynchronize();
					cudaMemcpy( &aver_score, d_score, sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy( denominator, d_denominator, sizeof(ui), cudaMemcpyDeviceToHost);
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
  				record.est_path = aver_score/denominator[0]* fixednum;
  				auto sampling_end = std::chrono::high_resolution_clock::now();
				record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
          	}

          	return 0;
}

template <const ui blocksize>
ui testratio (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    double * sample_ratio;
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
    sample_ratio = new double [1];
    score[0] = 0;
    score_count[0] = 0;
    denominator[0] = 0;
    sample_ratio[0] = 0.999;


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
    double* d_sample_ratio;
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
    allocateGPU1D(d_sample_ratio,sample_ratio,sizeof(double));
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
  				ui sample_time = record. sample_time;
  				// record the possibility weight to sample in the current（first） layer
  				ui round = (sample_time - 1)/ taskPerRound + 1;
  				double aver_score = 0;
  				double pass_ratio = 0;
  				ui h_score_count = 0;
  				// run some iterations to deside the proper ratio.
				printf("set ratio %f \n", sample_ratio[0]);
				auto sampling_start = std::chrono::high_resolution_clock::now();
				decideRatio<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,blocksize,d_denominator, d_sample_ratio);
				cudaDeviceSynchronize();
				auto sampling_end = std::chrono::high_resolution_clock::now();
				// use  h_score_count to store the number of success samples
				cudaMemcpy( &h_score_count, d_score_count, sizeof(ui), cudaMemcpyDeviceToHost);
				pass_ratio = (double) h_score_count/ (10*blocksize);
				printf("success : %d, total: %d, ratio: %f \n", h_score_count, 10*blocksize, pass_ratio);
				h_score_count = 0;
				cudaMemcpy( d_score_count, &h_score_count, sizeof(ui), cudaMemcpyHostToDevice);
				//100% success ratio
				record.full_ratio = pass_ratio;
				printf("pass_ratio for 100 is %f \n", pass_ratio);
				sample_ratio[0] = 0.9;
				//compute time
				auto sampling_time =  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
				std::string msg_time = std::to_string(sampling_time) + "\t";
				// tuning ratio.
				record.adapt_ratio = pass_ratio;
				record.sample_ratio = 0.1;
				std::string msg = std::to_string(pass_ratio) + "\t";
				double record_ratio = pass_ratio;
				double samplelist [] = {0.5, 0.3, 0.1, 0.05, 0.02};
				for (int i = 0; i< 5; i++){
					sample_ratio[0] = samplelist[i];
					cudaMemcpy( d_sample_ratio, sample_ratio, sizeof(double), cudaMemcpyHostToDevice);
					printf(" set ratio %f \n", sample_ratio[0]);
					sampling_start = std::chrono::high_resolution_clock::now();
					decideRatio<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,blocksize,d_denominator, d_sample_ratio);
					cudaDeviceSynchronize();
					sampling_end = std::chrono::high_resolution_clock::now();
					// use  h_score_count to store the number of success samples
					cudaMemcpy( &h_score_count, d_score_count, sizeof(ui), cudaMemcpyDeviceToHost);
					pass_ratio = (double) h_score_count/ (10*blocksize);
					printf("success : %d, total: %d, ratio: %f \n", h_score_count, numBlocks*blocksize, pass_ratio);
					h_score_count = 0;
					cudaMemcpy( d_score_count, &h_score_count, sizeof(double), cudaMemcpyHostToDevice);
					msg += std::to_string(pass_ratio) + "\t";
					sample_ratio[0] = sample_ratio[0]-0.1;
					sampling_time =  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
					msg_time += std::to_string(sampling_time) + "\t";
				}
				std::cout << msg << std::endl;
				std::cout << msg_time << std::endl;
				record.msg = msg;
				record.msg_time = msg_time;
          	}
          	return 0;
}


template <const ui blocksize>
ui outputCandidateSet (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					wanderJoinWithoutCheck<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
ui outputworkloads (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    double* d_workload_early_exit;
    double* d_intersection_count;
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
    allocateGPU1D( d_workload_early_exit ,score, 1* sizeof(double));
    allocateGPU1D( d_intersection_count ,score, 1* sizeof(double));
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
  				double exworkload = 0;
  				double intersection_count = 0;
  				// round is 1 in fact
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					workloadEst<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock,d_denominator,d_workload_early_exit,d_intersection_count);
					cudaDeviceSynchronize();
					cudaMemcpy( &aver_score, d_score, sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy( denominator, d_denominator, sizeof(ui), cudaMemcpyDeviceToHost);
					cudaMemcpy( &exworkload, d_workload_early_exit, sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy( &intersection_count, d_intersection_count, sizeof(double), cudaMemcpyDeviceToHost);
	//				cudaMemcpy( &h_score_count, d_score_count, sizeof(ui), cudaMemcpyDeviceToHost);
	//				std::cout << "total_score: " << aver_score << "path count " << h_score_count <<std::endl;
					auto err = cudaGetLastError();
					if (err != cudaSuccess){
						std::cout <<"An error ocurrs when sampling!"<<std::endl;
					}else{

						std::cout <<"Sampling end!"<<std::endl;
					}
  				}
  				printf("intersection count on average : %f", intersection_count/denominator[0]);
				// beacuse 1st only run once, so * fixednum
  				record.est_path = aver_score/denominator[0]* fixednum;
  				record.est_workload = (exworkload + aver_score)/denominator[0]* fixednum;
  				record.set_intersection_count = (intersection_count)/denominator[0]* fixednum;
  				auto sampling_end = std::chrono::high_resolution_clock::now();
				record.sampling_time +=  std::chrono::duration_cast<std::chrono::nanoseconds>(sampling_end - sampling_start).count();
          	}
          	return 0;
}

//other ordering method
std::pair<VertexID, VertexID>  selectQSIStartEdge(const Graph *query_graph, Edges ***edge_matrix) {
     /**
      * Select the edge with the minimum number of candidate edges.
      * Tie Handling:
      *  1. the sum of the degree values of the end points of an edge (prioritize the edge with smaller degree).
      *  2. label id
      */
     ui min_value = std::numeric_limits<ui>::max();
     ui min_degree_sum = std::numeric_limits<ui>::max();

     std::pair<VertexID, VertexID> start_edge = std::make_pair(0, 1);
     for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
          VertexID begin_vertex = i;
          ui nbrs_cnt;
          const ui* nbrs = query_graph->getVertexNeighbors(begin_vertex, nbrs_cnt);

          for (ui j = 0; j < nbrs_cnt; ++j) {
               VertexID end_vertex = nbrs[j];
               ui cur_value = (*edge_matrix[begin_vertex][end_vertex]).edge_count_;
               ui cur_degree_sum = query_graph->getVertexDegree(begin_vertex) + query_graph->getVertexDegree(end_vertex);

               if (cur_value < min_value || (cur_value == min_value
                   && (cur_degree_sum < min_degree_sum))) {
                    min_value = cur_value;
                    min_degree_sum = cur_degree_sum;

                    start_edge = query_graph->getVertexDegree(begin_vertex) < query_graph->getVertexDegree(end_vertex) ?
                            std::make_pair(end_vertex, begin_vertex) : std::make_pair(begin_vertex, end_vertex);
               }
          }
     }

     return start_edge;
}
void generateQSIQueryPlan(const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix,
                                             ui *&order, ui *&pivot) {
     /**
      * Generate a minimum spanning tree.
      */
     std::vector<bool> visited_vertices(query_graph->getVerticesCount(), false);
     std::vector<bool> adjacent_vertices(query_graph->getVerticesCount(), false);
     order = new ui[query_graph->getVerticesCount()];
     pivot = new ui[query_graph->getVerticesCount()];

     std::pair<VertexID, VertexID> start_edge = selectQSIStartEdge(query_graph, edge_matrix);
     order[0] = start_edge.first;
     order[1] = start_edge.second;
     pivot[1] = start_edge.first;
     updateValidVertices(query_graph, start_edge.first, visited_vertices, adjacent_vertices);
     updateValidVertices(query_graph, start_edge.second, visited_vertices, adjacent_vertices);

     for (ui l = 2; l < query_graph->getVerticesCount(); ++l) {
          ui min_value = std::numeric_limits<ui>::max();
          ui max_degree = 0;
          ui max_adjacent_selected_vertices = 0;
          std::pair<VertexID, VertexID> selected_edge;
          for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
               VertexID begin_vertex = i;
               if (visited_vertices[begin_vertex]) {
                    ui nbrs_cnt;
                    const VertexID *nbrs = query_graph->getVertexNeighbors(begin_vertex, nbrs_cnt);
                    for (ui j = 0; j < nbrs_cnt; ++j) {
                         VertexID end_vertex = nbrs[j];

                         if (!visited_vertices[end_vertex]) {
                              ui cur_value = (*edge_matrix[begin_vertex][end_vertex]).edge_count_;
                              ui cur_degree = query_graph->getVertexDegree(end_vertex);
                              ui adjacent_selected_vertices = 0;
                              ui end_vertex_nbrs_count;
                              const VertexID *end_vertex_nbrs = query_graph->getVertexNeighbors(end_vertex,
                                                                                                end_vertex_nbrs_count);

                              for (ui k = 0; k < end_vertex_nbrs_count; ++k) {
                                   VertexID end_vertex_nbr = end_vertex_nbrs[k];

                                   if (visited_vertices[end_vertex_nbr]) {
                                        adjacent_selected_vertices += 1;
                                   }
                              }

                              if (cur_value < min_value || (cur_value == min_value && adjacent_selected_vertices < max_adjacent_selected_vertices)
                                      || (cur_value == min_value && adjacent_selected_vertices == max_adjacent_selected_vertices && cur_degree > max_degree)) {
                                   selected_edge = std::make_pair(begin_vertex, end_vertex);
                                   min_value = cur_value;
                                   max_degree = cur_degree;
                                   max_adjacent_selected_vertices = adjacent_selected_vertices;
                              }
                         }
                    }
               }
          }

          order[l] = selected_edge.second;
          pivot[l] = selected_edge.first;
          updateValidVertices(query_graph, selected_edge.second, visited_vertices, adjacent_vertices);
     }
}


void bfsTraversal(const Graph *graph, VertexID root_vertex, TreeNode *&tree, VertexID *&bfs_order) {
    ui vertex_num = graph->getVerticesCount();

    std::queue<VertexID> bfs_queue;
    std::vector<bool> visited(vertex_num, false);

    tree = new TreeNode[vertex_num];
    for (ui i = 0; i < vertex_num; ++i) {
        tree[i].initialize(vertex_num);
    }
    bfs_order = new VertexID[vertex_num];

    ui visited_vertex_count = 0;
    bfs_queue.push(root_vertex);
    visited[root_vertex] = true;
    tree[root_vertex].level_ = 0;
    tree[root_vertex].id_ = root_vertex;

    while(!bfs_queue.empty()) {
        const VertexID u = bfs_queue.front();
        bfs_queue.pop();
        bfs_order[visited_vertex_count++] = u;

        ui u_nbrs_count;
        const VertexID* u_nbrs = graph->getVertexNeighbors(u, u_nbrs_count);
        for (ui i = 0; i < u_nbrs_count; ++i) {
            VertexID u_nbr = u_nbrs[i];

            if (!visited[u_nbr]) {
                bfs_queue.push(u_nbr);
                visited[u_nbr] = true;
                tree[u_nbr].id_ = u_nbr;
                tree[u_nbr].parent_ = u;
                tree[u_nbr].level_ = tree[u] .level_ + 1;
                tree[u].children_[tree[u].children_count_++] = u_nbr;
            }
        }
    }
}

void dfs(TreeNode *tree, VertexID cur_vertex, VertexID *dfs_order, ui &count) {
    dfs_order[count++] = cur_vertex;

    for (ui i = 0; i < tree[cur_vertex].children_count_; ++i) {
        dfs(tree, tree[cur_vertex].children_[i], dfs_order, count);
    }
}

void dfsTraversal(TreeNode *tree, VertexID root_vertex, ui node_num, VertexID *&dfs_order) {
    dfs_order = new VertexID[node_num];
    ui count = 0;
    dfs(tree, root_vertex, dfs_order, count);
}


VertexID selectTSOFilterStartVertex(const Graph *data_graph, const Graph *query_graph) {
    auto rank_compare = [](std::pair<VertexID, double> l, std::pair<VertexID, double> r) {
        return l.second < r.second;
    };
    // Maximum priority queue.
    std::priority_queue<std::pair<VertexID, double>, std::vector<std::pair<VertexID, double>>, decltype(rank_compare)> rank_queue(rank_compare);

    // Compute the ranking.
    for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
        VertexID query_vertex = i;
        LabelID label = query_graph->getVertexLabel(query_vertex);
        ui degree = query_graph->getVertexDegree(query_vertex);
        ui frequency = data_graph->getLabelsFrequency(label);
        double rank = frequency / (double)degree;
        rank_queue.push(std::make_pair(query_vertex, rank));
    }

    // Keep the top-3.
    while (rank_queue.size() > 3) {
        rank_queue.pop();
    }

    // Pick the one with the smallest number of candidates.
    VertexID start_vertex = 0;
    ui min_candidates_num = data_graph->getGraphMaxLabelFrequency() + 1;
    while (!rank_queue.empty()) {
        VertexID query_vertex = rank_queue.top().first;

        if (rank_queue.size() == 1) {
            ui count;
            computeCandidateWithNLF(data_graph, query_graph, query_vertex, count,NULL);
            if (count < min_candidates_num) {
                start_vertex = query_vertex;
            }
        }
        else {
            LabelID label = query_graph->getVertexLabel(query_vertex);
            ui frequency = data_graph->getLabelsFrequency(label);
            if (frequency / (double)data_graph->getVerticesCount() <= 0.05) {
                ui count;
                computeCandidateWithNLF(data_graph, query_graph, query_vertex, count,NULL);
                if (count < min_candidates_num) {
                    start_vertex = query_vertex;
                    min_candidates_num = count;
                }
            }
        }
        rank_queue.pop();
    }

    return start_vertex;
}

void generateTSOFilterPlan(const Graph *data_graph, const Graph *query_graph, TreeNode *&tree, VertexID *&order) {
    VertexID start_vertex = selectTSOFilterStartVertex(data_graph, query_graph);
    VertexID* bfs_order;
    bfsTraversal(query_graph, start_vertex, tree, bfs_order);
    dfsTraversal(tree, start_vertex, query_graph->getVerticesCount(), order);
    delete[] bfs_order;
}

void estimatePathEmbeddsingsNum(std::vector<ui> &path, Edges ***edge_matrix,
                                                   std::vector<size_t> &estimated_embeddings_num) {
    assert(path.size() > 1);
    std::vector<size_t> parent;
    std::vector<size_t> children;

    estimated_embeddings_num.resize(path.size() - 1);
    Edges& last_edge = *edge_matrix[path[path.size() - 2]][path[path.size() - 1]];
    children.resize(last_edge.vertex_count_);

    size_t sum = 0;
    for (ui i = 0; i < last_edge.vertex_count_; ++i) {
        children[i] = last_edge.offset_[i + 1] - last_edge.offset_[i];
        sum += children[i];
    }

    estimated_embeddings_num[path.size() - 2] = sum;

    for (int i = path.size() - 2; i >= 1; --i) {
        ui begin = path[i - 1];
        ui end = path[i];

        Edges& edge = *edge_matrix[begin][end];
        parent.resize(edge.vertex_count_);

        sum = 0;
        for (ui j = 0; j < edge.vertex_count_; ++j) {

            size_t local_sum = 0;
            for (ui k = edge.offset_[j]; k < edge.offset_[j + 1]; ++k) {
                ui nbr = edge.edge_[k];
                local_sum += children[nbr];
            }

            parent[j] = local_sum;
            sum += local_sum;
        }

        estimated_embeddings_num[i - 1] = sum;
        parent.swap(children);
    }
}

void generateRootToLeafPaths(TreeNode *tree_node, VertexID cur_vertex, std::vector<ui> &cur_path,
                                                std::vector<std::vector<ui>> &paths) {
    TreeNode& cur_node = tree_node[cur_vertex];
    cur_path.push_back(cur_vertex);

    if (cur_node.children_count_ == 0) {
        paths.emplace_back(cur_path);
    }
    else {
        for (ui i = 0; i < cur_node.children_count_; ++i) {
            VertexID next_vertex = cur_node.children_[i];
            generateRootToLeafPaths(tree_node, next_vertex, cur_path, paths);
        }
    }

    cur_path.pop_back();
}

ui generateNoneTreeEdgesCount(const Graph *query_graph, TreeNode *tree_node, std::vector<ui> &path) {
    ui non_tree_edge_count = query_graph->getVertexDegree(path[0]) - tree_node[path[0]].children_count_;

    for (ui i = 1; i < path.size(); ++i) {
        VertexID vertex = path[i];
        non_tree_edge_count += query_graph->getVertexDegree(vertex) - tree_node[vertex].children_count_ - 1;
    }

    return non_tree_edge_count;
}

void generateTSOQueryPlan(const Graph *query_graph, Edges ***edge_matrix, ui *&order, ui *&pivot,
                                             TreeNode *tree, ui *dfs_order) {
    /**
     * Order the root to leaf paths according to the estimated number of embeddings and generate the matching order.
     */
    ui query_vertices_num = query_graph->getVerticesCount();
    std::vector<std::vector<ui>> paths;
    paths.reserve(query_vertices_num);

    std::vector<ui> single_path;
    single_path.reserve(query_vertices_num);

    generateRootToLeafPaths(tree, dfs_order[0], single_path, paths);

    std::vector<std::pair<double, std::vector<ui>*>> path_orders;

    for (std::vector<ui>& path : paths) {

        std::vector<size_t> estimated_embeddings_num;

        ui non_tree_edges_count = generateNoneTreeEdgesCount(query_graph, tree, path);
        estimatePathEmbeddsingsNum(path, edge_matrix, estimated_embeddings_num);
        double score = estimated_embeddings_num[0] / (double) (non_tree_edges_count + 1);
        path_orders.emplace_back(std::make_pair(score, &path));
    }

    std::sort(path_orders.begin(), path_orders.end(), [](std::pair<double, std::vector<ui>*> l, std::pair<double, std::vector<ui>*> r)
    { return l.first < r.first; });

    std::vector<bool> visited_vertices(query_vertices_num, false);
    order = new ui[query_vertices_num];
    pivot = new ui[query_vertices_num];

    ui count = 0;
    for (auto& path : path_orders) {
        for (ui i = 0; i < path.second->size(); ++i) {
            VertexID vertex = path.second->at(i);
            if (!visited_vertices[vertex]) {
                order[count] = vertex;
                if (i != 0) {
                    pivot[count] = path.second->at(i - 1);
                }
                count += 1;
                visited_vertices[vertex] = true;
            }
        }
    }
}

void generateCorePaths(const Graph *query_graph, TreeNode *tree_node, VertexID cur_vertex,
                                          std::vector<ui> &cur_core_path, std::vector<std::vector<ui>> &core_paths) {
    TreeNode& node = tree_node[cur_vertex];
    cur_core_path.push_back(cur_vertex);

    bool is_core_leaf = true;
    for (ui i = 0; i < node.children_count_; ++i) {
        VertexID child = node.children_[i];
        if (query_graph->getCoreValue(child) > 1) {
            generateCorePaths(query_graph, tree_node, child, cur_core_path, core_paths);
            is_core_leaf = false;
        }
    }

    if (is_core_leaf) {
        core_paths.emplace_back(cur_core_path);
    }
    cur_core_path.pop_back();
}

void generateTreePaths(const Graph *query_graph, TreeNode *tree_node, VertexID cur_vertex,
                                          std::vector<ui> &cur_tree_path, std::vector<std::vector<ui>> &tree_paths) {
    TreeNode& node = tree_node[cur_vertex];
    cur_tree_path.push_back(cur_vertex);

    bool is_tree_leaf = true;
    for (ui i = 0; i < node.children_count_; ++i) {
        VertexID child = node.children_[i];
        if (query_graph->getVertexDegree(child) > 1) {
            generateTreePaths(query_graph, tree_node, child, cur_tree_path, tree_paths);
            is_tree_leaf = false;
        }
    }

    if (is_tree_leaf && cur_tree_path.size() > 1) {
        tree_paths.emplace_back(cur_tree_path);
    }
    cur_tree_path.pop_back();
}

void generateLeaves(const Graph *query_graph, std::vector<ui> &leaves) {
    for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
        VertexID cur_vertex = i;
        if (query_graph->getVertexDegree(cur_vertex) == 1) {
            leaves.push_back(cur_vertex);
        }
    }
}

VertexID selectCFLFilterStartVertex(const Graph *data_graph, const Graph *query_graph) {
    auto rank_compare = [](std::pair<VertexID, double> l, std::pair<VertexID, double> r) {
        return l.second < r.second;
    };

    std::priority_queue<std::pair<VertexID, double>, std::vector<std::pair<VertexID, double>>, decltype(rank_compare)> rank_queue(rank_compare);

    // Compute the ranking.
    for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
        VertexID query_vertex = i;

        if (query_graph->get2CoreSize() == 0 || query_graph->getCoreValue(query_vertex) > 1) {
            LabelID label = query_graph->getVertexLabel(query_vertex);
            ui degree = query_graph->getVertexDegree(query_vertex);
            ui frequency = data_graph->getLabelsFrequency(label);
            double rank = frequency / (double) degree;
            rank_queue.push(std::make_pair(query_vertex, rank));
        }
    }

    // Keep the top-3.
    while (rank_queue.size() > 3) {
        rank_queue.pop();
    }

    VertexID start_vertex = 0;
    double min_score = data_graph->getGraphMaxLabelFrequency() + 1;

    while (!rank_queue.empty()) {
        VertexID query_vertex = rank_queue.top().first;
        ui count;
        computeCandidateWithNLF(data_graph, query_graph, query_vertex, count,NULL);
        double cur_score = count / (double) query_graph->getVertexDegree(query_vertex);

        if (cur_score < min_score) {
            start_vertex = query_vertex;
            min_score = cur_score;
        }
        rank_queue.pop();
    }

    return start_vertex;
}

void generateCFLFilterPlan(const Graph *data_graph, const Graph *query_graph, TreeNode *&tree,
                                                  VertexID *&order, int &level_count, ui *&level_offset) {
    ui query_vertices_num = query_graph->getVerticesCount();
    VertexID start_vertex = selectCFLFilterStartVertex(data_graph, query_graph);
    GraphOperations::bfsTraversal(query_graph, start_vertex, tree, order);

    std::vector<ui> order_index(query_vertices_num);
    for (ui i = 0; i < query_vertices_num; ++i) {
        VertexID query_vertex = order[i];
        order_index[query_vertex] = i;
    }

    level_count = -1;
    level_offset = new ui[query_vertices_num + 1];

    for (ui i = 0; i < query_vertices_num; ++i) {
        VertexID u = order[i];
        tree[u].under_level_count_ = 0;
        tree[u].bn_count_ = 0;
        tree[u].fn_count_ = 0;

        if (tree[u].level_ != level_count) {
            level_count += 1;
            level_offset[level_count] = 0;
        }

        level_offset[level_count] += 1;

        ui u_nbrs_count;
        const VertexID* u_nbrs = query_graph->getVertexNeighbors(u, u_nbrs_count);
        for (ui j = 0; j < u_nbrs_count; ++j) {
            VertexID u_nbr = u_nbrs[j];

            if (tree[u].level_ == tree[u_nbr].level_) {
                if (order_index[u_nbr] < order_index[u]) {
                    tree[u].bn_[tree[u].bn_count_++] = u_nbr;
                }
                else {
                    tree[u].fn_[tree[u].fn_count_++] = u_nbr;
                }
            }
            else if (tree[u].level_ > tree[u_nbr].level_) {
                tree[u].bn_[tree[u].bn_count_++] = u_nbr;
            }
            else {
                tree[u].under_level_[tree[u].under_level_count_++] = u_nbr;
            }
        }
    }

    level_count += 1;

    ui prev_value = 0;
    for (ui i = 1; i <= level_count; ++i) {
        ui temp = level_offset[i];
        level_offset[i] = level_offset[i - 1] + prev_value;
        prev_value = temp;
    }
    level_offset[0] = 0;
}

void generateCFLQueryPlan(const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix,
                                             ui *&order, ui *&pivot, TreeNode *tree, ui *bfs_order, ui *candidates_count) {
    ui query_vertices_num = query_graph->getVerticesCount();
    VertexID root_vertex = bfs_order[0];
    order = new ui[query_vertices_num];
    pivot = new ui[query_vertices_num];
    std::vector<bool> visited_vertices(query_vertices_num, false);

    std::vector<std::vector<ui>> core_paths;
    std::vector<std::vector<std::vector<ui>>> forests;
    std::vector<ui> leaves;

    generateLeaves(query_graph, leaves);
    if (query_graph->getCoreValue(root_vertex) > 1) {
        std::vector<ui> temp_core_path;
        generateCorePaths(query_graph, tree, root_vertex, temp_core_path, core_paths);
        for (ui i = 0; i < query_vertices_num; ++i) {
            VertexID cur_vertex = i;
            if (query_graph->getCoreValue(cur_vertex) > 1) {
                std::vector<std::vector<ui>> temp_tree_paths;
                std::vector<ui> temp_tree_path;
                generateTreePaths(query_graph, tree, cur_vertex, temp_tree_path, temp_tree_paths);
                if (!temp_tree_paths.empty()) {
                    forests.emplace_back(temp_tree_paths);
                }
            }
        }
    }
    else {
        std::vector<std::vector<ui>> temp_tree_paths;
        std::vector<ui> temp_tree_path;
        generateTreePaths(query_graph, tree, root_vertex, temp_tree_path, temp_tree_paths);
        if (!temp_tree_paths.empty()) {
            forests.emplace_back(temp_tree_paths);
        }
    }

    // Order core paths.
    ui selected_vertices_count = 0;
    order[selected_vertices_count++] = root_vertex;
    visited_vertices[root_vertex] = true;

    if (!core_paths.empty()) {
        std::vector<std::vector<size_t>> paths_embededdings_num;
        std::vector<ui> paths_non_tree_edge_num;
        for (auto& path : core_paths) {
            ui non_tree_edge_num = generateNoneTreeEdgesCount(query_graph, tree, path);
            paths_non_tree_edge_num.push_back(non_tree_edge_num + 1);

            std::vector<size_t> path_embeddings_num;
            estimatePathEmbeddsingsNum(path, edge_matrix, path_embeddings_num);
            paths_embededdings_num.emplace_back(path_embeddings_num);
        }

        // Select the start path.
        double min_value = std::numeric_limits<double>::max();
        ui selected_path_index = 0;

        for (ui i = 0; i < core_paths.size(); ++i) {
            double cur_value = paths_embededdings_num[i][0] / (double) paths_non_tree_edge_num[i];

            if (cur_value < min_value) {
                min_value = cur_value;
                selected_path_index = i;
            }
        }


        for (ui i = 1; i < core_paths[selected_path_index].size(); ++i) {
            order[selected_vertices_count] = core_paths[selected_path_index][i];
            pivot[selected_vertices_count] = core_paths[selected_path_index][i - 1];
            selected_vertices_count += 1;
            visited_vertices[core_paths[selected_path_index][i]] = true;
        }

        core_paths.erase(core_paths.begin() + selected_path_index);
        paths_embededdings_num.erase(paths_embededdings_num.begin() + selected_path_index);
        paths_non_tree_edge_num.erase(paths_non_tree_edge_num.begin() + selected_path_index);

        while (!core_paths.empty()) {
            min_value = std::numeric_limits<double>::max();
            selected_path_index = 0;

            for (ui i = 0; i < core_paths.size(); ++i) {
                ui path_root_vertex_idx = 0;
                for (ui j = 0; j < core_paths[i].size(); ++j) {
                    VertexID cur_vertex = core_paths[i][j];

                    if (visited_vertices[cur_vertex])
                        continue;

                    path_root_vertex_idx = j - 1;
                    break;
                }

                double cur_value = paths_embededdings_num[i][path_root_vertex_idx] / (double)candidates_count[core_paths[i][path_root_vertex_idx]];
                if (cur_value < min_value) {
                    min_value = cur_value;
                    selected_path_index = i;
                }
            }

            for (ui i = 1; i < core_paths[selected_path_index].size(); ++i) {
                if (visited_vertices[core_paths[selected_path_index][i]])
                    continue;

                order[selected_vertices_count] = core_paths[selected_path_index][i];
                pivot[selected_vertices_count] = core_paths[selected_path_index][i - 1];
                selected_vertices_count += 1;
                visited_vertices[core_paths[selected_path_index][i]] = true;
            }

            core_paths.erase(core_paths.begin() + selected_path_index);
            paths_embededdings_num.erase(paths_embededdings_num.begin() + selected_path_index);
        }
    }

    // Order tree paths.
    for (auto& tree_paths : forests) {
        std::vector<std::vector<size_t>> paths_embededdings_num;
        for (auto& path : tree_paths) {
            std::vector<size_t> path_embeddings_num;
            estimatePathEmbeddsingsNum(path, edge_matrix, path_embeddings_num);
            paths_embededdings_num.emplace_back(path_embeddings_num);
        }

        while (!tree_paths.empty()) {
            double min_value = std::numeric_limits<double>::max();
            ui selected_path_index = 0;

            for (ui i = 0; i < tree_paths.size(); ++i) {
                ui path_root_vertex_idx = 0;
                for (ui j = 0; j < tree_paths[i].size(); ++j) {
                    VertexID cur_vertex = tree_paths[i][j];

                    if (visited_vertices[cur_vertex])
                        continue;

                    path_root_vertex_idx = j == 0 ? j : j - 1;
                    break;
                }

                double cur_value = paths_embededdings_num[i][path_root_vertex_idx] / (double)candidates_count[tree_paths[i][path_root_vertex_idx]];
                if (cur_value < min_value) {
                    min_value = cur_value;
                    selected_path_index = i;
                }
            }

            for (ui i = 0; i < tree_paths[selected_path_index].size(); ++i) {
                if (visited_vertices[tree_paths[selected_path_index][i]])
                    continue;

                order[selected_vertices_count] = tree_paths[selected_path_index][i];
                pivot[selected_vertices_count] = tree_paths[selected_path_index][i - 1];
                selected_vertices_count += 1;
                visited_vertices[tree_paths[selected_path_index][i]] = true;
            }

            tree_paths.erase(tree_paths.begin() + selected_path_index);
            paths_embededdings_num.erase(paths_embededdings_num.begin() + selected_path_index);
        }
    }

    // Order the leaves.
    while (!leaves.empty()) {
        double min_value = std::numeric_limits<double>::max();
        ui selected_leaf_index = 0;

        for (ui i = 0; i < leaves.size(); ++i) {
            VertexID vertex = leaves[i];
            double cur_value = candidates_count[vertex];

            if (cur_value < min_value) {
                min_value = cur_value;
                selected_leaf_index = i;
            }
        }

        if (!visited_vertices[leaves[selected_leaf_index]]) {
            order[selected_vertices_count] = leaves[selected_leaf_index];
            pivot[selected_vertices_count] = tree[leaves[selected_leaf_index]].parent_;
            selected_vertices_count += 1;
            visited_vertices[leaves[selected_leaf_index]] = true;
        }
        leaves.erase(leaves.begin() + selected_leaf_index);
    }
}
void computeCandidateWithLDF(const Graph *data_graph, const Graph *query_graph, VertexID query_vertex,
                                             ui &count, ui *buffer) {
    LabelID label = query_graph->getVertexLabel(query_vertex);
    ui degree = query_graph->getVertexDegree(query_vertex);
    count = 0;
    ui data_vertex_num;
    const ui* data_vertices = data_graph->getVerticesByLabel(label, data_vertex_num);

    if (buffer == NULL) {
        for (ui i = 0; i < data_vertex_num; ++i) {
            VertexID v = data_vertices[i];
            if (data_graph->getVertexDegree(v) >= degree) {
                count += 1;
            }
        }
    }
    else {
        for (ui i = 0; i < data_vertex_num; ++i) {
            VertexID v = data_vertices[i];
            if (data_graph->getVertexDegree(v) >= degree) {
                buffer[count++] = v;
            }
        }
    }
}

VertexID selectDPisoStartVertex(const Graph *data_graph, const Graph *query_graph) {
    double min_score = data_graph->getVerticesCount();
    VertexID start_vertex = 0;

    for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
        ui degree = query_graph->getVertexDegree(i);
        if (degree <= 1)
            continue;

        ui count = 0;
        computeCandidateWithLDF(data_graph, query_graph, i, count,NULL);
        double cur_score = count / (double)degree;
        if (cur_score < min_score) {
            min_score = cur_score;
            start_vertex = i;
        }
    }

    return start_vertex;
}

void generateDPisoFilterPlan(const Graph *data_graph, const Graph *query_graph, TreeNode *&tree,
                                                    VertexID *&order) {
    VertexID start_vertex = selectDPisoStartVertex(data_graph, query_graph);
    GraphOperations::bfsTraversal(query_graph, start_vertex, tree, order);

    ui query_vertices_num = query_graph->getVerticesCount();
    std::vector<ui> order_index(query_vertices_num);
    for (ui i = 0; i < query_vertices_num; ++i) {
        VertexID query_vertex = order[i];
        order_index[query_vertex] = i;
    }

    for (ui i = 0; i < query_vertices_num; ++i) {
        VertexID u = order[i];
        tree[u].under_level_count_ = 0;
        tree[u].bn_count_ = 0;
        tree[u].fn_count_ = 0;

        ui u_nbrs_count;
        const VertexID* u_nbrs = query_graph->getVertexNeighbors(u, u_nbrs_count);
        for (ui j = 0; j < u_nbrs_count; ++j) {
            VertexID u_nbr = u_nbrs[j];
            if (order_index[u_nbr] < order_index[u]) {
                tree[u].bn_[tree[u].bn_count_++] = u_nbr;
            }
            else {
                tree[u].fn_[tree[u].fn_count_++] = u_nbr;
            }
        }
    }
}

void generateDSPisoQueryPlan(const Graph *query_graph, Edges ***edge_matrix, ui *&order, ui *&pivot,
                                                TreeNode *tree, ui *bfs_order, ui *candidates_count, ui **&weight_array) {
    ui query_vertices_num = query_graph->getVerticesCount();
    order = new ui[query_vertices_num];
    pivot = new ui[query_vertices_num];

    for (ui i = 0; i < query_vertices_num; ++i) {
        order[i] = bfs_order[i];
    }

    for (ui i = 1; i < query_vertices_num; ++i) {
        pivot[i] = tree[order[i]].parent_;
    }

    // Compute weight array.
    weight_array = new ui*[query_vertices_num];
    for (ui i = 0; i < query_vertices_num; ++i) {
        weight_array[i] = new ui[candidates_count[i]];
        std::fill(weight_array[i], weight_array[i] + candidates_count[i], std::numeric_limits<ui>::max());
    }

    for (int i = query_vertices_num - 1; i >= 0; --i) {
        VertexID vertex = order[i];
        TreeNode& node = tree[vertex];
        bool set_to_one = true;

        for (ui j = 0; j < node.fn_count_; ++j) {
            VertexID child = node.fn_[j];
            TreeNode& child_node = tree[child];

            if (child_node.bn_count_ == 1) {
                set_to_one = false;
                Edges& cur_edge = *edge_matrix[vertex][child];
                for (ui k = 0; k < candidates_count[vertex]; ++k) {
                    ui cur_candidates_count = cur_edge.offset_[k + 1] - cur_edge.offset_[k];
                    ui* cur_candidates = cur_edge.edge_ + cur_edge.offset_[k];

                    ui weight = 0;

                    for (ui l = 0; l < cur_candidates_count; ++l) {
                        ui candidates = cur_candidates[l];
                        weight += weight_array[child][candidates];
                    }

                    if (weight < weight_array[vertex][k])
                        weight_array[vertex][k] = weight;
                }
            }
        }

        if (set_to_one) {
            std::fill(weight_array[vertex], weight_array[vertex] + candidates_count[vertex], 1);
        }
    }
}

void generateCECIQueryPlan(const Graph *query_graph, TreeNode *tree, ui *bfs_order, ui *&order,
                                              ui *&pivot) {
    ui query_vertices_num = query_graph->getVerticesCount();
    order = new ui[query_vertices_num];
    pivot = new ui[query_vertices_num];

    for (ui i = 0; i < query_vertices_num; ++i) {
        order[i] = bfs_order[i];
    }

    for (ui i = 1; i < query_vertices_num; ++i) {
        pivot[i] = tree[order[i]].parent_;
    }
}

void generateRIQueryPlan(const Graph *data_graph, const Graph *query_graph, ui *&order, ui *&pivot) {
    ui query_vertices_num = query_graph->getVerticesCount();
    order = new ui[query_vertices_num];
    pivot = new ui[query_vertices_num];

    std::vector<bool> visited(query_vertices_num, false);
    // Select the vertex with the maximum degree as the start vertex.
    order[0] = 0;
    for (ui i = 1; i < query_vertices_num; ++i) {
        if (query_graph->getVertexDegree(i) > query_graph->getVertexDegree(order[0])) {
            order[0] = i;
        }
    }
    visited[order[0]] = true;
    // Order vertices.
    std::vector<ui> tie_vertices;
    std::vector<ui> temp;

    for (ui i = 1; i < query_vertices_num; ++i) {
        // Select the vertices with the maximum number of backward neighbors.
        ui max_bn = 0;
        for (ui u = 0; u < query_vertices_num; ++u) {
            if (!visited[u]) {
                // Compute the number of backward neighbors of u.
                ui cur_bn = 0;
                for (ui j = 0; j < i; ++j) {
                    ui uu = order[j];
                    if (query_graph->checkEdgeExistence(u, uu)) {
                        cur_bn += 1;
                    }
                }

                // Update the vertices under consideration.
                if (cur_bn > max_bn) {
                    max_bn = cur_bn;
                    tie_vertices.clear();
                    tie_vertices.push_back(u);
                } else if (cur_bn == max_bn) {
                    tie_vertices.push_back(u);
                }
            }
        }

        if (tie_vertices.size() != 1) {
            temp.swap(tie_vertices);
            tie_vertices.clear();

            ui count = 0;
            std::vector<ui> u_fn;
            for (auto u : temp) {
                // Compute the number of vertices in the matching order that has at least one vertex not in the matching order && connected with u.

                // Get the neighbors of u that are not in the matching order.
                ui un_count;
                const ui* un = query_graph->getVertexNeighbors(u, un_count);
                for (ui j = 0; j < un_count; ++j) {
                    if (!visited[un[j]]) {
                        u_fn.push_back(un[j]);
                    }
                }

                // Compute the valid number of vertices.
                ui cur_count = 0;
                for (ui j = 0; j < i; ++j) {
                    ui uu = order[j];
                    ui uun_count;
                    const ui* uun = query_graph->getVertexNeighbors(uu, uun_count);
                    ui common_neighbor_count = 0;
                    ComputeSetIntersection::ComputeCandidates(uun, uun_count, u_fn.data(), (ui)u_fn.size(), common_neighbor_count);
                    if (common_neighbor_count > 0) {
                        cur_count += 1;
                    }
                }

                u_fn.clear();

                // Update the vertices under consideration.
                if (cur_count > count) {
                    count = cur_count;
                    tie_vertices.clear();
                    tie_vertices.push_back(u);
                }
                else if (cur_count == count){
                    tie_vertices.push_back(u);
                }
            }
        }

        if (tie_vertices.size() != 1) {
            temp.swap(tie_vertices);
            tie_vertices.clear();

            ui count = 0;
            std::vector<ui> u_fn;
            for (auto u : temp) {
                // Compute the number of vertices not in the matching order && not the neighbor of vertices in the matching order, but is connected with u.

                // Get the neighbors of u that are not in the matching order.
                ui un_count;
                const ui* un = query_graph->getVertexNeighbors(u, un_count);
                for (ui j = 0; j < un_count; ++j) {
                    if (!visited[un[j]]) {
                        u_fn.push_back(un[j]);
                    }
                }

                // Compute the valid number of vertices.
                ui cur_count = 0;
                for (auto uu : u_fn) {
                    bool valid = true;

                    for (ui j = 0; j < i; ++j) {
                        if (query_graph->checkEdgeExistence(uu, order[j])) {
                            valid = false;
                            break;
                        }
                    }

                    if (valid) {
                        cur_count += 1;
                    }
                }

                u_fn.clear();

                // Update the vertices under consideration.
                if (cur_count > count) {
                    count = cur_count;
                    tie_vertices.clear();
                    tie_vertices.push_back(u);
                }
                else if (cur_count == count){
                    tie_vertices.push_back(u);
                }
            }
        }

        order[i] = tie_vertices[0];

        visited[order[i]] = true;
        for (ui j = 0; j < i; ++j) {
            if (query_graph->checkEdgeExistence(order[i], order[j])) {
                pivot[i] = order[j];
                break;
            }
        }

        tie_vertices.clear();
        temp.clear();
    }
}

void generateVF2PPQueryPlan(const Graph *data_graph, const Graph *query_graph, ui *&order, ui *&pivot) {
    ui query_vertices_num = query_graph->getVerticesCount();
    order = new ui[query_vertices_num];
    pivot = new ui[query_vertices_num];

    ui property_count = 0;
    std::vector<std::vector<ui>> properties(query_vertices_num);
    std::vector<bool> order_type(query_vertices_num, true);     // True: Ascending, False: Descending.
    std::vector<ui> vertices;
    std::vector<bool> in_matching_order(query_vertices_num, false);

    for (ui i = 0; i < query_vertices_num; ++i) {
        properties[i].resize(3);
    }

    // Select the root vertex with the rarest node labels and the largest degree.
    property_count = 2;
    order_type[0] = true;
    order_type[1] = false;

    for (ui u = 0; u < query_vertices_num; ++u) {
        vertices.push_back(u);
        properties[u][0] = data_graph->getLabelsFrequency(query_graph->getVertexLabel(u));
        properties[u][1] = query_graph->getVertexDegree(u);
    }

    auto order_lambda = [&properties, &order_type, property_count](ui l, ui r) -> bool {
        for (ui x = 0; x < property_count; ++x) {
            if (properties[l][x] == properties[r][x])
                continue;

            if (order_type[0]) {
                return properties[l][x] < properties[r][x];
            }
            else {
                return properties[l][x] > properties[r][x];
            }
        }

        return l < r;
    };

    std::stable_sort(vertices.begin(), vertices.end(), order_lambda);
    order[0] = vertices[0];
    in_matching_order[order[0]] = true;

    vertices.clear();
    TreeNode* tree;
    ui* bfs_order;
    GraphOperations::bfsTraversal(query_graph, order[0], tree, bfs_order);


    property_count = 3;
    order_type[0] = false;
    order_type[1] = false;
    order_type[2] = true;

    ui level = 1;
    ui count = 1;
    ui num_vertices_in_matching_order = 1;
    while (num_vertices_in_matching_order < query_vertices_num) {
        // Get the vertices in current level.
        while (count < query_vertices_num) {
            ui u = bfs_order[count];
            if (tree[u].level_ == level) {
                vertices.push_back(u);
                count += 1;
            }
            else {
                level += 1;
                break;
            }
        }

        // Process a level in the BFS tree.
        while(!vertices.empty()) {
            // Set property.
            for (auto u : vertices) {
                ui un_count;
                const ui* un = query_graph->getVertexNeighbors(u, un_count);

                ui bn_count = 0;
                for (ui i = 0; i < un_count; ++i) {
                    ui uu = un[i];
                    if (in_matching_order[uu]) {
                        bn_count += 1;
                    }
                }

                properties[u][0] = bn_count;
                properties[u][1] = query_graph->getVertexDegree(u);
                properties[u][2] = data_graph->getLabelsFrequency(query_graph->getVertexLabel(u));
            }


            std::sort(vertices.begin(), vertices.end(), order_lambda);
            pivot[num_vertices_in_matching_order] = tree[vertices[0]].parent_;
            order[num_vertices_in_matching_order++] = vertices[0];

            in_matching_order[vertices[0]] = true;

            vertices.erase(vertices.begin());
        }
    }


    delete[] tree;
    delete[] bfs_order;
}

void checkQueryPlanCorrectness(const Graph *query_graph, ui *order) {
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
        VertexID u = order[i];

        bool valid = false;
        for (ui j = 0; j < i; ++j) {
            VertexID v = order[j];
            if (query_graph->checkEdgeExistence(u, v)) {
                valid = true;
                break;
            }
        }

        assert(valid);
        visited_vertices[u] = true;
    }
}

void generateOrderSpectrum(const Graph *query_graph, std::vector<std::vector<ui>> &spectrum,
                                              ui num_spectrum_limit) {
    ui query_vertices_num = query_graph->getVerticesCount();

    std::vector<bool> visited(query_vertices_num, false);
    std::vector<ui> matching_order;

    // Get the core vertices and the non-core vertices.
    std::vector<ui> core_vertices;
    std::vector<ui> noncore_vertices;
    for (ui u = 0; u < query_vertices_num; ++u) {
        if (query_graph->getCoreValue(u) >= 2) {
            core_vertices.push_back(u);
        }
        else {
            noncore_vertices.push_back(u);
        }
    }

    // Sort the vertices by the core value and the degree value.
    auto compare = [query_graph](ui l, ui r) -> bool {
        if (query_graph->getCoreValue(l) != query_graph->getCoreValue(r)) {
            return query_graph->getCoreValue(l) > query_graph->getCoreValue(r);
        }

        if (query_graph->getVertexDegree(l) != query_graph->getVertexDegree(r)) {
            return query_graph->getVertexDegree(l) > query_graph->getVertexDegree(r);
        }

        return l < r;
    };

    std::sort(core_vertices.begin(), core_vertices.end(), compare);
    std::sort(noncore_vertices.begin(), noncore_vertices.end(), compare);
    // Permutate the vertices. Keep the matching order is connected.

    std::vector<ui> cur_index(query_vertices_num, 0);

    std::vector<ui>& candidates = core_vertices.empty() ? noncore_vertices : core_vertices;

    for (ui i = 0; i < candidates.size(); ++i) {
        ui u = candidates[i];

        matching_order.push_back(u);
        visited[u] = true;

        int cur_depth = 1;
        cur_index[cur_depth] = 0;

        while (true) {
            while (cur_index[cur_depth] < candidates.size()) {
                u = candidates[cur_index[cur_depth]++];

                // If u has been in the matching order, then skip it.
                if (visited[u])
                    continue;

                // Determine whether u is connected to the vertices in the matching order.
                bool valid = false;

                for (auto v : matching_order) {
                    if (query_graph->checkEdgeExistence(u, v)) {
                        valid = true;
                        break;
                    }
                }

                if (!valid)
                    continue;

                // Update.
                matching_order.push_back(u);
                visited[u] = true;

                // All the core vertices are in the matching order.
                if (matching_order.size() == candidates.size()) {
                    spectrum.emplace_back(matching_order);

                    visited[matching_order.back()] = false;
                    matching_order.pop_back();

                    if (spectrum.size() >= num_spectrum_limit) {
                        goto EXIT;
                    }
                    else {
                        break;
                    }
                }
                else {
                    cur_depth += 1;
                    cur_index[cur_depth] = 0;
                }
            }

            visited[matching_order.back()] = false;
            matching_order.pop_back();
            cur_depth -= 1;

            if (cur_depth == 0)
                break;
        }
    }
    EXIT:
    // Order the non-core vertices. All matching orders in the spectrum have the same order of non-core vertices.
    if (!core_vertices.empty()) {
        matching_order.clear();

        for (auto u : core_vertices) {
            visited[u] = true;
        }
        while (matching_order.size() < noncore_vertices.size()) {
            for (auto u : noncore_vertices) {
                if (visited[u])
                    continue;

                bool valid = false;
                for (ui v = 0; v < query_vertices_num; ++v) {
                    if (visited[v] && query_graph->checkEdgeExistence(u, v)) {
                        valid = true;
                        break;
                    }
                }

                if (valid) {
                    matching_order.push_back(u);
                    visited[u] = true;
                    break;
                }
            }
        }

        for (auto &order : spectrum) {
            order.insert(order.end(), matching_order.begin(), matching_order.end());
        }
    }

    // Check the correctness of each matching order.
    for (auto& order : spectrum) {
        checkQueryPlanCorrectness(query_graph, order.data());
    }
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui) + max_candidates_num * sizeof(ui)) * threadnum;
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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					PartialIntersectionMS<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
  					//PartialIntersection<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
ui partialIntersectionMS (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui) + max_candidates_num * sizeof(ui)) * threadnum;
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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					PartialIntersectionMS<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
ui GGERSAL (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
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
//    allocateGPU1D( d_idx ,idx,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_count ,idx_count,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_embedding ,embedding,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_embedding ,idx_embedding,query_vertices_num * sizeof(ui));
    allocateGPU1D( d_order, order, query_vertices_num * sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);
//    allocateGPU1D( d_temp_buffer ,temp_buffer, max_candidates_num * sizeof(ui));
    allocateGPU1D( d_score ,score, 1* sizeof(double));
    allocateGPU1D( d_score_count ,score_count, 1* sizeof(double));
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
  					ggersal<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }
//
//
//
//      EXIT:
//  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
//  //                  visited_vertices,
//  //                  bn, bn_count);
//
//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}



template <const ui blocksize>
ui AL (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
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
//    allocateGPU1D( d_idx ,idx,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_count ,idx_count,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_embedding ,embedding,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_embedding ,idx_embedding,query_vertices_num * sizeof(ui));
    allocateGPU1D( d_order, order, query_vertices_num * sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);
//    allocateGPU1D( d_temp_buffer ,temp_buffer, max_candidates_num * sizeof(ui));
    allocateGPU1D( d_score ,score, 1* sizeof(double));
    allocateGPU1D( d_score_count ,score_count, 1* sizeof(double));
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
ui GGECOAL (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
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
//    allocateGPU1D( d_idx ,idx,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_count ,idx_count,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_embedding ,embedding,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_embedding ,idx_embedding,query_vertices_num * sizeof(ui));
    allocateGPU1D( d_order, order, query_vertices_num * sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);
//    allocateGPU1D( d_temp_buffer ,temp_buffer, max_candidates_num * sizeof(ui));
    allocateGPU1D( d_score ,score, 1* sizeof(double));
    allocateGPU1D( d_score_count ,score_count, 1* sizeof(double));
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
  					ggecoal<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
ui GGECOWJ (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
  					ggecowj<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock,d_denominator);
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
ui GGERSCOPR (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
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
//    allocateGPU1D( d_idx ,idx,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_count ,idx_count,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_embedding ,embedding,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_embedding ,idx_embedding,query_vertices_num * sizeof(ui));
    allocateGPU1D( d_order, order, query_vertices_num * sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);
//    allocateGPU1D( d_temp_buffer ,temp_buffer, max_candidates_num * sizeof(ui));
    allocateGPU1D( d_score ,score, 1* sizeof(double));
    allocateGPU1D( d_score_count ,score_count, 1* sizeof(double));
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
//    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp, fixednum * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
//    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui)) * threadnum;
//    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;


//    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
		record. successrun = false;
		std::cout <<"An error ocurrs when allocate memory!"<<std::endl;
	}else{
		std::cout <<"Pass memory assignment test!"<<std::endl;
	}
	// compute total bytes allocated.
//	ui* d_test;
//	auto fast_alloc_begin = std::chrono::high_resolution_clock::now();
//    cudaMalloc(&d_test,GPU_bytes);
//    auto fast_alloc_end = std::chrono::high_resolution_clock::now();
//    printf("fast alloc memory: %f s", (double)std::chrono::duration_cast<std::chrono::nanoseconds>(fast_alloc_end - fast_alloc_begin).count()/1000000000 );
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
  					ggecorspr<blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
ui GGECOPR (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
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
//    allocateGPU1D( d_idx ,idx,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_count ,idx_count,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_embedding ,embedding,query_vertices_num * sizeof(ui));
//    allocateGPU1D( d_idx_embedding ,idx_embedding,query_vertices_num * sizeof(ui));
    allocateGPU1D( d_order, order, query_vertices_num * sizeof(ui));
    GPU_bytes += query_vertices_num* sizeof(ui);
//    allocateGPU1D( d_temp_buffer ,temp_buffer, max_candidates_num * sizeof(ui));
    allocateGPU1D( d_score ,score, 1* sizeof(double));
    allocateGPU1D( d_score_count ,score_count, 1* sizeof(double));
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
ui blockLayerBalanceWJ (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
	std::cout << "blocksize: "<< blocksize << " numBlocks: "<< numBlocks << " total threads: " << blocksize*numBlocks << " max_candidates_num " << max_candidates_num<<std::endl;

	// for each thread we assign its own global memoory.
    allocateMemoryPerThread(d_idx ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_range ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_count ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui) + max_candidates_num * sizeof(ui)) * threadnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
		std::cout <<"An error ocurrs when allocate memory!"<<std::endl;
	}else{
		std::cout <<"Pass memory assignment test!"<<std::endl;
	}
	// compute total bytes allocated.


	// test candidate

//    std::cout << "pitch : " << valid_candidate_idx_pitch << " ";
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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					BlockLayerBalanceWJ< blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }



//      EXIT:
  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
  //                  visited_vertices,
  //                  bn, bn_count);

//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}

template <const ui blocksize>
ui blockLayerBalancePI (const Graph *data_graph, const Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
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
    /* score length is equal to number of threads*/
//    ui score_length = idx_count[0];
//    score = new double [score_length];
//    memset (score , 0 , score_length* sizeof (double));
    score = new double [1];
    score_count = new ui [1];
    score[0] = 0;;
    score_count[0] = 0;

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
	std::cout << "blocksize: "<< blocksize << " numBlocks: "<< numBlocks << " total threads: " << blocksize*numBlocks << " max_candidates_num " << max_candidates_num<<std::endl;

	// for each thread we assign its own global memoory.
    allocateMemoryPerThread(d_idx ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_embedding ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_range ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_idx_count ,query_vertices_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_intersection ,max_candidates_num * sizeof(ui), threadnum);
    allocateMemoryPerThread(d_temp ,query_vertices_num* fixednum * sizeof(ui), threadnum);
//    allocateMemoryPerThread(d_temp ,query_vertices_num* max_candidates_num * sizeof(ui), threadnum);
    cudaDeviceSynchronize();
    GPU_bytes += (query_vertices_num * sizeof(ui) * 5 + query_vertices_num* fixednum * sizeof(ui) + max_candidates_num * sizeof(ui)) * threadnum;
    std::cout << "total GPU memory: " << GPU_bytes/ 1024 / 1024 << " M" <<std::endl;
    cudaDeviceSynchronize();
    // test cuda err after memory is assigned
    auto err = cudaGetLastError();
	if (err != cudaSuccess){
		std::cout <<"An error ocurrs when allocate memory!"<<std::endl;
	}else{
		std::cout <<"Pass memory assignment test!"<<std::endl;
	}
	// compute total bytes allocated.


	// test candidate

//    std::cout << "pitch : " << valid_candidate_idx_pitch << " ";
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
  				for (ui k = 0; k< round; ++k){
					//one thread one path
  					BlockLayerBalancePI< blocksize><<<numBlocks,blocksize>>>(start_vertex,d_offset_index,d_offsets, d_edge_index, d_edges ,d_order, d_candidates,d_candidates_count, d_bn ,d_bn_count, d_idx_count, d_idx,  d_range,  d_embedding, d_idx_embedding ,d_temp,d_intersection, query_vertices_num, max_candidates_num, threadnum , 0, max_depth - 1,fixednum, d_score, d_score_count,record.taskPerBlock);
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
//              ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
//              VertexID u = order[cur_depth];
//              VertexID v = candidates[u][valid_idx];
//
//              if (visited_vertices[v]) {
//                  idx[cur_depth] += 1;
//
//                  continue;
//              }
//
//              embedding[u] = v;
//              idx_embedding[u] = valid_idx;
//              visited_vertices[v] = true;
//              idx[cur_depth] += 1;
//
//
//              if (cur_depth == max_depth - 1) {
//                  embedding_cnt += 1;
//                  record. real_workload +=1;
//                  visited_vertices[v] = false;
//                  //print a path
////                  for (int i = 0; i<= cur_depth; i++){
////                	  std::cout << "i: " << i<<" index: " <<  valid_candidate_idx[i][idx[i] - 1]<< " range: " <<  idx_count[i] <<std::endl;
////                  }
//
//                  if (embedding_cnt >= output_limit_num) {
//                      goto EXIT;
//                  }
//              } else {
//
//
//                  call_count += 1;
//                  cur_depth += 1;
//
//                  idx[cur_depth] = 0;
//                  generateValidCandidateIndex2(cur_depth, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn,
//                                              bn_count, order, temp_buffer,record.set_intersection_count,record.total_compare);
//
//              	if(idx_count[cur_depth] == 0){
//              		record. real_workload +=1;
//              	}
//              }
//          }
//
//
//          cur_depth -= 1;
//          if (cur_depth < 0)
//              break;
//          else {
//              VertexID u = order[cur_depth];
//
//              visited_vertices[embedding[u]] = false;
//
//
//          }
//      }



//      EXIT:
  //    releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
  //                  visited_vertices,
  //                  bn, bn_count);

//  	auto end = std::chrono::high_resolution_clock::now();
//
//  	record. enumerating_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start).count() - record. sampling_time - record. cand_alloc_time;
//    return embedding_cnt;
          	return 0;
}
