#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "kernel.cuh"
#include "until.cpp"
#include <vector>
#include <time.h>

//user-defined APIs
//  Check if the sample is valid
__device__ bool validate(ui*  d_embedding, ui* d_order, ui valid_candidate_size, ui v, ui depth, ui offset_qn ) {
	// duplicate check & vertice id check
	if(valid_candidate_size == 0 ||  v== INVALID_ID || duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
		return true;
	}
	return false;
}

// sample one vertex from refine array with 50% chance.
__device__ void sample(ui* refine, ui count, ui pos, ui val, ui tid) {
	//select a random vertex from refine array
	if(count < 1 ){
		refine[pos+ count ] = val;
	}else{
		ui random_ui = (getrandomnum(tid))%(2);
		//each vertex has 50% chance to be selected 
		if(random_ui <= 0){
			refine[pos] = val;
		}
	}
}
// compute the reciprocal of sample probability 
__device__ void computeScore(ui* d_range, ui sl, ui el,ui offset_qn, ui& thread_score) {
	//compute score
	double score = 1;
	for (int i =sl ; i <= el; ++i){
		
		if(d_range[i + offset_qn] > 1){
			score *= (double)d_range[i + offset_qn]/1;

		}
	}
	thread_score = score;	
}

//Refine candidate array by set intersection
__device__ void Refine (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){
	// initalize u and its offsets
	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	ui neighbor_count = d_bn_count[depth];
	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];
	// prepare array for intersection
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
	//first array ListToBeIntersected from 0 to first_neighbor_count
	ui valid_candidate_count = first_neighbor_count;
	// fetch candidates from candidate graph
	ui count = 0;
	// for each edge search v in the candidate graph.
	for(ui i =0; i < first_neighbor_count; ++i){
		ui val = ListToBeIntersected[i];
		bool find = true;
		ui intersection_time = 1;
		while(find && (intersection_time < neighbor_count)){
			ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
			ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
			ui* secondListToBeIntersected;
			ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, 		 second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
			find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
			intersection_time ++;
		}
		if (find){
			sample( d_temp, count, query_vertices_num*tid + depth,val,tid );
			count ++;
		}
	}

	// save results to d_temp and d_idx_count
	d_idx_count [offset_qn + depth] = count;
}


