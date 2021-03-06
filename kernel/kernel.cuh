#pragma once
#include <mma.h>
#include "./configuration/types.h"
#include "./cub/cub/cub.cuh"
#include <curand.h>
#include <curand_kernel.h>
#define INVALID_ID 100000000
#define FULL_MASK 0xffffffff

using namespace nvcuda;

//typedef nvcuda::wmma::experimental::precision::b1 b1;

//typedef unsigned char uc;
//
//#define WARP_SIZE 32

// MMA matrix tile dimensions.
//#define M 16
//#define N 16
//#define K 16

__device__ __host__ bool BinarySearchCheck(ui* arr, ui len, ui k) {
  ui left = 0, right = len;
  while (left < right) {
    ui mid = (left + right) / 2;
    bool pred =  arr[mid] < k;
    if (pred)
      left = mid + 1;
    else
      right = mid;
  }
  return arr[left]==k;
}

__device__ __host__ bool SearchCheck(ui* arr, ui len, ui k) {
  ui left = 0, right = len;
  for (ui i = 0; i < right; ++i) {
    if(arr[i]==k){
    	return true;
    }
  }
  return false;
}

__global__ void GPUsetIntersection_count(ui* list, ui* listOffsets, int listnum,ui* results_count ){
	ui tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < listOffsets[1]){
		//each warp handle one element in the first list
		ui k =  list[tid];
		//search in other lists
		bool find = 1;
		for (int i = 1; i < listnum ; ++i){
			ui begin = listOffsets[i];
			ui end =  listOffsets[i+1];
			if (find == 0) {
				break;
			}else{
				find  = BinarySearchCheck(list+ begin,end-begin, k);
			}
		}
		// sync
		results_count[tid] = find;

	}
}
__device__ ui getrandomnum(ui tid){
      return clock64()+tid;
}

__global__ void GPUsetIntersection(ui* list, ui* results_count , ui*results, ui lengthOfFIrstArr){
	ui tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < lengthOfFIrstArr){
		if(tid == 0 ){
			if(results_count[tid] == 0){

			}else{
				results[0] = list[tid];
			}
		}else{
			if (results_count[tid-1] != results_count[tid]){
				results[results_count[tid]-1] = list[tid];
			}
		}
	}

}

__global__ void test_first (ui* dist){
	printf("G:: %d \n",dist[0]);
}

//
__global__ void getCandidate (ui* d_candidates,ui max_candidates_num, ui u, ui tid, ui & v){
	ui index = (max_candidates_num*u) + tid;
	v =  d_candidates[index];
}

// each thread generate one random number
__device__ double generate_random_numbers(ui tid )
{

	curandState state;

	curand_init(clock64(), tid, 0, &state);

	return  curand_uniform(&state);

}

// this method contains a unknown bug!
__device__ bool deviceBinarySearchRec(ui* array, ui x, ui low, ui high) {

	// Repeat until the pointers low and high meet each other
  if (low <= high) {
    ui mid = low + (high - low) / 2;

    if (array[mid] == x)
      return true;

    if (array[mid] > x)
      return deviceBinarySearchRec(array, x, low, mid -1);

    return deviceBinarySearchRec(array, x, mid + 1,high );
  }

  return false;
}

__device__ bool deviceBinarySearch(ui* array, ui x, ui low, ui high) {
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

__device__ bool deviceSearch(ui* array, ui x, ui low, ui high) {

	// Repeat until the pointers low and high meet each other
	ui index = low;
	while (index <= high) {

		if (array[index] == x){
		  return true;
		}
		index++;
	}

	return false;
}

__device__ void binarySearchPerthread(ui* ListToBeIntersected, ui first_neighbor_count, ui* secondListToBeIntersected, ui second_neighbor_count, ui* d_intersection, ui & valid_candidate_count, ui tid, ui offset_cn, ui depth)
{
	ui count = 0;
	for(ui i =0; i < first_neighbor_count; ++i){
		ui val = ListToBeIntersected[i];

		bool find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);

		if (find == true){
			d_intersection[offset_cn + count] = val;
			count += 1;
		}
	}
	valid_candidate_count = count;
}

__device__ void binarySearchPerWarp(ui* ListToBeIntersected, ui first_neighbor_count, ui* secondListToBeIntersected, ui second_neighbor_count, ui* d_intersection, ui & valid_candidate_count, ui tid, ui offset_cn, ui depth)
{
	ui count = 0;
	typedef cub::WarpScan<ui> WarpScan;
	__shared__ typename WarpScan::TempStorage temp_storage;

	ui wid = tid /32;
	ui lid = tid % 32;
	for(ui i =0; i < (first_neighbor_count - 1)/32 + 1; ++i){
		ui index = i*32 + lid;
		ui val = -1;
		if(index < first_neighbor_count){
			val = ListToBeIntersected[index];
		}
		ui prefix_sum = 0;
		ui find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
		WarpScan(temp_storage).ExclusiveSum(find,prefix_sum);
		__syncwarp();
		ui add = __shfl(prefix_sum, 31);
		count += add;
		if (find != 0){
			d_intersection[offset_cn + count + prefix_sum] = val;
		}
	}
	valid_candidate_count = count;
}

__device__ ui PickOneRandomCandidate (ui* d_candidates,ui range ,ui max_candidates_num, ui u, ui tid, ui & v){
	float rand_f = generate_random_numbers (tid);
	ui rand_i = ceilf(rand_f * range ) - 1;
	ui index = (max_candidates_num*u) + rand_i;
	v =  d_candidates[index];
	return rand_i;
}

__device__ ui PickOneRandomCandidate (ui* d_candidates,ui range ,ui max_candidates_num, ui u, ui tid, ui & v, curandState & state){
	ui rand_i =  curand(&state) % (range);
//	printf("rand_i %d, range %d \n", rand_i,range);
	ui index = (max_candidates_num*u) + rand_i;
	v =  d_candidates[index];
	return rand_i;
}


// just sample one candidate one time
__device__ ui PickOneRandomCandidateFromTemp (ui* d_candidates, ui* d_temp, ui temp_size, ui depth, ui query_vertices_num ,ui max_candidates_num , ui u,ui tid, ui & v){

	float rand_f =  generate_random_numbers (tid);

	ui rand_i = ceilf(rand_f * temp_size ) - 1;

	ui index = query_vertices_num*max_candidates_num* tid + depth*max_candidates_num  + rand_i;

	ui valid_idx = d_temp[index];


	v = d_candidates[(max_candidates_num*u) + valid_idx];

	return  d_temp[index];
}

__device__ ui PickOneRandomCandidateFromTemp (ui* d_candidates, ui* d_temp, ui temp_size, ui depth, ui query_vertices_num ,ui max_candidates_num , ui u,ui tid, ui & v, curandState & state){

	ui rand_i =  curand(&state) % temp_size;

	ui index = query_vertices_num*max_candidates_num* tid + depth*max_candidates_num  + rand_i;

	ui valid_idx = d_temp[index];


	v = d_candidates[(max_candidates_num*u) + valid_idx];

	return  d_temp[index];
}

// use swap unformly sample fixednum caniddates
__global__ void random_pick_i_candidate(ui* sample_list, ui length ,ui tid, ui* temp_arr,ui fixednum , ui depth,ui threadsPerBlock){
	for (ui i = 0; i< fixednum; ++i){
		float rand_f = generate_random_numbers (tid);
		ui rand_i = ceilf(rand_f * (length - i) );
		ui index = i + depth*threadsPerBlock*fixednum;
		sample_list[index] = temp_arr [rand_i];
		//swap
		ui temp = temp_arr [length - i];
		temp_arr [length - i]    = temp_arr [rand_i];
		temp_arr [rand_i] = temp;

	}
}

__device__ bool duplicate_test(ui* d_embedding,ui v,ui depth, ui* d_order, ui offset_qn){
	for (ui i = 0; i < depth; i++){
		ui v_star = d_embedding[offset_qn + d_order[i]];
		if(v_star == v){
			return true;
		}
	}
	return false;
}


// compute the arr
__device__ ui getList  (ui* d_offset_index, ui* d_offsets, ui* d_edge_index,ui* d_edges, ui first_neigbor,ui u,ui first_neighbor_embedding_idx, ui query_vertices_num, ui* & ListToBeIntersected){
	ui n_u_idx_offset = d_offset_index [query_vertices_num * first_neigbor + u];
	ui n_u_idx_edge = d_edge_index [query_vertices_num * first_neigbor + u];
	ui begin = d_offsets [n_u_idx_offset + first_neighbor_embedding_idx ];
	ui count = d_offsets [n_u_idx_offset + first_neighbor_embedding_idx +1] - begin;
//	printf("first_neighbor_embedding_idx: %d\n", first_neighbor_embedding_idx);
	ListToBeIntersected = & d_edges [n_u_idx_edge + begin];
	return count;
}

__device__ void generateTemp (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid ){


	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);
	ui intersection_length = 0;
	//copy to d_intersection
	for( ui i = 0 ; i< first_neighbor_count ; ++i){
		d_intersection[i+offset_cn] = ListToBeIntersected[i];
	}
//	if(tid == 0&& depth ==5){
//		printf("tid %d, first_neighbor_count %d  \n", tid, first_neighbor_count);
//		for (int j = 0; j<first_neighbor_count; ++j ){
//			printf("tid %d, firstListToBeIntersected %d  \n", tid, ListToBeIntersected[j]);
//		}
//	}

	ui valid_candidate_count = first_neighbor_count;
	for (ui i = 1; i < neighbor_count; ++i){

		ui second_neighbor = d_bn[query_vertices_num* depth + i];
		ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
		ui* secondListToBeIntersected;
		ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
//if(tid == 0&& depth ==5){
//		printf("tid %d, second_neighbor_count %d  \n", tid, first_neighbor_count);
//		for (int j = 0; j<second_neighbor_count; ++j ){
//			printf("tid %d, secondListToBeIntersected arr %d  \n", tid, secondListToBeIntersected[j]);
//		}
//	}

		binarySearchPerthread(ListToBeIntersected,first_neighbor_count, secondListToBeIntersected,second_neighbor_count, d_intersection, valid_candidate_count, tid, offset_cn,depth);
		//set ListToBeIntersected to be the result of previous run
		ListToBeIntersected = &d_intersection[offset_cn];
		first_neighbor_count = valid_candidate_count;
//		if(tid == 0&& depth ==5){
//			printf("tid %d, i %d  \n", tid, i);
//			for (int j = 0; j<first_neighbor_count; ++j ){
//				printf("tid %d, result arr %d  \n", tid, ListToBeIntersected[j]);
//			}
//		}
	}

	// save to d_temp and d_idx_count
	d_idx_count [offset_qn + depth] = valid_candidate_count;
	for(int i = 0; i< valid_candidate_count; ++i){
		d_temp[offset_qmn + max_candidates_num*depth + i ] = d_intersection[i+offset_cn];
	}

}

__device__ void generateFixedsizeTemp (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){
	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);
	ui intersection_length = 0;
	//copy to d_intersection
	for( ui i = 0 ; i< first_neighbor_count ; ++i){
		d_intersection[i+offset_cn] = ListToBeIntersected[i];
	}
//	if(tid == 0&& depth ==5){
//		printf("tid %d, first_neighbor_count %d  \n", tid, first_neighbor_count);
//		for (int j = 0; j<first_neighbor_count; ++j ){
//			printf("tid %d, firstListToBeIntersected %d  \n", tid, ListToBeIntersected[j]);
//		}
//	}


	ui valid_candidate_count = first_neighbor_count;
	for (ui i = 1; i < neighbor_count; ++i){

		ui second_neighbor = d_bn[query_vertices_num* depth + i];
		ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
		ui* secondListToBeIntersected;
		ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
// if(tid == 0&& depth ==5){
//		printf("tid %d, second_neighbor_count %d  \n", tid, first_neighbor_count);
//		for (int j = 0; j<second_neighbor_count; ++j ){
//			printf("tid %d, secondListToBeIntersected arr %d  \n", tid, secondListToBeIntersected[j]);
//		}
//	}

		binarySearchPerthread(ListToBeIntersected,first_neighbor_count, secondListToBeIntersected,second_neighbor_count, d_intersection, valid_candidate_count, tid, offset_cn,depth);
		//set ListToBeIntersected to be the result of previous run
		ListToBeIntersected = &d_intersection[offset_cn];
		first_neighbor_count = valid_candidate_count;
//		if(tid == 0&& depth ==5){
//			printf("tid %d, i %d  \n", tid, i);
//			for (int j = 0; j<first_neighbor_count; ++j ){
//				printf("tid %d, result arr %d  \n", tid, ListToBeIntersected[j]);
//			}
//		}
	}

	// save to d_temp and d_idx_count
	d_idx_count [offset_qn + depth] = valid_candidate_count;
	for(int i = 0; i< fixednum; ++i){
		float rand_f =  generate_random_numbers (tid);
		ui rand_i = ceilf(rand_f * valid_candidate_count ) - 1;
		d_temp[query_vertices_num*fixednum*tid + fixednum*depth + i ] = d_intersection[rand_i + offset_cn];
//		printf("save %d, index %d \n",  d_intersection[rand_i + offset_cn],query_vertices_num*fixednum*tid + fixednum*depth + i  );
	}

}

__device__ void generateFixedsizeTemp (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum,curandState & state ){
	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);
	ui intersection_length = 0;
	//copy to d_intersection
	for( ui i = 0 ; i< first_neighbor_count ; ++i){
		d_intersection[i+offset_cn] = ListToBeIntersected[i];
	}
//	if(tid == 0&& depth ==5){
//		printf("tid %d, first_neighbor_count %d  \n", tid, first_neighbor_count);
//		for (int j = 0; j<first_neighbor_count; ++j ){
//			printf("tid %d, firstListToBeIntersected %d  \n", tid, ListToBeIntersected[j]);
//		}
//	}


	ui valid_candidate_count = first_neighbor_count;
	for (ui i = 1; i < neighbor_count; ++i){

		ui second_neighbor = d_bn[query_vertices_num* depth + i];
		ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
		ui* secondListToBeIntersected;
		ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
// if(tid == 0&& depth ==5){
//		printf("tid %d, second_neighbor_count %d  \n", tid, first_neighbor_count);
//		for (int j = 0; j<second_neighbor_count; ++j ){
//			printf("tid %d, secondListToBeIntersected arr %d  \n", tid, secondListToBeIntersected[j]);
//		}
//	}

		binarySearchPerthread(ListToBeIntersected,first_neighbor_count, secondListToBeIntersected,second_neighbor_count, d_intersection, valid_candidate_count, tid, offset_cn,depth);
		//set ListToBeIntersected to be the result of previous run
		ListToBeIntersected = &d_intersection[offset_cn];
		first_neighbor_count = valid_candidate_count;
//		if(tid == 0&& depth ==5){
//			printf("tid %d, i %d  \n", tid, i);
//			for (int j = 0; j<first_neighbor_count; ++j ){
//				printf("tid %d, result arr %d  \n", tid, ListToBeIntersected[j]);
//			}
//		}
	}
//	if(tid == 0&& depth ==5){
//		printf("test: %d depth %d,valid_candidate_count %d \n",tid, depth,valid_candidate_count);
//	}
	// save to d_temp and d_idx_count
	d_idx_count [offset_qn + depth] = valid_candidate_count;
	if(false){
		for(int i = 0; i< fixednum; ++i){

			ui rand_i =  curand (&state)% (valid_candidate_count - i);

			d_temp[query_vertices_num*fixednum*tid + fixednum*depth + i ] = d_intersection[rand_i + offset_cn];
			// swap
			ui lastEle = d_intersection[valid_candidate_count - i - 1 + offset_cn];
			d_intersection[valid_candidate_count - i - 1 + offset_cn] = d_intersection[rand_i + offset_cn];
			d_intersection[rand_i + offset_cn] = d_intersection[valid_candidate_count - i - 1 + offset_cn];
	//		printf("save %d, index %d \n",  d_intersection[rand_i + offset_cn],query_vertices_num*fixednum*tid + fixednum*depth + i  );
		}
	}else{
		for(int i = 0; i< fixednum; ++i){
			ui rand_i =  curand (&state)% (valid_candidate_count );
			d_temp[query_vertices_num*fixednum*tid + fixednum*depth + i ] = d_intersection[rand_i + offset_cn];
		}
	}

//	if((tid == 1 || tid == 10 ) && depth ==5){
//		printf("tid %d, range %d  \n", tid, valid_candidate_count);
//	}
}

__device__ void generateFixedsizeTemp_partially (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum, double & score){
	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);
	ui intersection_length = 0;
	//copy to d_intersection
	for( ui i = 0 ; i< first_neighbor_count ; ++i){
		d_intersection[i+offset_cn] = ListToBeIntersected[i];
	}
//	if(tid == 0&& depth ==5){
//		printf("tid %d, first_neighbor_count %d  \n", tid, first_neighbor_count);
//		for (int j = 0; j<first_neighbor_count; ++j ){
//			printf("tid %d, firstListToBeIntersected %d  \n", tid, ListToBeIntersected[j]);
//		}
//	}


	ui valid_candidate_count = first_neighbor_count;
	for (ui i = 1; i < neighbor_count; ++i){

		ui second_neighbor = d_bn[query_vertices_num* depth + i];
		ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
		ui* secondListToBeIntersected;
		ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
// if(tid == 0&& depth ==5){
//		printf("tid %d, second_neighbor_count %d  \n", tid, first_neighbor_count);
//		for (int j = 0; j<second_neighbor_count; ++j ){
//			printf("tid %d, secondListToBeIntersected arr %d  \n", tid, secondListToBeIntersected[j]);
//		}
//	}

		binarySearchPerthread(ListToBeIntersected,first_neighbor_count, secondListToBeIntersected,second_neighbor_count, d_intersection, valid_candidate_count, tid, offset_cn,depth);
		//set ListToBeIntersected to be the result of previous run
		ListToBeIntersected = &d_intersection[offset_cn];
		first_neighbor_count = valid_candidate_count;
//		if(tid == 0&& depth ==5){
//			printf("tid %d, i %d  \n", tid, i);
//			for (int j = 0; j<first_neighbor_count; ++j ){
//				printf("tid %d, result arr %d  \n", tid, ListToBeIntersected[j]);
//			}
//		}
	}

	// save to d_temp and d_idx_count
	d_idx_count [offset_qn + depth] = valid_candidate_count;
	for(int i = 0; i< fixednum; ++i){
		float rand_f =  generate_random_numbers (tid);
		ui rand_i = ceilf(rand_f * valid_candidate_count ) - 1;
		d_temp[query_vertices_num*fixednum*tid + fixednum*depth + i ] = d_intersection[rand_i + offset_cn];
//		printf("save %d, index %d \n",  d_intersection[rand_i + offset_cn],query_vertices_num*fixednum*tid + fixednum*depth + i  );
	}

}

__device__ void generateFixedsizeTempWarp (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){
	ui wid = tid / 32;
	ui u = d_order[depth];
	ui offset_qn = wid* query_vertices_num;
	ui offset_cn = wid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);
	ui intersection_length = 0;
	//copy to d_intersection
	for( ui i = 0 ; i< first_neighbor_count ; ++i){
		d_intersection[i+offset_cn] = ListToBeIntersected[i];
	}
//	if(tid == 0&& depth ==5){
//		printf("tid %d, first_neighbor_count %d  \n", tid, first_neighbor_count);
//		for (int j = 0; j<first_neighbor_count; ++j ){
//			printf("tid %d, firstListToBeIntersected %d  \n", tid, ListToBeIntersected[j]);
//		}
//	}


	ui valid_candidate_count = first_neighbor_count;
	for (ui i = 1; i < neighbor_count; ++i){

		ui second_neighbor = d_bn[query_vertices_num* depth + i];
		ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
		ui* secondListToBeIntersected;
		ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
// if(tid == 0&& depth ==5){
//		printf("tid %d, second_neighbor_count %d  \n", tid, first_neighbor_count);
//		for (int j = 0; j<second_neighbor_count; ++j ){
//			printf("tid %d, secondListToBeIntersected arr %d  \n", tid, secondListToBeIntersected[j]);
//		}
//	}
		// ????????????
//			binarySearchPerthread(ListToBeIntersected,first_neighbor_count, secondListToBeIntersected,second_neighbor_count, d_intersection, valid_candidate_count, wid, offset_cn,depth);
		// ????????????
			binarySearchPerWarp(ListToBeIntersected,first_neighbor_count, secondListToBeIntersected,second_neighbor_count, d_intersection, valid_candidate_count, tid, offset_cn,depth);



		//set ListToBeIntersected to be the result of previous run
		ListToBeIntersected = &d_intersection[offset_cn];
		first_neighbor_count = valid_candidate_count;
//		if(tid == 0&& depth ==5){
//			printf("tid %d, i %d  \n", tid, i);
//			for (int j = 0; j<first_neighbor_count; ++j ){
//				printf("tid %d, result arr %d  \n", tid, ListToBeIntersected[j]);
//			}
//		}
	}
//	if(tid == 0&& depth ==5){
//		printf("test: %d depth %d,valid_candidate_count %d \n",tid, depth,valid_candidate_count);
//	}
	// save to d_temp and d_idx_count
	d_idx_count [offset_qn + depth] = valid_candidate_count;
	for(int i = 0; i< fixednum; ++i){
		float rand_f =  generate_random_numbers (tid);
		ui rand_i = ceilf(rand_f * valid_candidate_count ) - 1;
		rand_i = __shfl(rand_i, 0);
		d_temp[query_vertices_num*fixednum*wid + fixednum*depth + i ] = d_intersection[rand_i + offset_cn];
//		printf("save %d, index %d \n",  d_intersection[rand_i + offset_cn],query_vertices_num*fixednum*tid + fixednum*depth + i  );
	}
//	if((tid == 1 || tid == 10 ) && depth ==5){
//		printf("tid %d, range %d  \n", tid, valid_candidate_count);
//	}
}

// cost less memory without using d_intersection
__device__ void generateFixedsizeTempWarpLessmem (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){
	ui wid = tid / 32;
	ui lid = tid % 32;
	ui u = d_order[depth];
	ui offset_qn = wid* query_vertices_num;
	ui offset_cn = wid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);

	//first array ListToBeIntersected from 0 to first_neighbor_count


	ui valid_candidate_count = first_neighbor_count;

	ui count = 0;


	for(ui i =0; i < (first_neighbor_count - 1)/32 + 1; ++i){
		ui index = i*32 + lid;
		ui val = -1;
		if(index < first_neighbor_count){
			val = ListToBeIntersected[index];
		}
		ui prefix_sum = 0;
		ui find = 1;
		ui intersection_time = 1;
		while(find && intersection_time < neighbor_count){
			intersection_time ++;
			ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
			ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
			ui* secondListToBeIntersected;
			ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
			find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
		}


		if (find != 0){
			for(int j = 0; j< fixednum; ++j){
				d_temp[query_vertices_num*fixednum*wid + fixednum*depth + j ] = val;
			}
		}
	}

//	for (ui i = 1; i < neighbor_count; ++i){
//
//		ui second_neighbor = d_bn[query_vertices_num* depth + i];
//		ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
//		ui* secondListToBeIntersected;
//		ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
//
//		binarySearchPerWarp(ListToBeIntersected,first_neighbor_count, secondListToBeIntersected,second_neighbor_count, d_intersection, valid_candidate_count, tid, offset_cn,depth);
//
//
//
//		//set ListToBeIntersected to be the result of previous run
//		ListToBeIntersected = &d_intersection[offset_cn];
//		first_neighbor_count = valid_candidate_count;
//
//	}
//	if(tid == 0&& depth ==5){
//		printf("test: %d depth %d,valid_candidate_count %d \n",tid, depth,valid_candidate_count);
//	}
	// save to d_temp and d_idx_count
	d_idx_count [offset_qn + depth] = count;
//	for(int i = 0; i< fixednum; ++i){
//		float rand_f =  generate_random_numbers (tid);
//		ui rand_i = ceilf(rand_f * valid_candidate_count ) - 1;
//		rand_i = __shfl(rand_i, 0);
//		d_temp[query_vertices_num*fixednum*wid + fixednum*depth + i ] = d_intersection[rand_i + offset_cn];
////		printf("save %d, index %d \n",  d_intersection[rand_i + offset_cn],query_vertices_num*fixednum*tid + fixednum*depth + i  );
//	}

}

// thtread based version
__device__ void generateFixedsizeTempThreadLessmem (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){

	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);

	//first array ListToBeIntersected from 0 to first_neighbor_count


	ui valid_candidate_count = first_neighbor_count;

	ui count = 0;

	for(ui i =0; i < first_neighbor_count; ++i){

		ui val = ListToBeIntersected[i];
		bool find = true;
		ui intersection_time = 1;
		while(find && (intersection_time < neighbor_count)){

			ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
			ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
			ui* secondListToBeIntersected;
			ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
			find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
			intersection_time ++;

		}


		if (find){

			if(count < fixednum ){
				d_temp[query_vertices_num*fixednum*tid + fixednum*depth + count ] = val;
			}else{
				// reservoir sampling
				ui random_ui = (clock64()+tid)%(count + 1);
				if(random_ui <= fixednum - 1){
					d_temp[query_vertices_num*fixednum*tid + fixednum*depth + random_ui ] = val;
				}

			}
			count ++;
		}
	}
	// if count < fixednum then fill

	// save to d_temp and d_idx_count
	d_idx_count [offset_qn + depth] = count;

}

__device__ void estimateIntersectionCount (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum, double* d_intersection_count){

	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);

	//first array ListToBeIntersected from 0 to first_neighbor_count


	ui valid_candidate_count = first_neighbor_count;

	ui count = 0;
	double intersect_times = 0;
	
	for(ui i =0; i < first_neighbor_count; ++i){

		ui val = ListToBeIntersected[i];
		bool find = true;
		ui intersection_time = 1;
		while(find && (intersection_time < neighbor_count)){

			ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
			ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
			ui* secondListToBeIntersected;
			ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
			find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
			intersection_time ++;

		}
		intersect_times += intersection_time;

		if (find){

			if(count < fixednum ){
				d_temp[query_vertices_num*fixednum*tid + fixednum*depth + count ] = val;
			}else{
				// reservoir sampling
				ui random_ui = (clock64()+tid)%(count + 1);
				if(random_ui <= fixednum - 1){
					d_temp[query_vertices_num*fixednum*tid + fixednum*depth + random_ui ] = val;
				}

			}
			count ++;
		}
	}
	//printf("intersection count %f \n", intersect_times);
	//write estimate to d_intersection_count
	atomicAdd (d_intersection_count , intersect_times);
	// if count < fixednum then fill

	// save to d_temp and d_idx_count
	d_idx_count [offset_qn + depth] = count;

}

// first reservoir sampling than intersection
__device__ void generateFixedsizeTempThreadLessmemV2 (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){

	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);

	//first array ListToBeIntersected from 0 to first_neighbor_count


	ui valid_candidate_count = first_neighbor_count;

	ui count = 0;
	ui random_count = 0;
	// sometimes first_neighbor_count = 0 which
    if(first_neighbor_count > 0 ){
		while ( random_count < fixednum){
			ui random_ui = clock64()% (first_neighbor_count);
			ui val = ListToBeIntersected[random_ui];
			bool find = true;
			ui intersection_time = 1;
			while(find && (intersection_time < neighbor_count)){
				ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
				ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
				ui* secondListToBeIntersected;
				ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
				find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
				intersection_time ++;

			}
			if (find){

				if(count < fixednum ){
					d_temp[query_vertices_num*fixednum*tid + fixednum*depth + count ] = val;
				}
				count ++;

			}
			random_count++;
		}
	}
	// if count < fixednum then fill
	// save to d_temp and d_idx_count
	if(count == 0 ){
		first_neighbor_count = 0;
	}
	d_idx_count [offset_qn + depth] = first_neighbor_count;

}



// first reservoir sampling than intersection + bloom filter?
__device__ void generateFixedsizeTempThreadLessmemV3 (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){

	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);

	//first array ListToBeIntersected from 0 to first_neighbor_count


	ui valid_candidate_count = first_neighbor_count;

	ui count = 0;
	ui length = 0;
	ui random_count = 0;
	unsigned long long mask = 0;
	// compute mask;
	if(neighbor_count > 1){
		ui random_ui = clock64()% (neighbor_count -1) +1;
		ui second_neighbor = d_bn[query_vertices_num* depth + random_ui];
		ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
		ui* secondListToBeIntersected;
		ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);

		for(ui i =0; i < second_neighbor_count; ++i){
			ui val = secondListToBeIntersected[i];
			ui pos = val % 64;
			mask |= 1ULL << pos;

		}
		for(ui i =0; i < first_neighbor_count; ++i){
				ui val = ListToBeIntersected[i];
				ui pos = val % 64;
				bool bitval = mask &(1ULL << pos);
//				printf("~mask: %llu, pos %d \n", mask,pos);
				if(bitval == true){

					length++;
				}
			}

	}else{
		length = valid_candidate_count;
	}
	ui continue_count = 0;
	while ( random_count < fixednum && continue_count < first_neighbor_count){
		continue_count ++;
		ui random_ui = clock64()% (first_neighbor_count);
		ui val = ListToBeIntersected[random_ui];

		ui pos = val % 64;
		bool bitval = mask &(1ULL << pos);
		if(bitval != true && neighbor_count > 1){
			continue;
		}
		bool find = true;
		ui intersection_time = 1;
		while(find && (intersection_time < neighbor_count)){
			ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
			ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
			ui* secondListToBeIntersected;
			ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
			find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
			intersection_time ++;

		}
		if (find){

			if(count < fixednum ){
				d_temp[query_vertices_num*fixednum*tid + fixednum*depth + count ] = val;
			}
			count ++;

		}
		random_count++;

	}
	// if count < fixednum then fill
	// save to d_temp and d_idx_count
	if(count == 0 ){
		length = 0;
	}
	d_idx_count [offset_qn + depth] = length;

}

// estimate length instead
__device__ void generateFixedsizeTempThreadLessmemV4 (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){

	ui u = d_order[depth];
		ui offset_qn = tid* query_vertices_num;
		ui offset_cn = tid* max_candidates_num;
	//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
		ui neighbor_count = d_bn_count[depth];

		ui first_neighbor = d_bn[query_vertices_num* depth];
		ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

		// CSR's offset & list
		ui* ListToBeIntersected;
		ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
	//	printf("count: %d \n", first_neighbor_count);

		//first array ListToBeIntersected from 0 to first_neighbor_count


		ui valid_candidate_count = first_neighbor_count;

		ui count = 0;

		ui length = first_neighbor_count;
		ui sample_time = length*0.1 + 1;
		ui cur_sample_cnt = 0;
		bool if_found = 0;
		if(first_neighbor_count > 0){
			for (ui cur_sample_cnt =0; cur_sample_cnt < sample_time;++cur_sample_cnt ){
				ui random_count = 0;
				while ( random_count < fixednum){
					ui random_ui = (clock64() + tid)% (first_neighbor_count);
					ui val = ListToBeIntersected[random_ui];
					bool find = true;
					ui intersection_time = 1;
					while(find && (intersection_time < neighbor_count)){
						ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
						ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
						ui* secondListToBeIntersected;
						ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
						find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
						intersection_time ++;
					}

					if (find){

						if(count == 0 ){
							d_temp[query_vertices_num*fixednum*tid + fixednum*depth + count ] = val;
						}
						count ++;

					}
					random_count++;
				}
			}
		}
		// if count < fixednum then fill
		// save to d_temp and d_idx_count
		if(count == 0 ){
			first_neighbor_count = 0;
		}else{
			first_neighbor_count = count* first_neighbor_count/sample_time;
		}
		d_idx_count [offset_qn + depth] = first_neighbor_count;

}

__device__ void generateFixedsizeTempPR (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){
	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);
	ui intersection_length = 0;
	//copy to d_intersection
	ui clen = 0.1* first_neighbor_count + 1;
	ui j = 0;
	for( ui i = 0 ; i< first_neighbor_count ; ++i){
		ui random_ui = (clock64() + tid)% (first_neighbor_count);
		if (random_ui <= clen ){
		    d_intersection[j+offset_cn] = ListToBeIntersected[i];
		     ListToBeIntersected[j] = d_intersection[j+offset_cn];
		    j++;
		}
	}
	clen = j;
	// random select

	ui valid_candidate_count = j;
	for (ui i = 1; i < neighbor_count; ++i){

		ui second_neighbor = d_bn[query_vertices_num* depth + i];
		ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
		ui* secondListToBeIntersected;
		ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);


		binarySearchPerthread(ListToBeIntersected,j, secondListToBeIntersected,second_neighbor_count, d_intersection, valid_candidate_count, tid, offset_cn,depth);
		//set ListToBeIntersected to be the result of previous run
		ListToBeIntersected = &d_intersection[offset_cn];
		j = valid_candidate_count;
	}
    first_neighbor_count = clen* j/first_neighbor_count;
	// save to d_temp and d_idx_count
	d_idx_count [offset_qn + depth] = valid_candidate_count;
	for(int i = 0; i< fixednum; ++i){
		float rand_f =  generate_random_numbers (tid);
		ui rand_i = ceilf(rand_f * valid_candidate_count ) - 1;
		d_temp[query_vertices_num*fixednum*tid + fixednum*depth + i ] = d_intersection[rand_i + offset_cn];
//		printf("save %d, index %d \n",  d_intersection[rand_i + offset_cn],query_vertices_num*fixednum*tid + fixednum*depth + i  );
	}

}

__device__ void generateFixedsizeTempThreadV4 (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){

		ui u = d_order[depth];
		ui offset_qn = tid* query_vertices_num;
		ui offset_cn = tid* max_candidates_num;
	//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
		ui neighbor_count = d_bn_count[depth];

		ui first_neighbor = d_bn[query_vertices_num* depth];
		ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

		// CSR's offset & list
		ui* ListToBeIntersected;
		ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
	//	printf("count: %d \n", first_neighbor_count);

		//copy to d_intersection arry.
		for( ui i = 0 ; i< first_neighbor_count ; ++i){
		    d_intersection[i+offset_cn] = ListToBeIntersected[i];
	    }

		ui valid_candidate_count = first_neighbor_count;

		ui count = 0;

		ui length = first_neighbor_count;
		ui sample_time = length + 1;
		ui cur_sample_cnt = 0;

		if(first_neighbor_count > 0){
			for (ui cur_sample_cnt =0; cur_sample_cnt < sample_time;++cur_sample_cnt ){
				ui random_count = 0;
				while ( random_count < fixednum){
					ui random_ui = (clock64() + tid)% (first_neighbor_count);
					ui val = d_intersection[random_ui+offset_cn];
					bool find = true;
					ui intersection_time = 1;
					while(intersection_time < neighbor_count){
						ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
						ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
						ui* secondListToBeIntersected;
						ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
						bool f = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
						if(f == false ){
							find = false;
						}
						intersection_time ++;
					}

					if (find){

						d_temp[query_vertices_num*fixednum*tid + fixednum*depth + count ] = val;
						
						count ++;

					}
					random_count++;
				}
			}
		}
		// if count < fixednum then fill
		// save to d_temp and d_idx_count
		if(count == 0 ){
			first_neighbor_count = 0;
		}else{
			first_neighbor_count = count* first_neighbor_count/sample_time;
		}
		d_idx_count [offset_qn + depth] = first_neighbor_count;

}

__device__ void generateFixedsizeTempThreadLessmemAdapt (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum,double* d_sample_ratio){
	ui u = d_order[depth];
	double s_ratio = d_sample_ratio[0];
//	if(tid == 0)
//	printf("s_ratio %f \n", s_ratio);
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
	//	printf("count: %d \n", first_neighbor_count);

	//first array ListToBeIntersected from 0 to first_neighbor_count


	ui valid_candidate_count = first_neighbor_count;

	ui count = 0;
	ui length = first_neighbor_count;
	ui sample_time = length*s_ratio + 1;

	ui cur_sample_cnt = 0;
	bool if_found = 0;
	if(first_neighbor_count > 0){
		for (ui cur_sample_cnt =0; cur_sample_cnt < sample_time;++cur_sample_cnt ){
			ui random_count = 0;
			while ( random_count < fixednum){
				ui random_ui = (clock64() + tid)% (first_neighbor_count);
				ui val = ListToBeIntersected[random_ui];
				bool find = true;
				ui intersection_time = 1;
				while(find && (intersection_time < neighbor_count)){
					ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
					ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
					ui* secondListToBeIntersected;
					ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
					find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
					intersection_time ++;
				}

				if (find){

					if(count == 0 ){
						d_temp[query_vertices_num*fixednum*tid + fixednum*depth + count ] = val;
					}
					count ++;

				}
				random_count++;
			}
		}
	}
	// if count < fixednum then fill
	// save to d_temp and d_idx_count
	if(count == 0 ){
		first_neighbor_count = 0;
	}else{
		first_neighbor_count = count* first_neighbor_count/sample_time;
	}
	d_idx_count [offset_qn + depth] = first_neighbor_count;

}

__device__ void generateFixedsizeTempThreadLessmemAuto (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum,double* d_sample_ratio){
	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
	//	printf("count: %d \n", first_neighbor_count);

	//first array ListToBeIntersected from 0 to first_neighbor_count


	ui valid_candidate_count = first_neighbor_count;

	ui count = 0;

	ui length = first_neighbor_count;
	ui sample_time = length*d_sample_ratio[depth] + 1;
	ui cur_sample_cnt = 0;
	bool if_found = 0;
	if(first_neighbor_count > 0){
		for (ui cur_sample_cnt =0; cur_sample_cnt < sample_time;++cur_sample_cnt ){
			ui random_count = 0;
			while ( random_count < fixednum){
				ui random_ui = (clock64() + tid)% (first_neighbor_count);
				ui val = ListToBeIntersected[random_ui];
				bool find = true;
				ui intersection_time = 1;
				while(find && (intersection_time < neighbor_count)){
					ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
					ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
					ui* secondListToBeIntersected;
					ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
					find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
					intersection_time ++;
				}

				if (find){

					if(count == 0 ){
						d_temp[query_vertices_num*fixednum*tid + fixednum*depth + count ] = val;
					}
					count ++;

				}
				random_count++;
			}
		}
	}
	// if count < fixednum then fill
	// save to d_temp and d_idx_count
	if(count == 0 ){
		first_neighbor_count = 0;
	}else{
		first_neighbor_count = count* first_neighbor_count/sample_time;
	}
	d_idx_count [offset_qn + depth] = first_neighbor_count;

}

// use bayse update
__device__ void generateFixedsizeTempThreadLessmemV5 (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){

	ui u = d_order[depth];
		ui offset_qn = tid* query_vertices_num;
		ui offset_cn = tid* max_candidates_num;
	//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
		ui neighbor_count = d_bn_count[depth];

		ui first_neighbor = d_bn[query_vertices_num* depth];
		ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

		// CSR's offset & list
		ui* ListToBeIntersected;
		ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
	//	printf("count: %d \n", first_neighbor_count);

		//first array ListToBeIntersected from 0 to first_neighbor_count


		ui valid_candidate_count = first_neighbor_count;

		ui count = 0;

		ui length = first_neighbor_count;

		ui maxtime = 5;
		ui sample_time = length/10 + 1;
		ui cur_sample_cnt = 0;
		bool if_found = 0;
		if(first_neighbor_count > 0){
			for (; cur_sample_cnt < sample_time;++cur_sample_cnt ){
				ui random_count = 0;
				while ( random_count < fixednum){
					ui random_ui = (clock64() + tid)% (first_neighbor_count);
					ui val = ListToBeIntersected[random_ui];
					bool find = true;
					ui intersection_time = 1;
					while(find && (intersection_time < neighbor_count)){
						ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
						ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
						ui* secondListToBeIntersected;
						ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
						find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
						intersection_time ++;
					}

					if (find){

						if(count == 0 ){
							d_temp[query_vertices_num*fixednum*tid + fixednum*depth + count ] = val;
						}
						count ++;

					}
					random_count++;
				}
			}
			// second sampling
			if(count==0 || count== sample_time){
				sample_time += length/10 + 1;
				if_found = 0;
				for (; cur_sample_cnt < sample_time;++cur_sample_cnt ){
					ui random_count = 0;
					while ( random_count < fixednum){
						ui random_ui = (clock64() + tid)% (first_neighbor_count);
						ui val = ListToBeIntersected[random_ui];
						bool find = true;
						ui intersection_time = 1;
						while(find && (intersection_time < neighbor_count)){
							ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
							ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
							ui* secondListToBeIntersected;
							ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
							find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
							intersection_time ++;
						}

						if (find){

							if(count == 0 ){
								d_temp[query_vertices_num*fixednum*tid + fixednum*depth + count ] = val;
							}
							count ++;

						}
						random_count++;
					}
				}
			}
		}


		// if count < fixednum then fill
		// save to d_temp and d_idx_count
		if(count == 0 ){
			first_neighbor_count = 0;
		}else{
			first_neighbor_count = count* first_neighbor_count/sample_time;
		}
		d_idx_count [offset_qn + depth] = first_neighbor_count;

}

// estimate length instead
__device__ void generateFixedsizeTempThreadLessmemV4_res (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){

	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);

	//first array ListToBeIntersected from 0 to first_neighbor_count


	ui valid_candidate_count = first_neighbor_count;

	ui count = 0;

	ui length = first_neighbor_count;
	ui sample_time = length/10 + 1;
	ui cur_sample_cnt = 0;
	bool if_found = 0;
	if(first_neighbor_count > 0){
		for (ui cur_sample_cnt =0; cur_sample_cnt < sample_time;++cur_sample_cnt ){
			ui random_count = 0;
			while ( random_count < fixednum){
				ui random_ui = (clock64() + tid)% (first_neighbor_count);
				ui val = ListToBeIntersected[random_ui];
				bool find = true;
				ui intersection_time = 1;
				while(find && (intersection_time < neighbor_count)){
					ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
					ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
					ui* secondListToBeIntersected;
					ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
					find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
					intersection_time ++;
				}

				if (find){

					if(count == 0 ){
						d_temp[query_vertices_num*fixednum*tid + fixednum*depth + count ] = val;
					}else{
//							 reservoir sampling
							ui random_ui = (clock64()+tid)%(count + 1);
							if(random_ui <= fixednum - 1){
								d_temp[query_vertices_num*fixednum*tid + fixednum*depth + random_ui ] = val;
							}

					}
					count ++;

				}
				random_count++;
			}
		}
	}
	// if count < fixednum then fill
	// save to d_temp and d_idx_count
	if(count == 0 ){
		first_neighbor_count = 0;
	}else{
		first_neighbor_count = count* first_neighbor_count/sample_time;
	}
	d_idx_count [offset_qn + depth] = first_neighbor_count;

}

__device__ void generateFixedsizeTempThreadLessmemV4_test (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){

	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);

	//first array ListToBeIntersected from 0 to first_neighbor_count


	ui valid_candidate_count = first_neighbor_count;

	ui count = 0;

		ui length = first_neighbor_count;
		ui sample_time = length;
		ui cur_sample_cnt = 0;

		if(first_neighbor_count > 0){
			for (ui cur_sample_cnt =0; cur_sample_cnt < sample_time;++cur_sample_cnt ){
				ui random_count = 0;

					ui random_ui = (clock64() + tid)% (first_neighbor_count);
//					printf("tid: %d, random_ui %d \n in %d", tid,random_ui,first_neighbor_count);
//					ui random_ui = cur_sample_cnt;
					ui val = ListToBeIntersected[random_ui];
					bool find = true;
					ui intersection_time = 1;
					while(find && (intersection_time < neighbor_count)){
						ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
						ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
						ui* secondListToBeIntersected;
						ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
						find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
						intersection_time ++;
					}

					if (find){

						if(count == 0 ){
							d_temp[query_vertices_num*fixednum*tid + fixednum*depth + count ] = val;
						}else{
//							 reservoir sampling
							ui random_ui = (clock64()+tid)%(count + 1);
							if(random_ui <= fixednum - 1){
								d_temp[query_vertices_num*fixednum*tid + fixednum*depth + random_ui ] = val;
							}

						}
						count ++;

					}
					random_count++;

			}
		}
		// if count < fixednum then fill
		// save to d_temp and d_idx_count
		if(count == 0 ){
			first_neighbor_count = 0;
		}else{
			first_neighbor_count = count* first_neighbor_count/sample_time;
		}
		d_idx_count [offset_qn + depth] = first_neighbor_count;
}

// thtread based version
__device__ void wanderjoinThreadLessmem (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){

	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);
	//first array ListToBeIntersected from 0 to first_neighbor_count
	ui valid_candidate_count = first_neighbor_count;
	ui random_ui = (clock64()+tid)%(first_neighbor_count + 1);
	d_temp[query_vertices_num*fixednum*tid + fixednum*depth ] = ListToBeIntersected[random_ui];
	// if count < fixednum then fill
	// save to d_temp and d_idx_count
	d_idx_count [offset_qn + depth] = first_neighbor_count;
}

__device__ bool wanderjoinCheck(ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui tid , ui fixednum){
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;

	for (int d = 1; d<=depth; ++d ){
		ui u = d_order[d];
		ui val = d_idx_embedding[offset_qn + u] ;
		ui neighbor_count = d_bn_count[d];
		for(int intersection_time = 1; intersection_time< neighbor_count; ++intersection_time){
			ui second_neighbor = d_bn[query_vertices_num* d + intersection_time];
			ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
			ui* secondListToBeIntersected;
			ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
			auto find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
			if(!find ){
				return false;
			}
		}

	}
	return true;
}

__device__ bool wanderjoinCheckOneNode(ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui tid , ui fixednum){
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;

	ui d = depth;
	ui u = d_order[d];
	ui val = d_idx_embedding[offset_qn + u] ;
	ui neighbor_count = d_bn_count[d];
	for(int intersection_time = 1; intersection_time< neighbor_count; ++intersection_time){
		ui second_neighbor = d_bn[query_vertices_num* d + intersection_time];
		ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
		ui* secondListToBeIntersected;
		ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
		auto find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
		if(!find ){
			return false;
		}
	}

	return true;
}


__device__ void generateFixedsizeTemp_test (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum, ui* d_arr_range_count){
	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);
	ui intersection_length = 0;
	//copy to d_intersection
	for( ui i = 0 ; i< first_neighbor_count ; ++i){
		d_intersection[i+offset_cn] = ListToBeIntersected[i];
	}
//	if(tid == 0&& depth ==5){
//		printf("tid %d, first_neighbor_count %d  \n", tid, first_neighbor_count);
//		for (int j = 0; j<first_neighbor_count; ++j ){
//			printf("tid %d, firstListToBeIntersected %d  \n", tid, ListToBeIntersected[j]);
//		}
//	}

//	if(first_neighbor_count <= 32){
//		atomicAdd (&d_arr_range_count[0] , 1);
//	}else{
//		if(first_neighbor_count <= 128){
//			atomicAdd (&d_arr_range_count[1] , 1);
//		}else{
//			if(first_neighbor_count <= 512){
//				atomicAdd (&d_arr_range_count[2] , 1);
//			}else{
//				if(first_neighbor_count <= 2048){
//					atomicAdd (&d_arr_range_count[3] , 1);
//				}else{
//					atomicAdd (&d_arr_range_count[4] , 1);
//				}
//			}
//		}
//	}


	ui valid_candidate_count = first_neighbor_count;
	for (ui i = 1; i < neighbor_count; ++i){

		ui second_neighbor = d_bn[query_vertices_num* depth + i];
		ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
		ui* secondListToBeIntersected;
		ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
// if(tid == 0&& depth ==5){
//		printf("tid %d, second_neighbor_count %d  \n", tid, first_neighbor_count);
//		for (int j = 0; j<second_neighbor_count; ++j ){
//			printf("tid %d, secondListToBeIntersected arr %d  \n", tid, secondListToBeIntersected[j]);
//		}
//	}

		binarySearchPerthread(ListToBeIntersected,first_neighbor_count, secondListToBeIntersected,second_neighbor_count, d_intersection, valid_candidate_count, tid, offset_cn,depth);
		//set ListToBeIntersected to be the result of previous run
		ListToBeIntersected = &d_intersection[offset_cn];
		first_neighbor_count = valid_candidate_count;
//		if(tid == 0&& depth ==5){
//			printf("tid %d, i %d  \n", tid, i);
//			for (int j = 0; j<first_neighbor_count; ++j ){
//				printf("tid %d, result arr %d  \n", tid, ListToBeIntersected[j]);
//			}
//		}
	}
//	if(tid == 0&& depth ==5){
//		printf("test: %d depth %d,valid_candidate_count %d \n",tid, depth,valid_candidate_count);
//	}
	// save to d_temp and d_idx_count
	d_idx_count [offset_qn + depth] = valid_candidate_count;
	for(int i = 0; i< fixednum; ++i){
		float rand_f =  generate_random_numbers (tid);
		ui rand_i = ceilf(rand_f * valid_candidate_count ) - 1;
		d_temp[query_vertices_num*fixednum*tid + fixednum*depth + i ] = d_intersection[rand_i + offset_cn];
//		printf("save %d, index %d \n",  d_intersection[rand_i + offset_cn],query_vertices_num*fixednum*tid + fixednum*depth + i  );
	}
//	if((tid == 1 || tid == 10 ) && depth ==5){
//		printf("tid %d, range %d  \n", tid, valid_candidate_count);
//	}
}

// one thread one candidate
template < ui threadsPerBlock>
__global__  void samplingByGPUThread(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count){
	ui depth = sl;
	ui u = root;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	if (tid < threadnum){
		// each thread gets a v.
		ui v =0;
		ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
//		//remove
//		valid_idx = 12;
//		v = d_candidates[max_candidates_num*u + valid_idx];
		while (true) {
			ui valid_candidate_size = d_candidates_count[u];
			if(depth != sl){
				valid_candidate_size = d_idx_count[ offset_qn+ depth];
			}
			ui min_size = min (valid_candidate_size,fixednum);
//			printf("$$tid: %d, min_size %d,depth %d \n", tid, min_size,depth);
			while (d_idx[depth + offset_qn] < min_size){
//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){

					valid_idx = PickOneRandomCandidateFromTemp (d_candidates, d_temp, valid_candidate_size, depth, query_vertices_num , max_candidates_num ,u, tid, v);
				}
//				printf("tid: %d, valid_idx %d, depth %d, v %d \n", tid,valid_idx,depth,v);
//				if(depth == 1) {
//					valid_idx = 19;
//				}
//				if(depth == 2) {
//					valid_idx = 25;
//				}
//				if(depth == 3) {
//					valid_idx = 40;
//				}
//				if(depth == 4) {
//					valid_idx = 33;
//				}
//				if(depth == 5) {
//					valid_idx = 46;
//				}
//				if(depth == 6) {
//					valid_idx = 48;
//				}
//				if(depth == 7) {
//					valid_idx = 80;
//				}
				//remove
				v = d_candidates[max_candidates_num*u + valid_idx];


				if( v== INVALID_ID){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd (&d_score_count[0], 1);
//					printf("tid: %d,INVALID_ID%d\n", tid);
					continue;
				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd ( &d_score_count[0], 1);
//					printf("tid: %d,duplicate %d\n", tid);
					continue;
				}

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;
				d_idx[offset_qn + depth] +=1;


				if (depth == el) {
					double score = 1;
					for (int i =sl ; i <= el; ++i){
//						printf("reach end!");
//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
						if(d_range[i + offset_qn] > fixednum){
							score *= (double)d_range[i + offset_qn]/fixednum;

						}
					}

//					printf("thread sscore: %f tid %d \n ", score,tid);

					atomicAdd (d_score, score);

//					atomicAdd (d_score_count, 1);
//					printf("tid : %d, score %f \n ", tid, score);
				}

				if(depth < el){

					depth = depth + 1;
					d_idx[offset_qn + depth] = 0;


					generateTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid );

					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);

					if(valid_candidate_size == 0){
						d_idx[ offset_qn+depth - 1] ++;
//						atomicAdd (d_score_count, 1);
					}
//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
				}
			}
			// backtrack
			depth --;
			u = d_order[depth];
			if(depth <= sl ){
				break ;
			}

		}
	}
}

template < ui threadsPerBlock>
__global__  void samplingByGPUThread_v2(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count){
	ui depth = sl;
	ui u = root;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	if (tid < threadnum){
		// each thread gets a v.
		ui v =0;
		ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
//		//remove
		v = d_candidates[max_candidates_num*u + valid_idx];
		while (true) {
			ui valid_candidate_size = d_candidates_count[u];
			if(depth != sl){
				valid_candidate_size = d_idx_count[ offset_qn+ depth];
			}
			ui min_size = min (valid_candidate_size,fixednum);

			while (d_idx[depth + offset_qn] < min_size){
//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}

//				printf("tid: %d, valid_idx %d, depth %d, v %d \n", tid,valid_idx,depth,v);
//				if(depth == 1) {
//					valid_idx = 19;
//				}
//				if(depth == 2) {
//					valid_idx = 25;
//				}
//				if(depth == 3) {
//					valid_idx = 40;
//				}
//				if(depth == 4) {
//					valid_idx = 33;
//				}
//				if(depth == 5) {
//					valid_idx = 46;
//				}
//				if(depth == 6) {
//					valid_idx = 48;
//				}
//				if(depth == 7) {
//					valid_idx = 80;
//				}
				//remove
				v = d_candidates[max_candidates_num*u + valid_idx];


				if( v== INVALID_ID){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd (&d_score_count[0], 1);
//					printf("tid: %d,INVALID_ID%d\n", tid);
					continue;
				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd ( &d_score_count[0], 1);
//					printf("tid: %d,duplicate %d\n", tid);
					continue;
				}

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;
				d_idx[offset_qn + depth] +=1;


				if (depth == el) {
					double score = 1;
					for (int i =sl ; i <= el; ++i){
//						printf("reach end!");
//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
						if(d_range[i + offset_qn] > fixednum){
							score *= (double)d_range[i + offset_qn]/fixednum;

						}
					}

//					printf("thread sscore: %f tid %d \n ", score,tid);

					atomicAdd (d_score, score);

//					atomicAdd (d_score_count, 1);
//					printf("tid : %d, score %f \n ", tid, score);
				}

				if(depth < el){

					depth = depth + 1;
					d_idx[offset_qn + depth] = 0;


					generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);

					if(valid_candidate_size == 0){
						d_idx[ offset_qn+depth - 1] ++;
//						atomicAdd (d_score_count, 1);
					}
//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
				}
			}
			// backtrack
			depth --;
			u = d_order[depth];
			if(depth <= sl ){
				break ;
			}

		}
	}
}

template < ui threadsPerBlock>
__global__  void samplingByGPUWarp(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count){
	ui depth = sl;
	ui u = root;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui wid = tid / 32;
	ui offset_qn = wid* query_vertices_num;
	ui offset_cn = wid* max_candidates_num;
	if (wid < threadnum){
		// each thread gets a v.
		ui v =0;
		ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, wid, v);
		valid_idx = __shfl(valid_idx, 0);
		v = __shfl(v, 0);

		while (true) {
			ui valid_candidate_size = d_candidates_count[u];
			if(depth != sl){
				valid_candidate_size = d_idx_count[ offset_qn+ depth];
			}
			ui min_size = min (valid_candidate_size,fixednum);

			while (d_idx[depth + offset_qn] < min_size){
//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* wid  + depth* fixednum + d_idx[depth + offset_qn]];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}

//				printf("tid: %d, valid_idx %d, depth %d, v %d \n", tid,valid_idx,depth,v);
//				if(depth == 1) {
//					valid_idx = 19;
//				}
//				if(depth == 2) {
//					valid_idx = 25;
//				}
//				if(depth == 3) {
//					valid_idx = 40;
//				}
//				if(depth == 4) {
//					valid_idx = 33;
//				}
//				if(depth == 5) {
//					valid_idx = 46;
//				}
//				if(depth == 6) {
//					valid_idx = 48;
//				}
//				if(depth == 7) {
//					valid_idx = 80;
//				}
				//remove
				v = d_candidates[max_candidates_num*u + valid_idx];


				if( v== INVALID_ID){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd (&d_score_count[0], 1);
//					printf("tid: %d,INVALID_ID%d\n", tid);
					continue;
				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd ( &d_score_count[0], 1);
//					printf("tid: %d,duplicate %d\n", tid);
					continue;
				}

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;
				d_idx[offset_qn + depth] +=1;


				if (depth == el) {
					double score = 1;
					for (int i =sl ; i <= el; ++i){
//						printf("reach end!");
//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
						if(d_range[i + offset_qn] > fixednum){
							score *= (double)d_range[i + offset_qn]/fixednum;

						}
					}

//					printf("thread sscore: %f tid %d \n ", score,tid);
					if(tid %32 == 0){
//						printf("score %f",score);
						atomicAdd (d_score, score);
					}


//					atomicAdd (d_score_count, 1);
//					printf("tid : %d, score %f \n ", tid, score);
				}

				if(depth < el){

					depth = depth + 1;
					d_idx[offset_qn + depth] = 0;


					generateFixedsizeTempWarp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);

					if(valid_candidate_size == 0){
						d_idx[ offset_qn+depth - 1] ++;
//						atomicAdd (d_score_count, 1);
					}
//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
				}
			}
			// backtrack
			depth --;
			u = d_order[depth];
			if(depth <= sl ){
				break ;
			}

		}
	}
}

// cost less memory
template < ui threadsPerBlock>
__global__  void samplingByGPUWarpLessmem(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection, ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count){
	ui depth = sl;
	ui u = root;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui wid = tid / 32;
	ui offset_qn = wid* query_vertices_num;
	ui offset_cn = wid* max_candidates_num;
	if (wid < threadnum){
		// each thread gets a v.
		ui v =0;
		ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, wid, v);
		valid_idx = __shfl(valid_idx, 0);
		v = __shfl(v, 0);

		while (true) {
			ui valid_candidate_size = d_candidates_count[u];
			if(depth != sl){
				valid_candidate_size = d_idx_count[ offset_qn+ depth];
			}
			ui min_size = min (valid_candidate_size,fixednum);

			while (d_idx[depth + offset_qn] < min_size){
//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* wid  + depth* fixednum + d_idx[depth + offset_qn]];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}

//				printf("tid: %d, valid_idx %d, depth %d, v %d \n", tid,valid_idx,depth,v);
//				if(depth == 1) {
//					valid_idx = 19;
//				}
//				if(depth == 2) {
//					valid_idx = 25;
//				}
//				if(depth == 3) {
//					valid_idx = 40;
//				}
//				if(depth == 4) {
//					valid_idx = 33;
//				}
//				if(depth == 5) {
//					valid_idx = 46;
//				}
//				if(depth == 6) {
//					valid_idx = 48;
//				}
//				if(depth == 7) {
//					valid_idx = 80;
//				}
				//remove
				v = d_candidates[max_candidates_num*u + valid_idx];


				if( v== INVALID_ID){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd (&d_score_count[0], 1);
//					printf("tid: %d,INVALID_ID%d\n", tid);
					continue;
				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd ( &d_score_count[0], 1);
//					printf("tid: %d,duplicate %d\n", tid);
					continue;
				}

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;
				d_idx[offset_qn + depth] +=1;


				if (depth == el) {
					double score = 1;
					for (int i =sl ; i <= el; ++i){
//						printf("reach end!");
//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
						if(d_range[i + offset_qn] > fixednum){
							score *= (double)d_range[i + offset_qn]/fixednum;

						}
					}

//					printf("thread sscore: %f tid %d \n ", score,tid);
					if(tid %32 == 0){
//						printf("score %f",score);
						atomicAdd (d_score, score);
					}


//					atomicAdd (d_score_count, 1);
//					printf("tid : %d, score %f \n ", tid, score);
				}

				if(depth < el){

					depth = depth + 1;
					d_idx[offset_qn + depth] = 0;


					generateFixedsizeTempWarpLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);

					if(valid_candidate_size == 0){
						d_idx[ offset_qn+depth - 1] ++;
//						atomicAdd (d_score_count, 1);
					}
//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
				}
			}
			// backtrack
			depth --;
			u = d_order[depth];
			if(depth <= sl ){
				break ;
			}

		}
	}
}


template < ui threadsPerBlock>
__global__  void BlockPathBalance(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){
	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
//	curandState state;
//	curand_init(clock64(), tid, 0, &state);

	while(s < taskPerBlock){
		ui depth = sl;
		ui u = root;
		for (int d = sl ; d < el; ++d  ){
			d_idx[d + offset_qn] = 0;
		}
		if (tid < threadnum){
			atomicAdd (&s, 1);
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
			// copy state is a little slower
//			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v, state);
	//		//remove
			v = d_candidates[max_candidates_num*u + valid_idx];
			while (true) {
				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);

				while (d_idx[depth + offset_qn] < min_size){
	//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
					u = d_order[depth];
					d_range[depth + offset_qn]  = valid_candidate_size;

					// if depth is not beginning depth.
					if(depth != sl){
						valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

						v = d_candidates[max_candidates_num*u + valid_idx];
					}



					if( v== INVALID_ID){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd (&d_score_count[0], 1);
	//					printf("tid: %d,INVALID_ID%d\n", tid);
						continue;
					}

					if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd ( &d_score_count[0], 1);
	//					printf("tid: %d,duplicate %d\n", tid);
						continue;
					}

					d_embedding[offset_qn + u] = v;
					d_idx_embedding[offset_qn + u] = valid_idx;
					d_idx[offset_qn + depth] +=1;


					if (depth == el) {
						double score = 1;
						for (int i =sl ; i <= el; ++i){
	//						printf("reach end!");
	//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}

	//					printf("thread sscore: %f tid %d \n ", score,tid);

//						atomicAdd (d_score, score);
						thread_score += score;

					}

					if(depth < el){

						depth = depth + 1;
						d_idx[offset_qn + depth] = 0;

//						generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
						generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

						valid_candidate_size = d_idx_count[ offset_qn+ depth];

						min_size  = min (valid_candidate_size,fixednum);

						if(valid_candidate_size == 0){
							d_idx[ offset_qn+depth - 1] ++;
	//						atomicAdd (d_score_count, 1);
						}
	//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
	//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
					}
				}
				// backtrack
				depth --;
				u = d_order[depth];
				if(depth <= sl ){
					break ;
				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void PartialIntersection(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){
	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
//	curandState state;
//	curand_init(clock64(), tid, 0, &state);

	while(s < taskPerBlock){
		ui depth = sl;
		ui u = root;
		for (int d = sl ; d < el; ++d  ){
			d_idx[d + offset_qn] = 0;
		}
		if (tid < threadnum){
			atomicAdd (&s, 1);
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
			// copy state is a little slower
//			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v, state);
	//		//remove
			v = d_candidates[max_candidates_num*u + valid_idx];
			while (true) {
				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);

				while (d_idx[depth + offset_qn] < min_size){
	//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
					u = d_order[depth];
					d_range[depth + offset_qn]  = valid_candidate_size;

					// if depth is not beginning depth.
					if(depth != sl){
						valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

						v = d_candidates[max_candidates_num*u + valid_idx];
					}



					if( v== INVALID_ID){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd (&d_score_count[0], 1);
	//					printf("tid: %d,INVALID_ID%d\n", tid);
						continue;
					}

					if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd ( &d_score_count[0], 1);
	//					printf("tid: %d,duplicate %d\n", tid);
						continue;
					}

					d_embedding[offset_qn + u] = v;
					d_idx_embedding[offset_qn + u] = valid_idx;
					d_idx[offset_qn + depth] +=1;


					if (depth == el) {
						double score = 1;
						for (int i =sl ; i <= el; ++i){
	//						printf("reach end!");
	//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}

	//					printf("thread sscore: %f tid %d \n ", score,tid);

//						atomicAdd (d_score, score);
						thread_score += score;

					}

					if(depth < el){

						depth = depth + 1;
						d_idx[offset_qn + depth] = 0;

//						generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
						generateFixedsizeTempThreadV4( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

						valid_candidate_size = d_idx_count[ offset_qn+ depth];

						min_size  = min (valid_candidate_size,fixednum);

						if(valid_candidate_size == 0){
							d_idx[ offset_qn+depth - 1] ++;
	//						atomicAdd (d_score_count, 1);
						}
	//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
	//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
					}
				}
				// backtrack
				depth --;
				u = d_order[depth];
				if(depth <= sl ){
					break ;
				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void PartialIntersectionMS(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){
	__shared__ unsigned int s;
	double thread_score = 0.0; 
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
//	curandState state;
//	curand_init(clock64(), tid, 0, &state);

	while(s < taskPerBlock){
		ui depth = sl;
		ui u = root;
		for (int d = sl ; d < el; ++d  ){
			d_idx[d + offset_qn] = 0;
		}
		if (tid < threadnum){
			atomicAdd (&s, 1);
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
			// copy state is a little slower
//			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v, state);
	//		//remove
			v = d_candidates[max_candidates_num*u + valid_idx];
			while (true) {
				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);

				while (d_idx[depth + offset_qn] < min_size){
	//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
					u = d_order[depth];
					d_range[depth + offset_qn]  = valid_candidate_size;

					// if depth is not beginning depth.
					if(depth != sl){
						valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

						v = d_candidates[max_candidates_num*u + valid_idx];
					}



					if( v== INVALID_ID){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd (&d_score_count[0], 1);
	//					printf("tid: %d,INVALID_ID%d\n", tid);
						continue;
					}

					if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd ( &d_score_count[0], 1);
	//					printf("tid: %d,duplicate %d\n", tid);
						continue;
					}

					d_embedding[offset_qn + u] = v;
					d_idx_embedding[offset_qn + u] = valid_idx;
					d_idx[offset_qn + depth] +=1;


					if (depth == el) {
						double score = 1;
						for (int i =sl ; i <= el; ++i){
	//						printf("reach end!");
	//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}

	//					printf("thread sscore: %f tid %d \n ", score,tid);

//						atomicAdd (d_score, score);
						thread_score += score;

					}

					if(depth < el){

						depth = depth + 1;
						d_idx[offset_qn + depth] = 0;
                        //generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
	//					generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
						generateFixedsizeTempPR( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

						valid_candidate_size = d_idx_count[ offset_qn+ depth];

						min_size  = min (valid_candidate_size,fixednum);

						if(valid_candidate_size == 0){
							d_idx[ offset_qn+depth - 1] ++;
	//						atomicAdd (d_score_count, 1);
						}
	//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
	//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
					}
				}
				// backtrack
				depth --;
				u = d_order[depth];
				if(depth <= sl ){
					break ;
				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void BlockPathBalanceLessmem(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){

	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;

	while(s < taskPerBlock){
		ui depth = sl;
		ui u = root;
		for (int d = sl ; d < el; ++d  ){
			d_idx[d + offset_qn] = 0;
		}
		if (tid < threadnum){
			atomicAdd (&s, 1);
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
	//		//remove
//			v = d_candidates[max_candidates_num*u + valid_idx];
			while (true) {
				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);

				while (d_idx[depth + offset_qn] < min_size){
	//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
					u = d_order[depth];
					d_range[depth + offset_qn]  = valid_candidate_size;

					// if depth is not beginning depth.
					if(depth != sl){
						valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

						v = d_candidates[max_candidates_num*u + valid_idx];
					}



					if( v== INVALID_ID){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd (&d_score_count[0], 1);
	//					printf("tid: %d,INVALID_ID%d\n", tid);
						continue;
					}

					if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd ( &d_score_count[0], 1);
	//					printf("tid: %d,duplicate %d\n", tid);
						continue;
					}

					d_embedding[offset_qn + u] = v;
					d_idx_embedding[offset_qn + u] = valid_idx;
					d_idx[offset_qn + depth] +=1;


					if (depth == el) {
						double score = 1;
						for (int i =sl ; i <= el; ++i){
	//						printf("reach end!");
	//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}

	//					printf("thread sscore: %f tid %d \n ", score,tid);

//						atomicAdd (d_score, score);
						thread_score += score;

					}

					if(depth < el){

						depth = depth + 1;
						d_idx[offset_qn + depth] = 0;

						generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//						generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

						valid_candidate_size = d_idx_count[ offset_qn+ depth];

						min_size  = min (valid_candidate_size,fixednum);

						if(valid_candidate_size == 0){
							d_idx[ offset_qn+depth - 1] ++;
	//						atomicAdd (d_score_count, 1);
						}
	//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
	//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
					}
				}
				// backtrack
				depth --;
				u = d_order[depth];
				if(depth <= sl ){
					break ;
				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void BlockPathBalanceLessmemV2(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){
	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;

	while(s < taskPerBlock){
		ui depth = sl;
		ui u = root;
		for (int d = sl ; d < el; ++d  ){
			d_idx[d + offset_qn] = 0;
		}
		if (tid < threadnum){
			atomicAdd (&s, 1);
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
	//		//remove
			v = d_candidates[max_candidates_num*u + valid_idx];
			while (true) {
				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);

				while (d_idx[depth + offset_qn] < min_size){
	//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
					u = d_order[depth];
					d_range[depth + offset_qn]  = valid_candidate_size;

					// if depth is not beginning depth.
					if(depth != sl){
						valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

						v = d_candidates[max_candidates_num*u + valid_idx];
					}



					if( v== INVALID_ID){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd (&d_score_count[0], 1);
	//					printf("tid: %d,INVALID_ID%d\n", tid);
						continue;
					}

					if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd ( &d_score_count[0], 1);
	//					printf("tid: %d,duplicate %d\n", tid);
						continue;
					}

					d_embedding[offset_qn + u] = v;
					d_idx_embedding[offset_qn + u] = valid_idx;
					d_idx[offset_qn + depth] +=1;


					if (depth == el) {
						double score = 1;
						for (int i =sl ; i <= el; ++i){
	//						printf("reach end!");
	//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}

	//					printf("thread sscore: %f tid %d \n ", score,tid);

//						atomicAdd (d_score, score);
						thread_score += score;

					}

					if(depth < el){

						depth = depth + 1;
						d_idx[offset_qn + depth] = 0;

						generateFixedsizeTempThreadLessmemV2( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//						generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

						valid_candidate_size = d_idx_count[ offset_qn+ depth];

						min_size  = min (valid_candidate_size,fixednum);

						if(valid_candidate_size == 0){
							d_idx[ offset_qn+depth - 1] ++;
	//						atomicAdd (d_score_count, 1);
						}
	//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
	//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
					}
				}
				// backtrack
				depth --;
				u = d_order[depth];
				if(depth <= sl ){
					break ;
				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void BlockPathBalanceLessmemV3(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){
	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;

	while(s < taskPerBlock){
		ui depth = sl;
		ui u = root;
		for (int d = sl ; d < el; ++d  ){
			d_idx[d + offset_qn] = 0;
		}
		if (tid < threadnum){
			atomicAdd (&s, 1);
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
	//		//remove
			v = d_candidates[max_candidates_num*u + valid_idx];
			while (true) {
				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);

				while (d_idx[depth + offset_qn] < min_size){
	//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
					u = d_order[depth];
					d_range[depth + offset_qn]  = valid_candidate_size;

					// if depth is not beginning depth.
					if(depth != sl){
						valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

						v = d_candidates[max_candidates_num*u + valid_idx];
					}



					if( v== INVALID_ID){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd (&d_score_count[0], 1);
	//					printf("tid: %d,INVALID_ID%d\n", tid);
						continue;
					}

					if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd ( &d_score_count[0], 1);
	//					printf("tid: %d,duplicate %d\n", tid);
						continue;
					}

					d_embedding[offset_qn + u] = v;
					d_idx_embedding[offset_qn + u] = valid_idx;
					d_idx[offset_qn + depth] +=1;


					if (depth == el) {
						double score = 1;
						for (int i =sl ; i <= el; ++i){
	//						printf("reach end!");
	//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}

	//					printf("thread sscore: %f tid %d \n ", score,tid);

//						atomicAdd (d_score, score);
						thread_score += score;

					}

					if(depth < el){

						depth = depth + 1;
						d_idx[offset_qn + depth] = 0;

						generateFixedsizeTempThreadLessmemV3( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//						generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

						valid_candidate_size = d_idx_count[ offset_qn+ depth];

						min_size  = min (valid_candidate_size,fixednum);

						if(valid_candidate_size == 0){
							d_idx[ offset_qn+depth - 1] ++;
	//						atomicAdd (d_score_count, 1);
						}
	//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
	//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
					}
				}
				// backtrack
				depth --;
				u = d_order[depth];
				if(depth <= sl ){
					break ;
				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void BlockPathBalanceLessmemV4(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){
	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;

	while(s < taskPerBlock){
		ui depth = sl;
		ui u = root;
		for (int d = sl ; d < el; ++d  ){
			d_idx[d + offset_qn] = 0;
		}
		if (tid < threadnum){
			atomicAdd (&s, 1);
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
	//		//remove
			v = d_candidates[max_candidates_num*u + valid_idx];
			while (true) {
				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);

				while (d_idx[depth + offset_qn] < min_size){
	//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
					u = d_order[depth];
					d_range[depth + offset_qn]  = valid_candidate_size;

					// if depth is not beginning depth.
					if(depth != sl){
						valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

						v = d_candidates[max_candidates_num*u + valid_idx];
					}



					if( v== INVALID_ID){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd (&d_score_count[0], 1);
	//					printf("tid: %d,INVALID_ID%d\n", tid);
						continue;
					}

					if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd ( &d_score_count[0], 1);
	//					printf("tid: %d,duplicate %d\n", tid);
						continue;
					}

					d_embedding[offset_qn + u] = v;
					d_idx_embedding[offset_qn + u] = valid_idx;
					d_idx[offset_qn + depth] +=1;


					if (depth == el) {
						double score = 1;
						for (int i =sl ; i <= el; ++i){
	//						printf("reach end!");
	//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}

	//					printf("thread sscore: %f tid %d \n ", score,tid);

//						atomicAdd (d_score, score);
						thread_score += score;

					}

					if(depth < el){

						depth = depth + 1;
						d_idx[offset_qn + depth] = 0;

						generateFixedsizeTempThreadLessmemV4( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//						generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

						valid_candidate_size = d_idx_count[ offset_qn+ depth];

						min_size  = min (valid_candidate_size,fixednum);

						if(valid_candidate_size == 0){
							d_idx[ offset_qn+depth - 1] ++;
	//						atomicAdd (d_score_count, 1);
						}
	//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
	//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
					}
				}
				// backtrack
				depth --;
				u = d_order[depth];
				if(depth <= sl ){
					break ;
				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void BlockLayerBalance(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){
	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
	ui depth = sl;
	ui u = root;

	if (tid < threadnum){
		atomicAdd (&s, 1);
		// each thread gets a v.
		ui v =0;
		ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
//		//remove
//		v = d_candidates[max_candidates_num*u + valid_idx];
		while (true) {
			ui valid_candidate_size = d_candidates_count[u];
			if(depth != sl){
				valid_candidate_size = d_idx_count[ offset_qn+ depth];
			}
			ui min_size = min (valid_candidate_size,fixednum);

			while (d_idx[depth + offset_qn] < min_size){
//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}


				//remove
				v = d_candidates[max_candidates_num*u + valid_idx];


				if( v== INVALID_ID){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd (&d_score_count[0], 1);
//					printf("tid: %d,INVALID_ID%d\n", tid);
					continue;
				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd ( &d_score_count[0], 1);
//					printf("tid: %d,duplicate %d\n", tid);
					continue;
				}

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;
				d_idx[offset_qn + depth] +=1;


				if (depth == el) {
					double score = 1;
					for (int i =sl ; i <= el; ++i){
//						printf("reach end!");
//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
						if(d_range[i + offset_qn] > fixednum){
							score *= (double)d_range[i + offset_qn]/fixednum;

						}
					}

//					printf("thread sscore: %f tid %d \n ", score,tid);

//					atomicAdd (d_score, score);
					thread_score += score;

				}

				if(depth < el){

					depth = depth + 1;
					d_idx[offset_qn + depth] = 0;


					generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);

					if(valid_candidate_size == 0){
						d_idx[ offset_qn+depth - 1] ++;
//						atomicAdd (d_score_count, 1);
					}
//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
				}
			}
			// backtrack
			depth --;
			u = d_order[depth];
			if(depth <= sl ){
				// sample end
				atomicAdd (&s, 1);
				if(s >= taskPerBlock){
					break ;
				}else{
					depth = sl;
					u = root;
					valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
					for (int d = sl ; d < el; ++d  ){
						d_idx[d + offset_qn] = 0;
					}
				}

			}

		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }

}

template < ui threadsPerBlock>
__global__  void BlockLayerBalanceWJ(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){
	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
	ui depth = sl;
	ui u = root;

	if (tid < threadnum){
		atomicAdd (&s, 1);
		// each thread gets a v.
		ui v =0;
		ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
//		//remove
//		v = d_candidates[max_candidates_num*u + valid_idx];
		while (true) {
			ui valid_candidate_size = d_candidates_count[u];
			if(depth != sl){
				valid_candidate_size = d_idx_count[ offset_qn+ depth];
			}
			ui min_size = min (valid_candidate_size,fixednum);

			while (d_idx[depth + offset_qn] < min_size){
//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}


				//remove
				v = d_candidates[max_candidates_num*u + valid_idx];


				if( v== INVALID_ID){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd (&d_score_count[0], 1);
//					printf("tid: %d,INVALID_ID%d\n", tid);
					continue;
				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd ( &d_score_count[0], 1);
//					printf("tid: %d,duplicate %d\n", tid);
					continue;
				}

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;
				d_idx[offset_qn + depth] +=1;

				if (depth == el) {
					// check whether this path is vaild
					bool valid_path = wanderjoinCheckOneNode ( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,tid , fixednum);
					//compute score
					if(!valid_path){
						break;
	   				}
					double score = 1;
					for (int i =sl ; i <= el; ++i){
					//	printf("reach end!");
						//printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
						if(d_range[i + offset_qn] > fixednum){
							score *= (double)d_range[i + offset_qn]/fixednum;
							

						}
					}
					thread_score += score;
					continue;
				}

				if(depth < el){

					depth = depth + 1;
					d_idx[offset_qn + depth] = 0;

					wanderjoinThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);

					if(valid_candidate_size == 0){
						d_idx[ offset_qn+depth - 1] ++;
//						atomicAdd (d_score_count, 1);
					}
//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
				}
			}
			// backtrack
			depth --;
			u = d_order[depth];
			if(depth <= sl ){
				// sample end
				atomicAdd (&s, 1);
				if(s >= taskPerBlock){
					break ;
				}else{
					depth = sl;
					u = root;
					valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
					for (int d = sl ; d < el; ++d  ){
						d_idx[d + offset_qn] = 0;
					}
				}

			}

		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }

}




template < ui threadsPerBlock>
__global__  void WarpPathBalance(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerWarp){
	//max block size <= 512
	extern __shared__ unsigned int s_int[];
	int * s = (int* ) s_int;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui wid = threadIdx.x/32;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;

	if(threadIdx.x < threadsPerBlock/32){
		s[threadIdx.x] = 0;
	}

//	for(int i =0; i< threadsPerBlock/32 ;++i){
//		s[i] = 0;
//	}
	if (tid < threadnum){
		while(s[wid] < taskPerWarp){
			// if there are remaining tasks continue.
			ui depth = sl;
			ui u = root;
			for (int d = sl ; d < el; ++d  ){
				d_idx[d + offset_qn] = 0;
			}

			ui old = atomicAdd (&s[wid], 1);
            // if < taskPerWarp break

			if(old +1 > taskPerWarp){
				continue;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			while (true) {
				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);

				while (d_idx[depth + offset_qn] < min_size){
	//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
					u = d_order[depth];
					d_range[depth + offset_qn]  = valid_candidate_size;

					// if depth is not beginning depth.
					if(depth != sl){
						valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

						v = d_candidates[max_candidates_num*u + valid_idx];
					}

	//				printf("tid: %d, valid_idx %d, depth %d, v %d \n", tid,valid_idx,depth,v);
	//				if(depth == 1) {
	//					valid_idx = 19;
	//				}
	//				if(depth == 2) {
	//					valid_idx = 25;
	//				}
	//				if(depth == 3) {
	//					valid_idx = 40;
	//				}
	//				if(depth == 4) {
	//					valid_idx = 33;
	//				}
	//				if(depth == 5) {
	//					valid_idx = 46;
	//				}
	//				if(depth == 6) {
	//					valid_idx = 48;
	//				}
	//				if(depth == 7) {
	//					valid_idx = 80;
	//				}


					if( v== INVALID_ID){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd (&d_score_count[0], 1);
	//					printf("tid: %d,INVALID_ID%d\n", tid);
						continue;
					}

					if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd ( &d_score_count[0], 1);
	//					printf("tid: %d,duplicate %d\n", tid);
						continue;
					}

					d_embedding[offset_qn + u] = v;
					d_idx_embedding[offset_qn + u] = valid_idx;
					d_idx[offset_qn + depth] +=1;


					if (depth == el) {
						double score = 1;
						for (int i =sl ; i <= el; ++i){
	//						printf("reach end!");
	//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}

						//write to global memory
//						atomicAdd (d_score, score);

						thread_score += score;
					}

					if(depth < el){

						depth = depth + 1;
						d_idx[offset_qn + depth] = 0;


						generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

						valid_candidate_size = d_idx_count[ offset_qn+ depth];

						min_size  = min (valid_candidate_size,fixednum);

						if(valid_candidate_size == 0){
							d_idx[ offset_qn+depth - 1] ++;
	//						atomicAdd (d_score_count, 1);
						}
	//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
	//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
					}
				}
				// backtrack
				depth --;
				u = d_order[depth];
				if(depth <= sl ){
					break ;
				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void WarpLayerBalance(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerWarp){
	extern __shared__ unsigned int s_int[];
	int * s = (int* ) s_int;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui wid = threadIdx.x/32;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;

	if(threadIdx.x < threadsPerBlock/32){
		s[threadIdx.x] = 0;
	}
	// setup random numbers for each thread
//	curandState state;
//	curand_init(clock64(), tid, 0, &state);
	ui depth = sl;
	ui u = root;
	if (tid < threadnum){
		// each thread gets a v.
		ui v =0;
		ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
//		//remove
//		v = d_candidates[max_candidates_num*u + valid_idx];
		while (true) {
			ui valid_candidate_size = d_candidates_count[u];
			if(depth != sl){
				valid_candidate_size = d_idx_count[ offset_qn+ depth];
			}
			ui min_size = min (valid_candidate_size,fixednum);

			while (d_idx[depth + offset_qn] < min_size){

//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}

//				printf("tid: %d, valid_idx %d, depth %d, v %d \n", tid,valid_idx,depth,v);
//				if(depth == 1) {
//					valid_idx = 19;
//				}
//				if(depth == 2) {
//					valid_idx = 25;
//				}
//				if(depth == 3) {
//					valid_idx = 40;
//				}
//				if(depth == 4) {
//					valid_idx = 33;
//				}
//				if(depth == 5) {
//					valid_idx = 46;
//				}
//				if(depth == 6) {
//					valid_idx = 48;
//				}
//				if(depth == 7) {
//					valid_idx = 80;
//				}


				if( v== INVALID_ID){
					d_idx[ offset_qn+depth] ++;
					continue;

				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn) ){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd ( &d_score_count[0], 1);
//					printf("tid: %d,duplicate %d\n", tid);
					continue;

				}

					d_embedding[offset_qn + u] = v;
					d_idx_embedding[offset_qn + u] = valid_idx;
					d_idx[offset_qn + depth] +=1;


					if (depth == el) {
						double score = 1;
						for (int i =sl ; i <= el; ++i){
	//						printf("reach end!");
	//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}

	//					printf("thread sscore: %f tid %d \n ", score,tid);

//						atomicAdd (d_score, score);
						thread_score += score;
					}


				// make sure all thread will reach here.
				if(depth < el){

					depth = depth + 1;
					d_idx[offset_qn + depth] = 0;


					generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);

					if(valid_candidate_size == 0){
						d_idx[ offset_qn+depth - 1] ++;
//						atomicAdd (d_score_count, 1);
					}


				}
			}
			// backtrack
			depth --;
			u = d_order[depth];
			if(depth <= sl ){
				// sample end
				if(s[wid] >= taskPerWarp - 32){
					break ;
				}else{
					atomicAdd (&s[wid], 1);
					depth = sl;
					u = root;
					valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
					for (int d = sl ; d < el; ++d  ){
						d_idx[d + offset_qn] = 0;
					}

				}

			}

		}
	}
	// block reduce for thread score

	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void WarpLayerBalance_v2(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerWarp){
	extern __shared__ unsigned int s_int[];
	int * s = (int* ) s_int;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui wid = threadIdx.x/32;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	if(threadIdx.x < threadsPerBlock/32){
		s[threadIdx.x] = 0;
	}
	// setup random numbers for each thread
//	curandState state;
//	curand_init(clock64(), tid, 0, &state);
	ui depth = sl;
	ui u = root;
	if (tid < threadnum){
		// each thread gets a v.
		ui v =0;
		ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
//		//remove
//		v = d_candidates[max_candidates_num*u + valid_idx];
		// mask whether this thread have any task
		bool if_end = false;
		while (__any_sync(0xffffffff,!if_end)) {

			ui valid_candidate_size = d_candidates_count[u];
			if(depth != sl){
				valid_candidate_size = d_idx_count[ offset_qn+ depth];
			}
			ui min_size = min (valid_candidate_size,fixednum);

			bool if_backtrack = false;
			while (__any_sync(0xffffffff,!if_backtrack)){

				bool if_continue = true;
				if(if_end == true){
					if_continue = false;
					if_backtrack = true;
				}
				if(d_idx[depth + offset_qn] >= min_size){
					if_backtrack = true;
					if_continue = false;
				}
//				printf("depth:%d, d_idx %d, min %d if_continue %d\n",  depth, d_idx[depth + offset_qn],min_size,if_continue);
				if(if_continue){
					u = d_order[depth];
					d_range[depth + offset_qn]  = valid_candidate_size;
				}

				// if depth is not beginning depth.
				if(depth != sl && if_continue){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}


				if( v== INVALID_ID && if_continue ){
					d_idx[ offset_qn+depth] ++;
//					continue;
					if_continue = false;
				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn) && if_continue){
					d_idx[ offset_qn+depth] ++;
//					atomicAdd ( &d_score_count[0], 1);
//					printf("tid: %d,duplicate %d\n", tid);
//					continue;
					if_continue = false;
				}
				if(if_continue){
					d_embedding[offset_qn + u] = v;
					d_idx_embedding[offset_qn + u] = valid_idx;
					d_idx[offset_qn + depth] +=1;


					if (depth == el) {
						double score = 1;
						for (int i =sl ; i <= el; ++i){
	//						printf("reach end!");
	//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}



//						atomicAdd (d_score, score);
						thread_score += score;
					}
				}
//				printf("mid contunue depth:%d, d_idx %d, min %d if_continue %d \n",  depth, d_idx[depth + offset_qn],min_size,if_continue);

				// make sure all thread will reach here.
				if(depth < el){
					if(if_continue){

						depth = depth + 1;
						d_idx[offset_qn + depth] = 0;

						generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

						valid_candidate_size = d_idx_count[ offset_qn+ depth];

						min_size  = min (valid_candidate_size,fixednum);

						if(valid_candidate_size == 0){
							d_idx[ offset_qn+depth - 1] ++;
	//						atomicAdd (d_score_count, 1);
						}

					}

				}
//				printf("end contunue depth:%d, d_idx %d, min %d if_continue %d \n",  depth, d_idx[depth + offset_qn],min_size,if_continue);

			}

//			// backtrack


			if(depth <= sl ){

//				// sample end
				if(s[wid] >= taskPerWarp - 32){
					if_end = true;
				}else{
					atomicAdd (&s[wid], 1);
					depth = sl;
					u = root;
					valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
					for (int d = sl ; d < el; ++d  ){
						d_idx[d + offset_qn] = 0;
					}

				}
			}else{
				depth --;
				u = d_order[depth];
			}

		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void test_intersection(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock,ui* d_arr_range_count){
	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;

	while(s < taskPerBlock){
		ui depth = sl;
		ui u = root;
		for (int d = sl ; d < el; ++d  ){
			d_idx[d + offset_qn] = 0;
		}
		if (tid < threadnum){
			atomicAdd (&s, 1);
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
	//		//remove
			v = d_candidates[max_candidates_num*u + valid_idx];
			while (true) {
				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);

				while (d_idx[depth + offset_qn] < min_size){
	//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
					u = d_order[depth];
					d_range[depth + offset_qn]  = valid_candidate_size;

					// if depth is not beginning depth.
					if(depth != sl){
						valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum + d_idx[depth + offset_qn]];

						v = d_candidates[max_candidates_num*u + valid_idx];
					}

	//				printf("tid: %d, valid_idx %d, depth %d, v %d \n", tid,valid_idx,depth,v);
	//				if(depth == 1) {
	//					valid_idx = 19;
	//				}
	//				if(depth == 2) {
	//					valid_idx = 25;
	//				}
	//				if(depth == 3) {
	//					valid_idx = 40;
	//				}
	//				if(depth == 4) {
	//					valid_idx = 33;
	//				}
	//				if(depth == 5) {
	//					valid_idx = 46;
	//				}
	//				if(depth == 6) {
	//					valid_idx = 48;
	//				}
	//				if(depth == 7) {
	//					valid_idx = 80;
	//				}



					if( v== INVALID_ID){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd (&d_score_count[0], 1);
	//					printf("tid: %d,INVALID_ID%d\n", tid);
						continue;
					}

					if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
						d_idx[ offset_qn+depth] ++;
	//					atomicAdd ( &d_score_count[0], 1);
	//					printf("tid: %d,duplicate %d\n", tid);
						continue;
					}

					d_embedding[offset_qn + u] = v;
					d_idx_embedding[offset_qn + u] = valid_idx;
					d_idx[offset_qn + depth] +=1;


					if (depth == el) {
						double score = 1;
						for (int i =sl ; i <= el; ++i){
	//						printf("reach end!");
	//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}

	//					printf("thread sscore: %f tid %d \n ", score,tid);

//						atomicAdd (d_score, score);
						thread_score += score;

					}

					if(depth < el){

						depth = depth + 1;
						d_idx[offset_qn + depth] = 0;


						generateFixedsizeTemp_test( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum,d_arr_range_count);

						valid_candidate_size = d_idx_count[ offset_qn+ depth];

						min_size  = min (valid_candidate_size,fixednum);

						if(valid_candidate_size == 0){
							d_idx[ offset_qn+depth - 1] ++;
	//						atomicAdd (d_score_count, 1);
						}
	//					printf("next range: %d, go to depth %d \n", valid_candidate_size, depth);
	//					printf("tid: %d,depth %d ,min_size %d, fixednum %d\n", tid,depth,min_size,fixednum );
					}
				}
				// backtrack
				depth --;
				u = d_order[depth];
				if(depth <= sl ){
					break ;
				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

// only support one path
template < ui threadsPerBlock>
__global__  void Help_backup(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){

	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
	//

	while(s < taskPerBlock){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
			atomicAdd (&s, 1);
			if(s >= taskPerBlock) {
				break;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				if_end = false;
				if_interrupt = false;
				if(valid_candidate_size == 0 ){
					if_interrupt = true;


				}

				if( v== INVALID_ID){
					if_interrupt = true;

				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;

				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//Find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver =0;
				auto elected_thread = -1;
				auto old_depth =  0;

				if(notEndingCnt > 0 ){
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
//					printf("depth : %d, old_depth: %d, tid %d,elected_thread %d  \n",depth, old_depth,tid, elected_thread );
//					if(elected_lane != 0 )
//					printf("elected_lane %d elected_thread %d \n", elected_lane, elected_thread);
				}
				//go to threads that will reach the end, so elected_thread is not active any more
				bool if_help = false;
				if(if_end){


					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score / divergence;

					}


					if(notEndingCnt > 0 && s < taskPerBlock){

//						atomicAdd (&s, 1);

//						if(s < taskPerBlock){
							if_help = true;
							// count threads that can be used
							ui end_count = __popc(__activemask());
							// get divergence of main
							// to sub thread that join main thread
							divergence = old_diver* (end_count + 1);
							// help elected lane
							// get depth from elected lane

							depth = old_depth;
							//copy d_range, d_embedding, d_idx_embedding
							ui t_offset_qn = elected_thread* query_vertices_num;
							for (int i =sl ; i <= depth; ++i){
								d_range[i + offset_qn] = d_range[i + t_offset_qn];
								d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
								d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
								d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
							}


							if_end = false;
							if_interrupt = false;
						}
//					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
//					divergence *= (total+1);
				}

				if(!if_end ){


					depth = depth + 1;

					generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//					generateFixedsizeTempThreadLessmemV4( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);


					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);
//					if(depth == 5 && tid < 64){
//						if(if_help) {
//							printf("threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d, notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ]);
//
//						}else{
//							if(tid == elected_thread ){
//								printf("**threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d,  notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ] );
//							}
//						}
//					}

//					printf("threadid : %d, if_help %d,elected_lane %d, depth %d, divergence %d, valid_candidate_size %d \n", tid, if_help, elected_lane,depth, divergence, valid_candidate_size );
//					break;

				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
	 // printf("s %d , taskPerBlock %d \n", s ,taskPerBlock );
}

// test same thread
template < ui threadsPerBlock>
__global__  void Help(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){

	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
	//
	for( int it = 0; it < taskPerBlock/threadsPerBlock; ++it  ){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
//			atomicAdd (&s, 1);

			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				if_end = false;
				if_interrupt = false;
				if(valid_candidate_size == 0 ){
					if_interrupt = true;


				}

				if( v== INVALID_ID){
					if_interrupt = true;

				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;

				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//Find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver =0;
				auto elected_thread = -1;
				auto old_depth =  0;

				if(notEndingCnt > 0 ){
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
//					printf("depth : %d, old_depth: %d, tid %d,elected_thread %d  \n",depth, old_depth,tid, elected_thread );
//					if(elected_lane != 0 )
//					printf("elected_lane %d elected_thread %d \n", elected_lane, elected_thread);
				}
				//go to threads that will reach the end, so elected_thread is not active any more
				bool if_help = false;
				if(if_end){


					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score / divergence;

					}


					if(notEndingCnt > 0 ){
//						atomicAdd (&s, 1);

//						if(s < taskPerBlock){
							if_help = true;
							// count threads that can be used
							ui end_count = __popc(__activemask());
							// get divergence of main
							// to sub thread that join main thread
							divergence = old_diver* (end_count + 1);
							// help elected lane
							// get depth from elected lane

							depth = old_depth;
							//copy d_range, d_embedding, d_idx_embedding
							ui t_offset_qn = elected_thread* query_vertices_num;
							for (int i =sl ; i <= depth; ++i){
								d_range[i + offset_qn] = d_range[i + t_offset_qn];
								d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
								d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
								d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
							}


							if_end = false;
							if_interrupt = false;
						}
//					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
//					divergence *= (total+1);
				}

				if(!if_end ){


					depth = depth + 1;

					generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//					generateFixedsizeTempThreadLessmemV4( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);


					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);
//					if(depth == 5 && tid < 64){
//						if(if_help) {
//							printf("threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d, notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ]);
//
//						}else{
//							if(tid == elected_thread ){
//								printf("**threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d,  notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ] );
//							}
//						}
//					}

//					printf("threadid : %d, if_help %d,elected_lane %d, depth %d, divergence %d, valid_candidate_size %d \n", tid, if_help, elected_lane,depth, divergence, valid_candidate_size );
//					break;

				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
	 //
	 //printf("s %d , taskPerBlock %d \n", s ,taskPerBlock );
}


template < ui threadsPerBlock>
__global__  void Help2(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){

	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
	//

	while(s < taskPerBlock){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
			atomicAdd (&s, 1);
			if(s >= taskPerBlock) {
				break;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				if_end = false;
				if_interrupt = false;
				if(valid_candidate_size == 0 ){
					if_interrupt = true;


				}

				if( v== INVALID_ID){
					if_interrupt = true;

				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;

				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//Find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver = 1;
				auto elected_thread = -1;
				auto old_depth =  0;

				if(notEndingCnt > 0 ){
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
//					printf("depth : %d, old_depth: %d, tid %d,elected_thread %d  \n",depth, old_depth,tid, elected_thread );
//					if(elected_lane != 0 )
//					printf("elected_lane %d elected_thread %d \n", elected_lane, elected_thread);
				}
				//go to threads that will reach the end, so elected_thread is not active any more
				bool if_help = false;
				if(if_end){


					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score;

					}


					if(notEndingCnt > 0 && s < taskPerBlock){

//						atomicAdd (&s, 1);

//						if(s < taskPerBlock){
							if_help = true;
							// count threads that can be used
							ui end_count = __popc(__activemask());
							// get divergence of main
							// to sub thread that join main thread
							divergence = old_diver* (end_count + 1);
							// help elected lane
							// get depth from elected lane

							depth = old_depth;
							//copy d_range, d_embedding, d_idx_embedding
							ui t_offset_qn = elected_thread* query_vertices_num;
							for (int i =sl ; i <= depth; ++i){
								d_range[i + offset_qn] = d_range[i + t_offset_qn];
								d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
								d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
								d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
							}


							if_end = false;
							if_interrupt = false;
//						}
					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
//					divergence *= (total+1);
				}

				if(!if_end ){


					depth = depth + 1;


					generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//						generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);
//					if(depth == 5 && tid < 64){
//						if(if_help) {
//							printf("threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d, notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ]);
//
//						}else{
//							if(tid == elected_thread ){
//								printf("**threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d,  notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ] );
//							}
//						}
//					}

//					printf("threadid : %d, if_help %d,elected_lane %d, depth %d, divergence %d, valid_candidate_size %d \n", tid, if_help, elected_lane,depth, divergence, valid_candidate_size );
//					break;

				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
//		 printf("add d_score %f \n", aggregate);
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void HelpPlus(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){

	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
	//

	while(s < taskPerBlock){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
			atomicAdd (&s, 1);
			if(s >= taskPerBlock) {
				break;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				if_end = false;
				if_interrupt = false;
				if(valid_candidate_size == 0 ){
					if_interrupt = true;


				}

				if( v== INVALID_ID){
					if_interrupt = true;

				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;

				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//Find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver =0;
				auto elected_thread = -1;
				auto old_depth =  0;

				if(notEndingCnt > 0 ){
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
//					printf("depth : %d, old_depth: %d, tid %d,elected_thread %d  \n",depth, old_depth,tid, elected_thread );
//					if(elected_lane != 0 )
//					printf("elected_lane %d elected_thread %d \n", elected_lane, elected_thread);
				}
				//go to threads that will reach the end, so elected_thread is not active any more
				bool if_help = false;
				if(if_end){


					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score / divergence;

					}


					if(notEndingCnt > 0 && s < taskPerBlock){

//						atomicAdd (&s, 1);

//						if(s < taskPerBlock){
							if_help = true;
							// count threads that can be used
							ui end_count = __popc(__activemask());
							// get divergence of main
							// to sub thread that join main thread
							divergence = old_diver* (end_count + 1);
							// help elected lane
							// get depth from elected lane

							depth = old_depth;
							//copy d_range, d_embedding, d_idx_embedding
							ui t_offset_qn = elected_thread* query_vertices_num;
							for (int i =sl ; i <= depth; ++i){
								d_range[i + offset_qn] = d_range[i + t_offset_qn];
								d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
								d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
								d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
							}


							if_end = false;
							if_interrupt = false;
						}
//					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
//					divergence *= (total+1);
				}

				if(!if_end ){


					depth = depth + 1;


					generateFixedsizeTempThreadLessmemV4( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//						generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);
//					if(depth == 5 && tid < 64){
//						if(if_help) {
//							printf("threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d, notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ]);
//
//						}else{
//							if(tid == elected_thread ){
//								printf("**threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d,  notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ] );
//							}
//						}
//					}

//					printf("threadid : %d, if_help %d,elected_lane %d, depth %d, divergence %d, valid_candidate_size %d \n", tid, if_help, elected_lane,depth, divergence, valid_candidate_size );
//					break;

				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void HelpPlusWJ(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){

	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
	//

	while(s < taskPerBlock){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
			atomicAdd (&s, 1);
			if(s >= taskPerBlock) {
				break;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				if_end = false;
				if_interrupt = false;
				if(valid_candidate_size == 0 ){
					if_interrupt = true;


				}

				if( v== INVALID_ID){
					if_interrupt = true;

				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;

				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//Find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver =0;
				auto elected_thread = -1;
				auto old_depth =  0;

				if(notEndingCnt > 0 ){
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
//					printf("depth : %d, old_depth: %d, tid %d,elected_thread %d  \n",depth, old_depth,tid, elected_thread );
//					if(elected_lane != 0 )
//					printf("elected_lane %d elected_thread %d \n", elected_lane, elected_thread);
				}
				//go to threads that will reach the end, so elected_thread is not active any more
				bool if_help = false;
				if(if_end){


					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score / divergence;

					}


					if(notEndingCnt > 0 && s < taskPerBlock){

//						atomicAdd (&s, 1);

						if(s < taskPerBlock){
							if_help = true;
							// count threads that can be used
							ui end_count = __popc(__activemask());
							// get divergence of main
							// to sub thread that join main thread
							divergence = old_diver* (end_count + 1);
							// help elected lane
							// get depth from elected lane

							depth = old_depth;
							//copy d_range, d_embedding, d_idx_embedding
							ui t_offset_qn = elected_thread* query_vertices_num;
							for (int i =sl ; i <= depth; ++i){
								d_range[i + offset_qn] = d_range[i + t_offset_qn];
								d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
								d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
								d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
							}


							if_end = false;
							if_interrupt = false;
						}
					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
//					divergence *= (total+1);
				}

				if(!if_end ){
					depth = depth + 1;

					wanderjoinThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//						generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
					// check vaildlity for new sampled node
					bool if_valid = wanderjoinCheckOneNode ( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,tid , fixednum);
					if(if_valid){
						valid_candidate_size = d_idx_count[ offset_qn+ depth];
					} else{
						d_idx_count[ offset_qn+ depth] = 0;
						valid_candidate_size = 0;
					}
					min_size  = min (valid_candidate_size,fixednum);
//					if(depth == 5 && tid < 64){
//						if(if_help) {
//							printf("threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d, notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ]);
//
//						}else{
//							if(tid == elected_thread ){
//								printf("**threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d,  notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ] );
//							}
//						}
//					}

//					printf("threadid : %d, if_help %d,elected_lane %d, depth %d, divergence %d, valid_candidate_size %d \n", tid, if_help, elected_lane,depth, divergence, valid_candidate_size );
//					break;

				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void baselineLayer(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){

	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
	//
	while(s < taskPerBlock){

		ui depth = sl;
		ui u = root;

		// get info from nearby thread
		if (tid < threadnum){
			atomicAdd (&s, 1);
			if(s >= taskPerBlock){
				break;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}

				bool if_end = false;
				if(valid_candidate_size == 0){
					if_end = true;
				}

				if(!if_end && v== INVALID_ID){
					if_end = true;
				}

				if(!if_end && duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_end = true;
				}

				if(if_end){
					depth = sl;
					u = root;
					valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
					atomicAdd (&s, 1);
					if(s >= taskPerBlock){
						break;
					}

					continue;
				}

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;



				if (depth == el) {

					//compute score
					double score = 1;
					for (int i =sl ; i <= el; ++i){
//						printf("reach end!");
//						printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
						if(d_range[i + offset_qn] > fixednum){
							score *= (double)d_range[i + offset_qn]/fixednum;

						}
					}
					thread_score += score;
					depth = sl;
					u = root;
					valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
					atomicAdd (&s, 1);
					if(s >= taskPerBlock){
						break;
					}

					continue;
				}



				if(depth < el){

					depth = depth + 1;


					generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//						generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);
					if(valid_candidate_size == 0){

						depth = sl;
						u = root;
						valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
						atomicAdd (&s, 1);
						if(s >= taskPerBlock){
							break;
						}
	
						continue;
					}
				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

//wandor join
template < ui threadsPerBlock>
__global__  void wanderJoin(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){
	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
	//
	while(s < taskPerBlock){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;

		// get info from nearby thread
		if (tid < threadnum){
			atomicAdd (&s, 1);
			if(s >= taskPerBlock){
				break;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			while (true) {
				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


				//printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}

				if(valid_candidate_size == 0){

					break;
				}

				if( v== INVALID_ID){

					break;
				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){

					break;
				}


				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;



				if (depth == el) {
					// check whether this path is vaild
					bool valid_path = wanderjoinCheck ( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,tid , fixednum);
					//compute score
					if(!valid_path){
						break;
	   				}
					double score = 1;
					for (int i =sl ; i <= el; ++i){
					//	printf("reach end!");
						//printf("d_range[i + offset_qn]: %d, fixednum %d \n",d_range[i + offset_qn], fixednum );
						if(d_range[i + offset_qn] > fixednum){
							score *= (double)d_range[i + offset_qn]/fixednum;
							

						}
					}
					thread_score += score;
					break;
				}



				if(depth < el){

					depth = depth + 1;

//					generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
					wanderjoinThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//					generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);
					if(valid_candidate_size == 0){

						break;
					}
				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}




template < ui threadsPerBlock>
__global__  void HelpIndependentPlus(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock,ui* d_denominator){

	__shared__ unsigned int s;
	__shared__ unsigned int d;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
//	d = 0;
	//

	while(s < taskPerBlock){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
			atomicAdd (&s, 1);
//			atomicAdd (&d, 1);
			if(s >= taskPerBlock) {
				break;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				if_end = false;
				if_interrupt = false;
				if(valid_candidate_size == 0 ){
					if_interrupt = true;


				}

				if( v== INVALID_ID){
					if_interrupt = true;

				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;

				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//Find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver =0;
				auto elected_thread = -1;
				auto old_depth =  0;

				if(notEndingCnt > 0 ){
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
//					printf("depth : %d, old_depth: %d, tid %d,elected_thread %d  \n",depth, old_depth,tid, elected_thread );
//					if(elected_lane != 0 )
//					printf("elected_lane %d elected_thread %d \n", elected_lane, elected_thread);
				}
				//go to threads that will reach the end, so elected_thread is not active any more
				bool if_help = false;
				if(if_end){


					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score / divergence;

					}


					if(notEndingCnt > 0 && s < taskPerBlock){
					//this control the sample count method.
					//	atomicAdd (&s, 1);
					//	atomicAdd (&d, 1);
//						if(s < taskPerBlock){
							if_help = true;
							// count threads that can be used
							ui end_count = __popc(__activemask());
							// get divergence of main
							// to sub thread that join main thread
							divergence = old_diver* (end_count + 1);
							// help elected lane
							// get depth from elected lane

							depth = old_depth;
							//copy d_range, d_embedding, d_idx_embedding
							ui t_offset_qn = elected_thread* query_vertices_num;
							for (int i =sl ; i <= depth; ++i){
								d_range[i + offset_qn] = d_range[i + t_offset_qn];
								d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
								d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
								d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
							}


							if_end = false;
							if_interrupt = false;
						}
//					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
//					divergence *= (total+1);
				}

				if(!if_end ){


					depth = depth + 1;

//					generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
					generateFixedsizeTempThreadLessmemV4( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);


					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);
//					if(depth == 5 && tid < 64){
//						if(if_help) {
//							printf("threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d, notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ]);
//
//						}else{
//							if(tid == elected_thread ){
//								printf("**threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d,  notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ] );
//							}
//						}
//					}

//					printf("threadid : %d, if_help %d,elected_lane %d, depth %d, divergence %d, valid_candidate_size %d \n", tid, if_help, elected_lane,depth, divergence, valid_candidate_size );
//					break;

				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
//		 atomicAdd (d_denominator,d );
	 }
}

// wanderJoin
template < ui threadsPerBlock>
__global__  void HelpWJ(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock,ui* d_denominator){

	__shared__ unsigned int s;
	__shared__ unsigned int d;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
//	d = 0;
	//

	while(s < taskPerBlock){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
			atomicAdd (&s, 1);
			atomicAdd (&d, 1);
			if(s >= taskPerBlock) {
				break;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				if_end = false;
				if_interrupt = false;
				if(valid_candidate_size == 0 || v== INVALID_ID || duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;
				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//Find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver =0;
				auto elected_thread = -1;
				auto old_depth =  0;

				if(notEndingCnt > 0 ){
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
//					printf("depth : %d, old_depth: %d, tid %d,elected_thread %d  \n",depth, old_depth,tid, elected_thread );
//					if(elected_lane != 0 )
//					printf("elected_lane %d elected_thread %d \n", elected_lane, elected_thread);
				}
				//go to threads that will reach the end, so elected_thread is not active any more
				bool if_help = false;
				if(if_end){


					if (depth == el && !if_interrupt) {
						bool if_valid = wanderjoinCheckOneNode ( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, el,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,tid , fixednum);
						//compute score
						double score = 1;
						if(!if_valid){
							score = 0;
						} 
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score / divergence;

					}


					if(notEndingCnt > 0 && s < taskPerBlock){
					//this control the sample count method.
					//	atomicAdd (&s, 1);
					atomicAdd (&d, 1);
//						if(s < taskPerBlock){
							if_help = true;
							// count threads that can be used
							ui end_count = __popc(__activemask());
							// get divergence of main
							// to sub thread that join main thread
							divergence = old_diver* (end_count + 1);
							// help elected lane
							// get depth from elected lane

							depth = old_depth;
							//copy d_range, d_embedding, d_idx_embedding
							ui t_offset_qn = elected_thread* query_vertices_num;
							for (int i =sl ; i <= depth; ++i){
								d_range[i + offset_qn] = d_range[i + t_offset_qn];
								d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
								d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
								d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
							}


							if_end = false;
							if_interrupt = false;
						}
//					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
//					divergence *= (total+1);
				}

				if(!if_end ){


					depth = depth + 1;

					wanderjoinThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					// check vaildlity for new sampled node
					bool if_valid = wanderjoinCheckOneNode ( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth-1,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,tid , fixednum);
					if(if_valid){
						valid_candidate_size = d_idx_count[ offset_qn+ depth];
					} else{
						d_idx_count[ offset_qn+ depth] = 0;
						valid_candidate_size = 0;
					}

					min_size  = min (valid_candidate_size,fixednum);
//					if(depth == 5 && tid < 64){
//						if(if_help) {
//							printf("threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d, notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ]);
//
//						}else{
//							if(tid == elected_thread ){
//								printf("**threadid : %d, elected_thread %d, if_help %d,elected_lane %d, depth %d, divergence %d,  notEndingCnt %d, valid_candidate_size %d, val %d \n", tid,elected_thread, if_help, elected_lane,depth, divergence, notEndingCnt, valid_candidate_size, d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ] );
//							}
//						}
//					}

//					printf("threadid : %d, if_help %d,elected_lane %d, depth %d, divergence %d, valid_candidate_size %d \n", tid, if_help, elected_lane,depth, divergence, valid_candidate_size );
//					break;

				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
		 atomicAdd (d_denominator,d );
	 }
}




// thtread based version
__device__ void ggerefine (ui* d_offset_index, ui* d_offsets, ui* d_edge_index, ui* d_edges,ui* d_order,ui depth, ui el,ui* d_bn ,ui* d_bn_count, ui* d_idx_count,ui* d_embedding, ui* d_idx_embedding, ui query_vertices_num , ui max_candidates_num, ui* d_temp, ui* d_intersection, ui tid , ui fixednum){
	
	int laneid = threadIdx.x % 32;
	ui u = d_order[depth];
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
//	ui offset_qmn = query_vertices_num*max_candidates_num*tid;
	ui neighbor_count = d_bn_count[depth];

	ui first_neighbor = d_bn[query_vertices_num* depth];
	ui first_neighbor_embedding_idx = d_idx_embedding [offset_qn + first_neighbor];

	// CSR's offset & list
	ui* ListToBeIntersected;
	ui first_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,first_neighbor, u,first_neighbor_embedding_idx, query_vertices_num, ListToBeIntersected);
//	printf("count: %d \n", first_neighbor_count);
    ui clen = first_neighbor_count;
   	double total_w = 0;
    // mask that clen >32
    ui mask1 = __ballot_sync(0xffffffff,clen > 32);
//    printf("lane id %d mask %x \n",laneid, mask1);
    ui curiter = 0;
    while (mask1){
       // process the result when clen > 32
 		int leaderid = __ffs(mask1) - 1;
  	  // broadcast tid
  		ui targetid = __shfl_sync(mask1, tid, leaderid);
  	    // broadcast 1st array
  	    unsigned long long adr = (unsigned long long)&ListToBeIntersected[0];
  	    unsigned long long t_adr = __shfl_sync(mask1,adr, leaderid);
  	    ui* targetArr = (ui*)t_adr;
  	    ui target_offset_qn = targetid* query_vertices_num;
  	    ui val = targetArr[laneid + curiter];
  	    //refine test
  	    bool find = true;
		ui intersection_time = 1;
		while(find && (intersection_time < neighbor_count)){
			ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
			ui second_neighbor_embedding_idx = d_idx_embedding [target_offset_qn + second_neighbor];
			ui* secondListToBeIntersected;
			ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
			find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
			intersection_time ++;
		}
  	    // key= rand^(1/w) if w!=1
  	    double key = (clock64()+tid);
  	    if(!find ){
  	    	key = 0;
  	    }
		ui mask2 =  __ballot_sync(mask1,find);
		double cur_w= __popc(mask2);
		total_w += cur_w;
  	    //find maximum key in warp 
  	    typedef cub::WarpReduce<double> WarpReduce;
  	    __shared__ typename WarpReduce::TempStorage temp_storage[4];
        double aggregate = WarpReduce(temp_storage[laneid]).Reduce(key, cub::Max());
        if (aggregate == key){
           ui prob = ((clock64()+tid)%(int)total_w);
           if (prob < cur_w){
           	d_temp[query_vertices_num*targetid + depth  ] = val;
           }
        }
  		if(laneid == leaderid){
     		curiter = curiter + 32; 
     	}
        mask1 = __ballot_sync(mask1, clen-curiter > 32);
      
    }
    
    //if clen -curiter <32
    ui count = total_w;
	if(clen-curiter <= 32){
		
		for(ui i =curiter; i < clen; ++i){
	
			ui val = ListToBeIntersected[i];
			bool find = true;
			ui intersection_time = 1;
			while(find && (intersection_time < neighbor_count)){
	
				ui second_neighbor = d_bn[query_vertices_num* depth + intersection_time];
				ui second_neighbor_embedding_idx = d_idx_embedding [offset_qn + second_neighbor];
				ui* secondListToBeIntersected;
				ui second_neighbor_count = getList(d_offset_index, d_offsets, d_edge_index, d_edges,second_neighbor, u, second_neighbor_embedding_idx,query_vertices_num, secondListToBeIntersected);
				find = deviceBinarySearch(secondListToBeIntersected, val,0,second_neighbor_count - 1);
				intersection_time ++;
	
			}
	
	
			if (find){
	
				if(count < fixednum ){
					d_temp[query_vertices_num*fixednum*tid + fixednum*depth + count ] = val;
				}else{
					// reservoir sampling
					ui random_ui = (clock64()+tid)%(count + 1);
					if(random_ui <= fixednum - 1){
						d_temp[query_vertices_num*fixednum*tid + fixednum*depth + random_ui ] = val;
					}
	
				}
				count ++;
			}
		}
	}


	d_idx_count [offset_qn + depth] = count;

}



// test same thread
template < ui threadsPerBlock>
__global__  void ggersal(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){

	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
	//
	for( int it = 0; it < taskPerBlock/threadsPerBlock; ++it  ){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
//			atomicAdd (&s, 1);

			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}

				if(valid_candidate_size == 0 ){
					if_interrupt = true;


				}

				if( v== INVALID_ID){
					if_interrupt = true;

				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;

				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;
            
			
				if(if_end){
					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						thread_score = score;
					}
				}

				__syncwarp();
				if(__all_sync(0xffffffff ,if_end)){
					break;
				}
				depth = depth + 1;
				ggerefine( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth, el, d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
			    if(!if_end ){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);

				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
	
	 //printf("s %d , taskPerBlock %d \n", s ,taskPerBlock );
}


template < ui threadsPerBlock>
__global__  void ggecorsal(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){

	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
	//

	while(s < taskPerBlock){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
			atomicAdd (&s, 1);
			if(s >= taskPerBlock) {
				break;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				if_end = false;
				if_interrupt = false;
				if(valid_candidate_size == 0 ){
					if_interrupt = true;


				}

				if( v== INVALID_ID){
					if_interrupt = true;

				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;

				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//Find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver =0;
				auto elected_thread = -1;
				auto old_depth =  0;

				if(notEndingCnt > 0 ){
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
//					printf("depth : %d, old_depth: %d, tid %d,elected_thread %d  \n",depth, old_depth,tid, elected_thread );
//					if(elected_lane != 0 )
//					printf("elected_lane %d elected_thread %d \n", elected_lane, elected_thread);
				}
				//go to threads that will reach the end, so elected_thread is not active any more
				bool if_help = false;
				if(if_end){


					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score / divergence;

					}


					if(notEndingCnt > 0 && s < taskPerBlock){

//						atomicAdd (&s, 1);

//						if(s < taskPerBlock){
							if_help = true;
							// count threads that can be used
							ui end_count = __popc(__activemask());
							// get divergence of main
							// to sub thread that join main thread
							divergence = old_diver* (end_count + 1);
							// help elected lane
							// get depth from elected lane

							depth = old_depth;
							//copy d_range, d_embedding, d_idx_embedding
							ui t_offset_qn = elected_thread* query_vertices_num;
							for (int i =sl ; i <= depth; ++i){
								d_range[i + offset_qn] = d_range[i + t_offset_qn];
								d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
								d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
								d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
							}


							if_end = false;
							if_interrupt = false;
						}
//					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
//					divergence *= (total+1);
				}
				
				if(if_end){
					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						thread_score = score;
					}
				}

				__syncwarp();
				if(__all_sync(0xffffffff ,if_end)){
					break;
				}
				depth = depth + 1;
				//ggerefine( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth, el, d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
			    generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
			    if(!if_end ){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);

				}


			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}


template < ui threadsPerBlock>
__global__  void ggecoal(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){

	__shared__ unsigned int s;
	__shared__ unsigned int d;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
//	d = 0;
	//

	while(s < taskPerBlock){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
			atomicAdd (&s, 1);
//			atomicAdd (&d, 1);
			if(s >= taskPerBlock) {
				break;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				if_end = false;
				if_interrupt = false;
				if(valid_candidate_size == 0 ){
					if_interrupt = true;


				}

				if( v== INVALID_ID){
					if_interrupt = true;

				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;

				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//Find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver =0;
				auto elected_thread = -1;
				auto old_depth =  0;

				if(notEndingCnt > 0 ){
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
//					printf("depth : %d, old_depth: %d, tid %d,elected_thread %d  \n",depth, old_depth,tid, elected_thread );
//					if(elected_lane != 0 )
//					printf("elected_lane %d elected_thread %d \n", elected_lane, elected_thread);
				}
				//go to threads that will reach the end, so elected_thread is not active any more
				bool if_help = false;
				if(if_end){


					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score / divergence;

					}


					if(notEndingCnt > 0 && s < taskPerBlock){
					//this control the sample count method.
					//	atomicAdd (&s, 1);
					//	atomicAdd (&d, 1);
//						if(s < taskPerBlock){
							if_help = true;
							// count threads that can be used
							ui end_count = __popc(__activemask());
							// get divergence of main
							// to sub thread that join main thread
							divergence = old_diver* (end_count + 1);
							// help elected lane
							// get depth from elected lane

							depth = old_depth;
							//copy d_range, d_embedding, d_idx_embedding
							ui t_offset_qn = elected_thread* query_vertices_num;
							for (int i =sl ; i <= depth; ++i){
								d_range[i + offset_qn] = d_range[i + t_offset_qn];
								d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
								d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
								d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
							}


							if_end = false;
							if_interrupt = false;
						}
//					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
//					divergence *= (total+1);
				}

				if(!if_end ){


					depth = depth + 1;
					
//					generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//					generateFixedsizeTempThreadLessmemV4( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
					generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);


				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
//		 atomicAdd (d_denominator,d );
	 }
}

template < ui threadsPerBlock>
__global__  void ggecowj(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock,ui* d_denominator){

	__shared__ unsigned int s;
	__shared__ unsigned int d;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
//	d = 0;
	//

	while(s < taskPerBlock){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
			atomicAdd (&s, 1);
			atomicAdd (&d, 1);
			if(s >= taskPerBlock) {
				break;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				if_end = false;
				if_interrupt = false;
				if(valid_candidate_size == 0 ){
					if_interrupt = true;


				}

				if( v== INVALID_ID){
					if_interrupt = true;

				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;

				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//Find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver =0;
				auto elected_thread = -1;
				auto old_depth =  0;

				if(notEndingCnt > 0 ){
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
//					printf("depth : %d, old_depth: %d, tid %d,elected_thread %d  \n",depth, old_depth,tid, elected_thread );
//					if(elected_lane != 0 )
//					printf("elected_lane %d elected_thread %d \n", elected_lane, elected_thread);
				}
				//go to threads that will reach the end, so elected_thread is not active any more
				bool if_help = false;
				if(if_end){


					if (depth == el && !if_interrupt) {
						bool if_valid = wanderjoinCheckOneNode ( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, el,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,tid , fixednum);
						//compute score
						double score = 1;
						if(!if_valid){
							score = 0;
						} 
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score / divergence;

					}


					if(notEndingCnt > 0 && s < taskPerBlock){
					//this control the sample count method.
					//	atomicAdd (&s, 1);
					atomicAdd (&d, 1);
//						if(s < taskPerBlock){
							if_help = true;
							// count threads that can be used
							ui end_count = __popc(__activemask());
							// get divergence of main
							// to sub thread that join main thread
							divergence = old_diver* (end_count + 1);
							// help elected lane
							// get depth from elected lane

							depth = old_depth;
							//copy d_range, d_embedding, d_idx_embedding
							ui t_offset_qn = elected_thread* query_vertices_num;
							for (int i =sl ; i <= depth; ++i){
								d_range[i + offset_qn] = d_range[i + t_offset_qn];
								d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
								d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
								d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
							}


							if_end = false;
							if_interrupt = false;
						}
//					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
//					divergence *= (total+1);
				}

				if(!if_end ){


					depth = depth + 1;

					wanderjoinThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					// check vaildlity for new sampled node
					bool if_valid = wanderjoinCheckOneNode ( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth-1,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,tid , fixednum);
					if(if_valid){
						valid_candidate_size = d_idx_count[ offset_qn+ depth];
					} else{
						d_idx_count[ offset_qn+ depth] = 0;
						valid_candidate_size = 0;
					}

					min_size  = min (valid_candidate_size,fixednum);



				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
		 atomicAdd (d_denominator,d );
	 }
}

template < ui threadsPerBlock>
__global__  void ggecopr(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){

	__shared__ unsigned int s;
	__shared__ unsigned int d;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
//	d = 0;
	//

	while(s < taskPerBlock){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
			atomicAdd (&s, 1);
//			atomicAdd (&d, 1);
			if(s >= taskPerBlock) {
				break;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				if_end = false;
				if_interrupt = false;
				if(valid_candidate_size == 0 ){
					if_interrupt = true;


				}

				if( v== INVALID_ID){
					if_interrupt = true;

				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;

				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//Find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver =0;
				auto elected_thread = -1;
				auto old_depth =  0;

				if(notEndingCnt > 0 ){
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
//					printf("depth : %d, old_depth: %d, tid %d,elected_thread %d  \n",depth, old_depth,tid, elected_thread );
//					if(elected_lane != 0 )
//					printf("elected_lane %d elected_thread %d \n", elected_lane, elected_thread);
				}
				//go to threads that will reach the end, so elected_thread is not active any more
				bool if_help = false;
				if(if_end){


					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score / divergence;

					}


					if(notEndingCnt > 0 && s < taskPerBlock){
					//this control the sample count method.
					//	atomicAdd (&s, 1);
					//	atomicAdd (&d, 1);
//						if(s < taskPerBlock){
							if_help = true;
							// count threads that can be used
							ui end_count = __popc(__activemask());
							// get divergence of main
							// to sub thread that join main thread
							divergence = old_diver* (end_count + 1);
							// help elected lane
							// get depth from elected lane

							depth = old_depth;
							//copy d_range, d_embedding, d_idx_embedding
							ui t_offset_qn = elected_thread* query_vertices_num;
							for (int i =sl ; i <= depth; ++i){
								d_range[i + offset_qn] = d_range[i + t_offset_qn];
								d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
								d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
								d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
							}


							if_end = false;
							if_interrupt = false;
						}
//					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
//					divergence *= (total+1);
				}

				if(!if_end ){


					depth = depth + 1;
					
//					generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//					generateFixedsizeTempThreadLessmemV4( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
					generateFixedsizeTempThreadV4( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);

					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);


				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
//		 atomicAdd (d_denominator,d );
	 }
}

template < ui threadsPerBlock>
__global__  void ggecorspr(ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){

	__shared__ unsigned int s;
	double thread_score = 0.0;
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
	//

	while(s < taskPerBlock){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
			atomicAdd (&s, 1);
			if(s >= taskPerBlock) {
				break;
			}
			// each thread gets a v.
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);

			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


//				printf("depth:%d, d_idx %d, min %d \n",  depth, d_idx[depth + offset_qn],min_size);
				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if depth is not beginning depth.
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				if_end = false;
				if_interrupt = false;
				if(valid_candidate_size == 0 ){
					if_interrupt = true;


				}

				if( v== INVALID_ID){
					if_interrupt = true;

				}

				if(duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;

				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//Find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver =0;
				auto elected_thread = -1;
				auto old_depth =  0;

				if(notEndingCnt > 0 ){
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
//					printf("depth : %d, old_depth: %d, tid %d,elected_thread %d  \n",depth, old_depth,tid, elected_thread );
//					if(elected_lane != 0 )
//					printf("elected_lane %d elected_thread %d \n", elected_lane, elected_thread);
				}
				//go to threads that will reach the end, so elected_thread is not active any more
				bool if_help = false;
				if(if_end){


					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score / divergence;

					}


					if(notEndingCnt > 0 && s < taskPerBlock){

//						atomicAdd (&s, 1);

//						if(s < taskPerBlock){
							if_help = true;
							// count threads that can be used
							ui end_count = __popc(__activemask());
							// get divergence of main
							// to sub thread that join main thread
							divergence = old_diver* (end_count + 1);
							// help elected lane
							// get depth from elected lane

							depth = old_depth;
							//copy d_range, d_embedding, d_idx_embedding
							ui t_offset_qn = elected_thread* query_vertices_num;
							for (int i =sl ; i <= depth; ++i){
								d_range[i + offset_qn] = d_range[i + t_offset_qn];
								d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
								d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
								d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
							}


							if_end = false;
							if_interrupt = false;
						}
//					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
//					divergence *= (total+1);
				}
				
				if(if_end){
					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						thread_score = score;
					}
				}

				__syncwarp();
				if(__all_sync(0xffffffff ,if_end)){
					break;
				}
				depth = depth + 1;
				ggerefine_partial( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth, el, d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
			    if(!if_end ){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);

				}


			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

// ALLEY
template < ui threadsPerBlock>
__global__  void ggecoal2 (ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){
	// number of tasks per block
	__shared__ unsigned int s;
//	__shared__ unsigned int d;
    // estimated result for each sample. 
	double thread_score = 0.0;
	// thread id in current thread.
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	// offsets of each thread. 
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
//	d = 0;

	// when there are available tasks
	while(s < taskPerBlock){
		// reset to 1st layer
		ui depth = sl;
		ui u = root;
		ui divergence = 1;

		if (tid < threadnum){
			atomicAdd (&s, 1);
//			atomicAdd (&d, 1);
			if(s >= taskPerBlock) {
				break;
			}
			//  initial the starting vertex
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
			
			// flags for thread statues
			bool if_end = false;
			bool if_interrupt = false;
			while (true) {

				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,fixednum);


				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// fetch vertex for random walk
				if(depth != sl){
					valid_idx = d_temp[fixednum* query_vertices_num* tid  + depth* fixednum ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				if_end = false;
				if_interrupt = false;
				// update thread statues
				if(valid_candidate_size == 0 || v== INVALID_ID || duplicate_test(d_embedding,v, depth,d_order ,offset_qn)){
					if_interrupt = true;
				}

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver =0;
				auto elected_thread = -1;
				auto old_depth =  0;

				if(notEndingCnt > 0 ){
					// fetch sample from other thread.
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
//					printf("depth : %d, old_depth: %d, tid %d,elected_thread %d  \n",depth, old_depth,tid, elected_thread );
//					if(elected_lane != 0 )
//					printf("elected_lane %d elected_thread %d \n", elected_lane, elected_thread);
				}
				//
				bool if_help = false;
				if(if_end){


					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score / divergence;

					}


					if(notEndingCnt > 0 && s < taskPerBlock){
					//this control the sample count method.
					//	atomicAdd (&s, 1);
					//	atomicAdd (&d, 1);
//						if(s < taskPerBlock){
							if_help = true;
							// count threads that can be used
							ui end_count = __popc(__activemask());
							// get divergence of main
							// to sub thread that join main thread
							divergence = old_diver* (end_count + 1);
							// help elected lane
							// get depth from elected lane

							depth = old_depth;
							//copy d_range, d_embedding, d_idx_embedding
							ui t_offset_qn = elected_thread* query_vertices_num;
							for (int i =sl ; i <= depth; ++i){
								d_range[i + offset_qn] = d_range[i + t_offset_qn];
								d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
								d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
								d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
							}


							if_end = false;
							if_interrupt = false;
						}
//					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
//					divergence *= (total+1);
				}

				if(!if_end ){


					depth = depth + 1;
					
					generateFixedsizeTempThreadLessmem( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//					generateFixedsizeTempThreadLessmemV4( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//					generateFixedsizeTemp( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
//                  ggerefine( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth, el, d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  = min (valid_candidate_size,fixednum);


				}

			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}

template < ui threadsPerBlock>
__global__  void UserDefine (ui root,ui* d_offset_index,ui* d_offsets,ui* d_edge_index,ui* d_edges ,ui* d_order,ui* d_candidates,ui* d_candidates_count, ui* d_bn ,ui* d_bn_count, ui* d_idx_count,  ui* d_idx, ui* d_range, ui* d_embedding, ui* d_idx_embedding, ui* d_temp, ui* d_intersection,ui query_vertices_num ,ui max_candidates_num,ui threadnum,ui sl, ui el, ui fixednum, double* d_score,ui* d_score_count, ui taskPerBlock){
	// record the count for number of tasks per block.
	__shared__ unsigned int s;
	__shared__ unsigned int d;
	// each thread holds a estimation for one sample
	double thread_score = 0.0;
	// thread id
	ui tid = blockIdx.x * blockDim.x + threadIdx.x;
	// compute offsets 
	ui offset_qn = tid* query_vertices_num;
	ui offset_cn = tid* max_candidates_num;
	s = 0;
	// when there are available tasks
	while(s < taskPerBlock){
 		// initializing variables 
		ui depth = sl;
		ui u = root;
		ui divergence = 1;
		// for each available thread assign the work
		if (tid < threadnum){
			atomicAdd (&s, 1);
			if(s >= taskPerBlock) {
				break;
			}
			//  initializing starting vertices
			ui v =0;
			ui valid_idx = PickOneRandomCandidate ( d_candidates, d_candidates_count[u], max_candidates_num, u, tid, v);
            
			// flags for thread statues
			bool if_end = false;
			bool if_interrupt = false;
			while (true) {
				ui valid_candidate_size = d_candidates_count[u];
				if(depth != sl){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];
				}
				ui min_size = min (valid_candidate_size,1);

				u = d_order[depth];
				d_range[depth + offset_qn]  = valid_candidate_size;

				// if not the first iteration, fetch sampled vertex
				if(depth != sl){
					valid_idx = d_temp[query_vertices_num* tid  + depth ];

					v = d_candidates[max_candidates_num*u + valid_idx];
				}
				
				if_end = false;
				if_interrupt = false;
				//mask thread that holds invalid samples
				if_interrupt = validate( d_embedding,  d_order, valid_candidate_size,  v, depth, offset_qn );

				if_end = if_interrupt || (depth ==  el) ;

				d_embedding[offset_qn + u] = v;
				d_idx_embedding[offset_qn + u] = valid_idx;

				auto notEndingMask = __ballot_sync(__activemask(), !if_end);
				//check number of existing threads
				auto totalCnt =  __popc(__activemask());
				auto notEndingCnt = __popc(notEndingMask);
				//pick one unfinsihed thread
				//Find the lowest-numbered active lane
				int elected_lane = -1;
				//collect info when elected_lane is active
				auto old_diver =0;
				auto elected_thread = -1;
				auto old_depth =  0;
				// get samples from other threads
				if(notEndingCnt > 0 ){
					elected_lane = __ffs(notEndingMask) - 1;
					old_diver = __shfl( divergence, elected_lane);
					elected_thread = __shfl(tid, elected_lane);
					old_depth =  __shfl(depth, elected_lane);
				}
		
				bool if_help = false;
				
				if(if_end){
					if (depth == el && !if_interrupt) {

						//compute score
						double score = 1;
						for (int i =sl ; i <= el; ++i){

							if(d_range[i + offset_qn] > fixednum){
								score *= (double)d_range[i + offset_qn]/fixednum;

							}
						}
						if(divergence > 0 )
						thread_score += score / divergence;

					}
					// If there exists valid samples, continue. 
					if(notEndingCnt > 0 && s < taskPerBlock){
					//this control the sample count method.
						if_help = true;
						// count threads that can be used
						ui end_count = __popc(__activemask());
						// get divergence of main
						// to sub thread that join main thread
						divergence = old_diver* (end_count + 1);

						depth = old_depth;
						//copy d_range, d_embedding, d_idx_embedding
						ui t_offset_qn = elected_thread* query_vertices_num;
						for (int i =sl ; i <= depth; ++i){
							d_range[i + offset_qn] = d_range[i + t_offset_qn];
							d_embedding[offset_qn + d_order[i]] = d_embedding[t_offset_qn + d_order[i]];
							d_idx_embedding[offset_qn + d_order[i]] = d_idx_embedding[t_offset_qn + d_order[i]];
							d_idx_count[ offset_qn+ i] = d_idx_count[ t_offset_qn + i];
						}
						if_end = false;
						if_interrupt = false;
					}
				}
				if(notEndingCnt == 0  ){
					break;
				}
				__syncwarp();
				int help_count = totalCnt - notEndingCnt;
				if(tid == elected_thread){
					divergence *= (help_count+1);
				}
				if(if_end && depth == el && !if_interrupt){
					computeScore(d_range, sl, el,offset_qn, thread_score)
				
				}

				__syncwarp();
				if(__all_sync(0xffffffff ,if_end)){
					break;
				}
				depth = depth + 1;
				Refine( d_offset_index, d_offsets, d_edge_index, d_edges,d_order, depth,  d_bn , d_bn_count,  d_idx_count, d_embedding,d_idx_embedding, query_vertices_num, max_candidates_num,d_temp,d_intersection,tid , fixednum);
				valid_candidate_size = d_idx_count[ offset_qn+ depth];
				if(!if_end ){
					valid_candidate_size = d_idx_count[ offset_qn+ depth];

					min_size  =valid_candidate_size

				}
				
			}
		}
	}
	// block reduce for thread score
	 typedef cub::BlockReduce<double, threadsPerBlock> BlockReduce;
	 __shared__ typename BlockReduce::TempStorage temp_storage;
	 double aggregate = BlockReduce(temp_storage).Sum(thread_score, threadsPerBlock);
	 if(threadIdx.x == 0){
		 atomicAdd (d_score,aggregate );
	 }
}
