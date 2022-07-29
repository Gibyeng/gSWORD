#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <cuda_profiler_api.h>
#include <time.h>
#include <algorithm>
#include "Intersection.cu"
#include "until.cpp"
#include "graph/graph.cpp"
#include "matching/matching.cpp"
#include "matching/BuildTable.cpp"

#define maxListLength 1000

// main function
int main (int argc, char * argv[]){
	// set default variables  
	int opt = 0;
	std::string input_data_graph_file;
	std::string input_query_graph_file;
	int method = 2;
	ui step = 0;
	ui sample_time = 10;
	int inter_count = 5000;
	std::string output_file = "./output.txt";
	double b = 0.1;
	double alpha = 0.1;
	ui fixednum = 1;
	ui threadnum = sample_time;
	constexpr ui BlockSize = 128;
	constexpr ui WarpSize = 32;
	std::cout << "block size is "<<  BlockSize <<std::endl;
	// a small GPU test
	cudaDeviceSynchronize();
	bool successrun = true;
	auto err = cudaGetLastError();
	if (err != cudaSuccess){
		successrun = false;
		std::cout <<" error, restart GPU! "<<std::endl;
	}else{
		std::cout <<" Pass GPU test "<<std::endl;
	}
	ui orderid = 1;
	const char *optstring = "d:q:m:s:o:t:i:c:e:";
	while ((opt = getopt (argc, argv, optstring))!= -1){
		ParseInputPara(opt, argc, argv,optstring, input_data_graph_file,  input_query_graph_file, method, step, sample_time,  inter_count,output_file, threadnum, orderid);
	}
	// argument check
	ui task_per_thread = sample_time/ threadnum;
	ui taskPerBlock = task_per_thread* BlockSize;
	ui taskPerWarp = task_per_thread* WarpSize;
	std::cout << "taskPerThread: " << task_per_thread <<" taskperBlock: " << taskPerBlock << " taskPerWarp: " << taskPerWarp <<std::endl;
	// load graphs 
	Graph* query_graph = new Graph(true);
	Graph* data_graph = new Graph(true);
	data_graph->loadGraphFromFile(input_data_graph_file);
	query_graph->loadGraphFromFile(input_query_graph_file);
	//buildCoreTable
	query_graph->buildCoreTable();
	std::cout << "-----" << std::endl;
	std::cout << "Query Graph Meta Information" << std::endl;
	query_graph->printGraphMetaData();
	std::cout << "-----" << std::endl;
	data_graph->printGraphMetaData();
	// set step to max step if undefine 
	if(step == 0){
		step = query_graph->getVerticesCount();
	}

    /**
     * Define variables 
     */
	std::cout << "Start queries..." << std::endl;
	std::cout << "-----" << std::endl;
	std::cout << "Filter candidates..." << std::endl;
	ui** candidates = NULL;
	ui* candidates_count = NULL;
	ui* tso_order = NULL;
	TreeNode* tso_tree = NULL;
	ui* cfl_order = NULL;
	TreeNode* cfl_tree = NULL;
	ui* dpiso_order = NULL;
	TreeNode* dpiso_tree = NULL;
	TreeNode* ceci_tree = NULL;
	ui* ceci_order = NULL;
	
	// build candidate graph 
	GQLFilter(data_graph, query_graph, candidates, candidates_count);
	sortCandidates(candidates, candidates_count, query_graph->getVerticesCount());
	auto buildcand_start = std::chrono::high_resolution_clock::now();
	std::cout << "-----" << std::endl;
	std::cout << "Build indices..." << std::endl;
	Edges ***edge_matrix = NULL;
	edge_matrix = new Edges **[query_graph->getVerticesCount()];
	for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
		edge_matrix[i] = new Edges *[query_graph->getVerticesCount()];
	}
	BuildTable::buildTables(data_graph, query_graph, candidates, candidates_count, edge_matrix);
	size_t memory_cost_in_bytes = 0;
	memory_cost_in_bytes = BuildTable::computeMemoryCostInBytes(query_graph, candidates_count, edge_matrix);
	BuildTable::printTableCardinality(query_graph, edge_matrix);

	std::cout << "-----" << std::endl;
	std::cout << "Generate a matching order..." << std::endl;

	ui* matching_order = NULL;
	ui* pivots = NULL;
	ui** weight_array = NULL;

	size_t order_num = 0;

	std::vector<std::vector<ui>> spectrum;
	SelectQueryPlan(data_graph,  query_graph,edge_matrix, matching_order,  pivots,  tso_tree,tso_order, dpiso_tree, dpiso_order,cfl_tree, cfl_order ,spectrum,candidates_count, ceci_tree, ceci_order,weight_array)

	checkQueryPlanCorrectness(query_graph, matching_order, pivots);
	printSimplifiedQueryPlan(query_graph, matching_order);
	std::cout << "-----" << std::endl;
	auto buildcand_end = std::chrono::high_resolution_clock::now();
	std::cout<<"build candidates time: " << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(buildcand_end - buildcand_start).count() /1000000000<< std::endl;
	std::cout << "Initializing..." << std::endl;
	size_t output_limit = std::numeric_limits<size_t>::max();
	size_t embedding_count = 0;
	size_t call_count = 0;
	size_t time_limit = 0;
	timer record;
	record.sample_time = sample_time;
	record.inter_count = inter_count;
	record.b = b;
	record.alpha = alpha;
	record.fixednum = fixednum;
	record.taskPerBlock = taskPerBlock;
	record.taskPerWarp = taskPerWarp;
	record.threadnum = threadnum;
	record.successrun = successrun;

	switch (method){


		case 0:{
			//WanderJoin estimator
			embedding_count = WJ<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																																  matching_order, output_limit, call_count, step,record);
			break;
		}

		case 1:{
			//ALLEY estimator
			embedding_count = AL<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																																  matching_order, output_limit, call_count, step,record);
			
			break;
		}

		case 2:{
			//PartialRefine estimator
			embedding_count = PR<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																													  				matching_order, output_limit, call_count, step,record);
			break;
		}

		case 3:{
			//userDefine estimator
			embedding_count = UD<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																													  				matching_order, output_limit, call_count, step,record);
		
			break;
		}

	
	std::cout << "End... "<< std::endl;
	std::cout <<"Enumerate count: "<< embedding_count << std::endl;
	std::cout <<"Est path count: "<< record.est_path<< std::endl;
	std::cout <<"real workload count: "<< record.real_workload<< std::endl;
	std::cout <<"Est workload count: "<< record.est_workload << std::endl;
	std::cout << "Sampling_cost: " << record.sampling_time/1000000000 << std::endl;
	std::cout << "Enumerating_cost: " << record.enumerating_time/1000000000 << std::endl;
	std::cout << "candiate set cost: " << record.cand_alloc_time/1000000000 << std::endl;
	std::cout <<"call count: "<< call_count << std::endl;


	// reset GPU when exit
	cudaDeviceReset();
	return 0;
}
