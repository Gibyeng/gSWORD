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

int main (int argc, char * argv[]){
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
	//编译时常量, cub需要
	constexpr ui BlockSize = 128;
	constexpr ui WarpSize = 32;
	std::cout << "block size is "<<  BlockSize <<std::endl;
	//random seed if necessory
//	srand((unsigned)time(NULL));
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
	const char *optstring = "d:q:m:s:o:t:i:b:a:n:c:e:";
	while ((opt = getopt (argc, argv, optstring))!= -1){
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
			case 'b':{
				b = atof(optarg);
				std::cout << "branch var: " << b<<std::endl;
				break;
			}
			case 'a':{
				alpha = atof(optarg);
				std::cout << "alpha var: " <<alpha <<std::endl;
				break;
			}
			case 'n':{
				fixednum = atoi(optarg);
				std::cout << "fixnumber: " <<fixednum <<std::endl;
				break;
			}
//			case 'z':{
//				taskPerBlock = atoi(optarg);
//				std::cout << "taskPerBlock: " <<taskPerBlock <<std::endl;
//				break;
//			}
//			case 'x':{
//				taskPerWarp = atoi(optarg);
//				std::cout << "taskPerWarp: " <<taskPerWarp <<std::endl;
//				break;
//			}
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
	// argument check
	ui task_per_thread = sample_time/ threadnum;
	ui taskPerBlock = task_per_thread* BlockSize;
	ui taskPerWarp = task_per_thread* WarpSize;
	std::cout << "taskPerThread: " << task_per_thread <<" taskperBlock: " << taskPerBlock << " taskPerWarp: " << taskPerWarp <<std::endl;
	/* load graphs */
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
	// ** reset step to max if undefine **/
	if(step == 0){
		step = query_graph->getVerticesCount();
	}

    /**
     * Start queries.
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

	GQLFilter(data_graph, query_graph, candidates, candidates_count);
	sortCandidates(candidates, candidates_count, query_graph->getVerticesCount());
//
	auto buildcand_start = std::chrono::high_resolution_clock::now();
	std::cout << "-----" << std::endl;
	std::cout << "Build indices..." << std::endl;
	Edges ***edge_matrix = NULL;
	edge_matrix = new Edges **[query_graph->getVerticesCount()];
	for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
		edge_matrix[i] = new Edges *[query_graph->getVerticesCount()];
	}
	/*build edge_matrix [node1][node2] -> candidateofnode1*/
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
//	sscanf(input_order_num.c_str(), "%zu", &order_num);
	//select matching order
	// the ordering is 0:QSI 1:GQL 2:TSO 3:CFL 4:DPiso 5:CECI 6:RI 7:VF2PP 8:Spectrum
	std::vector<std::vector<ui>> spectrum;
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

	checkQueryPlanCorrectness(query_graph, matching_order, pivots);
	printSimplifiedQueryPlan(query_graph, matching_order);
	std::cout << "-----" << std::endl;
	auto buildcand_end = std::chrono::high_resolution_clock::now();
	std::cout<<"build candidates time: " << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(buildcand_end - buildcand_start).count() /1000000000<< std::endl;
	std::cout << "Enumerate..." << std::endl;
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
			embedding_count = userDefined<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
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
