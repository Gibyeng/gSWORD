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
			/*all CPU matching*/
			std::cout << "CPU enmu: "<<std::endl;
			embedding_count = LFTJ(data_graph, query_graph, edge_matrix, candidates, candidates_count,
	                                              matching_order, output_limit, call_count, record);
			break;
		}

		case 1:{
			/*GPU matching */

			std::cout << "GPU enmu: "<<std::endl;
			embedding_count = LFTJ_GPU(data_graph, query_graph, edge_matrix, candidates, candidates_count,
				                                              matching_order, output_limit, call_count);
			break;
		}
		case 2:{

			std::cout << "CPU enmu: "<<std::endl;
			embedding_count = LFTJ_uniform(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																		  matching_order, output_limit, call_count, step,record);

			break;
		}
		case 3:{

			std::cout << "GPU enmu: "<<std::endl;
			embedding_count = LFTJ_WJ(data_graph, query_graph, edge_matrix, candidates, candidates_count,
															  matching_order, output_limit, call_count, step,record);
			break;
		}
		/*need tp fix bugs*/
		case 4:{
			/*wandor join round rubin*/
			/*has bugs*/
			std::cout << "CPU enmu: "<<std::endl;
//			embedding_count = LFTJ_WJ_roundrubin(data_graph, query_graph, edge_matrix, candidates, candidates_count,
//															  matching_order, output_limit, call_count, step,record);
			embedding_count = LFTJ_partialSampling(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																		  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 5:{
			/*take aver*/
			/*multi- sample at least 50 times*/
			std::cout << "GPU enmu: "<<std::endl;
			embedding_count = LFTJ_adaptive_multi_v2(data_graph, query_graph, edge_matrix, candidates, candidates_count,
															  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 6:{
			/*take --- */
			/*0,001 update*/
			std::cout << "GPU enmu: "<<std::endl;
			embedding_count = LFTJ_adaptive_alpha(data_graph, query_graph, edge_matrix, candidates, candidates_count,
															  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 7:{
			/*take --- */

			std::cout << "GPU enmu: "<<std::endl;
			embedding_count = LFTJ_adaptive_rank(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																		  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 8:{
			/*take --- */
			/*branch*/
			std::cout << "GPU enmu: "<<std::endl;
			embedding_count = LFTJ_branch(data_graph, query_graph, edge_matrix, candidates, candidates_count,
															  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 9:{
			/*take --- */
			std::cout << "GPU enmu: "<<std::endl;
				embedding_count = LFTJ_combine(data_graph, query_graph, edge_matrix, candidates, candidates_count,
			  matching_order, output_limit, call_count, step,record);
			break;
		}

		case 10:{
			/*take --- */
			/*branch*/
			std::cout << "GPU enmu: "<<std::endl;
			embedding_count = LFTJ_fixedbranch(data_graph, query_graph, edge_matrix, candidates, candidates_count,
															  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 11:{
			std::cout << "GPU enmu: "<<std::endl;
			embedding_count = LFTJ_half(data_graph, query_graph, edge_matrix, candidates, candidates_count,
															  matching_order, output_limit, call_count, step,record);
			break;
		}
		// all in GPU memory.
		case 12: {

			embedding_count = LFTJ_GPU_all(data_graph, query_graph, edge_matrix, candidates, candidates_count,
							                                              matching_order, output_limit, call_count, step,record);
			break;
		}

		case 13: {

			embedding_count = LFTJ_GPU_warp(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																		  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 14: {

			embedding_count = LFTJ_GPU_warp_lessmem(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																		  matching_order, output_limit, call_count, step,record);
			break;
		}

		case 15: {

			embedding_count = blockPathBalance<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																		  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 16: {

			embedding_count = blockLayerBalance<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																		  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 17: {

			embedding_count = warpPathBalance<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																	  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 18: {

			embedding_count = warpLayerBalance<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																	  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 19:{
			embedding_count = test_intersection_count<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																				  matching_order, output_limit, call_count, step,record);
					break;
		}
		case 20:{
			embedding_count = blockPathBalanceLessmem<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																				  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 21:{
			embedding_count = blockPathBalanceLessmemV2<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																				  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 22:{
			embedding_count = blockPathBalanceLessmemV3<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																				  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 23:{
			embedding_count = blockPathBalanceLessmemV4<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																				  matching_order, output_limit, call_count, step,record);
			break;
		}
		//
		case 24:{
			embedding_count = help<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																				  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 25:{
			embedding_count = help_Baseline_pathsync<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																				  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 26:{
			embedding_count = help_Baseline_layersync<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																				  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 27:{
			embedding_count = GPUWandorJoin<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																				  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 31: {
			/* test if we assign prefect possiblity...*/
			ui cand_len = candidates_count[matching_order[0]];
			size_t* subtree = new size_t [cand_len];
			std::memset (subtree,0,sizeof(size_t)*cand_len);
			embedding_count = LFTJ_subtreesize(data_graph, query_graph, edge_matrix, candidates, candidates_count,
							                                              matching_order, output_limit, call_count, subtree);
			/* do sampling according to perfect P "subtree"*/
			LFTJ_perfectP(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																		  matching_order, output_limit, call_count, step,record,subtree);
			/*write the sampling possibity first 20 to a file */
			ofstream out;
			out.open("./p_ac.txt", std::ios_base::app);
			double sum = 0;
			for (int i = 0; i< cand_len; i++){
				sum += subtree[i];
			}
			for (int i = 0; i< 20; i++){
				int space = (cand_len-1) / 20 + 1;
				double poss = 0;
				for (int j = 0 ; j< space; j++){
					int pos = i*space + j;
					poss +=  subtree[pos];
				}
				out << i << " " << poss/sum << "\n";
			}
			out.close();
			break;
		}
		//experiment methods
		//
		case 40:{
			//baseline1
			// simple wanderjoin without intersection
			embedding_count = GPUWandorJoin<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																							  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 41:{
			//baseline2
			// GPU one path sampling with intermediate result
			embedding_count = blockPathBalance<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																					  matching_order, output_limit, call_count, step,record);
			break;

		}
		case 42:{
			//baseline3 the results are underestimated for unknown reason.
			// width = k% nodes
			// GPU one path sampling with intermediate result
			embedding_count = branchJoin<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																					  matching_order, output_limit, call_count, step,record);
			break;

		}
		case 43:{
			//opt 1 use less GPU memory
			// GPU one path sampling without intermediate result
			embedding_count = blockPathBalanceLessmem<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																							  matching_order, output_limit, call_count, step,record);
			break;

		}
		case 44:{
			//opt 2 use partial intersection with op1
			embedding_count = blockPathBalanceLessmemV4<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																							  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 45:{
			//opt 3 thread help each other.with op1 but NO op2
			embedding_count = help<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																							  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 46:{
			//opt 3 thread help each other.with op1 & op2
			embedding_count = helpplus<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																							  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 47:{
			// keep paths the same case 45
			embedding_count = helpIndependent<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																							  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 48:{
			// keep paths the same case 46
			embedding_count = helpIndependentplus<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																							  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 49:{
			// use auto tune
		embedding_count = helpIndepentauto<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																							  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 50:{
			/*all CPU matching*/
			std::cout << "CPU enmu: "<<std::endl;
			embedding_count = LFTJ_waitingtime(data_graph, query_graph, edge_matrix, candidates, candidates_count,
												  matching_order, output_limit, call_count, record);
			break;
		}
		// try different sample methods for partially intersection
		case 51:{
			embedding_count = helpplusres<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																										  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 52:{
			embedding_count = helpIndependentplusres<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																									  matching_order, output_limit, call_count, step,record);
			break;
		}
		// final version
		case 53:{
			embedding_count = helpplusadaptratio<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																										  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 54:{
			embedding_count = testratio<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																										  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 55:{
			// print running time of random walks without intersection check , then print candidate graph to a file.
			embedding_count = outputCandidateSet<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																										  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 56:{
			// print running time of random walks without intersection check , then print candidate graph to a file.
			embedding_count = outputworkloads<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																										  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 63:{
			//GPU pr
			embedding_count = partialIntersection<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																													  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 65:{
			//GGE pr
			embedding_count = partialIntersectionMS<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																													  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 59:{
			// GGE wj
			embedding_count = helpWJ<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																																  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 60:{
			// GGE balance workload within a warp
			embedding_count = GGERSAL<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																																  matching_order, output_limit, call_count, step,record);
			break;
		}

		case 61:{
			// alley 开合作 无warp内优化
			embedding_count = GGECOAL<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																																			  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 62:{
			// alley 全开
			embedding_count = GGERSCOAL<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																																  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 64:{
			// PR 开合作 无warp内优化
			embedding_count = GGECOPR<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,																															  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 66:{
			// PR 全开
			embedding_count = GGERSCOPR<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,																															  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 67:{

			embedding_count = blockLayerBalanceWJ<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																					  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 68:{

			embedding_count = blockLayerBalancePI<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
					  matching_order, output_limit, call_count, step,record);
			break;
		}
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

	//write files
	ofstream out;
	if(method != 54){
	out.open(output_file, std::ios_base::app);
	}
	if (out.is_open())
	{
		//header add header manally
//		out << "data file\t" <<"query file\t"<< "enumerate count\t"<< "step\t"<< "recursion count\t" << "sampling cost\t" << "reorder cost\t"<< "enumerate cost" <<std::endl;
		// data_file
		std::size_t found = input_data_graph_file.find_last_of("/\\");
		out << input_data_graph_file.substr(found+1) << "\t";
		//query file
		found = input_query_graph_file.find_last_of("/\\");
		out << input_query_graph_file.substr(found+1) << "\t";
		// step
		out<< step << "\t";
//		//call count
//		out << call_count << "\t";
		// 2nd -  sampling -t
		out << sample_time <<"\t";
		out << taskPerBlock <<"\t";
		out << taskPerWarp <<"\t";
		// inter  -i
//		out << record.total_sample_count <<"\t";
		// b
//		out << record.b <<"\t";
		// a
//		out << record.alpha <<"\t";
		//enumerate count
		out << embedding_count<< "\t";
		// est_emb
		out << record.est_path<< "\t";
		//real_workload
//		out << record.real_workload<< "\t";
		// est_workload
//		out << record.est_workload<< "\t";
		// number of set intersections
//		out << record.set_intersection_count << "\t";
		// total_compare
//		out << record.total_compare << "\t";
		// total paths
//		out << record.total_path << "\t";
		// Q error ratio of emb
		size_t emb_c = embedding_count;
		size_t est_c = record.est_path;

		if(emb_c == 0){
			emb_c = 1;
		}
		if(est_c == 0){
			est_c = 1;
		}
		double qerr = (double) emb_c / est_c;
		if(qerr < 1){
			qerr = 1/qerr;
		}
		// overestimate or underestimate
		if(embedding_count > record.est_path){
			out << "-";
		}else{
			out << "+";
		}

//		out << (double)abs((long long)embedding_count -  (long long)record.est_path)/embedding_count<< "\t";

		out << qerr<< "\t";
		out << record.cand_alloc_time/1000000000 << "\t";
		// simpling cost_ by_GPU
		out << record.sampling_time/1000000000 << "\t";
//		// reorder cost_ by_GPU
//		out << record.reorder_time/1000000000 << "\t";
		// enumerating cost
		out << record.enumerating_time/1000000000 << "\t";
		// if gpu run successfully
		out << record.successrun << "\t";

		// 》 64
		if(method == 19){
			out << record.arr_range_count[0] << "\t";
			out << record.arr_range_count[1] << "\t";
			out << record.arr_range_count[2] << "\t";
			out << record.arr_range_count[3] << "\t";
			out << record.arr_range_count[4] << "\t";
		}

		out << std::endl;
		out.close();
	}
	if (method == 53){
		ofstream out;
		std::string partialinter_output_file = "ratio_"+output_file;
		out.open(partialinter_output_file, std::ios_base::app);
		if (out.is_open())
		{
			std::size_t found = input_data_graph_file.find_last_of("/\\");
			out << input_data_graph_file.substr(found+1) << "\t";
			//query file
			found = input_query_graph_file.find_last_of("/\\");
			out << input_query_graph_file.substr(found+1) << "\t";
			// step
			out<< step << "\t";
			// ratio
			out<<record.full_ratio << "\t";
			out<< record.base_ratio << "\t";
			out<< record.adapt_ratio << "\t";
			out<< record.sample_ratio << "\t";
		}
		out << std::endl;
		out.close();
	}
	if (method == 54){
		ofstream out;
		// find ".txt"
//		std::string outname =  output_file.substr(0,output_file.find("."));
//		std::string partialinter_output_file = outname+"_sr"+".txt";
		out.open(output_file, std::ios_base::app);
		if (out.is_open())
		{
			std::size_t found = input_data_graph_file.find_last_of("/\\");
			out << input_data_graph_file.substr(found+1) << "\t";
			//query file
			found = input_query_graph_file.find_last_of("/\\");
			out << input_query_graph_file.substr(found+1) << "\t";
			// step
			out<< step << "\t";
			// msg
			out << record.msg_time;
		}
		out << std::endl;
		out.close();
	}
	//
	if (method == 55){
		ofstream out;
		ofstream out2;
		// find ".txt"
		std::string outname =  output_file.substr(0,output_file.find(".txt"));
		std::string partialinter_output_file = outname+"_cg"+".graph";
		std::string partialinter_output_file2 = outname+"_cg"+".txt";
		out.open(partialinter_output_file, std::ios_base::app| std::ios :: binary );
		out2.open(partialinter_output_file2, std::ios_base::app);
		if (out.is_open() && out2.is_open())
		{
			//write edge list to bin, src dist weight
			std::cout <<"cset write to file " << partialinter_output_file <<endl;
			int data[3];
			data[2] = 1;
			ui query_vertices_num = step;
			for (ui i = 0; i < query_vertices_num; ++i) {
			        VertexID begin_vertex = i;
			        for (ui j = i + 1; j < query_vertices_num; ++j) {
			            VertexID end_vertex = j;
			            if (query_graph->checkEdgeExistence(begin_vertex, end_vertex)) {
			            	 ui cardinality = (*edge_matrix[begin_vertex][end_vertex]).edge_count_;
			            	 for (ui k = 0; k < candidates_count[j]; ++k) {
								 VertexID v = candidates[j][k];
								 auto edge_start = (*edge_matrix[begin_vertex][end_vertex]).offset_[k];
								 auto edge_end = (*edge_matrix[begin_vertex][end_vertex]).offset_[k+1];
								 for (auto m = edge_start; m< edge_end; ++m){
									 VertexID u_idx = (*edge_matrix[begin_vertex][end_vertex]).edge_[m];
									 VertexID u = candidates[i][u_idx];
									 data[0] = u;
									 data[1] = v;
									 if(u < 100000000 && v < 100000000){
										 out.write(reinterpret_cast<const char *>(data), 4*3);
										 out2<< u <<" "<<v<<std::endl;
									 }

			            		 }
			            	 }
			            }

			        }
			}

		}else{
			std::cout <<"fail to open" << partialinter_output_file <<endl;
		}
		out.close();
	}
	if (method == 56){
			ofstream out;
			// find ".txt"
			std::string outname =  output_file.substr(0,output_file.find(".txt"));
			std::string output_file = outname+"_workload"+".txt";
			out.open(output_file, std::ios_base::app);
			if (out.is_open())
			{
				std::size_t found = input_data_graph_file.find_last_of("/\\");
				out << input_data_graph_file.substr(found+1) << "\t";
				//query file
				found = input_query_graph_file.find_last_of("/\\");
				out << input_query_graph_file.substr(found+1) << "\t";
				// workload
				out<< record.est_workload << "\t";
				// intersection
				out << record.set_intersection_count;
			}
			out << std::endl;
			out.close();
		}
	// reset GPU when exit
	cudaDeviceReset();
	return 0;
}
