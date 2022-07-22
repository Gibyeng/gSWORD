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
template < ui threadsPerBlock>
__global__  ui Refine (Sample* s,ui d, ui cand, ui clen, ui* refine){
refine = cand;
return clen;
}

template < ui threadsPerBlock>
__global__  pair<ui,double> Sample (Sample* s,ui d, ui cand, ui clen, ui* refine){
refine = cand;
return clen;
}

template < ui threadsPerBlock>
__global__  bool <ui,double> Validate (Sample* s,ui d, ui cand, ui clen, ui* refine){
	if(DupCheck(s,v)){
		Vertex u = getQueryVertex(cg, d);
		/*Get edges from cg*/
		for (Vertex u’ : getEdges(cg,u)){
			int k = GetOrderIndex(cg,u’);
			Vertex v’ = s.ins[k];
		/*check if (v’,v) exists*/
		if(!IsEdge(v’, v ,cg)){
			return false;
		}}
		s.prob = s.prob* prob;
		s.ins[d] = v;
		return true;
}}
