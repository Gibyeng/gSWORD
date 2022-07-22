# GSWORD
GSWORD is a GPU framework for subgraph counting problem. It provides RW estimators to estimate the number of subgraphs for a data graph which are isomorphic with a given query graph.

Organization
--------
Code for "gSWORD: GPU-accelerated Sampling for Subgraph Counting"

Download
--------
There is a public tool to download the code from anonymous4open. (https://github.com/ShoufaChen/clone-anonymous4open)

Compilation
--------

Requirements

* CMake &gt;= 2.8
* CUDA environment

Compilation is done within the Root/ directory with CMake. 
Please make sure you have installed CMake software compilation tool.
Configure cmakelist.txt appropriately before you start compile. 
To compile, simple run:

```
$ cmake .
$ make .
```

Running Code in GSWORD
--------
Running code is also done within the Root/ directory. 
Use "./build/gsword -d DataGraph -q QueryGraph -m method -s NumberOfQueryVetice" to estimate the count of QueryGraph in DataGraph.

| Method | code |Description|
|------|------|------|
|WJ|0|the WanderJoin RW Estimator|
|AL|1|the ALLEY Estimator|
|PR|2|the PartialRefine estimator|
|UD|3|the User-defined Estimator|

We also provide mpre advanced arguments for experienced users. 
-t NumberOfSamples,  -c NumberOfThreads, -e MatchOrder

| MatchOrder | code |Description|
|------|------|------|
|QSI|0|the ordering method of QuickSI|
|GQL|1|the ordering method of GraphQL|
|TSO|2|the ordering method of TurboIso|
|CFL|3|the ordering method of CFL|
|DPi|4|the ordering method of DP-iso|
|CECI|5|the ordering method of CECI|
|RI|6|the ordering method of RI|
|VF2|7|the ordering method of VF2++|

If you want to implement your own RW estimators.
Please overwrite following APIs provide by us.

Examples
```
$ ./gsword -d datagraph.graph -q query.graph -m 1 -s 16
or
./gsword -d datagraph.graph -q query.graph -m 1 -s 16 -t 128000 -c 5120 -e 6
```

Input Format for GSWORD
--------
 Graph starts with 't VertexNum EdgeNum' where VertexNum is the number of vertices and EdgeNum is the number of edges. Each vertex is represented as 'v VertexID LabelId Degree' and 'e VertexId VertexId' for edges. We give an input example in the following.

```
t 5 6
v 0 0 2
v 1 1 3
v 2 2 3
v 3 1 2
v 4 2 2
e 0 1
e 0 2
e 1 2
e 1 3
e 2 4
e 3 4
```

User-defined APIs
--------

__int Refine(Sample* s, int d, Vertex* cand, int clen,Vertex* refine)__ : Given a sample s with sample iteration d, a candidate array cand with length clen, it fills a refined candidate array refine and returns the length of refine.

__pair<Vertex, double> Sample(Sample* s, int d, Vertex* refine, int rlen)__ : Given a sample s with sample iteration d, a refined candidate array refine with length rlen, it samples a vertex v from refine. v is returned with its sampling probability/weight.

__bool Validate(Sample* s, int d, Vertex v, double prob)__ :Given a sample s with sample iteration d, if s remains valid after adding v as the dth vertex, the function returns true and updates the sampling probability of s given prob, the probability of sampling v. Otherwise, it returns false to indict invalid sample.

you can overwrite those APIs in file Root/kernel/APIs.cu.