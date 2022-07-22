# GSWORD
A light GPU system for Subgraph counting problem

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

Running code in GSWORD
--------
Running code is also done within the Root/ directory. 
Use "./build/gsword -d DataGraph -q QueryGraph -m method -s NumberOfQueryVetice" to estimate the count of QueryGraph in DataGraph.
if "method" is set to 1, we estimate the count by wanderJoin
if "method" is set to 2, we estimate the count by Alley
if "method" is set to 3, we estimate the count by PartialRefine

We also provide mpre advanced arguments for experienced users. 
-t NumberOfSamples,  -c NumberOfThreads, -e PruningStrategy

If you want to implement your own version of LP.
Please overwrite following APIs provide by us.

Examples
```
$ ./gsword -d datagraph.graph -q query.graph -m 1 -s 16
or
-d ./gsword -d datagraph.graph -q query.graph -m 1 -s 16 -t 128000 -c 5120 -e 6
```

Input Format for GLP
--------
 Graph starts with 't VertexNum EdgeNum' where VertexNum is the number of vertices and EdgeNum is the number of edges. Each vertex is represented as 'v VertexID LabelId Degree' and 'e VertexId VertexId' for edges. We give an input example in the following.

AdjacencyGraph
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

User-defined APIs
--------
**int Refine(Sample* s, int d, Vertex* cand, int clen,Vertex* refine)**: Given a sample s with sample iteration d, a candidate array cand with length clen, it fills a refined
candidate array refine and returns the length of refine.
**pair<Vertex, double> Sample(Sample* s, int d, Vertex* refine, int rlen)**:Given a sample s with sample iteration d, a refined candidate array refine with length rlen, it samples a vertex v from refine. v is returned with its sampling probability/weight.
**bool Validate(Sample* s, int d, Vertex v, double prob)**:Given a sample s with sample iteration d, if s remains valid after adding v as the dth vertex, the function returns true and updates the sampling probability of s given prob, the probability of sampling v. Otherwise, it returns false to indict invalid sample.
you can overwrite those APIs in file Root/header/kernel.cuh.