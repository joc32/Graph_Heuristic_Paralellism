## Multi-Core CPU Parallelisation of NetworkX Graph Heuristics for Link Prediction

![Alt text](./top.png?raw=true "Shell Output")

Version 1.0

Computation of NetworkX Graph Heuristics for Link Prediction uses one CPU core by default (due to Python's Global Interpreter Lock), which makes their computation on larger graph datasets uncomfortable and very slow.

Splitting the graph into multiple pieces, allocating them to separate CPU processess cuts down the processing to managable time. The multi-core workflow in this setup cuts the processing of the 'BlogCatalog' dataset [1] (N = 10,312, E = 333,983) down to 10 minutes compared to a single-core workflow which ran for 101 minutes.

The repository contains a collection of 9 datasets which were tested and can be used for experiments. [1-9]

![Alt text](./datasets.png?raw=true "Datasets")

### Experimental Setup

The multi-core setup ran on a Vast.ai Linux machine with AMD Epyc 7542 CPU (14 cores used) (Single Core Geekbench 5 Score - 950).
The single-core setup ran on a 2017 Macbook Pro 15' i7-7700HQ (Single Core Geekbench 5 Score - 840).

The dataset of choice was 'BlogCatalog' [1], a static homogeneous graph in node-edge format with 10,312 nodes and 333,983 edges. The chosen heuristic was Resource Allocation Index.

### Available Heuristics

Resource Allocation Index, Jaccard Coefficient, Adamic Adar Index, Preferential Attachment.

### Command to Run
```
python3 compute.py --n_cores 5 --dataset 'path/to/dataset' --plots 'YES' --test_size 0.1 --heur 'JA' --thresh_bin 0.5 --confusion_mat 'NO' 
```

### Requirements

* CPU with multiple cores (depends on the size of the dataset)
* Good Amount of RAM (depends on the size of the dataset)
* Python 3.X
* Homogeneous, Static and Undirected Graph (in Node - Edge list format)
* NetworkX & MultiProcessing library.

### Issues

As the solution uses mutiple-cores, each process has an independent memory space, which can take up a significant amount of RAM.

Another issue is when the graph is too large, the internal pickling module cannot pickle a file larger than N Bytes, which subsequently throws an exception and stops the computation half-way through. In this case, it is possible to sub-sample the graph down to K nodes before the computation and repeat the workflow on the subsampled graph.

### In case you use this code

Please reference this repository if you are using this code in your work. Released under MIT licence.

### References

[1] Reza Zafarani and Huan Liu. Social computing data repository at asu, 2009. URL http://socialcomputing.asu.edu, 2009. \
[2] V Batagelj and A Mrvar. Pajek datasets http://vlado.fmf.uni-lj.si/pub/networks/data/mix. USAir97. net, 2006. \
[3] Mark EJ Newman. Finding community structure in networks using the eigenvectors of matrices. Physical review E, 74(3 :036104, 2006. \
[4] Robert Ackland et al. Mapping the us political blogosphere: Are conservative bloggers more prominent? In BlogTalk Downunder 2005 Conference, Sydney. BlogTalk Downunder 2005 Conference, Sydney, 2005. \
[5] Christian Von Mering, Roland Krause, Berend Snel, Michael Cornell, Stephen G Oliver, Stanley Fields, and Peer Bork. Comparative assessment
of large-scale data sets of protein–protein interactions. Nature, 417(6887):399–403, 2002. \
[6] Duncan J Watts and Steven H Strogatz. Collective dynamics of ‘small-world’ networks. nature, 393(6684):440–442, 1998. \
[7] Neil Spring, Ratul Mahajan, David Wetherall, and Thomas Anderson. Measuring isp topologies with rocketfuel. IEEE/ACM Transactions on networking, 12(1):2–16, 2004. \
[8] Jure Leskovec and Andrej Krevl. Snap datasets: Stanford large network dataset collec- tion, 2014.
