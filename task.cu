#include <iostream>
#include <string>
#include <vector>

#include "utility.h"
#include "parallel_louvain.h"

using std::cout;
using std::endl;




int main(int argc, char* argv[]){
  

	char *edgelist_file = argv[1];
	int *nodes, *neighbors, *out_weights, *in_weights;
	int n;

	// To store 


	// Initialize 
	parseEdgelist(edgelist_file, nodes, neighbors, out_weights, in_weights, n);
	printInfo(nodes, neighbors, out_weights, in_weights, n);


	// 
	parallelLouvain(nodes, neighbors, out_weights, in_weights, n);



	return 0;
}
