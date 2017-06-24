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
	// printInfo(nodes, neighbors, out_weights, in_weights, n);



    Dec_vec  d_nodes(nodes, nodes+n); 
    Dec_vec  d_neighs(neighbors, neighbors+n); 
    Dec_vec  d_oWeights(out_weights, out_weights+n); 
    Dec_vec  d_iWeights(in_weights, in_weights+n);

	parallelLouvain(d_nodes, d_neighs, d_oWeights, d_iWeights);



	return 0;
}
