#include <iostream>

#include "global.h"
#include "preprocessing.h"
#include "parallel_louvain.h"

using std::cout;
using std::endl;





int main(int argc, char* argv[]){
    
    // Parse edgelist end store them in vectors.
	char *edgelist_file = argv[1];
    Host_vec h_nodes;
    Host_vec h_neighs;
    Host_vec h_oWeights;
    Host_vec h_iWeights;
    parseEdgelist(edgelist_file, h_nodes, h_neighs, h_oWeights, h_iWeights);

    // Convert host vectors to device vectors
    Dec_vec  d_nodes(h_nodes); 
    Dec_vec  d_neighs(h_neighs); 
    Dec_vec  d_oWeights(h_oWeights); 
    Dec_vec  d_iWeights(h_iWeights);   


    // Parallel Louvain approach
	parallelLouvain(d_nodes, d_neighs, d_oWeights, d_iWeights);


    // thrust::copy(d_oWeights.begin(), d_oWeights.end(), std::ostream_iterator<int>(std::cout,"  "));
    // cout<<endl;
    // thrust::copy(d_iWeights.begin(), d_iWeights.end(), std::ostream_iterator<int>(std::cout,"  "));
    // cout<<endl;



	return 0;
}




















