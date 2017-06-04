#include <iostream>
#include <string>
#include <vector>

#include "utility.h"

using std::cout;
using std::endl;




int main(int argc, char* argv[]){
  

	char *edgelist_file = argv[1];
	int *nodes, *neighborNodes, *outWeights, *inWeights;
	int n;

	// Initialize 
	parseEdgelist(edgelist_file, nodes, neighborNodes, outWeights, inWeights, n);
	printInfo(nodes, neighborNodes, outWeights, inWeights, n);


	return 0;
}
