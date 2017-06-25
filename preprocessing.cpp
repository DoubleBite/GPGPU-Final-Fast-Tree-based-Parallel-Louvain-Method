#include <fstream>
#include <sstream>

#include "preprocessing.h"


void parseEdgelist(char* eglist_file, Host_vec &h_nodes, Host_vec &h_neighs, Host_vec &h_oWeights, Host_vec &h_iWeights){

	std::ifstream eglist_data;
	eglist_data.open(eglist_file);
	
    std::string line;
	std::stringstream ss;
    int node1, node2, out_weight;

	if (eglist_data.is_open())
	{
		while ( getline (eglist_data, line) ){

			// cout << line << '\n';
	
			ss<<line;
			ss>>node1>>node2;
	
			if(ss.rdbuf()->in_avail() == 0)
				out_weight = 1;
			else
				ss>>out_weight;

            // // Print edgelist lines
			// cout << node1 << "," << node2 << "," << outWeight << '\n';

            h_nodes.push_back(node1);
            h_neighs.push_back(node2);
            h_oWeights.push_back(out_weight);
            h_iWeights.push_back(0);

            h_nodes.push_back(node2);
            h_neighs.push_back(node1);
            h_oWeights.push_back(0);
            h_iWeights.push_back(out_weight);
			
            // Reset ss 
			ss.str("");
			ss.clear();
		}
		eglist_data.close();
	}

	return;
}
