#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <utility>

#include "utility.h"
#include <stdio.h>

using std::cout;
using std::endl;





void parseEdgelist(char* eglist_file, int* &nodes, int* &neighborNodes, int* &outWeights, int* &inWeights, int &N){

	std::ifstream eglist_data;
	eglist_data.open(eglist_file);

	std::string line;
	std::stringstream ss;

	int node1, node2, outWeight, inWeight;


	// std::vector<int> nodes_vec;
	// std::vector<int> neighborNodes_vec;
	// std::vector<int> outWeights_vec;
	// std::vector<int> inWeights_vec;


	typedef std::pair<int, int> Nodes_pair;
	std::map<Nodes_pair, int> weight_map;

	std::vector<Nodes_pair> ctn1;


	if (eglist_data.is_open())
	{
		while ( getline (eglist_data, line) ){

			// cout << line << '\n';
	
			ss<<line;
			ss>>node1>>node2;
	
			if(ss.rdbuf()->in_avail() == 0)
				outWeight = 1;
			else
				ss>>outWeight;

			cout << node1 << "," << node2 << "," << outWeight << '\n';

			Nodes_pair temp(node1, node2);
			weight_map[temp] = outWeight;
			ctn1.push_back(temp);
			

			ss.str("");
			ss.clear();
		}
		eglist_data.close();
	}

	N = ctn1.size();
	nodes = new int[N];
	neighborNodes = new int[N];
	outWeights = new int[N];
	inWeights = new int[N];



	int counter = 0;
	for ( auto &pair : ctn1 ) {

		Nodes_pair temp(pair.second, pair.first);


		if ( weight_map.find(temp) == weight_map.end() )
			inWeight = 0;
		else
			inWeight = weight_map[temp];

		nodes[counter] = pair.first;
		neighborNodes[counter] = pair.second;
		outWeights[counter] = weight_map[pair];
		inWeights[counter] = inWeight;
		counter++;

	}


	return;
}



/*  */
void printInfo(int* nodes, int* neighborNodes, int* outWeights, int* inWeights, int N){

	cout<<endl;
	cout<< std::string(50, '=') << endl;
	
	// Print nodes
	cout<< std::setw(12)<< std::left << "Nodes:";
	cout<< std::right;
	for(int i=0; i<N ;i++){
		std::cout<< std::setw(4) << nodes[i];
	}

	// Print neighboring nodes
	cout<<std::endl;
	cout<< std::setw(12)<< std::left << "Neighbors:";
	cout<< std::right;
	for(int i=0; i<N ;i++){
		std::cout<< std::setw(4) << neighborNodes[i];
	}

	// Print outgoing weights
	cout<<std::endl;
	cout<< std::setw(12)<< std::left << "Out weights:";
	cout<< std::right;
	for(int i=0; i<N ;i++){
		std::cout<< std::setw(4) << outWeights[i];
	}

	// Print ingoing weights
	cout<<std::endl;
	cout<< std::setw(12)<< std::left << "In weights:";
	cout<< std::right;
	for(int i=0; i<N ;i++){
		cout<< std::setw(4) << inWeights[i];
	}

	cout<<endl;
	cout<<endl;

	return;
}


