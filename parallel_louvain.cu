#include <iostream>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#include "parallel_louvain.h"
#include "louvain_utils.h"

using std::cout;
using std::endl;


bool onePassLouvain(Dec_vec &d_comm_map, Dec_vec d_nodes, Dec_vec d_neighs, Dec_vec d_oWeights, Dec_vec d_iWeights, int n, int m);
int reassign(Dec_vec &vec);
void mergeCommunity(const Dec_vec &d_comm_map, Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights);




/**
	Host function to perform Louvain's method.
*/
void parallelLouvain(Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights){

    // Calculate n and m
    Dec_vec::iterator dev_ptr = thrust::max_element(d_nodes.begin(), d_nodes.end());
    int n1 = *dev_ptr;
    dev_ptr = thrust::max_element(d_neighs.begin(), d_neighs.end());
    int n2 = *dev_ptr;
    int n = std::max(n1, n2)+1;
    int m = thrust::reduce(d_oWeights.begin(), d_oWeights.end()) + thrust::reduce(d_iWeights.begin(), d_iWeights.end());
    cout<<"n = "<<n<<endl;
    cout<<"m = "<<m<<endl;

    // Generate initial partition and current community map
    Dec_vec current_partition(n); 
    thrust::sequence(current_partition.begin(), current_partition.end());
    Dec_vec d_comm_map(current_partition);

    // Start iteration
    int counter=0;
    int current_n = n;
    while(1){

        counter+=1;
        cout<<"======================================"<<endl;
        cout<<"Pass: "<<counter<<endl;

        // Get the partition generated by this pass.
        bool is_terminate = onePassLouvain(d_comm_map, d_nodes, d_neighs, d_oWeights, d_iWeights, n, m);
        if(is_terminate)
            break;

		// Reassign community map
		n = reassign(d_comm_map);
        if(n==current_n)
            break;
        else
            current_n = n;
        // Convert to this partition
        convertIDToCommunity(d_comm_map, current_partition);

        // Generate data for next round
        mergeCommunity(d_comm_map, d_nodes, d_neighs, d_oWeights, d_iWeights);
        d_comm_map.resize(n);
        thrust::sequence(d_comm_map.begin(), d_comm_map.end());

        

    }
    cout<<"Final partition: ";
    thrust::copy(current_partition.begin(), current_partition.end(), std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;

    return;
}



void mergeCommunity(const Dec_vec &d_comm_map, Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights){
    convertIDToCommunity(d_comm_map, d_nodes);
    convertIDToCommunity(d_comm_map, d_neighs);
    sortByFirstTwo(d_nodes, d_neighs, d_oWeights, d_iWeights);
    reduceByFirstTwo(d_nodes, d_neighs, d_oWeights, d_iWeights);
    return;
}




//////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////
bool oneIterLouvain(Dec_vec &d_comm_map, Dec_vec d_nodes, Dec_vec d_neighs, Dec_vec d_oWeights, Dec_vec d_iWeights
        , const Dec_vec &d_node_oWeights, const Dec_vec &d_node_iWeights, int m, float &sum_gain);
void computeModularityGain(thrust::device_vector<float> &d_mod_gains 
    , const Dec_vec &d_nodes, const Dec_vec &d_neighs, const Dec_vec &d_oWeights, const Dec_vec &d_iWeights
    , const Dec_vec &d_map, const Dec_vec &d_nodeOWeights, const Dec_vec &d_nodeIWeights
    , const Dec_vec &d_commOWeights, const Dec_vec &d_commIWeights, const int m);
bool assignNewCommunity(Dec_vec &d_comm_map, Dec_vec &d_nodes, Dec_vec &d_neighs, thrust::device_vector<float> &d_mod_gains, float &sum_gain);
void calculateCommunityWeights(Dec_vec  &d_comm_oWeights, Dec_vec  &d_comm_iWeights
        , const Dec_vec &d_comm_map, Dec_vec d_nodes, const Dec_vec &d_oWeights, const Dec_vec &d_iWeights);
void FNC(const Dec_vec &d_comm_map, Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights);
bool FBM(Dec_vec &d_nodes, Dec_vec &d_neighs, const Dec_vec &d_oWeights, const Dec_vec &d_iWeights
    , Dec_vec &d_comm_map, const Dec_vec &d_node_oWeights, const Dec_vec &d_node_iWeights
    , const Dec_vec &d_comm_oWeights, const Dec_vec &d_comm_iWeights, const int m, float &sum_gain);

/*

*/
bool onePassLouvain(Dec_vec &d_comm_map, Dec_vec d_nodes, Dec_vec d_neighs, Dec_vec d_oWeights, Dec_vec d_iWeights, int n, int m){

    // Calculate node weights since node weights would change with every pass
	Dec_vec d_node_oWeights(n+1);
	Dec_vec d_node_iWeights(n+1);
	calculateWeights(d_node_oWeights, d_nodes, d_oWeights);
	calculateWeights(d_node_iWeights, d_nodes, d_iWeights);

    // Start iteration of every pass, if counter=1, the while process should be ended.
    int counter = 0;
    bool is_assign;
    float sum_gain = 0;
    do{
        counter+=1;
        cout<<endl;
        cout<<"   Iteration: "<<counter<<endl;

        is_assign = oneIterLouvain(d_comm_map, d_nodes, d_neighs, d_oWeights, d_iWeights, d_node_oWeights, d_node_iWeights, m, sum_gain);

    }while(is_assign);

    return (counter==1);
}


bool oneIterLouvain(Dec_vec &d_comm_map, Dec_vec d_nodes, Dec_vec d_neighs, Dec_vec d_oWeights, Dec_vec d_iWeights
        , const Dec_vec &d_node_oWeights, const Dec_vec &d_node_iWeights, int m, float &sum_gain){


    // Find Neighboring Communities
	FNC(d_comm_map, d_nodes, d_neighs, d_oWeights, d_iWeights);


    // Get the max number of community and calculate community weights
    Dec_vec::iterator dev_ptr = thrust::max_element(d_comm_map.begin(), d_comm_map.end());
    int max_comm = *dev_ptr;
    Dec_vec d_comm_oWeights(max_comm+1);
    Dec_vec d_comm_iWeights(max_comm+1);
    calculateCommunityWeights(d_comm_oWeights, d_comm_iWeights, d_comm_map, d_nodes, d_oWeights, d_iWeights);
    
    // Find Best Move
	bool is_assign = FBM(d_nodes, d_neighs, d_oWeights, d_iWeights
    					, d_comm_map, d_node_oWeights, d_node_iWeights, d_comm_oWeights, d_comm_iWeights, m, sum_gain);

    // cout<< is_assign<<endl;
    return is_assign;
}


void FNC(const Dec_vec &d_comm_map, Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights){

	convertIDToCommunity(d_comm_map, d_neighs);
    sortByFirstTwo(d_nodes, d_neighs, d_oWeights, d_iWeights);
    reduceByFirstTwo(d_nodes, d_neighs, d_oWeights, d_iWeights);
	return;
}

bool FBM(Dec_vec &d_nodes, Dec_vec &d_neighs, const Dec_vec &d_oWeights, const Dec_vec &d_iWeights
    , Dec_vec &d_comm_map, const Dec_vec &d_node_oWeights, const Dec_vec &d_node_iWeights
    , const Dec_vec &d_comm_oWeights, const Dec_vec &d_comm_iWeights, const int m, float &sum_gain){

    // Allocate a vector to store modularity gains
    thrust::device_vector<float> d_mod_gains(d_nodes.size());

    computeModularityGain(d_mod_gains, d_nodes, d_neighs, d_oWeights, d_iWeights
    , d_comm_map, d_node_oWeights, d_node_iWeights, d_comm_oWeights, d_comm_iWeights, m);
	
	// // For debug
    cout<<"      Nodes: ";
    thrust::copy(d_nodes.begin(), d_nodes.end(), std::ostream_iterator<float>(std::cout," "));
    cout<<endl;
    cout<<"      Neigh: ";
    thrust::copy(d_neighs.begin(), d_neighs.end(), std::ostream_iterator<float>(std::cout," "));
    cout<<endl;
    cout<<"      Gains: ";
    thrust::copy(d_mod_gains.begin(), d_mod_gains.end(), std::ostream_iterator<float>(std::cout," "));
    cout<<endl;


    bool is_assign = assignNewCommunity(d_comm_map, d_nodes, d_neighs, d_mod_gains, sum_gain);
    cout<<"      is_assign:"<<is_assign<<endl;
    // For debug
    cout<<"      dmap: ";
    thrust::copy(d_comm_map.begin(), d_comm_map.end(), std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;
    return is_assign;
}



void calculateCommunityWeights(Dec_vec  &d_comm_oWeights, Dec_vec  &d_comm_iWeights
        , const Dec_vec &d_comm_map, Dec_vec d_nodes, const Dec_vec &d_oWeights, const Dec_vec &d_iWeights){

    convertIDToCommunity(d_comm_map, d_nodes);
	calculateWeights(d_comm_oWeights, d_nodes, d_oWeights);
	calculateWeights(d_comm_iWeights, d_nodes, d_iWeights);

    return;
}


/*
*/
void computeModularityGain(thrust::device_vector<float> &d_mod_gains 
    , const Dec_vec &d_nodes, const Dec_vec &d_neighs, const Dec_vec &d_oWeights, const Dec_vec &d_iWeights
    , const Dec_vec &d_map, const Dec_vec &d_nodeOWeights, const Dec_vec &d_nodeIWeights
    , const Dec_vec &d_commOWeights, const Dec_vec &d_commIWeights, const int m){

    int length = d_mod_gains.size();

    // Calculate first part: kic + kci
    thrust::transform(d_oWeights.begin(), d_oWeights.begin()+length, d_iWeights.begin(), d_mod_gains.begin(), thrust::plus<int>());

    // Calculate second part: kiout*kcin + kiin*kcout
    thrust::device_vector<float> tmp(length);
    const int* d_nOWeights_ptr = thrust::raw_pointer_cast(&d_nodeOWeights[0]);
    const int* d_nIWeights_ptr = thrust::raw_pointer_cast(&d_nodeIWeights[0]);
    const int* d_cOWeights_ptr = thrust::raw_pointer_cast(&d_commOWeights[0]);
    const int* d_cIWeights_ptr = thrust::raw_pointer_cast(&d_commIWeights[0]);

    auto ff = [=]  __device__ (int node, int nei_comm) {
        float result = d_nOWeights_ptr[node]*d_cIWeights_ptr[nei_comm] 
                        + d_nIWeights_ptr[node]*d_cOWeights_ptr[nei_comm];
        result = result/m;
        return result;
    };
    thrust::transform(d_nodes.begin(), d_nodes.begin()+length, d_neighs.begin(), tmp.begin(), ff);

    // Calculate first part + second part
    auto ff2 = [=]  __device__ (float first, float second) {  return (first-second)/m; };
    thrust::transform(d_mod_gains.begin(), d_mod_gains.begin()+length, tmp.begin(), d_mod_gains.begin(), ff2);
    tmp.clear();
    thrust::device_vector<float>().swap(tmp);

    // Calculate status map to indicate whether node is in the same community.
    Dec_vec status_map(length);
    const int* d_map_ptr = thrust::raw_pointer_cast(&d_map[0]);
    auto ff3 = [=]  __device__ (int node, int nei_comm) {  return d_map_ptr[node]!=nei_comm; };
    thrust::transform(d_nodes.begin(), d_nodes.begin()+length, d_neighs.begin(), status_map.begin(), ff3);


    // Calculate final result by specifying the node which is in the same community as its neighbor to zero.
    thrust::transform(d_mod_gains.begin(), d_mod_gains.begin()+length, status_map.begin(), d_mod_gains.begin(), thrust::multiplies<float>());

    return;

}


bool assignNewCommunity(Dec_vec &d_comm_map, Dec_vec &d_nodes, Dec_vec &d_neighs, thrust::device_vector<float> &d_mod_gains
    , float &sum_gain){

	sortByFirstTwo3(d_nodes, d_mod_gains, d_neighs);
    thrust::pair<Dec_vec::iterator, thrust::device_vector<float>::iterator> new_end1;
    thrust::pair<Dec_vec::iterator,Dec_vec::iterator> new_end2;


    Dec_vec tmp(d_nodes.size());
    new_end1 = thrust::unique_by_key_copy(d_nodes.begin(), d_nodes.end(), d_mod_gains.begin(), tmp.begin(), d_mod_gains.begin());

    int new_length = new_end1.first - tmp.begin();
    float sum = thrust::reduce(d_mod_gains.begin(), d_mod_gains.begin() + new_length);
    cout<<"      Iter gain: "<<sum<<endl;
    if(sum<=0 or sum<=sum_gain)
        return false;
    else{
        sum_gain = sum;
        new_end2 = thrust::unique_by_key_copy(d_nodes.begin(), d_nodes.end(), d_neighs.begin(), d_nodes.begin(), d_comm_map.begin());
        return true;
    }
    
}