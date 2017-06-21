#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>

using std::cout;
using std::endl;

typedef thrust::device_vector<int> I_vec;



void computeCommunityWeights(I_vec &d_nodes, I_vec &d_neighs, I_vec &d_oWeights, I_vec &d_iWeights, int vec_length);




int main(int argc, char* argv[]){
    
    // Test data on host
    int vec_length = 18;
    std::vector<int> h_nodes = {0,0,0,0,1,1,2,2,2,3,3,3,3,3,3,4,4,4};
    std::vector<int> h_neighs = {1,2,3,4,0,3,0,3,5,0,1,2,4,6,11,0,3,5};
    std::vector<int> h_oWeights = {1,1,1,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1};
    std::vector<int> h_iWeights = {0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,1,0,0};

    // Test data on device
    I_vec  d_nodes(h_nodes); 
    I_vec  d_neighs(h_neighs); 
    I_vec  d_oWeights(h_oWeights); 
    I_vec  d_iWeights(h_iWeights);


    // Print to check
    // thrust::copy(d_nodes.begin(),d_nodes.end(),std::ostream_iterator<int>(std::cout," "));
    computeCommunityWeights(d_nodes, d_neighs, d_oWeights, d_iWeights, vec_length);
	return 0;
}





void computeCommunityWeights(I_vec &d_nodes, I_vec &d_neighs, I_vec &d_oWeights, I_vec &d_iWeights, int vec_length){

    I_vec  d_comm_oWeights(d_nodes.size()); 
    I_vec  d_comm_iWeights(d_nodes.size()); 
    
    thrust::pair<thrust::device_vector<int>::iterator,thrust::device_vector<int>::iterator> new_end;

    new_end = thrust::reduce_by_key(d_nodes.begin(),d_nodes.end(), d_oWeights.begin(), d_comm_iWeights.begin(), d_comm_oWeights.begin());
    // Compute out-degree weights


    // Compute in-degree weights


    thrust::copy(d_comm_iWeights.begin(),d_comm_iWeights.end(),std::ostream_iterator<int>(std::cout," "));
    return;
}