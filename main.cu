#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>


using std::cout;
using std::endl;

typedef thrust::device_vector<int> Dec_vec;




int main(int argc, char* argv[]){
    
    // Test data on host
    std::vector<int> h_nodes = {0,0,0,0,1,1,2,2,2,3,3,3,3,3,3,4,4,4};
    std::vector<int> h_neighs = {1,2,3,4,0,3,0,3,5,0,1,2,4,6,11,0,3,5};
    std::vector<int> h_oWeights = {1,1,1,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1};
    std::vector<int> h_iWeights = {0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,1,0,0};

    std::vector<int> h_map = {0,1,0,1,2,2,1,3,3,3,3,3,3};

    // Test data on device
    Dec_vec  d_nodes(h_nodes); 
    Dec_vec  d_neighs(h_neighs); 
    Dec_vec  d_oWeights(h_oWeights); 
    Dec_vec  d_iWeights(h_iWeights);

    Dec_vec  d_comm_map(h_map);


    std::vector<int> h_nOWeights = {2,0,3,5,3,1};
    std::vector<int> h_nIWeights = {2,0,2,6,3,0};

    Dec_vec  d_nodeOWeights(h_nOWeights); 
    Dec_vec  d_nodeIWeights(h_nIWeights); 



    int m =10;

    thrust::copy(d_nodes.begin(), d_nodes.end(), std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;
    thrust::copy(d_neighs.begin(), d_neighs.end(), std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;
    thrust::copy(d_oWeights.begin(), d_oWeights.end(), std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;
    thrust::copy(d_iWeights.begin(), d_iWeights.end(), std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;


    onePassLouvain(d_comm_map, d_nodes, d_neighs, d_oWeights, d_iWeights, d_nodeOWeights, d_nodeIWeights, m);


    thrust::copy(d_nodes.begin(), d_nodes.end(), std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;
    thrust::copy(d_neighs.begin(), d_neighs.end(), std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;
    thrust::copy(d_oWeights.begin(), d_oWeights.end(), std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;
    thrust::copy(d_iWeights.begin(), d_iWeights.end(), std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;

    // thrust::copy(d_mod_gains.begin(), d_mod_gains.end(), std::ostream_iterator<float>(std::cout," "));
    // cout<<endl;




	return 0;
}




















