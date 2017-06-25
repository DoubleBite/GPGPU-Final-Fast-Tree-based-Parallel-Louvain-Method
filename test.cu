#include <vector>

#include "louvain_utils.h"
// #include <thrust/transform.h>
// #include <thrust/copy.h>
// #include <thrust/reduce.h>
// #include <thrust/sort.h>
// #include <thrust/scatter.h>
// #include <thrust/transform.h>
// #include <thrust/sequence.h>
// #include <thrust/unique.h>
// #include <thrust/execution_policy.h>
// #include <thrust/logical.h>
// #include <thrust/partition.h>
// #include <thrust/extrema.h>
// #include <thrust/device_ptr.h>

using std::cout;
using std::endl;
typedef thrust::device_vector<int> Dec_vec;



// void convertIDToCommunity(const Dec_vec  &d_comm_map, Dec_vec  &d_IDs){

//     const int* d_map_ptr = thrust::raw_pointer_cast(&d_comm_map[0]);
//     auto ff = [=]  __device__ (int x) {return d_map_ptr[x];};
//     thrust::transform(d_IDs.begin(), d_IDs.end(), d_IDs.begin(), ff);
//     return;
// }


// int reassign(Dec_vec &vec){

//     Dec_vec tmp_vec(vec);
//     thrust::sort(tmp_vec.begin(),tmp_vec.end());
//     Dec_vec::iterator new_end = thrust::unique(tmp_vec.begin(),tmp_vec.end());
//     tmp_vec.resize(new_end-tmp_vec.begin());
    
    
//     Dec_vec indices(tmp_vec.size()); 
//     thrust::sequence(indices.begin(), indices.end());    

//     Dec_vec::iterator dev_ptr = thrust::max_element(tmp_vec.begin(), tmp_vec.end());
//     int max_comm = *dev_ptr;
//     Dec_vec tmp_mapping(10+1);


//     thrust::scatter(indices.begin(), indices.end(), tmp_vec.begin(), tmp_mapping.begin());
//     convertIDToCommunity(tmp_mapping, vec);

//     return max_comm+1;
// }


int main(void){


    std::vector<int> h_nodes = {5, 3, 0, 5, 1, 2, 4, 3, 1};
    std::vector<int> h_weights = {1, 1, 0, 3, 4, 2, 6, 1, 6};

    Dec_vec d_nodes(h_nodes); 
    Dec_vec d_weights(h_weights); 
    Dec_vec d_weight_map(6);

    convertIDToCommunity(d_nodes, d_weights);

    // // Test calculateWeights
    // calculateWeights(d_weight_map, d_nodes, d_weights);
    thrust::copy(d_weights.begin(), d_weights.end(), std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;



    return 0;
}