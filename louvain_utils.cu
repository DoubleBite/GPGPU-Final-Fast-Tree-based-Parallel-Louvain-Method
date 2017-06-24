#include <thrust/fill.h>

#include <thrust/execution_policy.h>


void calculateWeights(Dec_vec & d_weight_map, Dec_vec d_nodeID, Dec_vec d_weight_vec){

    // Reset weights to zero
    thrust::fill(thrust::device, d_weight_map.begin(), d_weight_map.end(), 0);
    
    // Sort and reduce the weights according to ID
    thrust::stable_sort_by_key(d_nodeID.begin(), d_nodeID.end(), d_weight_vec.begin());
    
    thrust::pair<thrust::device_vector<Tuple>::iterator,Dec_vec::iterator> new_end;
    new_end = thrust::reduce_by_key(keys_input.begin(), keys_input.end(), d_iWeights.begin(), tmp.begin(), d_iWeights.begin()



    return;
}