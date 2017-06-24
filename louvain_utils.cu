#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/scatter.h>
#include "louvain_utils.h"


void calculateWeights(Dec_vec & d_weight_map, Dec_vec d_nodeID, Dec_vec d_weights){

    // Reset weights to zero
    thrust::fill(thrust::device, d_weight_map.begin(), d_weight_map.end(), 0);
    
    // Sort and reduce the weights according to ID
    thrust::stable_sort_by_key(d_nodeID.begin(), d_nodeID.end(), d_weights.begin());
    
    thrust::pair<Dec_vec::iterator,Dec_vec::iterator> new_end;
    new_end = thrust::reduce_by_key(d_nodeID.begin(), d_nodeID.end(), d_weights.begin(), d_nodeID.begin(), d_weights.begin());
    int new_length = new_end.first - d_nodeID.begin();

    // Assign the calculated weights to map according to their ID.
    thrust::scatter(d_weights.begin(), d_weights.begin()+new_length, d_nodeID.begin(), d_weight_map.begin());

    return;
}



void convertIDToCommunity(const Dec_vec  &d_comm_map, Dec_vec  &d_IDs){

    const int* d_map_ptr = thrust::raw_pointer_cast(&d_comm_map[0]);
    auto ff = [=]  __device__ (int x) {return d_map_ptr[x];};
    thrust::transform(d_IDs.begin(), d_IDs.end(), d_IDs.begin(), ff);
    return;
}



void sortByFirstTwo(Dec_vec &d_first, Dec_vec &d_second, Dec_vec &d_third, Dec_vec &d_fourth){

    // Initialize an indices vector
    Dec_vec indices(d_first.size()); 
    thrust::sequence(indices.begin(), indices.end());

    Dec_vec tmp_second1(d_second); 
    Dec_vec tmp_second2(d_second); 

    thrust::stable_sort_by_key(tmp_second1.begin(), tmp_second1.end(), indices.begin());
    thrust::stable_sort_by_key(tmp_second2.begin(), tmp_second2.end(), d_first.begin());
    thrust::stable_sort_by_key(d_first.begin(), d_first.end(), indices.begin());

    thrust::gather(indices.begin(), indices.end(), d_second.begin(), d_second.begin());
    thrust::gather(indices.begin(), indices.end(), d_third.begin(), d_third.begin());
    thrust::gather(indices.begin(), indices.end(), d_fourth.begin(), d_fourth.begin());

    return;
}


void sortByFirstTwo3(Dec_vec &d_first, Dec_vec &d_second, Dec_vec &d_third){

    // Initialize an indices vector
    Dec_vec indices(d_first.size()); 
    thrust::sequence(indices.begin(), indices.end());

    Dec_vec tmp_second1(d_second); 
    Dec_vec tmp_second2(d_second); 

    thrust::stable_sort_by_key(tmp_second1.begin(), tmp_second1.end(), indices.begin());
    thrust::stable_sort_by_key(tmp_second2.begin(), tmp_second2.end(), d_first.begin());
    thrust::stable_sort_by_key(d_first.begin(), d_first.end(), indices.begin());

    thrust::gather(indices.begin(), indices.end(), d_second.begin(), d_second.begin());
    thrust::gather(indices.begin(), indices.end(), d_third.begin(), d_third.begin());

    return;
}


void reduceByFirstTwo(Dec_vec &d_first, Dec_vec &d_second, Dec_vec &d_third, Dec_vec &d_fourth){

    // Define the equivalent for Tuple
    typedef thrust::tuple<int,int> Tuple;
    auto BinaryPredicate = [=]  __device__ (const Tuple& lhs, const Tuple& rhs) {
        return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) && (thrust::get<1>(lhs) == thrust::get<1>(rhs));
    };

    int current_length = d_first.size();
    thrust::pair<thrust::device_vector<Tuple>::iterator,Dec_vec::iterator> new_end;
    thrust::device_vector<Tuple> input_keys(current_length);
    thrust::device_vector<Tuple> tmp(current_length); 

    // Combine the first two vector and store them in input_keys
    auto combine = [=]  __device__ (int &first, int &second) {  return thrust::make_tuple(first, second); };
    thrust::transform(d_first.begin(), d_first.end(), d_second.begin(), input_keys.begin(), combine);

    // Sort the third and fourth vector by input keys
    new_end = thrust::reduce_by_key(input_keys.begin(), input_keys.end(), d_third.begin(), tmp.begin(), d_third.begin()
               , BinaryPredicate, thrust::plus<int>() );
    new_end = thrust::reduce_by_key(input_keys.begin(), input_keys.end(), d_fourth.begin(), tmp.begin(), d_fourth.begin()
               , BinaryPredicate, thrust::plus<int>() );

    // Unique the first and second vector by input_keys
    new_end = thrust::unique_by_key_copy(input_keys.begin(), input_keys.end(), d_first.begin(), tmp.begin(), d_first.begin());
    new_end = thrust::unique_by_key_copy(input_keys.begin(), input_keys.end(), d_second.begin(), tmp.begin(), d_second.begin());
    current_length = new_end.first - tmp.begin(); 

    // Resize these four vectors by new length
    d_first.resize(current_length);
    d_second.resize(current_length);
    d_third.resize(current_length);
    d_fourth.resize(current_length);

    return;
}


