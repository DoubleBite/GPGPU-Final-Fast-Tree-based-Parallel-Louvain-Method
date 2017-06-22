#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>


using std::cout;
using std::endl;

typedef thrust::device_vector<int> Dec_vec;

Dec_vec computeCommunityOutWeights(Dec_vec d_nodes, Dec_vec d_oWeights, int comm_size);
Dec_vec computeCommunityInWeights(Dec_vec d_neighs, Dec_vec d_iWeights, int comm_size);


void convertToCommunity(const Dec_vec &d_iWeights, Dec_vec &d_neighs);
void sortByNeighborCommunity(Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights);
int reduceCommunityWeights(Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights);
void computeModularityGain(thrust::device_vector<float> &d_mod_gains 
    , const Dec_vec &d_nodes, const Dec_vec &d_neighs, const Dec_vec &d_oWeights, const Dec_vec &d_iWeights
    , const Dec_vec &d_map, const Dec_vec &d_nWeights, const Dec_vec &d_cWeights);

int main(int argc, char* argv[]){
    
    int comm_size = 17;
    int array_length = 18;

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

    Dec_vec  d_map(h_map);

    convertToCommunity(d_map, d_neighs);
    sortByNeighborCommunity(d_nodes, d_neighs, d_oWeights, d_iWeights);

    // // 
    // Dec_vec d_comm_oWeights = computeCommunityOutWeights(d_nodes, d_oWeights, comm_size);
    // Dec_vec d_comm_iWeights = computeCommunityInWeights(d_neighs, d_iWeights, comm_size);
    thrust::copy(d_nodes.begin(),d_nodes.end(),std::ostream_iterator<int>(std::cout," "));
    cout<<endl;
    thrust::copy(d_neighs.begin(),d_neighs.end(),std::ostream_iterator<int>(std::cout," "));
    cout<<endl;
    thrust::copy(d_oWeights.begin(),d_oWeights.end(),std::ostream_iterator<int>(std::cout," "));
    cout<<endl;
    thrust::copy(d_iWeights.begin(),d_iWeights.end(),std::ostream_iterator<int>(std::cout," "));
    cout<<endl;



    int new_length = reduceCommunityWeights(d_nodes, d_neighs, d_oWeights, d_iWeights);


    thrust::copy(d_nodes.begin(), d_nodes.begin()+new_length, std::ostream_iterator<int>(std::cout," "));
    cout<<endl;
    thrust::copy(d_neighs.begin(), d_neighs.begin()+new_length, std::ostream_iterator<int>(std::cout," "));
    cout<<endl;
    thrust::copy(d_oWeights.begin(), d_oWeights.begin()+new_length, std::ostream_iterator<int>(std::cout," "));
    cout<<endl;
    thrust::copy(d_iWeights.begin(), d_iWeights.begin()+new_length, std::ostream_iterator<int>(std::cout," "));
    cout<<endl;

    ////////////////////////////
    std::vector<int> h_nWeights = {2,0,0,2,3,0,0,3,4,0};
    std::vector<int> h_cWeights = {0,2,2,0,0,3,3,0,0,4};

    Dec_vec  d_nWeights(h_nWeights); 
    Dec_vec  d_cWeights(h_cWeights); 

    thrust::device_vector<float> d_mod_gains(new_length);

    computeModularityGain(d_mod_gains, d_nodes, d_neighs, d_oWeights, d_iWeights, d_map, d_nWeights, d_cWeights);
    thrust::copy(d_mod_gains.begin(), d_mod_gains.end(), std::ostream_iterator<float>(std::cout," "));







	return 0;
}


void convertToCommunity(const Dec_vec  &d_map, Dec_vec  &d_neighs){

    const int* d_map_ptr = thrust::raw_pointer_cast(&d_map[0]);
    auto ff = [=]  __device__ (int x) {return d_map_ptr[x];};
    thrust::transform(d_neighs.begin(),d_neighs.end(),d_neighs.begin(),ff);
    return;
}


int reduceCommunityWeights(Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights){

    int array_length = d_nodes.size();
    Dec_vec tmp(array_length); 
    int new_length;

    thrust::pair<Dec_vec::iterator,Dec_vec::iterator> new_end;
    new_end = thrust::reduce_by_key(d_neighs.begin(), d_neighs.end(), d_oWeights.begin(), tmp.begin(), d_oWeights.begin());
    new_end = thrust::reduce_by_key(d_neighs.begin(), d_neighs.end(), d_iWeights.begin(), tmp.begin(), d_iWeights.begin());
    new_length = new_end.first - tmp.begin(); 
    new_end = thrust::unique_by_key(d_neighs.begin(), d_neighs.end(), d_nodes.begin());


    return new_length;
}


void sortByNeighborCommunity(Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights){

    int array_length = d_nodes.size();
    Dec_vec indices(array_length); 
    thrust::sequence(indices.begin(), indices.end());

    Dec_vec tmp_neighs1(d_neighs); 
    Dec_vec tmp_neighs2(d_neighs); 

    thrust::stable_sort_by_key(tmp_neighs1.begin(),tmp_neighs1.end(),indices.begin());
    thrust::stable_sort_by_key(tmp_neighs2.begin(),tmp_neighs2.end(),d_nodes.begin());
    thrust::stable_sort_by_key(d_nodes.begin(),d_nodes.end(),indices.begin());

    // thrust::gather(indices.begin(), indices.end(), d_nodes.begin(), d_nodes.end());
    thrust::gather(indices.begin(), indices.end(), d_neighs.begin(), d_neighs.begin());
    thrust::gather(indices.begin(), indices.end(), d_oWeights.begin(), d_oWeights.begin());
    thrust::gather(indices.begin(), indices.end(), d_iWeights.begin(), d_iWeights.begin());

    return;
}


/*
*/
void computeModularityGain(thrust::device_vector<float> &d_mod_gains 
    , const Dec_vec &d_nodes, const Dec_vec &d_neighs, const Dec_vec &d_oWeights, const Dec_vec &d_iWeights
    , const Dec_vec &d_map, const Dec_vec &d_nWeights, const Dec_vec &d_cWeights){

        int m=10;


    // Calculate first part: kic + kci
    thrust::transform(d_oWeights.begin(), d_oWeights.end(), d_iWeights.begin(), d_mod_gains.begin(), thrust::plus<int>());

    // Calculate second part: kiout*kcin + kiin*kcout
    int array_length = d_nodes.size();
    thrust::device_vector<float> tmp(array_length);
    const int* d_nWeights_ptr = thrust::raw_pointer_cast(&d_nWeights[0]);
    const int* d_cWeights_ptr = thrust::raw_pointer_cast(&d_cWeights[0]);

    auto ff = [=]  __device__ (int node, int nei_comm) {
        float result = d_nWeights_ptr[node*2+0]*d_cWeights_ptr[nei_comm*2+1] 
                        + d_nWeights_ptr[node*2+1]*d_cWeights_ptr[nei_comm*2+0];
        result = result/m;
        return result;
    };
    thrust::transform(d_nodes.begin(), d_nodes.end(), d_neighs.begin(), tmp.begin(), ff);

    // Calculate first part + second part
    thrust::transform(d_mod_gains.begin(), d_mod_gains.end(), tmp.begin(), d_mod_gains.begin(), thrust::minus<float>());
    tmp.clear();
    thrust::device_vector<float>().swap(tmp);

    // Calculate status map
    Dec_vec status_map(array_length);
    const int* d_map_ptr = thrust::raw_pointer_cast(&d_map[0]);
    auto ff2 = [=]  __device__ (int node, int nei_comm) {  return d_map_ptr[node]!=nei_comm; };
    thrust::transform(d_nodes.begin(), d_nodes.end(), d_neighs.begin(), status_map.begin(), ff2);
    thrust::copy(status_map.begin(), status_map.end(), std::ostream_iterator<int>(std::cout," "));
    cout<<endl;

    // Calculate final result
    thrust::transform(d_mod_gains.begin(), d_mod_gains.end(), status_map.begin(), d_mod_gains.begin(), thrust::multiplies<float>());

    return;

}




void findMaxGainNeighbor(Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_mod_gains){


    int array_length = d_nodes.size();
    Dec_vec indices(array_length); 
    thrust::sequence(indices.begin(), indices.end());

// , thrust::greater<int>()

    thrust::stable_sort_by_key(tmp_neighs1.begin(),tmp_neighs1.end(),indices.begin());
    thrust::stable_sort_by_key(tmp_neighs2.begin(),tmp_neighs2.end(),d_nodes.begin());


    return;
}










// /**
//     Compute the out-weights for each community.

//     Parameters
//     --------
//         d_nodes:
//         d_oWeights:
//         comm_size:

//     Example
//     --------
//     The input is something like:
//         d_nodes = {0,0,0,0,1,1,2,2,2,3,3,3,3,3,3,4,4,4};
//         d_neighs = {1,2,3,4,0,3,0,3,5,0,1,2,4,6,11,0,3,5};
//         d_oWeights = {1,1,1,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1};
//         d_iWeights = {0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,1,0,0};
// */
// Dec_vec computeCommunityOutWeights(Dec_vec d_nodes, Dec_vec d_oWeights, int comm_size){


//     Dec_vec d_comm_oWeights(comm_size); 
//     int range;
//     Dec_vec temp1(d_nodes.size());
//     Dec_vec temp2(d_nodes.size());
//     thrust::pair<thrust::device_vector<int>::iterator,thrust::device_vector<int>::iterator> new_end;
//     new_end = thrust::reduce_by_key(d_nodes.begin(),d_nodes.end(), d_oWeights.begin(), temp1.begin(), temp2.begin());

//     // Compute out-degree weights
//     range = new_end.first - temp1.begin();



//     thrust::scatter(temp2.begin(), temp2.begin()+range,temp1.begin(), d_comm_oWeights.begin());
//     return d_comm_oWeights;
// }



// /**
//     Compute the in-weights for each community.

//     The input is something like:
//         d_nodes = {0,0,0,0,1,1,2,2,2,3,3,3,3,3,3,4,4,4};
//         d_neighs = {1,2,3,4,0,3,0,3,5,0,1,2,4,6,11,0,3,5};
//         d_oWeights = {1,1,1,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1};
//         d_iWeights = {0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,1,0,0};
    


//     Parameters
//     --------
//         d_nodes:
//         d_oWeights:
//         comm_size:

//     Example
//     --------
// */
// Dec_vec computeCommunityInWeights(Dec_vec d_neighs, Dec_vec d_iWeights, int comm_size){
   
   
//     Dec_vec d_comm_iWeights(comm_size); 
    
//     //
//     int new_length; 
//     thrust::pair<Dec_vec::iterator,Dec_vec::iterator> new_end;
    
//     // 
//     Dec_vec tmp_keys(d_neighs.size());
//     Dec_vec tmp_values(d_iWeights.size());

//     thrust::stable_sort_by_key(d_neighs.begin(), d_neighs.end(), d_iWeights.begin());
//     new_end = thrust::reduce_by_key(d_neighs.begin(),d_neighs.end(), d_iWeights.begin(), tmp_keys.begin(), tmp_values.begin());
//     new_length = new_end.first - tmp_keys.begin();

//     thrust::scatter(tmp_values.begin(), tmp_values.begin()+new_length, tmp_keys.begin(), d_comm_iWeights.begin());
//     return d_comm_iWeights;
// }

