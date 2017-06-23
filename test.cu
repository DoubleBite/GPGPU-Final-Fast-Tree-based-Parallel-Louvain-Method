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
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/extrema.h>

using std::cout;
using std::endl;

typedef thrust::device_vector<int> Dec_vec;


void onePassLouvain(Dec_vec &d_comm_map, Dec_vec d_nodes, Dec_vec d_neighs, Dec_vec d_oWeights, Dec_vec d_iWeights
                    , const Dec_vec &d_nodeOWeights, const Dec_vec &d_nodeIWeights, int m);

bool oneIterLouvain(Dec_vec &d_comm_map, Dec_vec d_nodes, Dec_vec d_neighs, Dec_vec d_oWeights, Dec_vec d_iWeights
                    , const Dec_vec &d_nodeOWeights, const Dec_vec &d_nodeIWeights, int m);

void calculateCommunityWeights(Dec_vec  &d_commOWeights, Dec_vec  &d_commIWeights
        , const Dec_vec &d_comm_map, Dec_vec d_nodeOWeights, Dec_vec d_nodeIWeights);


void convertToCommunity(const Dec_vec &d_comm_map, Dec_vec &d_neighs);
void sortByNeighborCommunity(Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights);
int reduceCommunityWeights(Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights);

void computeModularityGain(thrust::device_vector<float> &d_mod_gains 
    , const Dec_vec &d_nodes, const Dec_vec &d_neighs, const Dec_vec &d_oWeights, const Dec_vec &d_iWeights
    , const Dec_vec &d_map, const Dec_vec &d_nOWeights, const Dec_vec &d_nIWeights
    , const Dec_vec &d_cOWeights, const Dec_vec &d_cIWeights, const int m);

bool isEnd(const thrust::device_vector<float> &d_mod_gains);
void assignNewCommunity(Dec_vec &d_comm_map, Dec_vec &d_nodes, Dec_vec &d_neighs, thrust::device_vector<float> &d_mod_gains);


void mergeCommunity(const Dec_vec &d_comm_map, Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights);




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

    mergeCommunity(d_comm_map, d_nodes, d_neighs, d_oWeights, d_iWeights);

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

void onePassLouvain(Dec_vec &d_comm_map, Dec_vec d_nodes, Dec_vec d_neighs, Dec_vec d_oWeights, Dec_vec d_iWeights
        , const Dec_vec &d_nodeOWeights, const Dec_vec &d_nodeIWeights, int m){

    bool proceed;

    // do{
        proceed = oneIterLouvain(d_comm_map, d_nodes, d_neighs, d_oWeights, d_iWeights, d_nodeOWeights, d_nodeIWeights, m);
    // }while(proceed);

    return;
}

bool oneIterLouvain(Dec_vec &d_comm_map, Dec_vec d_nodes, Dec_vec d_neighs, Dec_vec d_oWeights, Dec_vec d_iWeights
        , const Dec_vec &d_nodeOWeights, const Dec_vec &d_nodeIWeights, int m){


    // Get the max number of community.
    Dec_vec::iterator dev_ptr = thrust::max_element(d_comm_map.begin(), d_comm_map.end());
    int max_comm = *dev_ptr;
    
    // Calculate Community Weights
    Dec_vec d_commOWeights(max_comm+1);
    Dec_vec d_commIWeights(max_comm+1);
    calculateCommunityWeights(d_commOWeights, d_commIWeights, d_comm_map, d_nodeOWeights, d_nodeIWeights);


    convertToCommunity(d_comm_map, d_neighs);
    sortByNeighborCommunity(d_nodes, d_neighs, d_oWeights, d_iWeights);
    int new_length = reduceCommunityWeights(d_nodes, d_neighs, d_oWeights, d_iWeights);
    thrust::copy(d_nodes.begin(), d_nodes.begin()+new_length, std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;
    thrust::copy(d_neighs.begin(), d_neighs.begin()+new_length, std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;
    thrust::copy(d_oWeights.begin(), d_oWeights.begin()+new_length, std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;
    thrust::copy(d_iWeights.begin(), d_iWeights.begin()+new_length, std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;


    // Allocate a vector to store modularity gains
    thrust::device_vector<float> d_mod_gains(new_length);
    computeModularityGain(d_mod_gains, d_nodes, d_neighs, d_oWeights, d_iWeights
    , d_comm_map, d_nodeOWeights, d_nodeIWeights, d_commOWeights, d_commIWeights, m);
    thrust::copy(d_mod_gains.begin(), d_mod_gains.end(), std::ostream_iterator<float>(std::cout," "));
    cout<<endl;

    if(isEnd(d_mod_gains))
        return false;
    else{
        assignNewCommunity(d_comm_map, d_nodes, d_neighs, d_mod_gains);
        thrust::copy(d_comm_map.begin(), d_comm_map.end(), std::ostream_iterator<int>(std::cout,"  "));
        cout<<endl;
	    return true;
    }
    return false;
}


void convertToCommunity(const Dec_vec  &d_comm_map, Dec_vec  &d_neighs){

    const int* d_map_ptr = thrust::raw_pointer_cast(&d_comm_map[0]);
    auto ff = [=]  __device__ (int x) {return d_map_ptr[x];};
    thrust::transform(d_neighs.begin(),d_neighs.end(),d_neighs.begin(),ff);
    return;
}


void calculateCommunityWeights(Dec_vec  &d_commOWeights, Dec_vec  &d_commIWeights
        , const Dec_vec &d_comm_map, Dec_vec d_nodeOWeights, Dec_vec d_nodeIWeights){

    int length = d_nodeOWeights.size();

    Dec_vec indices(length); 
    thrust::sequence(indices.begin(), indices.end());
    convertToCommunity(d_comm_map, indices);
    Dec_vec indices2(indices); 

    thrust::stable_sort_by_key(indices.begin(), indices.end(), d_nodeOWeights.begin());
    thrust::stable_sort_by_key(indices2.begin(), indices2.end(), d_nodeIWeights.begin());

    thrust::pair<Dec_vec::iterator,Dec_vec::iterator> new_end;

    new_end = thrust::reduce_by_key(indices.begin(), indices.end(), d_nodeOWeights.begin(), indices.begin(), d_nodeOWeights.begin());
    new_end = thrust::reduce_by_key(indices2.begin(), indices2.end(), d_nodeIWeights.begin(), indices2.begin(), d_nodeIWeights.begin());
    int new_length = new_end.first - indices2.begin();


    thrust::scatter(d_nodeOWeights.begin(), d_nodeOWeights.begin()+new_length, indices.begin(), d_commOWeights.begin());
    thrust::scatter(d_nodeIWeights.begin(), d_nodeIWeights.begin()+new_length, indices2.begin(), d_commIWeights.begin());

    return;
}


void sortByNeighborCommunity(Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights){

    Dec_vec indices(d_nodes.size()); 
    thrust::sequence(indices.begin(), indices.end());

    Dec_vec tmp_neighs1(d_neighs); 
    Dec_vec tmp_neighs2(d_neighs); 

    thrust::stable_sort_by_key(tmp_neighs1.begin(),tmp_neighs1.end(),indices.begin());
    thrust::stable_sort_by_key(tmp_neighs2.begin(),tmp_neighs2.end(),d_nodes.begin());
    thrust::stable_sort_by_key(d_nodes.begin(),d_nodes.end(),indices.begin());

    thrust::gather(indices.begin(), indices.end(), d_neighs.begin(), d_neighs.begin());
    thrust::gather(indices.begin(), indices.end(), d_oWeights.begin(), d_oWeights.begin());
    thrust::gather(indices.begin(), indices.end(), d_iWeights.begin(), d_iWeights.begin());

    return;
}


typedef thrust::tuple<int,int> Tuple;
struct BinaryPredicate
{
  __host__ __device__ bool operator () 
                      (const Tuple& lhs, const Tuple& rhs) 
  {
    return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) && (thrust::get<1>(lhs) == thrust::get<1>(rhs));
  }
};

int reduceCommunityWeights(Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights){

    int new_length = d_nodes.size();
    thrust::pair<thrust::device_vector<Tuple>::iterator,Dec_vec::iterator> new_end;
    thrust::device_vector<Tuple> keys_input(new_length);
    auto ff = [=]  __device__ (int &first, int &second) {  return thrust::make_tuple(first, second); };
    thrust::transform(d_nodes.begin(), d_nodes.end(), d_neighs.begin(), keys_input.begin(), ff);
    thrust::device_vector<Tuple> tmp(new_length); 

    new_end = thrust::reduce_by_key(keys_input.begin(), keys_input.end(), d_oWeights.begin(), tmp.begin(), d_oWeights.begin()
               , BinaryPredicate(), thrust::plus<int>() );
    new_end = thrust::reduce_by_key(keys_input.begin(), keys_input.end(), d_iWeights.begin(), tmp.begin(), d_iWeights.begin()
               , BinaryPredicate(), thrust::plus<int>() );
    new_end = thrust::unique_by_key_copy(keys_input.begin(), keys_input.end(), d_nodes.begin(), tmp.begin(), d_nodes.begin());
    new_end = thrust::unique_by_key_copy(keys_input.begin(), keys_input.end(), d_neighs.begin(), tmp.begin(), d_neighs.begin());
    new_length = new_end.first - tmp.begin(); 

    return new_length;
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
    thrust::copy(status_map.begin(), status_map.begin()+length, std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;

    // Calculate final result by specifying the node which is in the same community as its neighbor to zero.
    thrust::transform(d_mod_gains.begin(), d_mod_gains.begin()+length, status_map.begin(), d_mod_gains.begin(), thrust::multiplies<float>());

    return;

}

bool isEnd(const thrust::device_vector<float> &d_mod_gains){

    // Check whether any of the reassignment gets modularity gain. If no, just return.
    auto greater_than_zero = [=]  __device__ (float gain) {  return gain>0; };
    bool is_proceed = thrust::any_of(d_mod_gains.begin(), d_mod_gains.end(), greater_than_zero);

    return !is_proceed;
}


void assignNewCommunity(Dec_vec &d_comm_map, Dec_vec &d_nodes, Dec_vec &d_neighs, thrust::device_vector<float> &d_mod_gains){

    int length = d_mod_gains.size();

    Dec_vec indices(length); 
    thrust::sequence(indices.begin(), indices.end());

    thrust::device_vector<float> tmp_gains(d_mod_gains);

    thrust::stable_sort_by_key(tmp_gains.begin(),tmp_gains.end(),indices.begin(), thrust::greater<float>());
    thrust::stable_sort_by_key(d_mod_gains.begin(),d_mod_gains.end(),d_nodes.begin(), thrust::greater<float>());
    thrust::stable_sort_by_key(d_nodes.begin(), d_nodes.begin()+length, indices.begin());
    thrust::gather(indices.begin(), indices.end(), d_neighs.begin(), d_neighs.begin());

    thrust::pair<Dec_vec::iterator,Dec_vec::iterator> new_end;
    new_end = thrust::unique_by_key_copy(d_nodes.begin(), d_nodes.begin()+length, d_neighs.begin(), d_nodes.begin(), d_comm_map.begin());

    return;
}



void mergeCommunity(const Dec_vec &d_comm_map, Dec_vec &d_nodes, Dec_vec &d_neighs, Dec_vec &d_oWeights, Dec_vec &d_iWeights){


    convertToCommunity(d_comm_map, d_nodes);
    convertToCommunity(d_comm_map, d_neighs);

    sortByNeighborCommunity(d_nodes, d_neighs, d_oWeights, d_iWeights);
    int new_length = reduceCommunityWeights(d_nodes, d_neighs, d_oWeights, d_iWeights);

    d_nodes.resize(new_length);
    d_neighs.resize(new_length);
    d_oWeights.resize(new_length);
    d_iWeights.resize(new_length);

    return;
}