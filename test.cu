#include <vector>
#include "louvain_utils.h"

using std::cout;
using std::endl;


int main(void){


    std::vector<int> h_nodes = {5, 3, 0, 5, 1, 2, 4, 3, 1};
    std::vector<int> h_weights = {1, 1, 0, 3, 4, 2, 6, 1, 6};

    Dec_vec d_nodes(h_nodes); 
    Dec_vec d_weights(h_weights); 
    Dec_vec d_weight_map(6);

    reassign(d_weights);

    // // Test calculateWeights
    // calculateWeights(d_weight_map, d_nodes, d_weights);
    thrust::copy(d_weights.begin(), d_weights.end(), std::ostream_iterator<int>(std::cout,"  "));
    cout<<endl;



    return 0;
}