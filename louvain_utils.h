#include <thrust/device_vector.h>


typedef thrust::device_vector<int> Dec_vec;

void calculateWeights(Dec_vec &, Dec_vec, Dec_vec);
void convertIDToCommunity(const Dec_vec &, Dec_vec &);
void sortByFirstTwo(Dec_vec &, Dec_vec &, Dec_vec &, Dec_vec &);
void sortByFirstTwo3(Dec_vec &, thrust::device_vector<float> &, Dec_vec &);
void reduceByFirstTwo(Dec_vec &, Dec_vec &, Dec_vec &, Dec_vec &);