#include "parallel_louvain.h"
#include "utility.h"


/*
	nodes解釋一下

*/

__global__ void countCommOutWeights(int *comm_out_weights, int *nodes, int *out_weights, int n);
__global__ void countCommInWeights(int *comm_in_weights, int *neightbors, int *in_weights, int n);

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

/**
	Host function to perform Louvain's method.
*/
void parallelLouvain(int* nodes, int* neighbors, int* out_weights, int* in_weights, int n){


	int *host_community_out_weights;
	int *host_community_in_weights;

	// The community-weight mapping arrays
	int *dev_community_out_weights;
	int *dev_community_in_weights;
	cudaMalloc((void**)&dev_community_out_weights, n);
	cudaMalloc((void**)&dev_community_in_weights, n);
	
 
	// Reset the mappings to 0
	cudaMemset(dev_community_out_weights, 0, n);
	cudaMemset(dev_community_in_weights, 0, n);

	
	// Count the community weights
	countCommOutWeights <<<1, 256>>> (dev_community_out_weights, nodes, out_weights, n);
	countCommInWeights <<<1, 256>>> (dev_community_in_weights, neighbors, in_weights, n);

	// printf("aaaaaaaaa");

	return;
}


/**
	Calculate the total out weight for each community.
	The output value is stored in comm_out_weights.
*/
__global__ void countCommOutWeights(int *comm_out_weights, int *nodes, int *out_weights, int n){

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	const int community_ID = nodes[idx]; 
	int out_weight = out_weights[idx];	

	// Accumulate the out-weight
	atomicAdd(&comm_out_weights[community_ID], out_weight);

	return;
}
 


/**
	Calculate the total in-weight for each community.
	The output value is stored in comm_in_weights.
*/
__global__ void countCommInWeights(int *comm_in_weights, int *neightbors, int *in_weights, int n){


	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	const int community_ID = neightbors[idx]; 
	int in_weight = in_weights[idx];	

	// Accumulate the out weight
	atomicAdd(&comm_in_weights[community_ID], in_weight);

	return;
}
