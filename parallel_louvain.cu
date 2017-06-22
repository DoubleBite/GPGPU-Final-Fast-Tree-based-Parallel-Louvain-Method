#include "parallel_louvain.h"
#include "utility.h"
#include <stdio.h>

/*
	nodes解釋一下

*/

__global__ void countCommOutWeights(int *, int *, int *, int);
__global__ void countCommInWeights(int *comm_in_weights, int *neightbors, int *in_weights, int n);
__global__ void computeDeltaModularity(float *modularities, int *d_nodes, int *d_neighbors, int *d_out_weights, int *d_in_weights
	, int *d_community_out_weights, int *d_community_in_weights, int m, int n);

/**
	Host function to perform Louvain's method.
*/
void parallelLouvain(int* h_nodes, int* h_neighbors, int* h_out_weights, int* h_in_weights, int n){




	return;
}







void mergeSameCommunity(){






}







/**
	Calculate the total out weight for each community.
	The output value is stored in comm_out_weights.
*/
__global__ void countCommOutWeights(int *comm_out_weights, int *nodes, int *out_weights, int n){

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx<n){
		const int community_ID = nodes[idx]; 
		int out_weight = out_weights[idx];	

		// Accumulate the out-weight
		atomicAdd(&comm_out_weights[community_ID], out_weight);

	}
	return;
}
 


/**
	Calculate the total in-weight for each community.
	The output value is stored in comm_in_weights.
*/
__global__ void countCommInWeights(int *comm_in_weights, int *nodes, int *in_weights, int n){

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx<n){
		const int community_ID = nodes[idx]; 
		int in_weight = in_weights[idx];	

		// Accumulate the out weight
		atomicAdd(&comm_in_weights[community_ID], in_weight);
	}
	return;
}


/**
*/
__global__ void computeDeltaModularity(float *modularities, int *d_nodes, int *d_neighbors, int *d_out_weights, int *d_in_weights
	, int *d_community_out_weights, int *d_community_in_weights, int m, int n){

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx<n){
		const int community_ID = d_nodes[idx]; 
		const int neightbor_comm_ID = d_neighbors[idx]; 
		const int k_ic = d_out_weights[idx];
		const int k_ci = d_in_weights[idx];
		const int k_i_out = d_community_out_weights[community_ID];
		const int k_i_in = d_community_in_weights[community_ID];
		const int k_c_out = d_community_out_weights[neightbor_comm_ID];
		const int k_c_in = d_community_in_weights[neightbor_comm_ID];
	
		modularities[idx] = k_ic + k_ci - (k_i_out*k_c_in + k_i_in*k_c_out)*1.0/m;
	}
	return;
}