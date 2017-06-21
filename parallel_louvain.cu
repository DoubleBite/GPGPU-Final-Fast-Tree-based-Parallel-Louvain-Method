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


	int m=17;

	//
	int *d_nodes, *d_neighbors, *d_out_weights, *d_in_weights;
	cudaMalloc((void**)&d_nodes, n*sizeof(int));
	cudaMalloc((void**)&d_neighbors, n*sizeof(int));
	cudaMalloc((void**)&d_out_weights, n*sizeof(int));
	cudaMalloc((void**)&d_in_weights, n*sizeof(int));
	cudaMemcpy(d_nodes, h_nodes, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_neighbors, h_neighbors, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out_weights, h_out_weights, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_weights, h_in_weights, n*sizeof(int), cudaMemcpyHostToDevice);


	// The community-weight mapping arrays
	int *h_community_out_weights = new int[n], *h_community_in_weights = new int[n];
	int *d_community_out_weights, *d_community_in_weights;
	cudaMalloc((void**)&d_community_out_weights, n*sizeof(int));
	cudaMalloc((void**)&d_community_in_weights, n*sizeof(int));
	cudaMemset(d_community_out_weights, 0, n* sizeof(int)); // Reset the mappings to 0
	cudaMemset(d_community_in_weights, 0, n* sizeof(int));


	//
	float *h_modularities = new float[n], *d_modularities;
	cudaMalloc((void**)&d_modularities, n*sizeof(float));
	
	/////////////////////////////////////////////////////////////////////////////////
	// // Count the community weights
	countCommOutWeights <<<1, 256>>> (d_community_out_weights, d_nodes, d_out_weights, n);
	countCommInWeights <<<1, 256>>> (d_community_in_weights, d_nodes, d_in_weights, n);



	/////////////////////////////////////////////////////////////////////////////////
	cudaMemcpy(h_community_out_weights, d_community_out_weights, n* sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_community_in_weights, d_community_in_weights, n* sizeof(int), cudaMemcpyDeviceToHost);
	// printf("%d  ",h_community_out_weights[0]);
	checkArrayValues(h_community_out_weights, n);
	checkArrayValues(h_community_in_weights, n);

	computeDeltaModularity <<<1, 256>>> (d_modularities, d_nodes, d_neighbors, d_out_weights, d_in_weights
	, d_community_out_weights, d_community_in_weights, m, n);
	cudaMemcpy(h_modularities, d_modularities, n* sizeof(int), cudaMemcpyDeviceToHost);
	checkArrayValues(h_modularities, n);


	return;
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