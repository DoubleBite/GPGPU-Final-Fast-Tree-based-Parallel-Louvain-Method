nvcc -std=c++11 --expt-extended-lambda -arch=sm_30 main.cu preprocessing.cpp parallel_louvain.cu louvain_utils.cu
./a.out dataset/test.edgelist >runtime.log