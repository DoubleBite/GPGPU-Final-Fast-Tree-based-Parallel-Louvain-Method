nvcc -std=c++11 -arch sm_30 task.cu utility.cu parallel_louvain.cu -o task.out
./task.out dataset/test.edgelist