nvcc -std=c++11  task.cu utility.cu parallel_louvain.cu -o task.out
./task.out dataset/test.edgelist