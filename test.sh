nvcc -std=c++11 --expt-extended-lambda -arch=sm_30 task.cu utility.cu parallel_louvain.cu louvain_utils.cu -o main.out &>log.log
./main.out dataset/web-Stanford.txt