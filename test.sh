nvcc -std=c++11 --expt-extended-lambda -arch=sm_30 task.cu utility.cu parallel_louvain.cu louvain_utils.cu -o main.out &>compile.log
./main.out dataset/email-Eu-core.txt 