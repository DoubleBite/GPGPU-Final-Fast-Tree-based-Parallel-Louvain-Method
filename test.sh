nvcc -std=c++11 --expt-extended-lambda -arch=sm_30 main.cu -o main.out &>log.log
./main.out