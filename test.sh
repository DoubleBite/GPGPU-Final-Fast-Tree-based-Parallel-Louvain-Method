nvcc -std=c++11 --expt-extended-lambda -arch=sm_30 test.cu -o test.out &>log.log
./test.out