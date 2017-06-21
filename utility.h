#include <iostream>
#include <iomanip> 



void parseEdgelist(char*, int*&, int*&, int*&, int*&, int&);

void printInfo(int*, int*, int*, int*, int);

/*  */
template <typename T>
void checkArrayValues(T* array, int N){

	std::cout<<std::endl;
	std::cout<< std::right;
	for(int i=0; i<N ;i++){
		std::cout<< std::setw(6) << i;
	}
	std::cout<<std::endl;
	std::cout<< std::right;
	for(int i=0; i<N ;i++){
		std::cout<< std::setw(12) <<array[i];
	}
	std::cout<<std::endl;

	return;
}