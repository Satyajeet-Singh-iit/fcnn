#include <vector>
#include <iostream>
#include "../include/print_variables/print_variable.h"

void printVariable2D(
	std::vector<std::vector<float>> var){
    int row = var.size();
    int col = var[0].size();
    std::cout<<"-------------------"<<std::endl;
    for(int i=0; i<row; i++){
    	for(int j=0; j<col; j++){
    		std::cout<<var[i][j]<<"\t";
    	}
    	std::cout<<std::endl;
    }
    std::cout<<"-------------------"<<std::endl;
}
void printVariable1D(
	std::vector<float> var){
    int row = var.size();
    std::cout<<"-------------------"<<std::endl;
    for(int i=0; i<row; i++){
    		std::cout<<var[i]<<"\t";
    }
    std::cout<<"-------------------"<<std::endl;
}
