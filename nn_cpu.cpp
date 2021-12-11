#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <fstream>

using namespace std;

#define N 784 
#define L1 1000 
#define L2 10 
#define D 1000 //total amount of train data 
#define epochs 60000

float a = 0.02;
float* TrainPixels = new float[D * N];
float* TrainLabels = new float[D * L2];
float* WL1 = new float[L1 * (N + 1)];
float* WL2 = new float[L2 * (L1+1)];
float* OL1 = new float[L1];
float* OL2 = new float[L2];
float* EL1 = new float[L1];
float* EL2 = new float[L2];



void init_Weights(void) {
	int i ,y;
	for ( i = 0; i < L1; i++)
	{
		for(y =0; y< N+1 ;y++)
			WL1[i*(N+1)+y] = 2 * ((rand() % RAND_MAX) / (float)RAND_MAX - 0.5);

	}

	for (i = 0; i < L2; i++)
	{
		for (y = 0; y < L1+1; y++)
			WL2[i*(L1+1)+y] = 2 * ((rand() % RAND_MAX) / (float)RAND_MAX - 0.5);

	}
	
}

void LoadTrainData(void){
	ifstream traindata("fashion-mnist_train.csv");	
	
	string label;
	string pixel[784];
	
	//skip traindata first row 
	getline(traindata, pixel[0],'\n');
	
	long int  line = 0 ;
	while(line<D){
		getline(traindata,label, ',');
		
		for(int i =0; i<783;i++)
			getline(traindata,pixel[i], ',');
		
		getline(traindata,pixel[783], '\n');
		
		
		float num = stod(label);
		
		//put data to label matrix
		
		for(int i=0; i <10; i++){
			if(num==i)
				TrainLabels[line*10+i] = 1;
			
			else
				TrainLabels[line*10+i] = 0;
		}
			
		//put data to pixel matrix
		
		for(int i=0; i <784; i++){
			TrainPixels[line*784+i] = stod(pixel[i]);
		}
		
		
		line++;
	}

	
	traindata.close();
}

float activation_Sigmoid(float y) {
	return 1 / (1 + exp(-y));
}

float derivative_Sigmoid(float y){
    return y * (1-y);
}

void activateNN(float* Input, int index){
	
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < L1; i++) {
		OL1[i] = 0;
		for (int y = 0; y < N; y++)
			OL1[i] += WL1[i*N+y] * Input[index*N+y];
		OL1[i] += WL1[i*N+N];
		OL1[i] = activation_Sigmoid(OL1[i]);
	}
	
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < L2; i++) {
		OL2[i] = 0;
		for (int y = 0; y < L1; y++)
			OL2[i] += WL2[i*L1+y] * OL1[y];
		OL2[i] += WL2[i*L1+L1];
		OL2[i] = activation_Sigmoid(OL2[i]);
	}
}

void calc_Error(float *target, int index) {
	//no reason for parallelization we lost time 
	for (int i = 0; i < L2; i++) {
		EL2[i] = (OL2[i] - target[index*L2+i]) * (derivative_Sigmoid(OL2[i])+a);
	}
	
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < L1; i++) {
		EL1[i] = 0;
		for (int i2 = 0; i2 < L2; i2++) {
			EL1[i] += EL2[i2] * WL2[i2*L2+i] * (derivative_Sigmoid(OL1[i])+a);
		}
	}
}

void trainNN(float* input, float* target, int index)
{


	#pragma omp parallel for schedule(static)
	for (int i = 0; i <L2 ; i++) {
		for (int j = 0; j < L1; j++) {
			WL2[i*L1+j] -= a * EL2[i] * OL1[j];
		}
        WL2[i*L1+L1] -= a * EL2[i];
	}
	

	#pragma omp parallel for schedule(static)
	for (int i = 0; i < L1; i++) {
		for (int j = 0; j < N; j++) {
			WL1[i*N+j] -= a * EL1[i] * input[index*N+j];
		}
        WL1[i*N+N] -= - a * EL1[i];
	}
	
	
}

void TrainDataAccuracy(void){
	//avrg accuracy of the hole train dataset
	
	float classifiedCorrectly = 0;
	float accuracy ;
	
	for(int y = 0;y < D;y++){
		
		activateNN(TrainPixels, y);
		
		int classifiedLebel;
		float max = 0;
		for (int i = 0; i < L2; i++) {
			if(OL2[i]>max){
				max = OL2[i];
				classifiedLebel= i;
			}
		}
		
		int correctLebel = 0;
		for(int i =1; i<L2; i++){
			if(TrainLabels[y*L2+i] == 1)
				correctLebel = i;
		}
		
		if(classifiedLebel == correctLebel)
			classifiedCorrectly++;
		
	}
	accuracy = classifiedCorrectly/D;
	
	printf("train data accuracy = %f \n", accuracy);
	
}

int main(){
    LoadTrainData();
    init_Weights();

    int rn;
	for (long int i = 0; i < epochs; i++)
	{
		
		rn = rand() % D;
		activateNN(TrainPixels, rn);
		calc_Error(TrainLabels, rn);
		trainNN(TrainPixels, TrainLabels, rn);
		
		if((i % 10000) == 0)
			TrainDataAccuracy();
			
	}

    //free 
    free(TrainPixels);
    free(TrainLabels);
    free(WL1);
    free(WL2);
    free(OL1);
    free(OL2);
    free(EL1);
    free(EL2);
    return 0;
}
