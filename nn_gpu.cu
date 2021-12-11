#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <fstream>

using namespace std;

#define N 784 
#define L1 4000
#define L2 10 
#define D 1000 //total amount of train data 
#define epochs 60000

__device__ const int d_N = N;
__device__ const int d_L1 = L1;
__device__ const int d_L2 = L2;
__device__ float a = 0.02;

float* TrainPixels = new float[D * N];
float* TrainLabels = new float[D * L2];
float* WL1 = new float[L1 * (N + 1)];
float* WL2 = new float[L2 * (L1+1)];
float* OL1 = new float[L1];
float* OL2 = new float[L2];
float* EL1 = new float[L1];
float* EL2 = new float[L2];

__host__ void LoadTrainData(void){
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

__host__ void init_Weights(void) {
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

__device__ float activation_Sigmoid(float y) {
	return 1 / (1 + exp(-y));
}

__device__ float derivative_Sigmoid(float y){
    return y * (1-y);
}

__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void activateNN_L1(float* Input, int index, float* WL1, float* WL2, float* OL1, float* OL2){
    __shared__ float sInput[d_N+240];
    unsigned int block_index = blockIdx.x;
    unsigned int thr_index = threadIdx.x;

    if(block_index<d_L1 && thr_index<d_N)
        sInput[thr_index] = WL1[block_index*d_N+thr_index] * Input[index*d_N+thr_index];
    else if (block_index<d_L1 && thr_index>=d_N && thr_index<d_N+240)
        sInput[thr_index] = 0;
    
    __syncthreads();

    if(block_index<d_L1) {
      if (thr_index < 512) sInput[thr_index]+= sInput[thr_index+512]; 
      __syncthreads(); 
      if (thr_index < 256) sInput[thr_index]+= sInput[thr_index+256]; 
      __syncthreads();
      if (thr_index < 128) sInput[thr_index]+= sInput[thr_index+128]; 
      __syncthreads();
      if (thr_index < 64) sInput[thr_index]+= sInput[thr_index+64]; 
      __syncthreads();
      if (thr_index < 32) warpReduce(sInput, thr_index);

      if(thr_index == 0){
          OL1[block_index] = sInput[0] + WL1[block_index*d_N+d_N];
          OL1[block_index] = activation_Sigmoid(OL1[block_index]);
      }
      
    }


}

__global__ void activateNN_L2(float* Input, int index, float* WL1, float* WL2, float* OL1, float* OL2){
    __shared__ float sInput[d_L1+24];
    unsigned int block_index = blockIdx.x;
    unsigned int thr_index = threadIdx.x;
    
    if(block_index<d_L2 && thr_index<d_L1)
        sInput[thr_index] = WL2[block_index * d_L1 + thr_index] * OL1[thr_index] + WL2[block_index * d_L1 + thr_index + 1000] * OL1[thr_index + 1000] + WL2[block_index * d_L1 + thr_index + 2000] * OL1[thr_index + 2000] + WL2[block_index * d_L1 + thr_index + 3000] * OL1[thr_index + 3000];
    else if (block_index<d_L2 && thr_index>=d_L1 && thr_index < d_L1+24)
        sInput[thr_index] = 0;
        
    __syncthreads();

    if(block_index<d_L2) {

        if (thr_index < 512) sInput[thr_index]+= sInput[thr_index+512]; 
        __syncthreads(); 
        if (thr_index < 256) sInput[thr_index]+= sInput[thr_index+256]; 
        __syncthreads();
        if (thr_index < 128) sInput[thr_index]+= sInput[thr_index+128]; 
        __syncthreads();
        if (thr_index < 64) sInput[thr_index]+= sInput[thr_index+64]; 
        __syncthreads();
        if(thr_index<32) warpReduce(sInput, thr_index);
        
        if(thr_index == 0){
            OL2[block_index] = sInput[0] + WL2[block_index*d_L1+d_L1];
            OL2[block_index] = activation_Sigmoid(OL2[block_index]);
        }

    }

}

__global__ void calc_Error(float *target, int index, float* WL2, float* OL1, float* OL2, float* EL1, float* EL2) {

    int thr_index = blockDim.x * blockIdx.x +threadIdx.x;
    int stride = blockDim.x * gridDim.x;

	for (int i = thr_index; i < d_L2; i+=stride) {
		EL2[i] = (OL2[i] - target[index*d_L2+i]) * (derivative_Sigmoid(OL2[i])+a);
	}
	
    __syncthreads();

	for (int i = thr_index; i < d_L1; i+=stride) {
		EL1[i] = 0;
		for (int i2 = 0; i2 < d_L2; i2++) {
			EL1[i] += EL2[i2] * WL2[i2*d_L2+i] * (derivative_Sigmoid(OL1[i])+a);
		}
	}

}

__global__ void trainNN(float* input, float* target, int index, float* WL1, float* WL2, float* OL1, float* OL2, float* EL1, float* EL2)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    
    if(i<d_L2 && j<1001){

        if(j < d_L1){
            WL2[i*d_L1+j] -= a * EL2[i] * OL1[j];
            WL2[i*d_L1+j+1000] -= a * EL2[i] * OL1[j+1000];
            WL2[i*d_L1+j+2000] -= a * EL2[i] * OL1[j+2000];
            WL2[i*d_L1+j+3000] -= a * EL2[i] * OL1[j+3000];

        }
            
        else
            WL2[i*d_L1+d_L1] -= a * EL2[i];

    }
  
      
    if(i<d_L1 && j<d_N+1){
        if(j < d_N)
            WL1[i*d_N+j] -= a * EL1[i] * input[index*d_N+j];
        else
            WL1[i*d_N+d_N] -= a * EL1[i];
    }
	
}

__global__ void Classify(float* TrainLabels, float* OL2, float *classifiedCorrectly, int y){
    
    int classifiedLebel;
    float max = 0;
    for (int i = 0; i < d_L2; i++) {
        if(OL2[i]>max){
            max = OL2[i];
            classifiedLebel= i;
        }
    }
    int correctLebel = 0;
    for(int i =1; i<d_L2; i++){
        if(TrainLabels[y*d_L2+i] == 1)
            correctLebel = i;
    }
    if(classifiedLebel == correctLebel)
        *classifiedCorrectly+=1;
    
}

void TrainDataAccuracy(float *d_TrainPixels, float *d_TrainLabels, float *d_WL1, float *d_WL2, float *d_OL1, float *d_OL2){
    //avrg accuracy of the hole train dataset
    float classifiedCorrectly = 0;
    float accuracy;
    float *d_classifiedCorrectly;

    cudaMalloc(&d_classifiedCorrectly, sizeof(float));
    cudaMemcpy(d_classifiedCorrectly, &classifiedCorrectly, sizeof(float), cudaMemcpyHostToDevice);

    for(int y = 0;y < D;y++){
      activateNN_L1<<<4096,1024>>>(d_TrainPixels, y, d_WL1, d_WL2, d_OL1, d_OL2);
      activateNN_L2<<<10,1024>>>(d_TrainPixels, y, d_WL1, d_WL2, d_OL1, d_OL2);
      Classify<<<1,1>>>(d_TrainLabels, d_OL2, d_classifiedCorrectly, y);
    }

    
    cudaMemcpy(&classifiedCorrectly, d_classifiedCorrectly, sizeof(float), cudaMemcpyDeviceToHost);
    accuracy = classifiedCorrectly/D;
    printf("train data accuracy = %f \n", accuracy);

}

int main(){
    // Allocate vectors in device memory
    float *d_TrainPixels;
    float *d_TrainLabels;
    float *d_WL1;
    float *d_WL2;
    float *d_OL1;
    float *d_OL2;
    float *d_EL1;
    float *d_EL2;

    cudaMalloc(&d_TrainPixels, D * N * sizeof(float));
    cudaMalloc(&d_TrainLabels, D * L2 * sizeof(float));
    cudaMalloc(&d_WL1,   L1 * (N + 1)* sizeof(float));
    cudaMalloc(&d_WL2,   L2 * (L1+1)* sizeof(float));
    cudaMalloc(&d_OL1,   L1* sizeof(float));
    cudaMalloc(&d_OL2,   L2* sizeof(float));
    cudaMalloc(&d_EL1,   L1* sizeof(float));
    cudaMalloc(&d_EL2,   L2* sizeof(float));

    LoadTrainData();
    init_Weights();

    //Copy vectors from host memory to device memory
    cudaMemcpy(d_TrainPixels, TrainPixels, D * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TrainLabels, TrainLabels, D * L2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_WL1, WL1, L1 * (N + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_WL2, WL2, L2 * (L1+1) * sizeof(float), cudaMemcpyHostToDevice);


    int rn;
    for(int i=0; i<epochs; i++){
        rn = rand() % D;
        activateNN_L1<<<4096,1024>>>(d_TrainPixels, rn, d_WL1, d_WL2, d_OL1, d_OL2);
        activateNN_L2<<<10,1024>>>(d_TrainPixels, rn, d_WL1, d_WL2, d_OL1, d_OL2);
        calc_Error<<<1,1024>>>(d_TrainLabels, rn, d_WL2, d_OL1 ,d_OL2, d_EL1, d_EL2);
        trainNN<<<4096,1001>>>(d_TrainPixels, d_TrainLabels, rn,  d_WL1,  d_WL2,  d_OL1,  d_OL2,  d_EL1, d_EL2);
        if(i%10000==0){
            TrainDataAccuracy(d_TrainPixels, d_TrainLabels, d_WL1, d_WL2, d_OL1, d_OL2);
        }
    }
    
    cudaDeviceSynchronize();

    //free host
    free(TrainPixels);
    free(TrainLabels);
    free(WL1);
    free(WL2);
    free(OL1);
    free(OL2);
    free(EL1);
    free(EL2);
    
    //free device 
    cudaFree(d_TrainPixels);
    cudaFree(d_TrainLabels);
    cudaFree(d_WL1);
    cudaFree(d_WL2);
    cudaFree(d_OL1);
    cudaFree(d_OL2);
    cudaFree(d_EL1);
    cudaFree(d_EL2);

    return 0; 
}
