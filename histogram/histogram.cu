#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "timer.h"
#include "device_functions.h"

long img_size;
int thread_block_size;

using namespace std;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/

static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}

void usage(){
    cout << "Usage: \n"
            "Set the 'option' flag to \n"
            "\t0: for running naive \n"
            "\t2: for 1 block with loop \n"
            "\t3: for shared memory histogram with reduction \n"
            "Change the 'length' flag to set the length\n"
            "Change the 'block_size' flag to set block size\n"
            "prun -v -1 -np 1 -native '-l gpu=GTX480' myhistogram <option> <length> <block_size>" << endl;
}

void verifyResults(unsigned int *histogramS, unsigned int *histogram, int hist_size){
    // verify the resuls
    for(int i=0; i<hist_size; i++) {
        if (histogram[i]!=histogramS[i]) {
            cout << "error in results! Bin " << i << " is "<< histogram[i] << ", but should be " << histogramS[i] << endl;
            exit(1);
        }
    }
    cout << "results OK!" << endl;
}

__global__ void histogramKernelAtomicLoop(unsigned char* image, long img_size, unsigned int* histogram, int hist_size) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int i, c;
    
    for (i = threadId; i < img_size; i += gridDim.x * blockDim.x) {
        c = image[i];
        atomicAdd(&histogram[c], 1);
    }
}

__global__ void histogramKernelAtomic(unsigned char* image, long img_size, unsigned int* histogram, int hist_size) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    int c;
    if (threadID < img_size) {
        c = image[threadID];
        atomicAdd(&histogram[c], 1);
    }
}

__global__ void histogramKernelShared(unsigned char* image, long img_size, unsigned int* histogram, int hist_size){
    extern __shared__ int local_histo[];
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    int c;

    if (threadIdx.x < hist_size){
        local_histo[threadIdx.x] = 0;
    }

    __syncthreads();
    if (threadID < img_size) {
        c = image[threadID];
        atomicAdd(&local_histo[c], 1);
    } else {
        return;
    }
    __syncthreads();

    if (threadIdx.x < hist_size){
        atomicAdd(&histogram[threadIdx.x], local_histo[threadIdx.x]);
    }
}

void histogramCuda(unsigned char* image, long img_size, unsigned int* histogram, int hist_size, int option) {
    cudaError_t err;

    // allocate the vectors on the GPU
    unsigned char* deviceImage = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceImage, img_size * sizeof(unsigned char)));
    if (deviceImage == NULL) {
        cout << "could not allocate memory!" << endl;
        return;
    }
    unsigned int* deviceHisto = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceHisto, hist_size * sizeof(unsigned int)));
    if (deviceHisto == NULL) {
        checkCudaCall(cudaFree(deviceImage));
        cout << "could not allocate memory!" << endl;
        return;
    }

    err = cudaMemset(deviceHisto, 0, hist_size * sizeof(unsigned int));
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMemset output: %s\n", cudaGetErrorString( err ));  }

    timer kernelTime1 = timer("kernelTime1");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceImage, image, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    if (option == 0) {
        cout << "\n----------------------Running 1 thread per pixel----------------------" << endl;
        histogramKernelAtomic <<< (img_size/thread_block_size)+1, thread_block_size >>>(deviceImage, img_size, deviceHisto, hist_size);
    } else if (option == 1) {
        cout << "\n----------------------Running looping inside block----------------------" << endl;
        histogramKernelAtomicLoop <<< 1, thread_block_size >>> (deviceImage, img_size, deviceHisto, hist_size);
    } else if (option == 2){
        cout << "\n----------------------Running with shared memory----------------------" << endl;
        histogramKernelShared <<< (img_size/thread_block_size)+1, thread_block_size, hist_size*sizeof(int) >>> (deviceImage, img_size, deviceHisto, hist_size);
    } else {
        cout << "\nInvalid option: " << option << endl;
    }

    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(histogram, deviceHisto, hist_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceImage));
    checkCudaCall(cudaFree(deviceHisto));

    cout << "histogram (kernel): \t\t" << kernelTime1  << endl;
    cout << "histogram (memory): \t\t" << memoryTime << endl;
}

void histogramSeq(unsigned char* image, long img_size, unsigned int* histogram, int hist_size) {
  int i; 

  timer sequentialTime = timer("Sequential");
  
  for (i=0; i<hist_size; i++) histogram[i]=0;

  sequentialTime.start();
  for (i=0; i<img_size; i++) {
	histogram[image[i]]++;
  }
  sequentialTime.stop();
  
  cout << "histogram (sequential): \t\t" << sequentialTime << endl;

}

int main(int argc, char* argv[]) {
    int hist_size = 256;
    int option = 0;

    if (argc < 4){
        usage();
        exit(-1);
    }
    option = atoi(argv[1]);
    img_size = atoi(argv[2]);
    thread_block_size = atoi(argv[3]);

    unsigned char *image = (unsigned char *)malloc(img_size * sizeof(unsigned char)); 
    unsigned int *histogramS = (unsigned int *)malloc(hist_size * sizeof(unsigned int));     
    unsigned int *histogram = (unsigned int *)malloc(hist_size * sizeof(unsigned int));

    // initialize the vectors.
    for(long i=0; i<img_size; i++) {
        //image[i] = (unsigned char) (i % hist_size);
        image[i] = (unsigned char) i % hist_size;
    }

    cout << "Compute the histogram of a gray image with "
         << img_size << " pixels using block size " << thread_block_size << endl;
    cout << "\n----------------------Running sequential----------------------" << endl;
    histogramSeq(image, img_size, histogramS, hist_size);

    histogramCuda(image, img_size, histogram, hist_size, option);

    verifyResults(histogramS, histogram, hist_size);

    free(image);
    free(histogram);
    free(histogramS);         
    
    return 0;
}
