#include <stdio.h>
#include <string.h>
#include "timer.h"

#define image_height 1024
#define image_width 1024
#define filter_height 5
#define filter_width 5

#define border_height ((filter_height/2)*2)
#define border_width ((filter_width/2)*2)
#define input_height (image_height + border_height)
#define input_width (image_width + border_width)

#define block_size_x 32
#define block_size_y 32

#define SEED 1234

// 0 = naive
// 1 = constant memory
// 2 = block optimized
// Set by adding argument at runtime
int option = 0;

__constant__ float d_filter[filter_height*filter_width];
__constant__ int SHARED_SIZE;

using namespace std;

void printImage(float *image, int width, int height){
    int i, x;
    int step = width;
    for (i = 0; i < height; i++) {
        for (x = i * step; x < i * step + width; x++) {
            printf("%f ", image[x]);
        }
        printf("\n");
    }
}

void convolutionSeq(float *output, float *input, float *filter) {
    //for each pixel in the output image

    timer sequentialTime = timer("Sequential");

    sequentialTime.start();

    for (int y = 0; y < image_height; y++) {
        for (int x = 0; x < image_width; x++) {

            //for each filter weight
            for (int i = 0; i < filter_height; i++) {
                for (int j = 0; j < filter_width; j++) {
                    output[y * image_width + x] += input[(y + i) * input_width + x + j] * filter[i * filter_width + j];
                }
            }

            output[y * image_width + x] = output[y * image_width + x] / (filter_width * filter_height);

        }
    }
    sequentialTime.stop();
    cout << "convolution (sequential): \t\t" << sequentialTime << endl;

}


__global__ void convolution_kernel_naive(float *output, float *input, float *filter) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (threadId > image_height*image_width)
        return;

    int height_pos = threadId / image_width;
    int width_pos = threadId % image_width;
    int i, j;
    float value = 0;

    // for each filter weight
    for (i = 0; i < filter_height; i++) {
        for (j = 0; j < filter_width; j++) {
            value += input[(height_pos + i) * input_width + width_pos + j] * filter[i * filter_width + j];
        }
    }
    output[threadId] = value / (filter_width * filter_height);
}

__global__ void convolution_kernel_constant_filter(float *output, float *input){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (threadId > image_height * image_width)
        return;

    int height_pos = threadId / image_width;
    int width_pos = threadId % image_width;
    int i, j;
    float value = 0;

    // for each filter weight
    for (i = 0; i < filter_height; i++) {
        for (j = 0; j < filter_width; j++) {
            value += input[(height_pos + i) * input_width + width_pos + j] * d_filter[i * filter_width + j];
        }
    }
    output[threadId] = value / (filter_width * filter_height);
}

__global__ void convolution_kernel_block_optimized(float *output, float *input, int shared_size) {
    // Create a block local shared array of Input, to increase read times
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (threadId > image_height*image_width)
        return;

    extern __shared__ float data[];

    int i, j;
    int n=0;
    float value = 0;

    int localId = (threadIdx.y * blockDim.x) + threadIdx.x;
    int height_pos = threadId / image_width;
    int width_pos = threadId % image_width;
    int local_height_pos = height_pos - (blockId * ((blockDim.x * blockDim.y) / image_width));

    if (localId == 0){
        for (i = height_pos * input_width; i < (height_pos * input_width) + ((blockDim.y+border_width) * input_width); i++){
            if (i >= input_width*input_height)
                break;
            data[n] = input[i];
            n++;
        }
    }
    __syncthreads();

    for (i = 0; i < filter_height; i++) {
        for (j = 0; j < filter_width; j++) {
            value += data[(local_height_pos + i) * input_width + width_pos + j] * d_filter[i * filter_width + j];
        }
    }

    output[threadId] = value / (filter_width * filter_height);

}

void convolutionCUDA(float *output, float *input, float *filter, int option) {
    float *d_input;
    float *d_output;
    float *local_filter;
    cudaError_t err;
    timer kernelTime = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // memory allocation
    err = cudaMalloc((void **) &d_input, input_height * input_width * sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMalloc d_input: %s\n", cudaGetErrorString(err)); }
    err = cudaMalloc((void **) &d_output, image_height * image_width * sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMalloc d_output: %s\n", cudaGetErrorString(err)); }

    if(option == 0){
        err = cudaMalloc((void **) &local_filter, filter_height * filter_width * sizeof(float));
        if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMalloc local_filter: %s\n", cudaGetErrorString(err)); }
    }

    memoryTime.start();
    // host to device
    err = cudaMemcpy(d_input, input, input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {fprintf(stderr, "Error in cudaMemcpy host to device input: %s\n", cudaGetErrorString(err));}

    if (option == 0){
        err = cudaMemcpy(local_filter, filter, filter_height * filter_width * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {fprintf(stderr, "Error in cudaMemcpy host to device filter: %s\n", cudaGetErrorString(err));}
    } else if(option == 1 || option == 2) {
        err = cudaMemcpyToSymbol(d_filter, filter, filter_height * filter_width * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "Error in cudaMemcpyToSymbol filter: %s\n", cudaGetErrorString(err));
        }
    }

    // zero the result array
    err = cudaMemset(d_output, 0, image_height * image_width * sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMemset output: %s\n", cudaGetErrorString(err)); }
    memoryTime.stop();

    //setup the grid and thread blocks
    //thread block size
    dim3 threads(block_size_x, block_size_y);
    //problem size divided by thread block size rounded up
    dim3 grid(int(ceilf(image_width / (float) threads.x)), int(ceilf(image_height / (float) threads.y)));

    int shared_size = (block_size_y+border_height) * input_width * sizeof(int);

    //measure the GPU function
    kernelTime.start();
    if(option == 0){
        printf("\nRunning block naive\n");
        kernelTime.start();
        convolution_kernel_naive<<<grid, threads>>> (d_output, d_input, local_filter);
        cudaDeviceSynchronize();
        kernelTime.stop();
    }else if (option == 1){
        printf("\nRunning block constant filter\n");
        kernelTime.start();
        convolution_kernel_constant_filter<<<grid, threads>>> (d_output, d_input);
        cudaDeviceSynchronize();
        kernelTime.stop();
    }else if(option == 2) {
        printf("\nRunning block optimized\n");
        kernelTime.start();
        convolution_kernel_block_optimized <<<grid, threads, shared_size>>> (d_output, d_input, shared_size);
        cudaDeviceSynchronize();
        kernelTime.stop();
    }
    cudaDeviceSynchronize();
    kernelTime.stop();

    //check to see if all went well
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error during kernel launch convolution_kernel: %s\n", cudaGetErrorString(err));
    }

    //copy the result back to host memory
    memoryTime.start();
    err = cudaMemcpy(output, d_output, image_height * image_width * sizeof(float), cudaMemcpyDeviceToHost);
    memoryTime.stop();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpy device to host output: %s\n", cudaGetErrorString(err));
    }

    err = cudaFree(d_input);
    if (err != cudaSuccess) { fprintf(stderr, "Error in freeing d_input: %s\n", cudaGetErrorString(err)); }
    err = cudaFree(d_output);
    if (err != cudaSuccess) { fprintf(stderr, "Error in freeing d_output: %s\n", cudaGetErrorString(err)); }
    if(option == 0){
        err = cudaFree(local_filter);
        if (err != cudaSuccess) { fprintf(stderr, "Error in freeing d_output: %s\n", cudaGetErrorString(err)); }
    }

    cout << "convolution (kernel): \t\t" << kernelTime << endl;
    cout << "convolution (memory): \t\t" << memoryTime << endl;

}

int compare_arrays(float *a1, float *a2, int n) {
    int errors = 0;
    int print = 0;

    for (int i = 0; i < n; i++) {

        if (isnan(a1[i]) || isnan(a2[i])) {
            errors++;
            if (print < 10) {
                print++;
                fprintf(stderr, "Error NaN detected at i=%d,\t a1= %10.7e \t a2= \t %10.7e\n", i, a1[i], a2[i]);
            }
        }

        float diff = (a1[i] - a2[i]) / a1[i];
        if (diff > 1e-6f) {
            errors++;
            if (print < 10) {
                print++;
                fprintf(stderr, "Error detected at i=%d, \t a1= \t %10.7e \t a2= \t %10.7e \t rel_error=\t %10.7e\n", i,
                        a1[i], a2[i], diff);
            }
        }

    }

    return errors;
}

int main(int argc, char* argv[] ){
    int i;
    int errors = 0;

    if (argc > 1){
        option = atoi(argv[1]);
    } else {
        printf("Usage:\n"
                       "Set option for choosing what CUDA version to run\n"
                       "  0: Naive\n"
                       "  1: Constant filter\n"
                       "  2: Shared memory (not working)\n"
                       "prun -np 1 -native '-l gpu=GTX480' myconvolution <option>\n");
        return -1;
    }

    //allocate arrays and fill them
    float *input = (float *) malloc(input_height * input_width * sizeof(float));
    float *output1 = (float *) calloc(image_height * image_width, sizeof(float));
    float *output2 = (float *) calloc(image_height * image_width, sizeof(float));
    float *filter = (float *) malloc(filter_height * filter_width * sizeof(float));

    for (i = 0; i < input_height * input_width; i++) {
        input[i] = (float) (i % SEED);
    }

    //THis is specific for a W==H smoothening filteri, where W and H are odd.

    for (i = 0; i < filter_height * filter_width; i++) {
        filter[i] = 1.0;
    }

    for (i = filter_width + 1; i < (filter_height - 1) * filter_width; i++) {
        if (i % filter_width > 0 && i % filter_width < filter_width - 1) filter[i] += 1.0;
    }

    filter[filter_width * filter_height / 2] = 3.0;
    //end initialization

    //measure the CPU function
    convolutionSeq(output1, input, filter);
    //measure the GPU function
    convolutionCUDA(output2, input, filter, option);

    //check the result
    errors += compare_arrays(output1, output2, image_height * image_width);
    if (errors > 0) {
        printf("TEST FAILED! %d errors!\n", errors);
    } else {
        printf("TEST PASSED!\n");
    }

    free(filter);
    free(input);
    free(output1);
    free(output2);

    return 0;
}


