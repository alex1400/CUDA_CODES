#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

extern "C"{
#include "input.h"
#include "output.h"
}

#include <cuda_runtime.h>
#include <sys/time.h>

__constant__ size_t d_sizes[2];
__constant__ double d_rotation[3];

static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
		fprintf(stderr,"Cuda error: %s\n", cudaGetErrorString(result));
		exit(1);
    }
}


/* Does the reduction step and return if the convergence has setteled */
static int fill_report(const struct parameters *p, struct results *r,
                        size_t h, size_t w, 
                        double * __restrict__ aa,
                        double * __restrict__ bb,
                        double iter,
                        struct timeval *before)
{
    const double (* __restrict__ a)[p->N+2][p->M+2] = (const double (*)[p->N+2][p->M+2])aa;
	const double (* __restrict__ b)[p->N+2][p->M+2] = (const double (*)[p->N+2][p->M+2])bb;
	/* compute min/max/avg */
    double tmin = INFINITY, tmax = -INFINITY;
    double sum = 0.0;
    double maxdiff = 0.0;
    struct timeval after;

    /* We have said that the final reduction does not need to be included. */
    gettimeofday(&after, NULL);

    for (size_t i = 1; i < h - 1; ++i)
        for (size_t j = 1; j < w - 1; ++j) 
        {
            double v = (*a)[i][j];
            double v_old = (*b)[i][j];
            double diff = fabs(v - v_old);
            sum += v;
            if (tmin > v) tmin = v;
            if (tmax < v) tmax = v;
            if (diff > maxdiff) maxdiff = diff;
        }

    r->niter = iter;
    r->maxdiff = maxdiff;
    r->tmin = tmin;
    r->tmax = tmax;
    r->tavg = sum / (p->N * p->M);

    r->time = (double)(after.tv_sec - before->tv_sec) + 
        (double)(after.tv_usec - before->tv_usec) / 1e6;

    return (maxdiff >= p->threshold) ? 0 : 1;
}



__global__ void SwapKernel(double *__restrict__ src, double *__restrict__ dst, int* convergence){
	
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t h=d_sizes[0];
	const size_t w=d_sizes[1];

	if(id==0){
		*convergence =1;
		size_t i;
		for (i = 0; i < h; ++i) {
			src[i*w+w-1] = src[i*w+1];
			src[i*w] = src[i*w+w-2];	
		}
   }
	
}



__global__ void vectorAddKernel(double* src, double* deviceCOND, double* dst, int* convergence) {
	
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t h=d_sizes[0];
	const size_t w=d_sizes[1];
	double c_cdir=d_rotation[0];
	double c_cdiag=d_rotation[1];
	double threshold=d_rotation[2];
	size_t i , j;
	i=id/w;
	j=id %w;

	if (i!=0 && j!=0 && i<h-1 && j<w-1){

		double val = deviceCOND[i*w+j];
		double restw = 1.0 - val;
		dst[i*w+j] = val * src[i*w+j] + 
				(src[(i+1)*w+j] + src[(i-1)*w+j] + 
				src[i*w+j+1] + src[i*w+j-1]) * (restw * c_cdir) +
				(src[(i-1)*w+j-1] + src[(i-1)*w+j+1] + 
				src[(i+1)*w+j-1] + src[(i+1)*w+j+1]) * (restw * c_cdiag);		
		double diff = fabs(dst[i*w+j] - src[i*w+j]);
		if (diff > threshold) 
			if (*convergence == 1)
				*convergence = 0;
	}

}

extern "C" 
void cuda_do_compute(const struct parameters* p, struct results *r) {
	
	struct timeval before;
    gettimeofday(&before, NULL);
    
	const double c_cdir = 0.25 * M_SQRT2 / (M_SQRT2 + 1.0);
	const double c_cdiag = 0.25 / (M_SQRT2 + 1.0);
	double *rotation = (double*)malloc(sizeof(double)*2);
	size_t *sizes = (size_t *)malloc(sizeof(size_t)*2);
	int *conver=(int *)malloc(sizeof(int));
	size_t N=p->N;
	size_t M=p->M;
	size_t i,j;
	size_t h_host=N+2;
    size_t w_host=M+2;
	size_t n=h_host*w_host;
	int threadBlockSize = 256 ;
	if(n/threadBlockSize >65535)
	   threadBlockSize=512;
	
	rotation[0]=c_cdir;
	rotation[1]=c_cdiag;
	rotation[2]=p->threshold;
	sizes[0]=h_host;
	sizes[1]=w_host;
	
    // alias input parameters
    const double (* __restrict__ tinit)[p->N][p->M] = (const double (*)[p->N][p->M])p->tinit;
    const double (* __restrict__ cinit)[p->N][p->M] = (const double (*)[p->N][p->M])p->conductivity;
	
	double (* __restrict__ g1)[h_host][w_host] = (double (*)[h_host][w_host]) malloc(h_host * w_host * sizeof(double));
    double (* __restrict__ g2)[h_host][w_host] = (double (*)[h_host][w_host]) malloc(h_host * w_host * sizeof(double));
    double (* __restrict__ c)[h_host][w_host] = (double (*)[h_host][w_host]) malloc(h_host * w_host * sizeof(double));

	// g2 is the same as result 
    for (i = 1; i < h_host - 1; ++i)
        for (j = 1; j < w_host - 1; ++j) {
            (*g1)[i][j] = (*tinit)[i - 1][j - 1];
            (*c)[i][j] = (*cinit)[i - 1][j - 1];
        }

    // smear outermost row to border
    for (j = 1; j < w_host - 1; ++j) {
        (*g1)[0][j] = (*g2)[0][j] = (*g1)[1][j];
        (*g1)[h_host - 1][j] = (*g2)[h_host - 1][j] = (*g1)[h_host - 2][j];
    }

	double (* __restrict__ src) = (double *) g2;
    double (* __restrict__ dst) = (double *) g1;
	

    // allocate the vectors on the GPU
    double* deviceCOND = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceCOND, n * sizeof(double)));
    if (deviceCOND == NULL) {
        printf("Error in cudaMalloc! \n");
		return;
    }

    double* deviceTEMP = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceTEMP, n * sizeof(double)));
    if (deviceTEMP == NULL) {
        checkCudaCall(cudaFree(deviceCOND));
        printf("Error in cudaMalloc! \n");
        return;
    }
    double* deviceResult = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceResult, n * sizeof(double)));
    if (deviceResult == NULL) {
        checkCudaCall(cudaFree(deviceTEMP));
        checkCudaCall(cudaFree(deviceCOND));
        printf("Error in cudaMalloc! \n");
        return;
    }
	
    int* convergence= NULL;
	checkCudaCall(cudaMalloc((void **)&convergence, sizeof(int)));
	if (deviceResult == NULL) {
		checkCudaCall(cudaFree(deviceTEMP));
		checkCudaCall(cudaFree(deviceCOND));
		checkCudaCall(cudaFree(deviceResult));
		printf("Error in cudaMalloc! \n");
	}

    // copy the original vectors to the GPU
    checkCudaCall(cudaMemcpy(deviceTEMP, src, n*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaCall(cudaMemcpy(deviceResult, dst, n*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceCOND, c, n*sizeof(double), cudaMemcpyHostToDevice));

	checkCudaCall(cudaMemcpyToSymbol(d_sizes, sizes, 2*sizeof(size_t)));
	checkCudaCall(cudaMemcpyToSymbol(d_rotation, rotation, 3*sizeof(double)));

    // execute kernel
	size_t iter;
	
	for (iter = 1; iter <= p->maxiter; ++iter){
		{double *temp=deviceTEMP; deviceTEMP=deviceResult; deviceResult=temp;}
		SwapKernel<<<1, 32>>>(deviceTEMP, deviceResult, convergence);
		cudaDeviceSynchronize();
		vectorAddKernel<<<n/threadBlockSize , threadBlockSize>>>(deviceTEMP, deviceCOND, deviceResult, convergence);
		cudaDeviceSynchronize();
		checkCudaCall(cudaMemcpy(conver, convergence, sizeof(int), cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		if(*conver==1)
			break;
		// conditional reporting
	   if (iter % p->period == 0) {				
			checkCudaCall(cudaMemcpy(dst, deviceResult, n * sizeof(double), cudaMemcpyDeviceToHost));
			checkCudaCall(cudaMemcpy(src, deviceTEMP, n * sizeof(double), cudaMemcpyDeviceToHost));
			fill_report(p, r, h_host, w_host, dst, src, iter, &before);
			if(p->printreports) report_results(p, r);
		}
	}
	
	if(*conver!=1)
		iter--;
	if(iter % p->period !=0){
		checkCudaCall(cudaMemcpy(dst, deviceResult, n * sizeof(double), cudaMemcpyDeviceToHost));
		checkCudaCall(cudaMemcpy(src, deviceTEMP, n * sizeof(double), cudaMemcpyDeviceToHost));
		fill_report(p, r, h_host, w_host, dst, src, iter, &before);
		report_results(p, r);
	}
	
	
    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    checkCudaCall(cudaMemcpy(g2, deviceResult, n * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(g1, deviceTEMP, n * sizeof(double), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(deviceCOND));
    checkCudaCall(cudaFree(deviceTEMP));
    checkCudaCall(cudaFree(deviceResult));
	checkCudaCall(cudaFree(convergence));
}