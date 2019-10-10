#define GLM_FORCE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "common.h"
#include "naivegpu.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <glm/glm.hpp>
#include <cublas_v2.h>
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\reduce.h>
#include "svd3.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define blockSize 128


__global__ void findCorrespondence(float* arr1, long numArr1, float* arr2, long numArr2, float* arr1_correspondence) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (numArr1/3)) {
		return;
	}
	glm::vec3 point(arr1[index * 3 + 0], arr1[index * 3 + 1], arr1[index * 3 + 2]);
	float min_dist = LONG_MAX;
	glm::vec3 closest_point;
	for (int j = 0; j < numArr2 / 3; j++) {
		glm::vec3 other_point(arr2[j * 3 + 0], arr2[j * 3 + 1], arr2[j * 3 + 2]);
		float dist = glm::distance(point, other_point);
		if (dist < min_dist) {
			closest_point = other_point;
			min_dist = dist;
		}
	}
	arr1_correspondence[index * 3 + 0] = closest_point.x;
	arr1_correspondence[index * 3 + 1] = closest_point.y;
	arr1_correspondence[index * 3 + 2] = closest_point.z;
}

__global__ void transpose(float* arr, float* arrTrans, int m, int n) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= m*n) {
		return;
	}
	int i = index / n;
	int j = index % n;

	arrTrans[m*j + i] = arr[n*i + j];
}


// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda = m, ldb = k, ldc = m;
	 float alf = 1;
	 
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

__global__ void matrix_subtraction(float* A, float* B, float* C, int m, int n) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= m*n) {
		return;
	}
	C[index] = A[index] - B[index];
}

__global__ void kernMatrixMultiply(float *dev_A, float *dev_B, float *dev_C, int m, int n, int k) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0;
	if (col < k && row < m)
	{
		for (int i = 0; i < n; i++)
			sum += dev_A[row * n + i] * dev_B[i * k + col];
		dev_C[row * k + col] = sum;
	}
}

// A - m x n || B - n x p and returns C - m x p
void multiplyMatrixCPU(float* A, float* B, float* C, int m, int n, int p) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			C[i*p + j] = 0;
			for (int k = 0; k < n; k++) {
				C[i*p + j] += (A[i*n + k] * B[k*p + j]);
			}
		}
	}
}


__global__ void addTranslation(float* A, float* trans, int num) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num) {
		return;
	}
	A[index * 3 + 0] += trans[0];
	A[index * 3 + 1] += trans[1];
	A[index * 3 + 2] += trans[2];
}

__global__ void upSweepOptimized(int n, int d, float* A) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);


	int other_index = 1 << d;
	int stride = other_index * 2;

	int new_index = stride * index;
	if (new_index >= n) {
		return;
	}
	A[new_index + stride - 1] += A[new_index + other_index - 1];
}

__global__ void meanCenter(float* arr, float* centered, int num, float mx, float my, float mz) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num) {
		return;
	}
	centered[index * 3 + 0] = arr[index * 3 + 0] - mx;
	centered[index * 3 + 1] = arr[index * 3 + 1] - my;
	centered[index * 3 + 2] = arr[index * 3 + 2] - mz;
}

__global__ void setValueOnDevice(float* device_var, float val) {
	*device_var = val;
}

//__global__ void get_svd(float* input, float* u, float* s, float* v) {
//	svd(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8],
//		u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8],
//		s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8],
//		v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
//}

__global__ void divide_sum_to_mean(float* sum, int num) {
	(*sum) = (*sum) / num;
}

void getArraySum(int n, float* input, float* sum) {
	float* padded_idata;
	int padded_size = 1 << (ilog2ceil(n));

	cudaMalloc((void**)&padded_idata, padded_size * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc padded_idata failed!");

	cudaMemset(padded_idata, 0, padded_size * sizeof(float));
	cudaMemcpy(padded_idata, input, sizeof(float) * n, cudaMemcpyDeviceToDevice);

	int iterations = ilog2(padded_size);

	int number_of_threads = padded_size;
	for (int d = 0; d < iterations; d++) {
		number_of_threads /= 2;
		dim3 fullBlocksPerGridUpSweep((number_of_threads + blockSize - 1) / blockSize);
		upSweepOptimized << <fullBlocksPerGridUpSweep, blockSize >> >(padded_size, d, padded_idata);
	}

	cudaMemcpy(sum, padded_idata + padded_size - 1, sizeof(float), cudaMemcpyDeviceToDevice);

	cudaFree(padded_idata);
}

void printMatrix(float* A, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << A[i*n + j] << " ";
		}
		std::cout << std::endl;
	}
}

namespace NaiveGPU {
	float* dev_x;
	float* dev_y;

	float* dev_x_corr;
	float* dev_R;
	float* dev_translation;


	void initScan(int numX) {
		cudaMalloc((void**)&dev_x_corr, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x_corr failed!");

		cudaMalloc((void**)&dev_R, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_R failed!");

		cudaMalloc((void**)&dev_translation, 3 * 1 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_translation failed!");
	}

	void match(float* x, float* y, int numX, int numY) {

		int eachX = numX / 3;
		int eachY = numY / 3;

		dim3 numBlocks((eachX + blockSize - 1) / blockSize);
		dim3 numBlocks1((numX+blockSize - 1) / blockSize);
		dim3 numBlocks2((3 * 3 + blockSize - 1) / blockSize);
		dim3 numBlocks3((3 * 1 + blockSize - 1) / blockSize);

		dim3 dimBlock(16, 16);
		dim3 dimGrid;

		//Copy data to GPU
		cudaMalloc((void**)&dev_x, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		cudaMemcpy(dev_x, x, sizeof(float) * numX, cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_y, numY * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_y failed!");
		cudaMemcpy(dev_y, y, sizeof(float) * numY, cudaMemcpyHostToDevice);

		
		//Find correspondence
		
		//Find Correspondence
		findCorrespondence << <numBlocks, blockSize >> >(dev_x, numX, dev_y, numY, dev_x_corr);

		//Transpose x_corr and x
		float* dev_x_tr;
		cudaMalloc((void**)&dev_x_tr, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		transpose << <numBlocks1, blockSize >> >(dev_x, dev_x_tr, eachX, 3);

		float* dev_x_corr_tr;
		cudaMalloc((void**)&dev_x_corr_tr, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		transpose << <numBlocks1, blockSize >> >(dev_x_corr, dev_x_corr_tr, eachX, 3);

		
		float meanX;
		meanX = thrust::reduce(thrust::device, dev_x_tr, dev_x_tr + eachX, 0.0f);
		meanX /= eachX;
		
		float meanY;
		meanY = thrust::reduce(thrust::device, dev_x_tr + eachX, dev_x_tr + (2 * eachX), 0.0f);
		meanY /= eachX;
		
		float meanZ;
		meanZ = thrust::reduce(thrust::device, dev_x_tr + (2 * eachX), dev_x_tr + numX, 0.0f);
		meanZ /= eachX;
		

		
		
		cudaFree(dev_x_tr);

		
		//Mean-center x_corr
		float meanXC;
		meanXC = thrust::reduce(thrust::device, dev_x_corr_tr, dev_x_corr_tr + eachX, 0.0f);
		meanXC /= eachX;
		
		float meanYC;
		meanYC = thrust::reduce(thrust::device, dev_x_corr_tr + eachX, dev_x_corr_tr + (2 * eachX), 0.0f);
		meanYC /= eachX;
		
		float meanZC;
		meanZC = thrust::reduce(thrust::device, dev_x_corr_tr + (2 * eachX), dev_x_corr_tr + numX, 0.0f);
		meanZC /= eachX;
		

		

		cudaFree(dev_x_corr_tr);

		float* dev_x_mean_center;
		cudaMalloc((void**)&dev_x_mean_center, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		

		float* dev_x_corr_mean_center;
		cudaMalloc((void**)&dev_x_corr_mean_center, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_y failed!");
		

		
		meanCenter <<<numBlocks, blockSize >>>(dev_x, dev_x_mean_center, eachX, meanX, meanY, meanZ);
		
		meanCenter <<<numBlocks, blockSize >>>(dev_x_corr, dev_x_corr_mean_center, eachX, meanXC, meanYC, meanZC);

		
		//Multiply x_corr_tr and x to get input to SVD
		cudaMalloc((void**)&dev_x_corr_tr, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		transpose << <numBlocks1, blockSize >> > (dev_x_corr_mean_center, dev_x_corr_tr, eachX, 3);

		
		

		float* dev_to_svd;
		cudaMalloc((void**)&dev_to_svd, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		float* to_svd = (float*)malloc(3 * 3 * sizeof(float));

		//multiplyMatrixCPU(y_tr_cpu, x_cpu, to_svd, 3, eachX, 3);

		//cudaMemcpy(dev_to_svd, to_svd, sizeof(float) * 9, cudaMemcpyHostToDevice);

		dimGrid.x = (3 + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (3 + dimBlock.y - 1) / dimBlock.y;
		kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_x_corr_tr, dev_x_mean_center, dev_to_svd, 3, eachX, 3);

		// Create a handle for CUBLAS
		cublasHandle_t handle;
		cublasCreate(&handle);
		//gpu_blas_mmul(handle, dev_x_corr_tr, dev_x, dev_to_svd, 3, eachX, 3);
		//multiplyMatrix<<<numBlocks2, blockSize>> > (dev_x_corr_tr, dev_x, dev_to_svd, 3, eachX, 3);

		
		cudaMemcpy(to_svd, dev_to_svd, sizeof(float) * 9, cudaMemcpyDeviceToHost);
		

		float* svd_u = (float*)malloc(3 * 3 * sizeof(float));
		memset(svd_u, 0.0f, 3 * 3 * sizeof(float));
		float* svd_v = (float*)malloc(3 * 3 * sizeof(float));
		memset(svd_v, 0.0f, 3 * 3 * sizeof(float));
		float* svd_s = (float*)malloc(3 * 3 * sizeof(float));
		memset(svd_s, 0.0f, 3 * 3 * sizeof(float));

		svd(to_svd[0], to_svd[1], to_svd[2], to_svd[3], to_svd[4], to_svd[5], to_svd[6], to_svd[7], to_svd[8],
			svd_u[0], svd_u[1], svd_u[2], svd_u[3], svd_u[4], svd_u[5], svd_u[6], svd_u[7], svd_u[8],
			svd_s[0], svd_s[1], svd_s[2], svd_s[3], svd_s[4], svd_s[5], svd_s[6], svd_s[7], svd_s[8],
			svd_v[0], svd_v[1], svd_v[2], svd_v[3], svd_v[4], svd_v[5], svd_v[6], svd_v[7], svd_v[8]);

		
		//Find SVD - U, V, S
		float* dev_svd_u;
		cudaMalloc((void**)&dev_svd_u, 3 * 3 * sizeof(float));
		cudaMemset(dev_svd_u, 0.0f, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		float* dev_svd_s;
		cudaMalloc((void**)&dev_svd_s, 3 * 3 * sizeof(float));
		cudaMemset(dev_svd_s, 0.0f, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		float* dev_svd_v;
		cudaMalloc((void**)&dev_svd_v, 3 * 3 * sizeof(float));
		cudaMemset(dev_svd_v, 0.0f, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		//get_svd << <1, 1 >> > (dev_to_svd, dev_svd_u, dev_svd_s, dev_svd_v);

		//float* u = (float*)malloc(3 * 3 * sizeof(float));
		cudaMemcpy(dev_svd_u, svd_u, sizeof(float) * 9, cudaMemcpyHostToDevice);
		

		//float* v = (float*)malloc(3 * 3 * sizeof(float));
		cudaMemcpy(dev_svd_v, svd_v, sizeof(float) * 9, cudaMemcpyHostToDevice);
		

		
		cudaFree(dev_svd_s);
		//Compute U x V_tr to get R
		float* dev_svd_v_tr;
		cudaMalloc((void**)&dev_svd_v_tr, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");
		transpose << <numBlocks2, blockSize >> > (dev_svd_v, dev_svd_v_tr, 3, 3);

		float* v_tr = (float*)malloc(3 * 3 * sizeof(float));
		cudaMemcpy(v_tr, dev_svd_v_tr, sizeof(float) * 9, cudaMemcpyDeviceToHost);
		

		cudaFree(dev_svd_v);
		

		
		dimGrid.x = (3 + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (3 + dimBlock.y - 1) / dimBlock.y;
		kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_svd_u, dev_svd_v_tr, dev_R, 3, 3, 3);

		//gpu_blas_mmul(handle, dev_svd_u, dev_svd_v_tr, dev_R, 3, 3, 3);

		float* R = (float*)malloc(3 * 3 * sizeof(float));
		cudaMemcpy(R, dev_R, sizeof(float) * 9, cudaMemcpyDeviceToHost);
		

		
		//Compute translation = x_corr_mean - R.x_mean
		float* dev_x_mean;
		cudaMalloc((void**)&dev_x_mean, 3 * sizeof(float));
		setValueOnDevice << <1, 1 >> > (&dev_x_mean[0], meanX);
		setValueOnDevice << <1, 1 >> > (&dev_x_mean[1], meanY);
		setValueOnDevice << <1, 1 >> > (&dev_x_mean[2], meanZ);

		float* dev_y_mean;
		cudaMalloc((void**)&dev_y_mean, 3 * sizeof(float));
		setValueOnDevice << <1, 1 >> > (&dev_y_mean[0], meanXC);
		setValueOnDevice << <1, 1 >> > (&dev_y_mean[1], meanYC);
		setValueOnDevice << <1, 1 >> > (&dev_y_mean[2], meanZC);

		

		float* dev_R_tr;
		cudaMalloc((void**)&dev_R_tr, 9 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		transpose << <numBlocks2, blockSize >> > (dev_R, dev_R_tr, 3, 3);

		float* inter;
		cudaMalloc((void**)&inter, 3 * 1 * sizeof(float));
		dimGrid.x = (3 + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (1 + dimBlock.y - 1) / dimBlock.y;
		kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_R, dev_x_mean, inter, 3, 3, 1);
		//gpu_blas_mmul(handle, dev_R, dev_x_mean, inter, 3, 3, 1);

		matrix_subtraction << <numBlocks3, blockSize >> > (dev_y_mean, inter, dev_translation, 1, 3);

		float* trans = (float*)malloc(3 * 1 * sizeof(float));
		cudaMemcpy(trans, dev_translation, sizeof(float) * 3, cudaMemcpyDeviceToHost);
		

		
		
		//Apply rotation on x
		float* dev_newX;
		cudaMalloc((void**)&dev_newX, numX * sizeof(float));
		dimGrid.x = (3 + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (eachX + dimBlock.y - 1) / dimBlock.y;
		kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_x, dev_R_tr, dev_newX, eachX, 3, 3);
		//gpu_blas_mmul(handle, dev_x, dev_R, dev_newX, eachX, 3, 3);
		cudaDeviceSynchronize();

		//Apply translation on x
		addTranslation << <numBlocks, blockSize >> > (dev_newX, dev_translation, eachX);

		cudaMemcpy(x, dev_newX, sizeof(float) * numX, cudaMemcpyDeviceToHost);
		
		cudaDeviceSynchronize();
	}
}

