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
#include "svd3_cuda.h"

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

__global__ void meanCenter(float* arr, int num, float mx, float my, float mz) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num) {
		return;
	}
	arr[index * 3 + 0] -= mx;
	arr[index * 3 + 1] -= my;
	arr[index * 3 + 2] -= mz;
}

__global__ void setValueOnDevice(float* device_var, int val) {
	*device_var = val;
}

__global__ void get_svd(float* input, float* u, float* s, float* v) {
	svd(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8],
		u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8],
		s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8],
		v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
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

		//Copy data to GPU
		cudaMalloc((void**)&dev_x, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		cudaMemcpy(dev_x, x, sizeof(float) * numX, cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_y, numY * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_y failed!");
		cudaMemcpy(dev_y, y, sizeof(float) * numY, cudaMemcpyHostToDevice);

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

		//Mean-center x
		float* meanX;
		cudaMalloc((void**)&meanX, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc sum failed!");
		getArraySum(eachX, dev_x_tr, meanX);

		float* meanY;
		cudaMalloc((void**)&meanY, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc sum failed!");
		getArraySum(eachX, dev_x_tr + eachX, meanY);

		float* meanZ;
		cudaMalloc((void**)&meanZ, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc sum failed!");
		getArraySum(eachX, dev_x_tr + (eachX * 2), meanZ);

		cudaFree(dev_x_tr);

		//Mean-center x_corr
		float* meanXC;
		cudaMalloc((void**)&meanXC, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc sum failed!");
		getArraySum(eachX, dev_x_corr_tr, meanXC);

		float* meanYC;
		cudaMalloc((void**)&meanYC, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc sum failed!");
		getArraySum(eachX, dev_x_corr_tr + eachX, meanYC);

		float* meanZC;
		cudaMalloc((void**)&meanZC, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc sum failed!");
		getArraySum(eachX, dev_x_corr_tr + (eachX * 2), meanZC);
		
		cudaFree(dev_x_corr_tr);

		meanCenter <<<numBlocks, blockSize >>>(dev_x, eachX, *meanX, *meanY, *meanZ);
		meanCenter <<<numBlocks, blockSize >>>(dev_x_corr, eachX, *meanXC, *meanYC, *meanZC);

		//Multiply x_corr_tr and x to get input to SVD
		cudaMalloc((void**)&dev_x_corr_tr, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		transpose << <numBlocks1, blockSize >> > (dev_x_corr, dev_x_corr_tr, eachX, 3);

		float* dev_to_svd;
		cudaMalloc((void**)&dev_to_svd, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		// Create a handle for CUBLAS
		cublasHandle_t handle;
		cublasCreate(&handle);
		gpu_blas_mmul(handle, dev_x_corr_tr, dev_x, dev_to_svd, 3, eachX, 3);

		//Find SVD - U, V, S
		float* dev_svd_u;
		cudaMalloc((void**)&dev_svd_u, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		float* dev_svd_s;
		cudaMalloc((void**)&dev_svd_s, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		float* dev_svd_v;
		cudaMalloc((void**)&dev_svd_v, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		get_svd << <1, 1 >> > (dev_to_svd, dev_svd_u, dev_svd_s, dev_svd_v);

		cudaFree(dev_svd_s);
		//Compute U x V_tr to get R
		float* dev_svd_v_tr;
		cudaMalloc((void**)&dev_svd_v_tr, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");
		transpose << <numBlocks2, blockSize >> > (dev_svd_v, dev_svd_v_tr, 3, 3);

		cudaFree(dev_svd_v);

		gpu_blas_mmul(handle, dev_svd_u, dev_svd_v_tr, dev_R, 3, 3, 3);

		//Compute translation = x_corr_mean - R.x_mean
		float* dev_x_mean;
		cudaMalloc((void**)&dev_x_mean, 3 * sizeof(float));
		dev_x_mean[0] = *meanX;
		dev_x_mean[1] = *meanY;
		dev_x_mean[2] = *meanZ;

		float* dev_y_mean;
		cudaMalloc((void**)&dev_y_mean, 3 * sizeof(float));
		dev_y_mean[0] = *meanXC;
		dev_y_mean[1] = *meanYC;
		dev_y_mean[2] = *meanZC;

		float* inter;
		cudaMalloc((void**)&inter, 3 * 1 * sizeof(float));
		gpu_blas_mmul(handle, dev_R, dev_x_mean, inter, 3, 3, 1);

		matrix_subtraction << <numBlocks3, blockSize >> > (dev_y_mean, inter, dev_translation, 1, 3);

		//Apply rotation on x
		float* dev_newX;
		cudaMalloc((void**)&dev_newX, numX * sizeof(float));
		gpu_blas_mmul(handle, dev_x, dev_R, dev_newX, eachX, 3, 3);

		//Apply translation on x
		addTranslation << <numBlocks, blockSize >> > (dev_newX, dev_translation, eachX);

		cudaMemcpy(x, dev_newX, sizeof(float) * numX, cudaMemcpyDeviceToHost);
	}
}

