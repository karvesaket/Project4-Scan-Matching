#define GLM_FORCE_CUDA
#include <cuda.h>
#include "common.h"
#include "naivegpu.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <glm/glm.hpp>

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
//void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
//	int lda = m, ldb = k, ldc = m;
//	const float alf = 1;
//	const float bet = 0;
//	const float *alpha = &alf;
//	const float *beta = &bet;
//
//	// Do the actual multiplication
//	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
//}

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

	void match(float* x, float* y, int numX, int numY, int max_iterations) {
		//Copy data to GPU
		cudaMalloc((void**)&dev_x, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		cudaMemcpy(dev_x, x, sizeof(float) * numX, cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_y, numY * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_y failed!");
		cudaMemcpy(dev_y, y, sizeof(float) * numY, cudaMemcpyHostToDevice);

		int iter = 0;
		while (iter < max_iterations) {
			//Find Correspondence
			dim3 numBlocks(((numX/3) + blockSize - 1) / blockSize);
			findCorrespondence << <numBlocks, blockSize >> >(dev_x, numX, dev_y, numY, dev_x_corr);

			//Transpose x_corr and x
			float* dev_x_tr;
			cudaMalloc((void**)&dev_x_tr, numX * sizeof(float));
			checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
			dim3 numBlocks1(((numX) + blockSize - 1) / blockSize);
			transpose << <numBlocks1, blockSize >> >(dev_x, dev_x_tr, numX/3, 3);

			float* dev_x_corr_tr;
			cudaMalloc((void**)&dev_x_corr_tr, numX * sizeof(float));
			checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
			
			transpose << <numBlocks1, blockSize >> >(dev_x_corr, dev_x_corr_tr, numX / 3, 3);

			int each = numX / 3;

			//Mean-center x
			float* meanX;
			cudaMalloc((void**)&meanX, sizeof(float));
			checkCUDAErrorWithLine("cudaMalloc sum failed!");
			getArraySum(each, dev_x_tr, meanX);

			float* meanY;
			cudaMalloc((void**)&meanY, sizeof(float));
			checkCUDAErrorWithLine("cudaMalloc sum failed!");
			getArraySum(each, dev_x_tr + each, meanY);

			float* meanZ;
			cudaMalloc((void**)&meanZ, sizeof(float));
			checkCUDAErrorWithLine("cudaMalloc sum failed!");
			getArraySum(each, dev_x_tr + (each * 2), meanZ);


			//Mean-center x_corr
			float* meanXC;
			cudaMalloc((void**)&meanXC, sizeof(float));
			checkCUDAErrorWithLine("cudaMalloc sum failed!");
			getArraySum(each, dev_x_corr_tr, meanXC);

			float* meanYC;
			cudaMalloc((void**)&meanYC, sizeof(float));
			checkCUDAErrorWithLine("cudaMalloc sum failed!");
			getArraySum(each, dev_x_corr_tr + each, meanYC);

			float* meanZC;
			cudaMalloc((void**)&meanZC, sizeof(float));
			checkCUDAErrorWithLine("cudaMalloc sum failed!");
			getArraySum(each, dev_x_corr_tr + (each * 2), meanZC);
			
			meanCenter <<<numBlocks, blockSize >>>(dev_x, each, *meanX, *meanY, *meanZ);
			meanCenter <<<numBlocks, blockSize >>>(dev_x_corr, each, *meanXC, *meanYC, *meanZC);

			//Multiply x_corr_tr and x to get input to SVD

			//Find SVD - U, V, S

			//Compute U x V_tr to get R

			//Compute translation = x_corr_mean - R.x_mean
		}
	}
}

