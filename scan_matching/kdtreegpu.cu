#define GLM_FORCE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "common.h"
#include "kdtreegpu.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <glm/glm.hpp>
#include <cublas_v2.h>
#include "svd3_cuda.h"
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define blockSize 128

__device__ glm::vec3 getClosestPointKDTree(glm::vec4* tree, int num, glm::vec3 target_point) {

}

__global__ void findCorrespondenceKDTree(float* arr1, long numArr1, glm::vec4* arr2, long numArr2, float* arr1_correspondence) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (numArr1 / 3)) {
		return;
	}
	glm::vec3 point(arr1[index * 3 + 0], arr1[index * 3 + 1], arr1[index * 3 + 2]);
	float min_dist = LONG_MAX;
	glm::vec3 closest_point = getClosestPointKDTree(arr2, numArr2, point);

	arr1_correspondence[index * 3 + 0] = closest_point.x;
	arr1_correspondence[index * 3 + 1] = closest_point.y;
	arr1_correspondence[index * 3 + 2] = closest_point.z;
}

void convertToVec3(float* arr, glm::vec4* arr_new, int num) {
	for (int i = 0; i < num; i++) {
		arr_new[i] = glm::vec4(arr[i * 3 + 0], arr[i * 3 + 1], arr[i * 3 + 2], 1.0f);
	}
	//printf("HERE %f\n", arr_new[index]);
}

struct XComp {
	inline bool operator() (const glm::vec4 a, const glm::vec4 b) {
		return a.x < b.x;
	}
};

struct YComp {
	inline bool operator() (const glm::vec4 a, const glm::vec4 b) {
		return a.y < b.y;
	}
};

struct ZComp {
	inline bool operator() (const glm::vec4 a, const glm::vec4 b) {
		return a.z < b.z;
	}
};

void buildKdTreeRecursive(glm::vec4* points, int beg, int end, int align, int root, glm::vec4* tree) {
	//Sort the array from beg to end based on its alignment
	//thrust::device_ptr<glm::vec4> thrust_points(points);
	if (beg > end) return;
	if (align == 0) {
		thrust::sort(thrust::host, points + beg, points + end, XComp());
	}
	else if (align == 1) {
		thrust::sort(thrust::host, points + beg, points + end, YComp());
	}
	else if (align == 2) {
		thrust::sort(thrust::host, points + beg, points + end, ZComp());
	}
	int mid = (beg + end) / 2;
	tree[root] = points[mid];
	//Recurse over left subtree
	buildKdTreeRecursive(points, beg, mid - 1, (align + 1) % 3, 2 * root + 1, tree);

	//Recurse over right subtree
	buildKdTreeRecursive(points, mid + 1, end, (align + 1) % 3, 2 * root + 2, tree);
}

void getKdTree(glm::vec4* tree, glm::vec4* points, int num) {
	std::cout << "here\n";
	buildKdTreeRecursive(points, 0, num-1, 0, 0, tree);
}

void printVec4(glm::vec4* print, int num) {
	for (int i = 0; i < num; i++) {
		std::cout << print[i].x << " " << print[i].y << " " << print[i].z << " "<<print[i].w << std::endl;
	}
	
}

namespace KDTreeGPU {
	float* dev_x;
	float* dev_y;
	glm::vec4* y_vec3;
	glm::vec4* tree;

	float* dev_x_corr;
	float* dev_R;
	float* dev_translation;

	void buildTree(float* points, int num) {
		//Convert float array to array of vec3
		int each = num / 3;
		y_vec3 = (glm::vec4*)malloc(each * sizeof(glm::vec4));
		std::cout << "Converting to vec3\n";
		convertToVec3 (points, y_vec3, each);
		printVec4(y_vec3, each);

		std::cout << "Building Tree\n";
		//Call recursive device function to build tree
		int treeSize = 1 << (ilog2ceil(each) + 1);
		std::cout << num << " " << treeSize << std::endl;
		tree = (glm::vec4*)malloc(treeSize * sizeof(glm::vec4));
		getKdTree(tree, y_vec3, each);
		printVec4(tree, treeSize);

	}

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
		dim3 numBlocks1((numX + blockSize - 1) / blockSize);
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

		//std::cout << "Computing correspondence gpu..." << std::endl;
		////Find correspondence
		//std::cout << "x[0] = " << x[0] << " x[1] = " << x[1] << " x[2] = " << x[2] << std::endl;
		////Find Correspondence
		//findCorrespondence << <numBlocks, blockSize >> > (dev_x, numX, dev_y, numY, dev_x_corr);

		////Transpose x_corr and x
		//float* dev_x_tr;
		//cudaMalloc((void**)&dev_x_tr, numX * sizeof(float));
		//checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		//transpose << <numBlocks1, blockSize >> > (dev_x, dev_x_tr, eachX, 3);

		//float* dev_x_corr_tr;
		//cudaMalloc((void**)&dev_x_corr_tr, numX * sizeof(float));
		//checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		//transpose << <numBlocks1, blockSize >> > (dev_x_corr, dev_x_corr_tr, eachX, 3);

		//std::cout << "Mean Centering x..." << std::endl;
		//float meanX;
		//meanX = thrust::reduce(thrust::device, dev_x_tr, dev_x_tr + eachX, 0.0f);
		//meanX /= eachX;

		//float meanY;
		//meanY = thrust::reduce(thrust::device, dev_x_tr + eachX, dev_x_tr + (2 * eachX), 0.0f);
		//meanY /= eachX;

		//float meanZ;
		//meanZ = thrust::reduce(thrust::device, dev_x_tr + (2 * eachX), dev_x_tr + numX, 0.0f);
		//meanZ /= eachX;


		//std::cout << "xm[0] = " << meanX << " xm[1] = " << meanY << " xm[2] = " << meanZ << std::endl;

		//cudaFree(dev_x_tr);

		//std::cout << "Mean Centering  x_corr..." << std::endl;
		////Mean-center x_corr
		//float meanXC;
		//meanXC = thrust::reduce(thrust::device, dev_x_corr_tr, dev_x_corr_tr + eachX, 0.0f);
		//meanXC /= eachX;

		//float meanYC;
		//meanYC = thrust::reduce(thrust::device, dev_x_corr_tr + eachX, dev_x_corr_tr + (2 * eachX), 0.0f);
		//meanYC /= eachX;

		//float meanZC;
		//meanZC = thrust::reduce(thrust::device, dev_x_corr_tr + (2 * eachX), dev_x_corr_tr + numX, 0.0f);
		//meanZC /= eachX;


		//std::cout << "xm[0] = " << meanXC << " xm[1] = " << meanYC << " xm[2] = " << meanZC << std::endl;

		//cudaFree(dev_x_corr_tr);

		//std::cout << "x..." << std::endl;
		//meanCenter << <numBlocks, blockSize >> > (dev_x, eachX, meanX, meanY, meanZ);
		//std::cout << "x_corr..." << std::endl;
		//meanCenter << <numBlocks, blockSize >> > (dev_x_corr, eachX, meanXC, meanYC, meanZC);

		//std::cout << "Computing input to SVD..." << std::endl;
		////Multiply x_corr_tr and x to get input to SVD
		//cudaMalloc((void**)&dev_x_corr_tr, numX * sizeof(float));
		//checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		//transpose << <numBlocks1, blockSize >> > (dev_x_corr, dev_x_corr_tr, eachX, 3);

		//float* x_cpu = (float*)malloc(numX * sizeof(float));
		//cudaMemcpy(x_cpu, dev_x, sizeof(float) * numX, cudaMemcpyDeviceToHost);
		//std::cout << "X : \n";


		//float* y_tr_cpu = (float*)malloc(numX * sizeof(float));
		//cudaMemcpy(y_tr_cpu, dev_x_corr_tr, sizeof(float) * numX, cudaMemcpyDeviceToHost);


		//float* dev_to_svd;
		//cudaMalloc((void**)&dev_to_svd, 3 * 3 * sizeof(float));
		//checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		//float* to_svd = (float*)malloc(3 * 3 * sizeof(float));

		////multiplyMatrixCPU(y_tr_cpu, x_cpu, to_svd, 3, eachX, 3);

		////cudaMemcpy(dev_to_svd, to_svd, sizeof(float) * 9, cudaMemcpyHostToDevice);

		//dimGrid.x = (3 + dimBlock.x - 1) / dimBlock.x;
		//dimGrid.y = (3 + dimBlock.y - 1) / dimBlock.y;
		//kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_x_corr_tr, dev_x, dev_to_svd, 3, eachX, 3);

		//// Create a handle for CUBLAS
		//cublasHandle_t handle;
		//cublasCreate(&handle);
		////gpu_blas_mmul(handle, dev_x_corr_tr, dev_x, dev_to_svd, 3, eachX, 3);
		////multiplyMatrix<<<numBlocks2, blockSize>> > (dev_x_corr_tr, dev_x, dev_to_svd, 3, eachX, 3);


		//cudaMemcpy(to_svd, dev_to_svd, sizeof(float) * 9, cudaMemcpyDeviceToHost);
		//std::cout << "Input to SVD : \n";
		//printMatrix(to_svd, 3, 3);
		//std::cout << std::endl;

		//std::cout << "SVD..." << std::endl;
		////Find SVD - U, V, S
		//float* dev_svd_u;
		//cudaMalloc((void**)&dev_svd_u, 3 * 3 * sizeof(float));
		//cudaMemset(dev_svd_u, 0.0f, 3 * 3 * sizeof(float));
		//checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		//float* dev_svd_s;
		//cudaMalloc((void**)&dev_svd_s, 3 * 3 * sizeof(float));
		//cudaMemset(dev_svd_s, 0.0f, 3 * 3 * sizeof(float));
		//checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		//float* dev_svd_v;
		//cudaMalloc((void**)&dev_svd_v, 3 * 3 * sizeof(float));
		//cudaMemset(dev_svd_v, 0.0f, 3 * 3 * sizeof(float));
		//checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		//get_svd << <1, 1 >> > (dev_to_svd, dev_svd_u, dev_svd_s, dev_svd_v);

		//float* u = (float*)malloc(3 * 3 * sizeof(float));
		//cudaMemcpy(u, dev_svd_u, sizeof(float) * 9, cudaMemcpyDeviceToHost);
		//std::cout << "U : \n";
		//printMatrix(u, 3, 3);
		//std::cout << std::endl;

		//float* v = (float*)malloc(3 * 3 * sizeof(float));
		//cudaMemcpy(v, dev_svd_v, sizeof(float) * 9, cudaMemcpyDeviceToHost);
		//std::cout << "V : \n";
		//printMatrix(v, 3, 3);
		//std::cout << std::endl;

		//std::cout << "SVD done..." << std::endl;
		//cudaFree(dev_svd_s);
		////Compute U x V_tr to get R
		//float* dev_svd_v_tr;
		//cudaMalloc((void**)&dev_svd_v_tr, 3 * 3 * sizeof(float));
		//checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");
		//transpose << <numBlocks2, blockSize >> > (dev_svd_v, dev_svd_v_tr, 3, 3);

		//float* v_tr = (float*)malloc(3 * 3 * sizeof(float));
		//cudaMemcpy(v_tr, dev_svd_v_tr, sizeof(float) * 9, cudaMemcpyDeviceToHost);
		//std::cout << "V TR : \n";
		//printMatrix(v_tr, 3, 3);
		//std::cout << std::endl;

		//cudaFree(dev_svd_v);
		//std::cout << "Computing R..." << std::endl;


		//dimGrid.x = (3 + dimBlock.x - 1) / dimBlock.x;
		//dimGrid.y = (3 + dimBlock.y - 1) / dimBlock.y;
		//kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_svd_u, dev_svd_v_tr, dev_R, 3, 3, 3);

		////gpu_blas_mmul(handle, dev_svd_u, dev_svd_v_tr, dev_R, 3, 3, 3);

		//float* R = (float*)malloc(3 * 3 * sizeof(float));
		//cudaMemcpy(R, dev_R, sizeof(float) * 9, cudaMemcpyDeviceToHost);
		//std::cout << "R : \n";
		//printMatrix(R, 3, 3);
		//std::cout << std::endl;

		//std::cout << "R done..." << std::endl;
		////Compute translation = x_corr_mean - R.x_mean
		//float* dev_x_mean;
		//cudaMalloc((void**)&dev_x_mean, 3 * sizeof(float));
		//setValueOnDevice << <1, 1 >> > (&dev_x_mean[0], meanX);
		//setValueOnDevice << <1, 1 >> > (&dev_x_mean[1], meanY);
		//setValueOnDevice << <1, 1 >> > (&dev_x_mean[2], meanZ);

		//float* dev_y_mean;
		//cudaMalloc((void**)&dev_y_mean, 3 * sizeof(float));
		//setValueOnDevice << <1, 1 >> > (&dev_y_mean[0], meanXC);
		//setValueOnDevice << <1, 1 >> > (&dev_y_mean[1], meanYC);
		//setValueOnDevice << <1, 1 >> > (&dev_y_mean[2], meanZC);

		//std::cout << "Computing translation..." << std::endl;
		//float* inter;
		//cudaMalloc((void**)&inter, 3 * 1 * sizeof(float));
		//dimGrid.x = (3 + dimBlock.x - 1) / dimBlock.x;
		//dimGrid.y = (1 + dimBlock.y - 1) / dimBlock.y;
		//kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_R, dev_x_mean, inter, 3, 3, 1);
		////gpu_blas_mmul(handle, dev_R, dev_x_mean, inter, 3, 3, 1);

		//matrix_subtraction << <numBlocks3, blockSize >> > (dev_y_mean, inter, dev_translation, 1, 3);

		//std::cout << "Applying transformation on x..." << std::endl;
		//float* dev_R_tr;
		//cudaMalloc((void**)&dev_R_tr, 9 * sizeof(float));
		//checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		//transpose << <numBlocks2, blockSize >> > (dev_R, dev_R_tr, 3, 3);
		////Apply rotation on x
		//float* dev_newX;
		//cudaMalloc((void**)&dev_newX, numX * sizeof(float));
		//dimGrid.x = (3 + dimBlock.x - 1) / dimBlock.x;
		//dimGrid.y = (eachX + dimBlock.y - 1) / dimBlock.y;
		//kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_x, dev_R_tr, dev_newX, eachX, 3, 3);
		////gpu_blas_mmul(handle, dev_x, dev_R, dev_newX, eachX, 3, 3);
		//cudaDeviceSynchronize();

		////Apply translation on x
		//addTranslation << <numBlocks, blockSize >> > (dev_newX, dev_translation, eachX);

		//cudaMemcpy(x, dev_newX, sizeof(float) * numX, cudaMemcpyDeviceToHost);
		//std::cout << "x[0] = " << x[0] << " x[1] = " << x[1] << " x[2] = " << x[2] << std::endl;
		//cudaDeviceSynchronize();
	}
}

