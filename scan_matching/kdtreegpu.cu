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
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "svd3.h"


#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define blockSize 128

struct stackElement {
	int index;
	bool is_good;
	int depth;
};

__global__ void transposeKD(float* arr, float* arrTrans, int m, int n) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= m * n) {
		return;
	}
	int i = index / n;
	int j = index % n;

	arrTrans[m*j + i] = arr[n*i + j];
}


__global__ void matrix_subtractionKD(float* A, float* B, float* C, int m, int n) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= m * n) {
		return;
	}
	C[index] = A[index] - B[index];
}

__global__ void kernMatrixMultiplyKD(float *dev_A, float *dev_B, float *dev_C, int m, int n, int k) {

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



__global__ void addTranslationKD(float* A, float* trans, int num) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num) {
		return;
	}
	A[index * 3 + 0] += trans[0];
	A[index * 3 + 1] += trans[1];
	A[index * 3 + 2] += trans[2];
}


__global__ void meanCenterKD(float* arr, float* centered, int num, float mx, float my, float mz) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num) {
		return;
	}
	centered[index * 3 + 0] = arr[index * 3 + 0] - mx;
	centered[index * 3 + 1] = arr[index * 3 + 1] - my;
	centered[index * 3 + 2] = arr[index * 3 + 2] - mz;
}

__global__ void setValueOnDeviceKD(float* device_var, float val) {
	*device_var = val;
}

//__global__ void get_svd(float* input, float* u, float* s, float* v) {
//	svd(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8],
//		u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8],
//		s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8],
//		v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
//}

__global__ void divide_sum_to_meanKD(float* sum, int num) {
	(*sum) = (*sum) / num;
}


void printMatrixKD(float* A, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << A[i*n + j] << " ";
		}
		std::cout << std::endl;
	}
}

__device__ glm::vec3 getClosestPointKDTree(glm::vec4* tree, int num, glm::vec3 target_point) {
	bool is_leaf = false;
	int root = 0;
	int left = 2 * root + 1;
	int right = 2 * root + 2;
	glm::vec3 closest_point = glm::vec3(tree[root].x, tree[root].y, tree[root].z);
	float min_dist = glm::distance(target_point, closest_point);
	float left_dist, right_dist;
	glm::vec3 current_point;
	while (!is_leaf) {
		//Check left child
		current_point = glm::vec3(tree[left].x, tree[left].y, tree[left].z);
		left_dist = glm::distance(target_point, current_point);
		
		//Check Right child
		current_point = glm::vec3(tree[right].x, tree[right].y, tree[right].z);
		right_dist = glm::distance(target_point, current_point);
		if (left_dist < right_dist) {
			min_dist = left_dist;
			closest_point = glm::vec3(tree[left].x, tree[left].y, tree[left].z);
			root = left;
		}
		else {
			min_dist = right_dist;
			closest_point = glm::vec3(tree[right].x, tree[right].y, tree[right].z);
			root = right;
		}
		left = 2 * root + 1;
		right = 2 * root + 2;
		if (left > num || right > num) break;
		if (tree[left].w == 0.0f || tree[right].w == 0.0f) is_leaf = true;
	}
	return closest_point;
}

__device__ void stackPush(stackElement* stack, stackElement to_push, int* stackTop) {
	//printf("Pushing Stack Content: %f %f %f\n", to_push.x, to_push.y, to_push.z);
	stack[*stackTop] = to_push;
	
	//printf("Pushed Stack Content: %f %f %f\n", stack[*stackTop].x, stack[*stackTop].y, stack[*stackTop].z);
	(*stackTop)++;
}

__device__ stackElement stackPop(stackElement* stack, int* stackTop) {
	(*stackTop)--;
	stackElement popped = stack[*stackTop];
	return popped;
}

__device__ bool is_worse(glm::vec4* tree, stackElement curr_node, float minDist, glm::vec3 target_point) {
	int parent_index = (curr_node.index-1) / 2;
	int align = (curr_node.depth % 3);
	if (align == 0) {
		return (abs(tree[parent_index].x - target_point.x) >= minDist);
	}
	else if (align == 1) {
		return (abs(tree[parent_index].y - target_point.y) >= minDist);
	}
	else if (align == 2) {
		return (abs(tree[parent_index].z - target_point.z) >= minDist);
	}
}

__device__ glm::vec3 getClosestPointKDTreeStack(glm::vec4* tree, int num, glm::vec3 target_point, stackElement* stack, int stackLength, int idx) {
	int root = 0;
	
	//x = index | y = is_good | z = depth
	stackElement rootNode;
	rootNode.index = root;
	rootNode.is_good = true;
	rootNode.depth = 0;
	//printf("Root Content: %f %f %f\n", rootNode.x, rootNode.y, rootNode.z);
	float min_dist = FLT_MAX;
	int stackTop = 0;
	//Push root to stack
	//printf("Stack Top before %d \n", stackTop);
	stackPush(stack, rootNode, &stackTop);
	//printf("Stack Top %d \n", stackTop);
	//printf("Stack Content: %f %f %f\n", stack[stackTop].x, stack[stackTop].y, stack[stackTop].z);
	glm::vec3 closest_point;

	int left;
	int right;

	while (stackTop > 0) {
		//printf("Here\n");
		//Pop element from top of stack
		stackElement current_node = stackPop(stack, &stackTop);
		if (idx == 0) {
			//printf("After popping Top : %d\n", stackTop);
		}
		int curr_index = current_node.index;
		if (tree[curr_index].w == 0.0f) {
			continue;
		}
		glm::vec3 current_node_tree = glm::vec3(tree[curr_index].x, tree[curr_index].y, tree[curr_index].z);
		
		float dist = glm::distance(current_node_tree, target_point);
		if (dist < min_dist) {
			min_dist = dist;
			closest_point = glm::vec3(current_node_tree.x, current_node_tree.y, current_node_tree.z);
		}
		//Check if current node is bad and worse (if yes, prune it)
		if (!current_node.is_good && is_worse(tree, current_node, min_dist, target_point)) {
			continue;
		}
		else {
			left = 2 * current_node.index + 1;
			right = 2 * current_node.index + 2;
			/*if (left >= num || right >= num) {
				continue;
			}*/

			int good_index = -1;
			int bad_index = -1;

			if (current_node.depth == 0) {
				if (current_node_tree.x <= target_point.x) {
					good_index = right;
					bad_index = left;
				}
				else {
					good_index = left;
					bad_index = right;
				}
			}
			if (current_node.depth == 1) {
				if (current_node_tree.y <= target_point.y) {
					good_index = right;
					bad_index = left;
				}
				else {
					good_index = left;
					bad_index = right;
				}
			}
			if (current_node.depth == 2) {
				if (current_node_tree.z <= target_point.z) {
					good_index = right;
					bad_index = left;
				}
				else {
					good_index = left;
					bad_index = right;
				}
			}
			stackElement goodNode;
			stackElement badNode;

			if (bad_index != -1 && bad_index < num && tree[bad_index].w == 1.0f) {
				badNode.depth = (current_node.depth + 1) % 3;
				badNode.is_good = false;
				badNode.index = bad_index;
				stackPush(stack, badNode, &stackTop);
			}
			if (good_index != -1 && good_index < num && tree[good_index].w == 1.0f) {
				goodNode.depth = (current_node.depth + 1) % 3;
				goodNode.is_good = true;
				goodNode.index = good_index;
				stackPush(stack, goodNode, &stackTop);
			}
			
		}
	}
	return closest_point;
}

__global__ void findCorrespondenceKDTree(float* arr1, long numArr1, glm::vec4* arr2, long numArr2, float* arr1_correspondence, stackElement* stack, int treeHeight) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (numArr1 / 3)) {
		return;
	}
	glm::vec3 point(arr1[index * 3 + 0], arr1[index * 3 + 1], arr1[index * 3 + 2]);
	int stackBegin = index * treeHeight;
	glm::vec3 closest_point = getClosestPointKDTreeStack(arr2, numArr2, point, stack + stackBegin, treeHeight, index);
	//glm::vec3 closest_point = getClosestPointKDTree(arr2, numArr2, point);

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
	//std::cout << "here\n";
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
	glm::vec4* dev_tree;

	float* dev_x_corr;
	float* dev_R;
	float* dev_translation;

	void buildTree(float* points, int num) {
		//Convert float array to array of vec3
		int each = num / 3;
		y_vec3 = (glm::vec4*)malloc(each * sizeof(glm::vec4));
		std::cout << "Converting to vec3\n";
		convertToVec3 (points, y_vec3, each);
		//printVec4(y_vec3, each);

		std::cout << "Building Tree\n";
		//Call recursive device function to build tree
		int treeSize = 1 << (ilog2ceil(each) + 1);
		std::cout << num << " " << treeSize << std::endl;
		tree = (glm::vec4*)malloc(treeSize * sizeof(glm::vec4));
		getKdTree(tree, y_vec3, each);
		//printVec4(tree, treeSize);

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
		int treeSize = 1 << (ilog2ceil(eachY) + 1);
		cudaMalloc((void**)&dev_tree, treeSize * sizeof(glm::vec4));
		checkCUDAErrorWithLine("cudaMalloc dev_y failed!");
		cudaMemcpy(dev_tree, tree, sizeof(glm::vec4) * treeSize, cudaMemcpyHostToDevice);

		//Find Correspondence
		
		stackElement* stack;
		int treeHeight = ilog2ceil(treeSize) + 1;
		cudaMalloc((void**)&stack, eachX * treeHeight * sizeof(stackElement));
		findCorrespondenceKDTree << <numBlocks, blockSize >> > (dev_x, numX, dev_tree, treeSize, dev_x_corr, stack, treeHeight);
		
		checkCUDAErrorWithLine("findCorrespondenceKDTree failed!");
		//Transpose x_corr and x
		float* dev_x_tr;
		cudaMalloc((void**)&dev_x_tr, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		transposeKD << <numBlocks1, blockSize >> > (dev_x, dev_x_tr, eachX, 3);

		float* dev_x_corr_tr;
		cudaMalloc((void**)&dev_x_corr_tr, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		transposeKD << <numBlocks1, blockSize >> > (dev_x_corr, dev_x_corr_tr, eachX, 3);

		
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


		
		meanCenterKD << <numBlocks, blockSize >> > (dev_x, dev_x_mean_center, eachX, meanX, meanY, meanZ);
		
		meanCenterKD << <numBlocks, blockSize >> > (dev_x_corr, dev_x_corr_mean_center, eachX, meanXC, meanYC, meanZC);

		
		//Multiply x_corr_tr and x to get input to SVD
		cudaMalloc((void**)&dev_x_corr_tr, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		transposeKD << <numBlocks1, blockSize >> > (dev_x_corr_mean_center, dev_x_corr_tr, eachX, 3);




		float* dev_to_svd;
		cudaMalloc((void**)&dev_to_svd, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		float* to_svd = (float*)malloc(3 * 3 * sizeof(float));

		//multiplyMatrixCPU(y_tr_cpu, x_cpu, to_svd, 3, eachX, 3);

		//cudaMemcpy(dev_to_svd, to_svd, sizeof(float) * 9, cudaMemcpyHostToDevice);

		dimGrid.x = (3 + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (3 + dimBlock.y - 1) / dimBlock.y;
		kernMatrixMultiplyKD << <dimGrid, dimBlock >> > (dev_x_corr_tr, dev_x_mean_center, dev_to_svd, 3, eachX, 3);

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
		transposeKD << <numBlocks2, blockSize >> > (dev_svd_v, dev_svd_v_tr, 3, 3);

		float* v_tr = (float*)malloc(3 * 3 * sizeof(float));
		cudaMemcpy(v_tr, dev_svd_v_tr, sizeof(float) * 9, cudaMemcpyDeviceToHost);
		

		cudaFree(dev_svd_v);
		


		dimGrid.x = (3 + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (3 + dimBlock.y - 1) / dimBlock.y;
		kernMatrixMultiplyKD << <dimGrid, dimBlock >> > (dev_svd_u, dev_svd_v_tr, dev_R, 3, 3, 3);

		//gpu_blas_mmul(handle, dev_svd_u, dev_svd_v_tr, dev_R, 3, 3, 3);

		float* R = (float*)malloc(3 * 3 * sizeof(float));
		cudaMemcpy(R, dev_R, sizeof(float) * 9, cudaMemcpyDeviceToHost);
		

		
		//Compute translation = x_corr_mean - R.x_mean
		float* dev_x_mean;
		cudaMalloc((void**)&dev_x_mean, 3 * sizeof(float));
		setValueOnDeviceKD << <1, 1 >> > (&dev_x_mean[0], meanX);
		setValueOnDeviceKD << <1, 1 >> > (&dev_x_mean[1], meanY);
		setValueOnDeviceKD << <1, 1 >> > (&dev_x_mean[2], meanZ);

		float* dev_y_mean;
		cudaMalloc((void**)&dev_y_mean, 3 * sizeof(float));
		setValueOnDeviceKD << <1, 1 >> > (&dev_y_mean[0], meanXC);
		setValueOnDeviceKD << <1, 1 >> > (&dev_y_mean[1], meanYC);
		setValueOnDeviceKD << <1, 1 >> > (&dev_y_mean[2], meanZC);

		

		float* dev_R_tr;
		cudaMalloc((void**)&dev_R_tr, 9 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		transposeKD << <numBlocks2, blockSize >> > (dev_R, dev_R_tr, 3, 3);

		float* inter;
		cudaMalloc((void**)&inter, 3 * 1 * sizeof(float));
		dimGrid.x = (3 + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (1 + dimBlock.y - 1) / dimBlock.y;
		kernMatrixMultiplyKD << <dimGrid, dimBlock >> > (dev_R, dev_x_mean, inter, 3, 3, 1);
		//gpu_blas_mmul(handle, dev_R, dev_x_mean, inter, 3, 3, 1);

		matrix_subtractionKD << <numBlocks3, blockSize >> > (dev_y_mean, inter, dev_translation, 1, 3);

		float* trans = (float*)malloc(3 * 1 * sizeof(float));
		cudaMemcpy(trans, dev_translation, sizeof(float) * 3, cudaMemcpyDeviceToHost);
		

		

		//Apply rotation on x
		float* dev_newX;
		cudaMalloc((void**)&dev_newX, numX * sizeof(float));
		dimGrid.x = (3 + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (eachX + dimBlock.y - 1) / dimBlock.y;
		kernMatrixMultiplyKD << <dimGrid, dimBlock >> > (dev_x, dev_R_tr, dev_newX, eachX, 3, 3);
		//gpu_blas_mmul(handle, dev_x, dev_R, dev_newX, eachX, 3, 3);
		cudaDeviceSynchronize();

		//Apply translation on x
		addTranslationKD << <numBlocks, blockSize >> > (dev_newX, dev_translation, eachX);

		cudaMemcpy(x, dev_newX, sizeof(float) * numX, cudaMemcpyDeviceToHost);
		
		cudaDeviceSynchronize();

		cudaFree(dev_x);
		cudaFree(dev_y);
		cudaFree(dev_newX);
		/*cudaFree(dev_x_tr);
		cudaFree(dev_x_corr_tr);
		cudaFree(dev_R);
		cudaFree(dev_R_tr);
		cudaFree(dev_to_svd);
		cudaFree(dev_x_corr_mean_center);
		cudaFree(dev_x_mean_center);*/
	}
}

