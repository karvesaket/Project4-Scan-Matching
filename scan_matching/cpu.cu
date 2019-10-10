#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <glm/glm.hpp>

#include "common.h"
#include "cpu.h"
#include "svd3.h"

namespace ScanMatching {
	float* x_corr;
	float* R;
	float* translation;

	void findCorrespondence(float* arr1, long numArr1, float* arr2, long numArr2, float* arr1_correspondence) {
		for (int i = 0; i < numArr1 / 3; i++) {
			glm::vec3 point(arr1[i * 3 + 0], arr1[i * 3 + 1], arr1[i * 3 + 2]);
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
			arr1_correspondence[i * 3 + 0] = closest_point.x;
			arr1_correspondence[i * 3 + 1] = closest_point.y;
			arr1_correspondence[i * 3 + 2] = closest_point.z;
		}
	}

	glm::vec3 meanCenter(float* arr, float* centered, int num) {
		float meanX = 0.0f;
		float meanY = 0.0f;
		float meanZ = 0.0f;

		for (int i = 0; i < num / 3; i++) {
			meanX += arr[i * 3 + 0];
			meanY += arr[i * 3 + 1];
			meanZ += arr[i * 3 + 2];
		}

		meanX = 3.0f * meanX / num;
		meanY = 3.0f * meanY / num;
		meanZ = 3.0f * meanZ / num;

		for (int i = 0; i < num / 3; i++) {
			centered[i * 3 + 0] = arr[i * 3 + 0] - meanX;
			centered[i * 3 + 1] = arr[i * 3 + 1] - meanY;
			centered[i * 3 + 2] = arr[i * 3 + 2] - meanZ;
		}
		return glm::vec3(meanX, meanY, meanZ);
	}

	// Takes m x n matrix and returns n x m
	void transpose(float* arr, float* arrTrans, int m, int n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				
				arrTrans[m*j + i] = arr[n*i + j];
			}
		}
	}

	// A - m x n || B - n x p and returns C - m x p
	void multiplyMatrix(float* A, float* B, float* C, int m, int n, int p) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < p; j++) {
				C[i*p + j] = 0;
				for (int k = 0; k < n; k++) {
					C[i*p + j] += (A[i*n + k] * B[k*p + j]);
				}
			}
		}
	}

	void subtractMatrices(float* A, float* B, float* C, int m, int n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				C[i*m + j] = A[i*m + j] - B[i*m + j];
			}
		}
	}

	//In-Place addition
	void addTranslation(float* A, float* trans, int num) {
		for (int i = 0; i < num; i++) {
			A[i * 3 + 0] += trans[0];
			A[i * 3 + 1] += trans[1];
			A[i * 3 + 2] += trans[2];
		}
	}

	void printMatrix(float* A, int m, int n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				std::cout << A[i*n + j] << " ";
			}
			std::cout << std::endl;
		}
	}

	void initScan(int numX) {
		x_corr = (float*)malloc(numX * sizeof(float));
		R = (float*)malloc(3 * 3 * sizeof(float));
		translation = (float*)malloc(3 * 1 * sizeof(float));

	}

	void match(float* x, float* y, int numX, int numY) {
		
		//Find correspondence
		

		findCorrespondence(x, numX, y, numY, x_corr);
		//std::cout << "x[0] = " << x_corr[3] << " x[1] = " << x_corr[4] << " x[2] = " << x_corr[5] << std::endl;
		

		float* x_mean_centered = (float*)malloc(numX * sizeof(float));
		float* x_corr_mean_centered = (float*)malloc(numX * sizeof(float));

		
		//Mean center x and x_corr
		glm::vec3 x_mean = meanCenter(x, x_mean_centered, numX);
	
		glm::vec3 x_corr_mean = meanCenter(x_corr, x_corr_mean_centered, numX);
		

		
		//Transpose X_corr
		float* x_corr_tr = (float*)malloc(numX * sizeof(float));
		transpose(x_corr_mean_centered, x_corr_tr, numX / 3, 3);

		

		
		//Compute C_corr_tr x X
		float* to_svd = (float*)malloc(3 * 3 * sizeof(float));
		multiplyMatrix(x_corr_tr, x_mean_centered, to_svd, 3, numX / 3, 3);

		
		
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

		
		
		//Compute U x V_tr to get rotation matrix
		float* v_tr = (float*)malloc(3 * 3 * sizeof(float));
		transpose(svd_v, v_tr, 3, 3);
		

		multiplyMatrix(svd_u, v_tr, R, 3, 3, 3);
		

		float* R_tr = (float*)malloc(9 * sizeof(float));
		transpose(R, R_tr, 3, 3);

		//Compute R x X_mean
		float* inter = (float*)malloc(3 * 1 * sizeof(float));
		float x_mean_arr[] = { x_mean.x, x_mean.y, x_mean.z };

		

		multiplyMatrix(R, x_mean_arr, inter, 3, 3, 1);

		
		//Compute Translation = x_corr_mean - R * x_mean
		float y_mean_arr[] = { x_corr_mean.x, x_corr_mean.y, x_corr_mean.z };

		
		printMatrix(y_mean_arr, 1, 3);
		std::cout << std::endl;

		

		subtractMatrices(y_mean_arr, inter, translation, 1, 3);
		

		

		//Apply the rotation matrix to current X
		float* newX = (float*)malloc(numX * sizeof(float));
		multiplyMatrix(x, R_tr, newX, numX/3 , 3, 3);

		//Add translation to every vertes
		addTranslation(newX, translation, numX / 3);
		

		//Copy updated X back
		memcpy(x, newX, numX * sizeof(float));
	}
	
}
