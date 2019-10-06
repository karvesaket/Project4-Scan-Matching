#pragma once

#include "common.h"

namespace ScanMatching {
	
	float* parsePly(std::string filename, long* num);
	void findCorrespondence(float* arr1, long numArr1, float* arr2, long numArr2, float* arr1_corr);
	glm::vec3 meanCenter(float* arr, int num);
	void transpose(float* arr, float* arrTrans, int m, int n);
	void multiplyMatrix(float* A, float* B, float* C, int m, int n, int p);
	void subtractMatrices(float* A, float* B, float* C, int m, int n);
	void printMatrix(float* A, int m, int n);

	// TODO: implement required elements for MLP sections 1 and 2 here
}