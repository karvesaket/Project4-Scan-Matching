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

namespace ScanMatching {
	
	float* parsePly(std::string filename, long* num) {
		float* arr;
		std::ifstream f(filename);
		std::string line;
		std::cout << "Opening file " << filename << std::endl;
		long numberOfVertices = -1;
		if (f.is_open()) {
			bool start = false;
			int i = 0;
			while (std::getline(f, line))
			{
				std::string::size_type sz;
				if (start && numberOfVertices > 0) {
					//Read vertices into array
					std::stringstream ssin(line);
					int j = 0;
					while (ssin.good() && j < 3) {
						std::string temp;
						ssin >> temp;
						arr[i * 3 + j] = std::stof(temp, &sz);
						++j;
					}
					//std::cout << x[i * 3 + 0] << " " << x[i * 3 + 1] << " " << x[i * 3 + 2] << std::endl;
					//std::cout << "i = " << i << std::endl;
					i++;
					if (i >= numberOfVertices) break;
				}
				if (line.compare(0, 14, "element vertex") == 0) {
					std::string num = line.substr(15);

					numberOfVertices = std::stol(num, &sz);
					std::cout << "Number of vertices " << numberOfVertices << std::endl;
					//Malloc array
					arr = (float*)malloc(3 * numberOfVertices * sizeof(float));
				}
				if (line.compare("end_header") == 0) {
					start = true;
				}
			}
		}
		std::cout << "Done reading " << numberOfVertices << " vertices" << std::endl;
		f.close();
		std::cout << "arr[0] = " << arr[0] << std::endl;
		*num = 3 * numberOfVertices;
		return arr;
	}

	void findCorrespondence(float* arr1, long numArr1, float* arr2, long numArr2, float* arr1_correspondence) {
		for (int i = 0; i < numArr1 / 3; i++) {
			glm::vec3 point(arr1[i * 3 + 0], arr1[i * 3 + 1], arr1[i * 3 + 2]);
			float min_dist = LONG_MAX;
			glm::vec3 closest_point;
			for (int j = 0; j < numArr1 / 3; j++) {
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

	glm::vec3 meanCenter(float* arr, int num) {
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
			arr[i * 3 + 0] -= meanX;
			arr[i * 3 + 1] -= meanY;
			arr[i * 3 + 2] -= meanZ;
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

	void printMatrix(float* A, int m, int n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				std::cout << A[i*m + j] << " ";
			}
			std::cout << std::endl;
		}
	}

	// TODO: __global__

	/**
		* Example of use case (follow how you did it in stream compaction)
		*/
		/*void scan(int n, int *odata, const int *idata) {
			timer().startGpuTimer();
			// TODO
			timer().endGpuTimer();
		}
		*/

		// TODO: implement required elements for MLP sections 1 and 2 here
}
