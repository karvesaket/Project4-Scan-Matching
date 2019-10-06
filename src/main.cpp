/**
 * @file      main.cpp
 * @brief     Scan Matching
 * @authors   Saket Karve
 * @date      2019
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <glm/glm.hpp>
#include <iostream>

#include <scan_matching/cpu.h>
#include <scan_matching/common.h>
#include "testing_helpers.hpp"
#include "svd3.h"



int main(int argc, char* argv[]) {
	// Read the two point clouds from a file
	std::string data_directory = "../data/";

	std::string filename1 = "bun000.ply";
	long numX;
	float* x = ScanMatching::parsePly(data_directory + filename1, &numX);
	std::cout << "x[0] = " << x[0] << std::endl;
	std::cout << "Num X = " << numX << std::endl;

	std::string filename2 = "bun045.ply";
	long numY;
	float* y = ScanMatching::parsePly(data_directory + filename2, &numY);
	std::cout << "y[0] = " << y[0] << std::endl;
	std::cout << "Num Y = " << numY << std::endl;

	std::cout << "Computing correspondence..." << std::endl;
	//Find correspondence
	float* x_corr = (float*)malloc(numX * sizeof(float));
	ScanMatching::findCorrespondence(x, numX, y, numY, x_corr);
	std::cout << "x_corr[0] = " << x_corr[0] << " x_corr[1] = " << x_corr[1] << " x_corr[2] = " << x_corr[2] << std::endl;

	std::cout << "Mean centering..." << std::endl;
	//Mean center x and x_corr
	glm::vec3 x_mean = ScanMatching::meanCenter(x, numX);
	std::cout << "x[0] = " << x[0] << std::endl;
	glm::vec3 x_corr_mean = ScanMatching::meanCenter(x_corr, numX);
	std::cout << "x_corr[0] = " << x_corr[0] << std::endl;

	std::cout << "Transposing..." << std::endl;
	//Transpose X_corr
	float* x_corr_tr = (float*)malloc(numX * sizeof(float));
	ScanMatching::transpose(x_corr, x_corr_tr, numX/3, 3);

	std::cout << "Multiplying tfor SVD.." << std::endl;
	//Compute C_corr_tr x X
	float* to_svd = (float*)malloc(3 * 3 * sizeof(float));
	ScanMatching::multiplyMatrix(x_corr_tr, x, to_svd, 3, numX/3, 3);

	std::cout << "Input to SVD : \n";
	ScanMatching::printMatrix(to_svd, 3, 3);
	std::cout << std::endl;

	std::cout << "Finding SVD..." << std::endl;
	float* svd_u = (float*)malloc(3 * 3 * sizeof(float));
	float* svd_v = (float*)malloc(3 * 3 * sizeof(float));
	float* svd_s = (float*)malloc(3 * 3 * sizeof(float));
	svd(to_svd[0], to_svd[1], to_svd[2], to_svd[3], to_svd[4], to_svd[5], to_svd[6], to_svd[7], to_svd[8],
		svd_u[0], svd_u[1], svd_u[2], svd_u[3], svd_u[4], svd_u[5], svd_u[6], svd_u[7], svd_u[8],
		svd_s[0], svd_s[1], svd_s[2], svd_s[3], svd_s[4], svd_s[5], svd_s[6], svd_s[7], svd_s[8], 
		svd_v[0], svd_v[1], svd_v[2], svd_v[3], svd_v[4], svd_v[5], svd_v[6], svd_v[7], svd_v[8]);

	std::cout << "SVD V: \n";
	ScanMatching::printMatrix(svd_v, 3, 3);
	std::cout << std::endl;

	std::cout << "Computing rotation matrix..." << std::endl;
	//Compute U x V_tr to get rotation matrix
	float* v_tr = (float*)malloc(3 * 3 * sizeof(float));
	ScanMatching::transpose(svd_v, v_tr, 3, 3);
	std::cout << "SVD V Transpose: \n";
	ScanMatching::printMatrix(v_tr, 3, 3);
	std::cout << std::endl;

	float* R = (float*)malloc(3 * 3 * sizeof(float));
	ScanMatching::multiplyMatrix(svd_u, v_tr, R, 3, 3, 3);
	std::cout << "Rotation matrix: \n";
	ScanMatching::printMatrix(R, 3, 3);
	std::cout << std::endl;

	//Compute R x X_mean
	float* inter = (float*)malloc(3 * 1 * sizeof(float));
	float x_mean_arr[] = {x_mean.x, x_mean.y, x_mean.z};
	
	std::cout << "X Mean ";
	ScanMatching::printMatrix(x_mean_arr, 1, 3);
	std::cout << std::endl;

	ScanMatching::multiplyMatrix(R, x_mean_arr, inter, 3, 3, 1);

	std::cout << "Computing translaton matrix..." << std::endl;
	//Compute Translation = x_corr_mean - R * x_mean
	float* translation = (float*)malloc(3 * 1 * sizeof(float));
	float y_mean_arr[] = { x_corr_mean.x, x_corr_mean.y, x_corr_mean.z };

	std::cout << "Y Mean ";
	ScanMatching::printMatrix(y_mean_arr, 1, 3);
	std::cout << std::endl;

	std::cout << "Inter";
	ScanMatching::printMatrix(inter, 1, 3);
	std::cout << std::endl;

	ScanMatching::subtractMatrices(y_mean_arr, inter, translation, 1, 3);
	std::cout << "Translation matrix : " << translation[0] << " " << translation[1] << " " << translation[2] << std::endl;
}

