#pragma once

#include "common.h"

namespace KDTreeGPU {

	void buildTree(float* points, int num);
	void initScan(int numX);
	void match(float* x, float*y, int numX, int numY);
}