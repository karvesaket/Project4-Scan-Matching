#pragma once

#include "common.h"

namespace NaiveGPU {

	void initScan(int numX);
	void match(float* x, float*y, int numX, int numY);
}