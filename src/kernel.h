#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>

namespace Boids {
    void initSimulation(float* x, float* y, int numX, int numY);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);
	void copyToDevice(float* x, float* y, int numX, int numY);
    void endSimulation();
    void unitTest();
}
