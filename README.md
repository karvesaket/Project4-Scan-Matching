CUDA Scan Matching
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Saket Karve
  * [LinkedIn](https://www.linkedin.com/in/saket-karve-43930511b/), [twitter](), etc.
* Tested on:  Windows 10 Education, Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, NVIDIA Quadro P1000 @ 4GB (Moore 100B Lab)


### Highlights

### Features Implemented

- Scan Matching on CPU
- Naive Scan Matching implementation on GPU
- KD-Tree implementation of Scan Matching on GPU
- Visualization of Output
- Performance comparison between different implementations

### Scan Matching Algorithm

Scan matching is a technique to align two independent point clouds representing an object with one of the two as a fixed reference. There are various algorithms which perform some form of scan matching given two independent point clouds. Essentially the goal is to estimate the transformation matrix (translation and rotation - assuming the two points clouds are rigid bodies) such that both the point clouds are aligned after applying the transformation to one of the point clouds.

Scan matching is widely used in robotics for various applications.

#### Algorithm

We use the Iterative Closest Point algorithm. While there are many variants, this algorithm essentially seeks to do the following:
1. For each point in the target pointcloud (a), find the closest point in the scene pointcloud (b). 
2. Compute a 3D transformation matrix that best aligns the point using Least Squares Regression (this is where most of the CUDA goes, a deep dive on the variations here)
3. Update all points in the target by the transformation matrix
4. Repeat steps 1-3 until some epsilon convergence

### Implementation overview

#### Input file format

#### Processing format

#### Libraries used

### CPU Implementation

### Naive GPU Implementation

### KD-Tree GPU Implementation

### Performance Analysis

