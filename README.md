CUDA Scan Matching
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Saket Karve
  * [LinkedIn](https://www.linkedin.com/in/saket-karve-43930511b/), [twitter](), etc.
* Tested on:  Windows 10 Education, Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, NVIDIA Quadro P1000 @ 4GB (Moore 100B Lab)


### Highlights

![]()

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

The program accepts input objects in the ```Polygon file format```. It is a format used to store 3D data generated from 3D scanners. Sample ply files are included in the *data* directory of this repository. The format of these files and some more samples can be refered [here](https://people.sc.fsu.edu/~jburkardt/data/ply/ply.html).

The program expects two file paths to be given as one comma-separated command line argument when running the code.

#### Processing format

The data is read from the input files. The first argument refers to the target pointcloud (the one which is transformed) and the second argument refers to the scene pointcloud (remains stationary). Data is stored as an array of floats with ```x```, ```y``` and ```z``` stored row wise. At the end of every iteration, the estimated transformation is applied to the target pointcloud and it is updated. The visualization shows every update iteratively.

#### Libraries used

Essentially I have used only two external libraries.
1. Thrust - For basic array manipulations.
2. [svd3](https://github.com/ericjang/svd3) - For computing the SVD of a matrix which is used to find the transformation matrix.

#### Running the code

- CMAKE and open the ```.sln``` in Visual Studio as usual.
- Give the command line arguments specifying the path to the scene and target pointcloud file (```.ply```) as a space separated list. For example, ```../data/bun000.ply ../data/bun045.ply```.
-  To run one of the three implementations of the project, set one of the flags (in ```main.cpp```) to 1 based on what implementation you choose to run.
   - For CPU implementation, set ```CPU = 1```
   - For Naive GPU implementation, set ```NAIVE_GPU = 1```
   - For KD-Tree GPU implementation, set ```KDTREE_GPU = 1```
   
Note that you should set the other flags to 0.

### CPU Implementation

#### Output

![]()

#### Analysis

In this implementation, when finding the correspondence points for each point in the target pointcloud every point in the scene pointcloud is checked against to find the closest point. This is the most important performance bottleneck. Moreover, since it is executed on the CPU, each point is run sequentially.

The FPS measured for visualization on various data sets with different number of points in the scene is as follows

| Target File | Scene File | Total number of points | FPS |
| ------------|----------- | ---------------------- | --- |
|             |            |                        |     |
|             |            |                        |     |
|             |            |                        |     |
|             |            |                        |     |

### Naive GPU Implementation

#### Output

![]()

#### Analysis

In this implementation, when finding the correspondence points for each point in the target pointcloud every point in the scene pointcloud is checked against to find the closest point. However, each point in the target pointcloud is executed on parallel threads. Hence, this implementation has a significant improvement in terms of performance.

The FPS measured for visualization on various data sets with different number of points in the scene is as follows

| Target File | Scene File | Total number of points | FPS |
| ------------|----------- | ---------------------- | --- |
|             |            |                        |     |
|             |            |                        |     |
|             |            |                        |     |
|             |            |                        |     |


### KD-Tree GPU Implementation

#### KD-Tree

KD-Tree is a binary search tree representation for multi-dimensional data. Points inserted in a KD-Tree partition the space into various hyperplanes based on the input points. It is a very useful data structure for applications involving search over a multi-dimensional space. When the points are inserted in the KD-Tree, the k-dimensional space is partitioned as can be seen from the following figure.

![]()

For finding the nearest neighbor of a given target point, the following algorithm is used,

![]()
[Reference](https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/kdtrees.pdf)

KD-Tree has an average case time complexity of O(log n) for search. This is the main reason for the significant improvment in performance.

#### Output

![]()

#### Analysis

The FPS measured for visualization on various data sets with different number of points in the scene is as follows

| Target File | Scene File | Total number of points | FPS |
| ------------|----------- | ---------------------- | --- |
|             |            |                        |     |
|             |            |                        |     |
|             |            |                        |     |
|             |            |                        |     |


### Performance Analysis

The overall performance is as expected the best for the KD-Tree implementation. In general, the main performance bottleneck is when finding correspondences. The CPU and the Naive GPU implementation check for the closest point against every other point. KD-Tree implementation limits this search space by diving the region by hyperplanes so that it does not need to search through those points which the KD-Tree gives gaurantee for not being a possible candidate for the closest point. This makes the search much faster and hence leads to a faster convergence.

Increasing the number of points in the point clouds will lead to a drop in performance as the number of points increases.
