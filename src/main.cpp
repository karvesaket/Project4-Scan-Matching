/**
* @file      main.cpp
* @brief     Example Boids flocking simulation for CIS 565
* @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
* @date      2013-2017
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
#include <scan_matching/naivegpu.h>
#include <scan_matching/kdtreegpu.h>
#include <scan_matching/common.h>
#include "testing_helpers.hpp"
#include "main.hpp"

// ================
// Configuration
// ================

// LOOK-2.1 LOOK-2.3 - toggles for UNIFORM_GRID and COHERENT_GRID
#define VISUALIZE 1
#define UNIFORM_GRID 1
#define COHERENT_GRID 0

const int CPU = 1;
const int NAIVE_GPU = 0;
const int KDTREE_GPU = 0;

// LOOK-1.2 - change this to adjust particle count in the simulation
int N_FOR_VIS = 0;
const float DT = 0.2f;

float* parsePly(std::string filename, long* num);

/**
* C main function.
*/
int main(int argc, char* argv[]) {
  projectName = "565 CUDA Scan Matching";

  // Read the two point clouds from a file
  //std::string data_directory = "../data/";

  //std::string filename1 = "bun045.ply";
  std::string filename1 = argv[1];
  long numX;
  float* x = parsePly(filename1, &numX);
  std::cout << "x[0] = " << x[0] << std::endl;
  std::cout << "Num X = " << numX << std::endl;

  std::string filename2 = argv[2];
  long numY;
  float* y = parsePly(filename2, &numY);
  std::cout << "y[0] = " << y[0] << std::endl;
  std::cout << "Num Y = " << numY << std::endl;
  N_FOR_VIS = numX + numY;

  if (init(argc, argv, x, y, numX, numY)) {
	  mainLoop(x, y, numX, numY);
	  Boids::endSimulation();
	  return 0;
	  } else {
	    return 1;
	  }
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

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


/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv, float* x, float* y, int numX, int numY) {
  // Set window title to "Student Name: [SM 2.0] GPU Name"
  cudaDeviceProp deviceProp;
  int gpuDevice = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpuDevice > device_count) {
    std::cout
    << "Error: GPU device number is greater than the number of devices!"
    << " Perhaps a CUDA-capable GPU is not installed?"
    << std::endl;
    return false;
  }
  cudaGetDeviceProperties(&deviceProp, gpuDevice);
  int major = deviceProp.major;
  int minor = deviceProp.minor;

  std::ostringstream ss;
  ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
  deviceName = ss.str();

  // Window setup stuff
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
    std::cout
    << "Error: Could not initialize GLFW!"
    << " Perhaps OpenGL 3.3 isn't available?"
    << std::endl;
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
  if (!window) {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    return false;
  }

  // Initialize drawing state
  initVAO();

  // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
  // change the device ID.
  cudaGLSetGLDevice(0);

  cudaGLRegisterBufferObject(boidVBO_positions);
  cudaGLRegisterBufferObject(boidVBO_velocities);

  // Initialize N-body simulation
  Boids::initSimulation(x, y, numX, numY);
  if (KDTREE_GPU == 1) {
	  //float sample[] = {4, 6, -3, -2, 0, 1, -4, 2, 5, 7, 8, 9};
	  KDTreeGPU::buildTree(y, numY);
	  //KDTreeGPU::buildTree(&sample[0], 12);
  }

  updateCamera();

  initShaders(program);

  glEnable(GL_DEPTH_TEST);

  return true;
}

void initVAO() {

  std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (N_FOR_VIS)] };
  std::unique_ptr<GLuint[]> bindices{ new GLuint[N_FOR_VIS] };

  glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
  glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

  for (int i = 0; i < N_FOR_VIS; i++) {
    bodies[4 * i + 0] = 0.0f;
    bodies[4 * i + 1] = 0.0f;
    bodies[4 * i + 2] = 0.0f;
    bodies[4 * i + 3] = 1.0f;
    bindices[i] = i;
  }


  glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
  glGenBuffers(1, &boidVBO_positions);
  glGenBuffers(1, &boidVBO_velocities);
  glGenBuffers(1, &boidIBO);

  glBindVertexArray(boidVAO);

  // Bind the positions array to the boidVAO by way of the boidVBO_positions
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer
  glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

  glEnableVertexAttribArray(positionLocation);
  glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  // Bind the velocities array to the boidVAO by way of the boidVBO_velocities
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
  glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(velocitiesLocation);
  glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

  glBindVertexArray(0);
}

void initShaders(GLuint * program) {
  GLint location;

  program[PROG_BOID] = glslUtility::createProgram(
    "shaders/boid.vert.glsl",
    "shaders/boid.geom.glsl",
    "shaders/boid.frag.glsl", attributeLocations, 2);
    glUseProgram(program[PROG_BOID]);

    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
      glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
      glUniform3fv(location, 1, &cameraPosition[0]);
    }
  }

  //====================================
  // Main loop
  //====================================
  void runCUDA(float* x, float* y, int numX, int numY) {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
    // use this buffer

    float4 *dptr = NULL;
    float *dptrVertPositions = NULL;
    float *dptrVertVelocities = NULL;

    cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
    cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);

    // execute the kernel
	if (CPU == 1) {
		ScanMatching::initScan(numX);
		ScanMatching::match(x, y, numX, numY);
	}
	else if(NAIVE_GPU == 1) {
		NaiveGPU::initScan(numX);
		NaiveGPU::match(x, y, numX, numY);
	}
	else if (KDTREE_GPU == 1) {
		KDTreeGPU::initScan(numX);
		KDTreeGPU::match(x, y, numX, numY);
	}
	Boids::copyToDevice(x, y, numX, numY);

    #if VISUALIZE
    Boids::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
    #endif
    // unmap buffer object
    cudaGLUnmapBufferObject(boidVBO_positions);
    cudaGLUnmapBufferObject(boidVBO_velocities);
  }

  void mainLoop(float* x, float* y, int numX, int numY) {
    double fps = 0;
    double timebase = 0;
    int frame = 0;

    Boids::unitTest(); // LOOK-1.2 We run some basic example code to make sure
                       // your CUDA development setup is ready to go.
	int iter = 0;
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      frame++;
      double time = glfwGetTime();

      if (time - timebase > 1.0) {
        fps = frame / (time - timebase);
        timebase = time;
        frame = 0;
      }
	  iter++;
	  if (iter > 10000) break;
      runCUDA(x, y, numX, numY);

      std::ostringstream ss;
      ss << "[";
      ss.precision(1);
      ss << std::fixed << fps;
      ss << " fps] " << deviceName;
      glfwSetWindowTitle(window, ss.str().c_str());

      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      #if VISUALIZE
      glUseProgram(program[PROG_BOID]);
      glBindVertexArray(boidVAO);
      glPointSize((GLfloat)pointSize);
      glDrawElements(GL_POINTS, N_FOR_VIS + 1, GL_UNSIGNED_INT, 0);
      glPointSize(1.0f);

      glUseProgram(0);
      glBindVertexArray(0);

      glfwSwapBuffers(window);
      #endif
    }
    glfwDestroyWindow(window);
    glfwTerminate();
  }


  void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
  }

  void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GL_TRUE);
    }
  }

  void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  }

  void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (leftMousePressed) {
      // compute new camera parameters
      phi += (xpos - lastX) / width;
      theta -= (ypos - lastY) / height;
      theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
      updateCamera();
    }
    else if (rightMousePressed) {
      zoom += (ypos - lastY) / height;
      zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
      updateCamera();
    }

	lastX = xpos;
	lastY = ypos;
  }

  void updateCamera() {
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.z = zoom * cos(theta);
    cameraPosition.y = zoom * cos(phi) * sin(theta);
    cameraPosition += lookAt;

    projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
    glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
    projection = projection * view;

    GLint location;

    glUseProgram(program[PROG_BOID]);
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
      glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
  }
