/**
 * @file ImageBlurGPU.h
 * @brief Use CUDA to declare the windowed averaging blur algorithm and Gaussian kernel to convert each image in the MNIST dataset into Blur images.
 *        The original images and blur images of the MNIST dataset are present in ../input and ../output folders.
 *        Go to the directory where the Makefile file is located, and then enter the make command at the command line to complete the compilation, and then generate the ImageBlurCPU executable file, finally run it.
 * @authors Yan Huang, Oluwabunmi Iwakin
 * @date 03-02-2024
 * @link https://github.com/YanHuang1994/CUDA/blob/main/Project3/ImageBlurGPU/include/ImageBlurGPU.h
 *
 */
#ifndef IMAGE_BLUR_GPU_H
#define IMAGE_BLUR_GPU_H

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h" //Provides a simple API to load, decode and query image information. Supports a variety of formats including JPEG, PNG, BMP, TGA, etc.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h" //Used to write image data to a file, supports a variety of image formats, such as PNG, TGA, BMP and so on.
#include "../include/timer.h"
#include "../include/files.h"
#include "../include/check.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

// Read and detect any inline Errors
inline cudaError_t checkCuda(cudaError_t result);

// Function to reverse the bytes of an integer
int reverseInt(int i);

// Function to check if the operating system is Windows
bool isWindows();

// Function to apply windowed average blur to an image on CPU
void applyWindowedAverageBlur(unsigned char *input, unsigned char *output, int width, int height, int channels, int windowSize);

// Function to apply windowed average blur to an image on GPU
__global__ void applyWindowedAverageBlurCUDA(unsigned char *input, unsigned char *output, int width, int height, int channels, int windowSize);

// Function to generate Gaussian kernel
void generateGaussian(float *kernel, int kernelSize, float sigma);

// Function to Gaussian blur to an image
void gaussianBlur(unsigned char *input, unsigned char *output, int width, int height, float *kernel, int kernelSize);

// Function to apply gaussian blur to an image on CPU
void applyGaussianKernelBlur(unsigned char *input, unsigned char *output, int width, int height, float *kernel, int channels, int windowSize);

// Function to apply gaussian blur to an image on GPU
__global__ void gaussianBlurCUDA(const unsigned char* input, unsigned char* output, int width, int height, const float* kernel, int kernelSize);

// Function to save an image to a png file, calling the function of stb library
void saveImage(const char* filename, unsigned char* buffer, int width, int height);

// Function to create a directory if it does not exist
void createDirectoryIfNotExists(const std::string &directoryPath);

// Function to remove a directory and its contents recursively
void removeDirectoryRecursively(const std::string &dirPath);

// Function to read MNIST image data from a file
void readMNISTImages(const char* filename, unsigned char** images, int* numberOfImages, int* nRows, int* nCols);

// Function to process input and output directory, ../input directory saves the original images and ../output directory saves the blurred images.
void processDirectory();

// cpu version code, include window average and gaussian kernel
void CpuVersion();

// gpu version code, include window average and gaussian kernel
void GpuVersion();

#endif // IMAGE_BLUR_GPU_H
