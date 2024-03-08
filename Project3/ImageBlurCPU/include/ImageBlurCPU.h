/**
 * @file ImageBlurCPU.h
 * @brief Use advanced features of C++11 to declare(Only CPU version, but it is incompatible with nvcc) windowed averaging blur algorithm and Gaussian kernel to convert each image in the MNIST dataset into Blur images.
 *        The original images and blur images of the MNIST dataset are present in ../input and ../output folders.
 *        Go to the directory where the Makefile file is located, and then enter the make command at the command line to complete the compilation, and then generate the ImageBlurCPU executable file, finally run it.
 * @authors Yan Huang, Oluwabunmi Iwakin
 * @date 03-02-2024
 * @link https://github.com/YanHuang1994/CUDA/blob/main/Project3/ImageBlurCPU/include/ImageBlurCPU.h
 *
 */
#pragma once

// Define to allow inclusion and compilation of stb_image_write functionality
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "timer.h"
#include "files.h"
#include <vector>
#include <tuple>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <typeinfo>

using namespace std;

// Function to read an ordinary image from file using stb_image.h
std::vector<std::tuple<int, int, int>> readImage(const std::string& filename, int& width, int& height);

// Function to read images from MNIST
std::vector<std::vector<unsigned char>> readMnistImages(const std::string& filename, int& numberOfImages);

// Function to implement windowed average blur algorithm
void winAvgImageBlur(const std::vector<std::tuple<int, int, int>>& inputImage, std::vector<std::tuple<int, int, int>>& outImage, int width, int height, int windowSize);

// Function to apply windowed average blur algorithm
void applyWindowedAverageBlur(const std::vector<unsigned char>& inputImage, std::vector<unsigned char>& outputImage, int width, int height, int windowSize);

// This function takes a linearized matrix in the form of a vector and
// calculates elements according to the 2D Gaussian distribution
void generateGaussian(std::vector<double> &K, int dim, int radius);

// Function to apply Gaussian blur
void gaussianKernelImageBlur(const std::vector<std::tuple<int, int, int>> &inputImage, std::vector<std::tuple<int, int, int>> &outImage, std::vector<double> K, int width, int height, int kdim);

// Function to apply gaussian kernel blur algorithm
void applyGaussianKernelImageBlur(const std::vector<unsigned char>& inputImage, std::vector<unsigned char>& outputImage, int width, int height, int windowSize);

// Utility function to reverse integer byte order
int reverseInt(int i);

bool isWindows();

// save an image to PNG file
void saveImageAsPNG(const std::vector<unsigned char>& image, const std::string& filename, int width, int height);

//create a directory
void createDirectoryIfNotExists(const std::string& directoryPath);

// remove directory recursively
void removeDirectoryRecursively(const std::string& dirPath);

// load MNIST Dataset
std::vector<std::vector<unsigned char>> loadMnistDataset(int& numberOfImages);