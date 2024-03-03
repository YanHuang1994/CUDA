/**
 * @file ImageBlurCPU.cpp
 * @brief Use windowed averaging blur algorithm and Gaussian kernel to convert each image in the MNIST dataset into Blur images
 * @author Yan Huang, Oluwabunmi Iwakin
 * @date 03-02-2024
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "timer.h"
#include "files.h"
#include <vector>
#include <tuple>
#include <string>

// Function to read an image from file using stb_image.h
std::vector<std::tuple<int, int, int>> readImage(const std::string& filename, int& width, int& height);

// Function to implement windowed average blur algorithm
void winAvgImageBlur(const std::vector<std::tuple<int, int, int>>& inputImage, std::vector<std::tuple<int, int, int>>& outImage, int width, int height, int windowSize);

// Function to implement guassian kernel blur algorithm (Not Implement)
void gaussianKernelImageBlur();

// Function to read images from MNIST
std::vector<std::vector<unsigned char>> readMnistImages(const std::string& filename, int& numberOfImages);

// Utility function to reverse integer byte order
int reverseInt(int i);

// save an image to PNG file
void saveImageAsPNG(const std::vector<unsigned char>& image, const std::string& filename, int width, int height);

// Function to apply windowed average blur algorithm
void applyWindowedAverageBlur(const std::vector<unsigned char>& inputImage, std::vector<unsigned char>& outputImage, int width, int height, int windowSize);

// Function to apply gaussian kernel blur algorithm (Not Implement)
void applyGaussianKernelImageBlur();

//create a directory
void createDirectoryIfNotExists(const std::string& directoryPath);

// remove directory recursively
void removeDirectoryRecursively(const std::string& dirPath);