/**
 * @file ImageBlurCPU.cpp
 * @brief Use windowed averaging blur algorithm and Gaussian kernel to convert each image in the MNIST dataset into Blur images
 *
 * This program reads images from the MNIST dataset and saves the first image
 * as a PNG file using the stb_image_write library. This demonstrates how to
 * work with the MNIST dataset in a C++ environment and provides a foundation
 * for further image processing and machine learning tasks.
 *
 * @author Yan Huang, Oluwabunmi Iwakin
 * @date 03-02-2024
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>
#include <tuple>

// Define blur size (adjust as needed)
#define BLUR_SIZE 10

// Function to read an image from file using stb_image.h
std::vector<std::tuple<int, int, int>> readImage(const std::string& filename, int& width, int& height);

// Function to apply windowed average blur
void winAvgImageBlur(const std::vector<std::tuple<int, int, int>>& inputImage, std::vector<std::tuple<int, int, int>>& outImage, int width, int height);

