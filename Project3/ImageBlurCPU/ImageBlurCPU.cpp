#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "ImageBlurCPU.h"

bool isWindows() {
    #ifdef _WIN32
    return true;
    #else
    return false;
    #endif
}

// Define blur size (adjust as needed)
#define BLUR_SIZE 3

const int FILE_HEADER_SIZE = 16; //size of MNIST image files

const int IMAGE_WIDTH = 28; //MNIST image width
const int IMAGE_HEIGHT = 28; //MNIST image height

int reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

std::vector<std::tuple<int, int, int>> readImage(const std::string& filename, int& width, int& height) {
    int channels;
    unsigned char* imageData = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb);
    if (!imageData) {
        std::cerr << "Error: Unable to read image file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::tuple<int, int, int>> image(width * height);
    for (int i = 0; i < width * height; ++i) {
        int index = i * 3; // Each pixel has 3 channels (RGB)
        image[i] = std::make_tuple(imageData[index], imageData[index + 1], imageData[index + 2]);
    }

    stbi_image_free(imageData);
    return image;
}

void winAvgImageBlur(const std::vector<std::tuple<int, int, int>>& inputImage, std::vector<std::tuple<int, int, int>>& outImage, int width, int height) {
    // Iterate over each pixel in the input image
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int r = 0, g = 0, b = 0;
            int pixels = 0;

            // Iterate over each pixel in the blur window
            for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
                for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    // Take care of the image edge and ensure valid image pixel
                    if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                        int curR = std::get<0>(inputImage[curRow * width + curCol]);
						int curG = std::get<1>(inputImage[curRow * width + curCol]);
						int curB = std::get<2>(inputImage[curRow * width + curCol]);
                        r += curR;
                        g += curG;
                        b += curB;
                        pixels++; // Number of pixels added
                    }
                }
            }

            // Write the new pixel value in outImage
            outImage[row * width + col] = std::make_tuple(r / pixels, g / pixels, b / pixels);
        }
    }
}

std::vector<std::vector<unsigned char>> readMnistImages(const std::string& filename, int& numberOfImages) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    int magicNumber = 0;
    int nRows = 0, nCols = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);

    if (magicNumber != 2051) {
        std::cerr << "Error: Invalid MNIST image file!" << std::endl;
        exit(EXIT_FAILURE);
    }

    file.read(reinterpret_cast<char*>(&numberOfImages), sizeof(numberOfImages));
    numberOfImages = reverseInt(numberOfImages);
    file.read(reinterpret_cast<char*>(&nRows), sizeof(nRows));
    nRows = reverseInt(nRows);
    file.read(reinterpret_cast<char*>(&nCols), sizeof(nCols));
    nCols = reverseInt(nCols);

    std::vector<std::vector<unsigned char>> images(numberOfImages, std::vector<unsigned char>(nRows * nCols));
    for (int i = 0; i < numberOfImages; ++i) {
        for (int j = 0; j < nRows * nCols; ++j) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[i][j] = pixel;
        }
    }

    return images;
}

void saveImageAsPNG(const std::vector<unsigned char>& image, const std::string& filename, int width, int height) {
    stbi_write_png(filename.c_str(), width, height, 1, image.data(), width);
}

void applyWindowedAverageBlur(const std::vector<unsigned char>& inputImage, std::vector<unsigned char>& outputImage, int width, int height, int windowSize) {
    outputImage.resize(width * height);

    std::vector<std::tuple<int, int, int>> inputImageTuples;
    for (int i = 0; i < width * height; ++i) {
        inputImageTuples.push_back(std::make_tuple(inputImage[i], inputImage[i], inputImage[i]));
    }

    std::vector<std::tuple<int, int, int>> outputImageTuples(width * height);

    winAvgImageBlur(inputImageTuples, outputImageTuples, width, height);

    for (int i = 0; i < width * height; ++i) {
        outputImage[i] = std::get<0>(outputImageTuples[i]);
    }
}

void createDirectoryIfNotExists(const std::string& directoryPath) {
    if (!std::ifstream(directoryPath)) {
        std::string command = "mkdir -p " + directoryPath;
        system(command.c_str());
    }
}

void removeDirectoryRecursively(const std::string& dirPath) {
    std::string command;
    if (isWindows()) {
        command = "rd /s /q \"" + dirPath + "\"";
    } else {
        command = "rm -r \"" + dirPath + "\"";
    }
    system(command.c_str());
}

int main()
{
    removeDirectoryRecursively("../input");
    removeDirectoryRecursively("../output");

    std::string inputDir = "../input";
    std::string outputDir = "../output";

    // Create input and output directories if they don't exist
    createDirectoryIfNotExists(inputDir);
    createDirectoryIfNotExists(outputDir);

    std::string inputFilename = "../data/train-images.idx3-ubyte";
    int numberOfImages = 0;
    auto images = readMnistImages(inputFilename, numberOfImages);

    // The path to the directory where the output images will be saved
    std::string outputDirectory = "../output";

    int windowSize = 3; // Blur windows size
    for (int i = 0; i < numberOfImages; ++i) {
        if (i % 100 == 0) { // Check if the index is a multiple of 100
            std::vector<unsigned char> blurredImage(28 * 28);
            applyWindowedAverageBlur(images[i], blurredImage, 28, 28, windowSize);

            // Save the original image
            std::string originalFilename = inputDir + "/original_image_" + std::to_string(i) + ".png";
            saveImageAsPNG(images[i], originalFilename, 28, 28);

            // Save the blurred image
            std::string blurredFilename = outputDir + "/blurred_image_" + std::to_string(i) + ".png";
            saveImageAsPNG(blurredImage, blurredFilename, 28, 28);

            std::cout << "Saved image " << i << std::endl;
        }
    }

    std::cout << "Processed " << numberOfImages << " images." << std::endl;

    return 0;
}
