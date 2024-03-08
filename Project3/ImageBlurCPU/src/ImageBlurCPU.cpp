/**
 * @file ImageBlurCPU.cpp
 * @brief Use advanced features of C++11 to declare(Only CPU version, but it is incompatible with nvcc). Implementation of windowed averaging blur algorithm and Gaussian kernel
 *        to blur images from the MNIST dataset. It reads original images from the
 *        ../input folder, applies blurring, and saves the result to the ../output folder.
 *        The comments for all the functions in this .cpp file are in the ../include/ImageBlurCPU.h
 * @authors Yan Huang, Oluwabunmi Iwakin
 * @date 03-02-2024
 */

#include "../include/ImageBlurCPU.h"

// Constants for MNIST images
const int FILE_HEADER_SIZE = 16; // size of MNIST image files
const int MNIST_IMAGE_WIDTH = 28;  // MNIST image width
const int MNIST_IMAGE_HEIGHT = 28; // MNIST image height

int reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

bool isWindows() {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

std::vector<std::tuple<int, int, int>> readImage(const std::string &filename, int &width, int &height) {
    int channels;
    unsigned char *imageData = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb);
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

std::vector<std::vector<unsigned char>> readMnistImages(const std::string &filename, int &numberOfImages) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    int magicNumber = 0;
    int nRows = 0, nCols = 0;
    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);

    if (magicNumber != 2051) {
        std::cerr << "Error: Invalid MNIST image file!" << std::endl;
        exit(EXIT_FAILURE);
    }

    file.read(reinterpret_cast<char *>(&numberOfImages), sizeof(numberOfImages));
    numberOfImages = reverseInt(numberOfImages);
    file.read(reinterpret_cast<char *>(&nRows), sizeof(nRows));
    nRows = reverseInt(nRows);
    file.read(reinterpret_cast<char *>(&nCols), sizeof(nCols));
    nCols = reverseInt(nCols);

    std::vector<std::vector<unsigned char>> images(numberOfImages, std::vector<unsigned char>(nRows * nCols));
    for (int i = 0; i < numberOfImages; ++i) {
        for (int j = 0; j < nRows * nCols; ++j) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char *>(&pixel), sizeof(pixel));
            images[i][j] = pixel;
        }
    }

    return images;
}

void winAvgImageBlur(const std::vector<std::tuple<int, int, int>> &inputImage, std::vector<std::tuple<int, int, int>> &outImage, int width, int height, int windowSize) {
    // Iterate over each pixel in the input image
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int r = 0, g = 0, b = 0;
            int pixels = 0;

            // Iterate over each pixel in the blur window
            for (int blurRow = -windowSize; blurRow <= windowSize; ++blurRow) {
                for (int blurCol = -windowSize; blurCol <= windowSize; ++blurCol) {
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

void applyWindowedAverageBlur(const std::vector<unsigned char> &inputImage, std::vector<unsigned char> &outputImage, int width, int height, int windowSize) {
    outputImage.resize(width * height);

    std::vector<std::tuple<int, int, int>> inputImageTuples;
    for (int i = 0; i < width * height; ++i) {
        inputImageTuples.push_back(std::make_tuple(inputImage[i], inputImage[i], inputImage[i]));
    }

    std::vector<std::tuple<int, int, int>> outputImageTuples(width * height);

    winAvgImageBlur(inputImageTuples, outputImageTuples, width, height, windowSize);

    for (int i = 0; i < width * height; ++i) {
        outputImage[i] = std::get<0>(outputImageTuples[i]);
    }
}

void generateGaussian(std::vector<double> &K, int dim, int radius) {
    double stdev = 0.2;
    double pi = 22.0 / 7.0;
    double constant = 1.0 / (2.0 * pi * pow(stdev, 2));
    double value;
    double sum = 0.0;

    for (int i = -radius; i < radius + 1; ++i) {
        for (int j = -radius; j < radius + 1; ++j) {
            int idx = (i + radius) * dim + (j + radius);
            value = constant * (1 / exp((pow(i, 2) + pow(j, 2)) / (2 * pow(stdev, 2))));
            K[idx] = value; // (i + radius) * dim + (j + radius)
            sum += value;
        }
    }
    // Normalize the kernel
    for (int i = 0; i < dim * dim; ++i) {
        K[i] /= sum;
    }
}

void gaussianKernelImageBlur(const std::vector<std::tuple<int, int, int>> &inputImage, std::vector<std::tuple<int, int, int>> &outImage, std::vector<double> K, int width, int height, int kdim) {
    // Iterate over each pixel in the input image
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int r = 0, g = 0, b = 0;
            int pixels = 0; // n pixels is basicallu kdim*kdim

            // Iterate over each pixel in the blur window
            for (int blurRow = -kdim; blurRow <= kdim; ++blurRow) {
                for (int blurCol = -kdim; blurCol <= kdim; ++blurCol) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    // Take care of the image edge and ensure valid image pixel
                    if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                        // Multiply each pixel by corresponding kernel weight
                        //auto [curR, curG, curB] = inputImage[curRow * width + curCol];
                        auto curPixel = inputImage[curRow * width + curCol];
                        int curR = std::get<0>(curPixel);
                        int curG = std::get<1>(curPixel);
                        int curB = std::get<2>(curPixel);


                        r += curR * K[(blurRow * kdim) + blurCol];
                        g += curG * K[(blurRow * kdim) + blurCol];
                        b += curB * K[(blurRow * kdim) + blurCol];
                        pixels++; // Number of pixels added
                    }
                }
            }
            // Write the new pixel value in outImage
            outImage[row * width + col] = std::make_tuple(r, g, b );
        }
    }
}

void applyGaussianKernelImageBlur(const std::vector<unsigned char>& inputImage, std::vector<unsigned char>& outputImage, int width, int height, int windowSize) {
    std::vector<double> kernel;
    int kRadius = (windowSize - 1) / 2;
    kernel.resize(windowSize*windowSize, 0);
    generateGaussian(kernel, windowSize, kRadius); // Fill the Gaussian kernel

    std::vector<std::tuple<int, int, int>> inputImageTuples(width * height);
    std::vector<std::tuple<int, int, int>> outputImageTuples(width * height);

    // Convert input image to RGB tuples format
    for (int i = 0; i < width * height; ++i) {
        inputImageTuples[i] = std::make_tuple(inputImage[i], inputImage[i], inputImage[i]);
    }

    // Apply Gaussian blur
    gaussianKernelImageBlur(inputImageTuples, outputImageTuples, kernel, width, height, windowSize);

    /*
     * Normalizing output matrix values
     */
    int maxR = 0, maxG = 0, maxB = 0;
    int r, g, b;

    // Find the maximum values for each channel
    for (auto& value : outputImageTuples) {
        std::tie(r, g, b) = value;
        maxR = std::max(r, maxR);
        maxG = std::max(g, maxG);
        maxB = std::max(b, maxB);
    }

    // Normalize each pixel value
    for (auto& value : outputImageTuples) {
        int r, g, b;
        std::tie(r, g, b) = value;
        r = (r * 255) / maxR;
        g = (g * 255) / maxG;
        b = (b * 255) / maxB;
        value = std::make_tuple(r, g, b);
    }

    // Prepare the output image
    outputImage.resize(width * height );
    for (int i = 0; i < width * height; ++i) {
        // Since the image is grayscale, we can take any channel's value as the output
        outputImage[i] = std::get<0>(outputImageTuples[i]);
    }

}

void saveImageAsPNG(const std::vector<unsigned char> &image, const std::string &filename, int width, int height) {
    stbi_write_png(filename.c_str(), width, height, 1, image.data(), width);
}

void createDirectoryIfNotExists(const std::string &directoryPath) {
    if (!std::ifstream(directoryPath)) {
        std::string command = "mkdir -p " + directoryPath;
        system(command.c_str());
    }
}

void removeDirectoryRecursively(const std::string &dirPath) {
    std::string command;
    if (isWindows()) {
        command = "rd /s /q \"" + dirPath + "\"";
    }
    else {
        command = "rm -r \"" + dirPath + "\"";
    }
    system(command.c_str());
}

std::vector<std::vector<unsigned char>> loadMnistDataset(int& numberOfImages) {
    std::string inputFilename = "../data/train-images.idx3-ubyte";
    return readMnistImages(inputFilename, numberOfImages);;
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

    int numberOfImages = 0;
    auto originalImages = loadMnistDataset(numberOfImages); // load train-images.idx3-ubyte

    if (!originalImages.size()) {
        std::cout << "load mnist dataset failed\n " << std::endl;
        return -1;
    }

    double totalTime = 0.0;

    // The path to the directory where the output images will be saved
    std::string outputDirectory = "../output";

    int windowSize = 3; // Blur windows size

    int choice = 0;
    std::cout << "Enter 1 for Windowed Average Blur or 2 for Gaussian Kernel Blur. ";
    std::cin >> choice;

    std::cout << "start processing " << std::endl;

    for (int i = 0; i < numberOfImages; ++i)
    {
        StartTimer();
        std::vector<unsigned char> blurredImage(MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT);

        auto currentImage = originalImages[i]; // Initially, use the original image

        for (int j = 0; j < 10; ++j)
        {
            if (choice == 1) {
                applyWindowedAverageBlur(currentImage, blurredImage, MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT, windowSize);
            } else if (choice == 2) {
                applyGaussianKernelImageBlur(currentImage, blurredImage, MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT, windowSize);
            }

            currentImage = blurredImage;

            if (i % 1000 == 0)
            { // Check if the index is a multiple of 100
                // Save the original image
                std::string originalFilename = inputDir + "/original_image_" + std::to_string(i) + ".png";
                saveImageAsPNG(currentImage, originalFilename, MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT);

                // Save the blurred image
                std::string blurredFilename = outputDir + "/blurred_image_" + std::to_string(i) + "_iter" + std::to_string(j) + ".png";
                saveImageAsPNG(blurredImage, blurredFilename, MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT);
            }
        }

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    std::cout << "Processed " << numberOfImages << " images." << std::endl;

    double avgTime = totalTime / (double)(numberOfImages);

    printf("totalTime: %0.3f second\n", totalTime);
    printf("avgTime %0.3f second\n", avgTime);

    return 0;
}
