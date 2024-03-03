
#include <iostream>
#include <cmath>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>
#include <tuple>

using namespace std;

std::string getFileName(const std::string& filepath) {
    size_t lastSlashIndex = filepath.find_last_of("/\\");
    if (lastSlashIndex != std::string::npos) {
        return filepath.substr(lastSlashIndex + 1);
    }
    return filepath; // No directory separator found, return the entire filepath
}

// Function to read an image from file using stb_image.h
std::vector<std::tuple<int, int, int>> readImage(const std::string &filename, int &width, int &height)
{
    int channels;
    unsigned char *imageData = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb);
    if (!imageData)
    {
        std::cerr << "Error: Unable to read image file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<std::tuple<int, int, int>> image(width * height);
    for (int i = 0; i < width * height; ++i)
    {
        int index = i * 3; // Each pixel has 3 channels (RGB)
        image[i] = std::make_tuple(imageData[index], imageData[index + 1], imageData[index + 2]);
    }

    stbi_image_free(imageData);
    return image;
}


// This function takes a linearized matrix in the form of a vector and
// calculates elements according to the 2D Gaussian distribution
void generateGaussian(std::vector<double> &K, int dim, int radius)
{
    double stdev = 1;
    double pi = 22.0 / 7.0;
    double constant = 1.0 / (2.0 * pi * pow(stdev, 2));
    double value;
    double sum = 0.0;

    for (int i = -radius; i < radius + 1; ++i)
    {
        for (int j = -radius; j < radius + 1; ++j)
        {   int idx = (i + radius) * dim + (j + radius);
            value = constant * (1 / exp((pow(i, 2) + pow(j, 2)) / (2 * pow(stdev, 2))));
            K[idx] = value; // (i + radius) * dim + (j + radius)
            sum += value;
        }
    }
    // Normalize the kernel
    for (int i = 0; i < dim * dim; ++i)
    {
        K[i] /= sum;
    }
}

// Function to apply Gaussian blur
void GaussianBlur(const std::vector<std::tuple<int, int, int>> &inputImage, std::vector<std::tuple<int, int, int>> &outImage, std::vector<double> K, int width, int height, int kdim)
{
    // Iterate over each pixel in the input image
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            int r = 0, g = 0, b = 0;
            int pixels = 0; // n pixels is basicallu kdim*kdim

            // Iterate over each pixel in the blur window
            for (int blurRow = -kdim; blurRow <= kdim; ++blurRow)
            {
                for (int blurCol = -kdim; blurCol <= kdim; ++blurCol)
                {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    // Take care of the image edge and ensure valid image pixel
                    if (curRow > -1 && curRow < height && curCol > -1 && curCol < width)
                    {  // Multiply each pixel by corresponding kernel weight
                        auto [curR, curG, curB] = inputImage[curRow * width + curCol];
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

int main()
{
    std::vector<double> kernel;
    int BLUR_SIZE, kRadius;
    int outCols, outRows;
    int max = 0;
    BLUR_SIZE = 3; // Kernel is square and odd in dimension, should be variable at some point

    // Read the input image
    std::string filename = "Images/example.jpeg"; // "original_image.png", "example2.png", "example.jpeg", "Pikachu.jpg"
    int width, height;

    std::vector<std::tuple<int, int, int>> inputImage = readImage(filename, width, height);

    // Output image vector to store blurred image
    std::vector<std::tuple<int, int, int>> outImage(width * height);

    /*
     * Determine whether image and filter dimensions are compatible
     */
    if ((height < 2 * BLUR_SIZE + 1) || (width < 2 * BLUR_SIZE + 1))
    {
        cout << "Image is too small to apply kernel effectively." << endl;
        exit(EXIT_FAILURE);
    }
    
    // Generate gaussian kernel
    kRadius = (BLUR_SIZE - 1) / 2; 
    kernel.resize(BLUR_SIZE*BLUR_SIZE, 0);
    generateGaussian(kernel, BLUR_SIZE, kRadius);

    /*
     * Apply Gaussian Kernel func
     */
    GaussianBlur(inputImage, outImage, kernel, width, height, BLUR_SIZE);

    /*
     * Normalizing output matrix values
     */
    int maxR = 0, maxG = 0, maxB = 0;

    // Find the maximum values for each channel
    for (auto& value : outImage) {
        auto [r, g, b] = value;
        maxR = std::max(r, maxR);
        maxG = std::max(g, maxG);
        maxB = std::max(b, maxB);
    }

    // Normalize each pixel value
    for (auto& value : outImage) {
        int r, g, b;
        std::tie(r, g, b) = value;
        r = (r * 255) / maxR;
        g = (g * 255) / maxG;
        b = (b * 255) / maxB;
        value = std::make_tuple(r, g, b);
    }

    // Save the blurred image as a JPEG file
    std::vector<unsigned char> imageData(width * height * 3); // Each pixel has 3 channels (RGB)
    for (int i = 0; i < width * height; ++i)
    {
        auto [r, g, b] = outImage[i];
        int index = i * 3;
        imageData[index] = static_cast<unsigned char>(r);
        imageData[index + 1] = static_cast<unsigned char>(g);
        imageData[index + 2] = static_cast<unsigned char>(b);
    }

    // std::string prefix = filename.substr(0, filename.find("."));
    std::string prefix = getFileName(filename);
    prefix = prefix.substr(0, prefix.find("."));
    std::string outname = "Res/" + prefix + "Blur.jpg";
    stbi_write_jpg(outname.c_str(), width, height, 3, imageData.data(), 100);

    std::cout << "Blurred image saved as " << outname << std::endl;

    exit(EXIT_SUCCESS);
}