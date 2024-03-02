#include "ImageBlurCPU.h"
#include <iostream>

// Define blur size (adjust as needed)
#define BLUR_SIZE 10

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

int main() {
    // Read the input image
    std::string filename = "original_image.png";
    int width, height;
    std::vector<std::tuple<int, int, int>> inputImage = readImage(filename, width, height);

    // Output image vector to store blurred image
    std::vector<std::tuple<int, int, int>> outImage(width * height);

    // Apply windowed average blur
    winAvgImageBlur(inputImage, outImage, width, height);

    // Save the blurred image as a JPEG file
    std::vector<unsigned char> imageData(width * height * 3); // Each pixel has 3 channels (RGB)
    for (int i = 0; i < width * height; ++i) {
		int r = std::get<0>(outImage[i]);
		int g = std::get<1>(outImage[i]);
		int b = std::get<2>(outImage[i]);
        int index = i * 3;
        imageData[index] = static_cast<unsigned char>(r);
        imageData[index + 1] = static_cast<unsigned char>(g);
        imageData[index + 2] = static_cast<unsigned char>(b);
    }

    stbi_write_jpg("blurred_image0.jpg", width, height, 3, imageData.data(), 100);

    std::cout << "Blurred image saved as blurred_image.jpg" << std::endl;

    return 0;
}
