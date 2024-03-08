 /**
 * @file ImageBlurCPU.cu
 * @brief Using CUDA C++ to finish the Implementation of windowed averaging blur algorithm and Gaussian kernel
 *        to blur images from the MNIST dataset. It reads original images from the
 *        ../input folder, applies blurring, and saves the result to the ../output folder.
 *        The comments for all the functions in this .cu file are in the ../include/ImageBlurGPU.h
 * @authors Yan Huang, Oluwabunmi Iwakin
 * @date 03-02-2024
 * @link https://github.com/YanHuang1994/CUDA/blob/main/Project3/ImageBlurGPU/src/ImageBlurGPU.cu
 */

#include "../include/ImageBlurGPU.h"

#define TILE_WIDTH 32
#define MAX_WINDOW_SIZE 16

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r)*(num)+(c))

// Read and detect any inline Errors
inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

bool isWindows() {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

void applyWindowedAverageBlur(unsigned char *input, unsigned char *output, int width, int height, int channels, int windowSize) {
    // Loop through each pixel in the image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            long long sum[4] = {0}; // Initialize sum of pixel values
            int count = 0; // Initialize count of pixels in the window

            // Loop through the pixels in the window
            for (int wy = -windowSize; wy <= windowSize; wy++) {
                for (int wx = -windowSize; wx <= windowSize; wx++) {
                    int nx = x + wx; // Calculate x-coordinate of neighbor pixel
                    int ny = y + wy; // Calculate y-coordinate of neighbor pixel

                    // Check if neighbor pixel is within image bounds
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        // Loop through channels and accumulate pixel values
                        for (int c = 0; c < channels; c++) {
                            sum[c] += input[(ny * width + nx) * channels + c];
                        }
                        count++; // Increment count of valid pixels in the window
                    }
                }
            }

            // Calculate average pixel value for each channel
            for (int c = 0; c < channels; c++) {
                output[(y * width + x) * channels + c] = sum[c] / count;
            }
        }
    }
}

__global__ void applyWindowedAverageBlurCUDA(uint8_t *inputImage, uint8_t *outImage, int width, int height, int channels, int windowSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int r = 0, g = 0, b = 0;
        int pixels = 0;

        for (int blurRow = -windowSize; blurRow <= windowSize; ++blurRow) {
            for (int blurCol = -windowSize; blurCol <= windowSize; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    int idx = curRow * width + curCol;
                    int curR = (inputImage[idx] >> 16) & 0xFF; // Extract red component
                    int curG = (inputImage[idx] >> 8) & 0xFF;  // Extract green component
                    int curB = inputImage[idx] & 0xFF;         // Extract blue component
                    r += curR;
                    g += curG;
                    b += curB;
                    pixels++;
                }
            }
        }

        int outIdx = row * width + col;
        outImage[outIdx] = ((r / pixels) << 16) | ((g / pixels) << 8) | (b / pixels); // Pack RGB values into a single integer
    }
}


void generateGaussian(float *kernel, int kernelSize, float sigma) {
    float sum = 0.0f;
    int i, j;
    float constant = 1.0f / (2.0f * M_PI * sigma * sigma);

    for (i = 0; i < kernelSize; ++i) {
        for (j = 0; j < kernelSize; ++j) {
            int x = i - (kernelSize / 2);
            int y = j - (kernelSize / 2);
            float exponent = -(x*x + y*y) / (2.0f * sigma * sigma);
            kernel[i * kernelSize + j] = constant * expf(exponent);
            sum += kernel[i * kernelSize + j];
        }
    }

    // Normalize the kernel
    for (i = 0; i < kernelSize * kernelSize; ++i) {
        kernel[i] /= sum;
    }
}

void gaussianBlur(unsigned char *input, unsigned char *output, int width, int height, float *kernel, int kernelSize) {
    int halfKernel = kernelSize / 2;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        sum += input[py * width + px] * kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
                    }
                }
            }
            output[y * width + x] = (unsigned char)sum;
        }
    }
}

void applyGaussianKernelBlur(unsigned char *input, unsigned char *output, int width, int height, float *kernel, int channels, int windowSize) {
    float sigma = 1.0f;
    generateGaussian(kernel, windowSize, sigma);
    gaussianBlur(input, output, width, height, kernel, windowSize);
}

__global__ void gaussianBlurCUDA(const unsigned char* input, unsigned char* output, int width, int height, const float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int halfKernel = kernelSize / 2;
        float sum = 0.0f;
        for (int ky = -halfKernel; ky <= halfKernel; ky++) {
            for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                int px = x + kx;
                int py = y + ky;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    sum += input[py * width + px] * kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
                }
            }
        }
        output[y * width + x] = static_cast<unsigned char>(sum);
    }
}

void saveImage(const char* filename, unsigned char* buffer, int width, int height) {
    stbi_write_png(filename, width, height, 1, buffer, width);
}

void createDirectoryIfNotExists(const std::string &directoryPath) {
    // Check if directory does not exist
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

void readMNISTImages(const char* filename, unsigned char** images, int* numberOfImages, int* nRows, int* nCols) {
    FILE *file = fopen(filename, "rb"); // Open file in binary mode
    if (file == NULL) {
        fprintf(stderr, "Cannot open file %s\n", filename); // Print error message
        exit(1);
    }

    int magicNumber; // Initialize variable to store magic number
    fread(&magicNumber, sizeof(magicNumber), 1, file); // Read magic number from file
    magicNumber = reverseInt(magicNumber); // Reverse bytes of magic number

    fread(numberOfImages, sizeof(*numberOfImages), 1, file); // Read number of images from file
    *numberOfImages = reverseInt(*numberOfImages); // Reverse bytes of number of images

    fread(nRows, sizeof(*nRows), 1, file); // Read number of rows from file
    *nRows = reverseInt(*nRows); // Reverse bytes of number of rows

    fread(nCols, sizeof(*nCols), 1, file); // Read number of columns from file
    *nCols = reverseInt(*nCols); // Reverse bytes of number of columns

    int imageSize = (*nRows) * (*nCols); // Calculate image size in bytes
    *images = (unsigned char*)malloc(*numberOfImages * imageSize); // Allocate memory for images

    fread(*images, 1, *numberOfImages * imageSize, file); // Read image data from file
    fclose(file); // Close file
}

void processDirectory() {
    removeDirectoryRecursively("../input");
    removeDirectoryRecursively("../output");

    std::string inputDir = "../input";
    std::string outputDir = "../output";

    // Create input and output directories if they don't exist
    createDirectoryIfNotExists(inputDir);
    createDirectoryIfNotExists(outputDir);
}

int main() {

    int choice = 0;
    std::cout << "Enter 1 for CpuVersion or 2 for GpuVersion: ";
    std::cin >> choice;

    if (choice == 1) {
        std::cout << "Start processing the cpu version of the blur algorithm." << std::endl;
        CpuVersion();
    } else if (choice == 2) {
        std::cout << "Start processing the gpu version of the blur algorithm." << std::endl;
        GpuVersion();
    }

    return 0;
}

void CpuVersion() {

    processDirectory();

    int choice = 0;
    std::cout << "Enter 1 for Windowed Average Blur or 2 for Gaussian Kernel Blur. ";
    std::cin >> choice;

    double totalTime = 0.0;

    unsigned char* images;
    int numberOfImages, nRows, nCols;
    readMNISTImages("../data/train-images.idx3-ubyte", &images, &numberOfImages, &nRows, &nCols); // get images, nRows, nCols from the Dataset

    if(tempImages == NULL || numberOfImages == 0 || nRows == 0 || nCols == 0) {
        printf("Dataset is null\n")
        return;
    }

    int imageSize = nRows * nCols;
    unsigned char* blurredImage = (unsigned char*)malloc(imageSize);
    unsigned char* tempImage = (unsigned char*)malloc(imageSize);

    if (blurredImage == NULL || tempImage == NULL) {
        printf("Memory allocation failed\n");
        free(images);
        if (blurredImage != NULL) free(blurredImage);
        if (tempImage != NULL) free(tempImage);
        return;
    }

    char originalFilename[256];
    char blurredFilename[256];
    for (int i = 0; i < numberOfImages; ++i) {
        StartTimer();
        unsigned char* currentImage = &images[i * imageSize];
        if (i % 1000 == 0) {
            memcpy(tempImage, currentImage, imageSize);
            snprintf(originalFilename, sizeof(originalFilename), "../input/original_image_%d.png", i);
            saveImage(originalFilename, currentImage, nCols, nRows);
        }
        // Iterate the blurred algorithm ten times
        for (int j = 0; j < 10; ++j) {
            if (choice == 1) {
                applyWindowedAverageBlur(tempImage, blurredImage, nRows, nCols, 1, 3);
            } else if (choice == 2) {
                float *kernel = (float *)malloc(3 * 3 * sizeof(float));
                if (kernel == NULL) {
                    printf("Memory allocation for Gaussian kernel failed\n");
                } else {
                    applyGaussianKernelBlur(tempImage, blurredImage, nRows, nCols, kernel, 1, 3);
                    free(kernel);
                }
            }

            if (i % 1000 == 0) {
                memcpy(tempImage, blurredImage, imageSize);

                snprintf(blurredFilename, sizeof(blurredFilename), "../output/blurred_image_%d_iter%d.png", i, j);
                saveImage(blurredFilename, blurredImage, nCols, nRows);
            }
        }
        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }
    std::cout << "Processed " << numberOfImages << " images." << std::endl;
    double avgTime = totalTime / (double)(numberOfImages);
    printf("totalTime: %0.3f second\n", totalTime);
    printf("avgTime %0.3f second\n", avgTime);

    free(images);
    free(blurredImage);
    free(tempImage);
}

void GpuVersion() {
    processDirectory();

    int choice = 0;
    std::cout << "Enter 1 for Windowed Average Blur or 2 for Gaussian Kernel Blur. ";
    std::cin >> choice;

    double totalTime = 0.0;

    unsigned char* images;
    unsigned char* tempImages;
    unsigned char* blurredImage;
    unsigned char* tempImage;
    int numberOfImages, nRows, nCols;

    readMNISTImages("../data/train-images.idx3-ubyte", &tempImages, &numberOfImages, &nRows, &nCols);

    if(tempImages == NULL || numberOfImages == 0 || nRows == 0 || nCols == 0) {
        printf("Dataset is null\n")
        return;
    }

    int imageSize = nRows * nCols;
    checkCuda(cudaMallocManaged(&images, numberOfImages * imageSize * sizeof(unsigned char)));
    checkCuda(cudaMallocManaged(&blurredImage, imageSize * sizeof(unsigned char)));
    checkCuda(cudaMallocManaged(&tempImage, imageSize * sizeof(unsigned char)));

    memcpy(images, tempImages, numberOfImages * imageSize * sizeof(unsigned char));
    free(tempImages);

    int deviceId;
    int numberOfSMs;
    int kernelSize = 3;

    checkCuda(cudaGetDevice(&deviceId));
    checkCuda(cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId));

    float sigma = 1.0f;

    char originalFilename[256];
    char blurredFilename[256];
    for (int i = 0; i < numberOfImages; ++i) {
        StartTimer();
		unsigned char* currentImage = &images[i * imageSize];
        if (i % 1000 == 0) {
            memcpy(tempImage, currentImage, imageSize);
            snprintf(originalFilename, sizeof(originalFilename), "../input/original_image_%d.png", i);
            saveImage(originalFilename, currentImage, nCols, nRows);

        }
        // Iterate the blurred algorithm ten times
        for (int j = 0; j < 10; ++j) {
            int BLOCK_SIZE = 32;
            dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
            dim3 gridSize((nRows + BLOCK_SIZE - 1) / BLOCK_SIZE, (nCols + BLOCK_SIZE - 1) / BLOCK_SIZE);

            if (choice == 1) {
                applyWindowedAverageBlurCUDA<<<gridSize, blockSize>>>(currentImage, blurredImage, nCols, nRows, 3, kernelSize);
            } else if (choice == 2) {
                float *kernel;
                //cudaMallocManaged(&kernel, 3 * 3 * sizeof(float));
                checkCuda(cudaMallocManaged(&kernel, 3 * 3 * sizeof(float)));
                generateGaussian(kernel, kernelSize, sigma);

                float *dkernel;
                checkCuda(cudaMalloc(&dkernel, kernelSize * kernelSize * sizeof(float)));
                checkCuda(cudaMemcpy(dkernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));

                generateGaussian(kernel, kernelSize, sigma);
                checkCuda(cudaDeviceSynchronize());
                //cudaDeviceSynchronize();

                if (kernel == NULL) {
                    printf("Memory allocation for Gaussian kernel failed\n");
                } else {
                    gaussianBlurCUDA<<<gridSize, blockSize>>>(currentImage, blurredImage, nCols, nRows, dkernel, kernelSize);

                    checkCuda(cudaFree(kernel));
                    checkCuda(cudaFree(dkernel));
                }
            }

            if (i % 1000 == 0) {
                memcpy(currentImage, blurredImage, imageSize);
                snprintf(blurredFilename, sizeof(blurredFilename), "../output/blurred_image_%d_iter%d.png", i, j);
                saveImage(blurredFilename, blurredImage, nCols, nRows);
            }
        }
        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }
    // cudaDeviceSynchronize();

    std::cout << "Processed " << numberOfImages << " images." << std::endl;
    double avgTime = totalTime / (double)(numberOfImages);
    printf("totalTime: %0.3f second\n", totalTime);
    printf("avgTime %0.3f second\n", avgTime);

    checkCuda(cudaFree(images));
    checkCuda(cudaFree(blurredImage));
    checkCuda(cudaFree(tempImage));
}