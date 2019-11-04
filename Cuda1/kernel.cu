
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>

#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

//11. ”величить размер изображени€, использу€ интерпол€цию по методу ближайшего соседа

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

// работает даже дл€ очень больших массивов
__global__ void cuda_scale_img(unsigned char* sourcePic, unsigned char* scalePic, int rowCount, int colCount, int scaleFactor)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= rowCount || j >= colCount)
		return;
	unsigned char r = sourcePic[(i * colCount + j) * 3 + 0];
	unsigned char g = sourcePic[(i * colCount + j) * 3 + 1];
	unsigned char b = sourcePic[(i * colCount + j) * 3 + 2];

	int startScaleI = i * scaleFactor;
	int endScaleI = startScaleI + scaleFactor;
	int startScaleJ = j * scaleFactor;
	int endScaleJ = startScaleJ + scaleFactor;
	int scaleColCount = colCount * scaleFactor;

	for (int scaleI = startScaleI; scaleI < endScaleI; scaleI++) {
		for (int scaleJ = startScaleJ; scaleJ < endScaleJ; scaleJ++) {
			scalePic[(scaleI * scaleColCount + scaleJ) * 3 + 0] = r;
			scalePic[(scaleI * scaleColCount + scaleJ) * 3 + 1] = g;
			scalePic[(scaleI * scaleColCount + scaleJ) * 3 + 2] = b;
		}
	}

}

int main(void)
{
	int scaleFactor = 2;
	Mat image = imread("cat.jpg", IMREAD_COLOR);   // Read the file
	Mat scalePic(image.rows * scaleFactor, image.cols * scaleFactor, CV_8UC3);
	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	/*namedWindow("IMAGE", WINDOW_AUTOSIZE);
	imshow("IMAGE", image);
	waitKey(0);*/

	clock_t startCPU = clock();

	const int IMAGE_BYTE_SIZE = 3 * image.rows * image.cols * sizeof(char);
	const int SCALE_IMAGE_BYTE_SIZE = scaleFactor * scaleFactor * IMAGE_BYTE_SIZE;

	for (int i = 0; i < image.rows; i++)
	{
		//pointer to 1st pixel in row
		Vec3b* picRow = image.ptr<Vec3b>(i);
		for (int j = 0; j < image.cols; j++)
		{
			int startScaleI = i * scaleFactor;
			int endScaleI = startScaleI + scaleFactor;
			int startScaleJ = j * scaleFactor;
			int endScaleJ = startScaleJ + scaleFactor;

			for (int scaleI = startScaleI; scaleI < endScaleI; scaleI++) {
				Vec3b* scalePicRow = scalePic.ptr<Vec3b>(scaleI);
				for (int scaleJ = startScaleJ; scaleJ < endScaleJ; scaleJ++) {
					scalePicRow[scaleJ] = picRow[j];
				}
			}

		}

	}
	float elapsedTimeCPU = (double)(clock() - startCPU) / CLOCKS_PER_SEC;
	cout << "CPU sum time = " << elapsedTimeCPU * 1000 << " ms\n";
	cout << "CPU memory throughput = " << IMAGE_BYTE_SIZE / elapsedTimeCPU / 1024 / 1024 / 1024 << " Gb/s\n";

	imwrite("cpu-size.jpg", scalePic);

	/*namedWindow("CPU SIZE", WINDOW_AUTOSIZE);
	imshow("CPU SIZE", scalePic);
	waitKey(0);*/

	unsigned char* source_image;
	CHECK(cudaMalloc(&source_image, IMAGE_BYTE_SIZE));
	CHECK(cudaMemcpy(source_image, image.data, IMAGE_BYTE_SIZE, cudaMemcpyHostToDevice));

	unsigned char* dist_image;
	CHECK(cudaMalloc(&dist_image, SCALE_IMAGE_BYTE_SIZE));

	cudaEvent_t startCUDA, stopCUDA;
	float elapsedTimeCUDA;

	cudaEventCreate(&startCUDA);
	cudaEventCreate(&stopCUDA);
	cudaEventRecord(startCUDA, 0);

	cuda_scale_img<<<dim3((image.rows+31)/32, (image.cols+31)/32,1), dim3(32,32,1)>>>(source_image, dist_image, image.rows, image.cols, scaleFactor);

	cudaEventRecord(stopCUDA, 0);
	cudaEventSynchronize(stopCUDA);
	CHECK(cudaGetLastError());

	cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

	cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
	cout << "CUDA memory throughput = " << IMAGE_BYTE_SIZE / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n";

	Mat scalePicGpu(image.rows * scaleFactor, image.cols * scaleFactor, CV_8UC3);
	CHECK(cudaMemcpy(scalePicGpu.data, dist_image, SCALE_IMAGE_BYTE_SIZE, cudaMemcpyDeviceToHost));

	imwrite("gpu-size.jpg", scalePicGpu);

	//namedWindow("GPU SIZE", WINDOW_AUTOSIZE);
	//imshow("GPU SIZE", scalePicGpu);
	//waitKey(0);

	CHECK(cudaFree(source_image));
	CHECK(cudaFree(dist_image));
	return 0;
}
