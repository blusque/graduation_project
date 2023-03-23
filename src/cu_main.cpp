#include "cuda/fdk.h"
#include "cuda/tensor.h"
#include "cuda/read_file.h"

#include <iostream>
#include <chrono>
// #include <windows.h>

#include <opencv2/opencv.hpp>

int main(int argc, char *argv)
{
    std::vector<float *> Matrices;
    auto start = std::chrono::system_clock::now();
    std::cout << "Loading Matrices..." << std::endl;
    cuda::readFile("../../senbai/data666.txt", Matrices);
    auto end = std::chrono::system_clock::now();
    std::cout << "time cost: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << std::endl;
    std::cout << "Matrices Size: " << Matrices.size() << std::endl;

    // on test
    cuda::FDK fdk(44.f, "r-l", Matrices, 256, 256);
    fdk.initialize();
    fdk.weighting();
    fdk.filtering();
    auto result = fdk.backprojecting();
    int nsteps = result.steps();
    int nrows = result.rows();
    int ncols = result.cols();
    float *resultHost = (float *)malloc(nsteps * ncols * nrows * sizeof(float));
    cudaMemcpy((void *)resultHost, (void *)result.getData(), nrows * ncols * ncols * sizeof(float), cudaMemcpyDeviceToHost);
    cv::Mat showImage(cv::Size(nrows, ncols), CV_32FC1);
    for (int k = 0; k < nsteps; k++)
    {
        float max = -1e15f;
        float min = 1e15f;
        std::string str = "layer: ";
        str += std::to_string(k);
        for (int i = 0; i < nrows; i++)
        {
            for (int j = 0; j < ncols; j++)
            {
                float num = resultHost[k * nrows * ncols + i * ncols + j];
                showImage.at<float>(i, j) = num;
                // if (num > max)
                // {
                //     max = num;
                // }
                // if (num < min)
                // {
                //     min = num;
                // }
            }
        }
        cv::putText(showImage, str, cv::Point(10, 40), 
            cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0), 1.0);
        // std::cout << "max: " << max << ", min: " << min << std::endl;
        cv::imshow("fdk z", showImage);
        cv::waitKey(100);
    }

    cv::waitKey();

    for (int i = 0; i < nrows; i++)
    {
        float max = -1e15f;
        float min = 1e15f;
        std::string str = "layer: ";
        str += std::to_string(i);
        for (int k = 0; k < nsteps; k++)
        {
            for (int j = 0; j < ncols; j++)
            {
                float num = resultHost[k * nrows * ncols + i * ncols + j];
                showImage.at<float>(k, j) = num;
                // if (num > max)
                // {
                //     max = num;
                // }
                // if (num < min)
                // {
                //     min = num;
                // }
            }
        }
        // std::cout << "max: " << max << ", min: " << min << std::endl;
        cv::putText(showImage, str, cv::Point(10, 40),
                    cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0), 1.0);
        cv::imshow("fdk x", showImage);
        cv::waitKey(100);
    }

    cv::waitKey();

    for (int j = 0; j < ncols; j++)
    {
        float max = -1e15f;
        float min = 1e15f;
        std::string str = "layer: ";
        str += std::to_string(j);
        for (int k = 0; k < nsteps; k++)
        {
            for (int i = 0; i < nrows; i++)
            {
                float num = resultHost[k * nrows * ncols + i * ncols + j];
                showImage.at<float>(k, i) = num;
                // if (num > max)
                // {
                //     max = num;
                // }
                // if (num < min)
                // {
                //     min = num;
                // }
            }
        }
        // std::cout << "max: " << max << ", min: " << min << std::endl;
        cv::putText(showImage, str, cv::Point(10, 40),
                    cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0), 1.0);
        cv::imshow("fdk y", showImage);
        cv::waitKey(100);
    }

    cv::waitKey();
    free(resultHost);
    resultHost = nullptr;
    return 0;
}