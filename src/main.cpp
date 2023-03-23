#include "cpu/fdk.h"
#include "cpu/read_file.h"

#include <iostream>
#include <chrono>
// #include <windows.h>

#include <opencv2/opencv.hpp>

float clip(float x)
{
    if (x < 0.f)
    {
        return 0.f;
    }
    else if (x > 1.f)
    {
        return 1.f;
    }
    else
    {
        return x;
    }
}

int main(int argc, char *argv)
{
    std::vector<Matrix> Matrices;
    auto start = std::chrono::system_clock::now();
    readFile("../../senbai/data666.txt", Matrices);
    auto end = std::chrono::system_clock::now();
    std::cout << "time cost: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << std::endl; 
    std::cout << "Matrices Size: " << Matrices.size() << std::endl;
    std::cout << "matrix size: " << Matrices[0].rows()
              << ", " << Matrices[0].cols() << std::endl;
    // float max = -1e15, min = 1e15;
    // for (int k = 0; k < Matrices.size(); k++)
    // {
    //     for (int i = 0; i < Matrices[0].rows(); i++)
    //     {
    //         for (int j = 0; j < Matrices[0].cols(); j++)
    //         {
    //             if (std::abs(Matrices[k](i, j)) > 1e-6)
    //             {
    //                 std::cout << "Non Zero: " << k << ", " << i << ", " << j << std::endl;
    //                 std::cout << Matrices[k](i, j) << std::endl;
    //                 Sleep(100);
    //             }
                
    //             if (Matrices[k](i, j) > max)
    //             {
    //                 max = Matrices[k](i, j);
    //             }
    //             if (Matrices[k](i, j) < min)
    //             {
    //                 min = Matrices[k](i, j);
    //             }
    //         }
    //     }
    //     std::cout << "max: " << max << ", min: " << min << std::endl;
    //     Sleep(500);
    // }

    // on test
    FDK fdk(44.f, "r-l", Matrices);
    fdk.initialize();
    fdk.weighting();
    fdk.filtering();
    auto result = fdk.backprojecting();
    int nrows = result[0].rows();
    int ncols = result[0].cols();
    cv::Mat showImage(cv::Size(nrows, ncols), CV_32FC1);
    for (int k = 0; k < result.size(); k++)
    {
        float max = -1e15f;
        float min = 1e15f;
        
        // std::cout << "inited, size: " << nrows << ", " << ncols << std::endl;
        for (int i = 0; i < nrows; i++)
        {
            for (int j = 0; j < ncols; j++)
            {
                float num = clip(result[k](i, j) * 3.1416 / 180.f * 64.f);
                showImage.at<float>(i, j) = num;
                if (num > max)
                {
                    max = num;
                }
                if (num < min)
                {
                    min = num;
                }
            }
        }
        std::cout << "max: " << max << ", min: " << min << std::endl;
        cv::imshow("fdk", showImage);
        cv::waitKey(100);
        // std::cout << "showed" << std::endl;
    }
    
    cv::waitKey();
    return 0;
}