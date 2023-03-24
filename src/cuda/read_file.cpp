#include "cuda/read_file.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#define NDEBUG
#include <assert.h>
#include <chrono>
#include <Windows.h>

using std::string;
using std::vector;

void split(const string &str, const char split, vector<string> &result)
{
    std::istringstream iss(str);
    string buffer;
    while (getline(iss, buffer, split))
    {
        result.push_back(buffer);
    }
    auto end = std::chrono::system_clock::now();
}

template <class Type>
Type stringToNum(const string &str)
{
    std::istringstream iss(str);
    Type num;
    iss >> num;
    return num;
}

bool str2float(const string &str, float &num)
{
    bool isNum = false;
    int power = 0;
    int flag = 1;
    num = 0.f;
    if (str.data()[0] == '.')
    {
        power = -1;
    }
    else if (str.data()[0] == '-')
    {
        flag = -1;
    }
    for (auto &&c : str)
    {
        if (c == '-')
        {
            flag = -1;
        }
        else if (c == '.')
        {
            power = -1;
        }
        else if (c >= '0' && c <= '9')
        {
            isNum = true;
            if (power == 0)
            {
                num *= 10.f;
                num += (float)(c - '0');
            }
            else if (power < 0)
            {
                num += std::pow(10.f, power) * (float)(c - '0');
                power -= 1;
            }
        }
    }
    num *= flag;
    return isNum;
}

bool str2int(const string &str, int &num)
{
    float numF = 0.f;
    auto isNum = str2float(str, numF);
    num = static_cast<int>(numF);
    return isNum;
}

const char *getLine(const char *buf, int &len)
{
    const char *tmp = buf;
    if (*tmp == 'E')
        return nullptr;
    while (*tmp && (*tmp != 0x0d && *tmp != 0x0a && *tmp != '\n'))
        ++tmp;
    len = tmp - buf; //

    // skip New-Line char
    if (*tmp == 0x0d)
    { // Windows style New-Line 0x0d 0x0a
        tmp += 2;
        // assert(*tmp == 0x0a);
    } // else Unix style New-Line 0x0a
    else
    {
        ++tmp;
    }

    return tmp;
}

namespace cuda
{
    void readFile(const char *filename, vector<float *> &Matrices)
    {
        std::cout << "Reading " << filename << std::endl;
        HANDLE hFile = CreateFile(filename,
                                  GENERIC_READ | GENERIC_WRITE,
                                  FILE_SHARE_READ,
                                  NULL,
                                  OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL,
                                  NULL);
        if (hFile == INVALID_HANDLE_VALUE)
        {
            std::cout << " CreateFile fail" << std::endl;
            exit(-1);
        }

        // 创建一个文件映射内核对象
        HANDLE hFileMap = CreateFileMapping(hFile,
                                            NULL,
                                            PAGE_READWRITE,
                                            NULL,
                                            NULL,
                                            NULL);
        if (hFileMap == NULL)
        {
            std::cout << "CreateFileMapping fail" << std::endl;
            exit(-1);
        }

        // 将文件数据映射到进程的地址空间
        char *pMapData = (char *)MapViewOfFile(hFileMap,
                                               FILE_MAP_ALL_ACCESS,
                                               NULL,
                                               NULL,
                                               NULL);
        if (pMapData == NULL)
        {
            std::cout << " MapViewOfFile fail" << std::endl;
            exit(-1);
        }

        // 读取数据
        const char *pBuf = pMapData;
        int index = 0;
        int len = 0;
        const char *startBuf = pMapData;
        std::cout << "Load Finished" << std::endl;
        float *mat;
        int nrow = 0, ncol = 0;
        auto sizeTheta = 0;
        auto sizeX = 0;
        auto sizeY = 0;
        auto matrixNum = 0;
        while (true)
        {
            // std::cout << "index1: " << index << std::endl;
            startBuf = getLine(pBuf, len);
            if (startBuf == nullptr)
            {
                index++;
                break;
            }
            string line(pBuf, len);
            if (line.data()[0] == '#')
            {
                vector<string> splited;
                split(line, ' ', splited);
                // std::cout << "splited size: " << splited.size() << "\n";
                // assert(str2int(splited[2], sizeTheta));
                // assert(str2int(splited[3], sizeX));
                // assert(str2int(splited[4], sizeY));
                str2int(splited[2], sizeTheta);
                str2int(splited[3], sizeX);
                str2int(splited[4], sizeY);
                mat = (float *)malloc(sizeX * sizeY * sizeof(float));
                Matrices.reserve(sizeTheta);
            }
            else if (line.size() == 0)
            {
                Matrices.push_back(mat);
                mat = (float *)malloc(sizeX * sizeY * sizeof(float));
                nrow = 0;
                ncol = 0;
            }
            else
            {
                vector<string> splited;
                split(line, '\t', splited);
                // ncol = 0;
                nrow = 0;
                for (auto &&str : splited)
                {
                    float num = 0.f;
                    // assert(str2float(str, num));
                    str2float(str, num);
                    mat[nrow * sizeX + ncol] = num;
                    // ncol++;
                    nrow++;
                }
                // nrow++;
                ncol++;
            }
            // std::cout << "line:\n" << line << std::endl;
            index++;
            std::cout << "\rprogress: " << std::setw(6) << std::setprecision(4) 
                << float(index) / float((sizeY + 1) * sizeTheta) * 100.f << "%";
            pBuf = startBuf;
        }
        std::cout << '\n';
        // 撤销文件视图
        UnmapViewOfFile(pBuf);
        // 关闭映射文件句柄
        CloseHandle(hFile);
        CloseHandle(hFileMap);
        // std::ifstream fin;
        // try
        // {
        //     fin.open(filename);
        //     if (!fin.is_open())
        //     {
        //         throw std::runtime_error("FILE OPEN ERROR: Please check the filename!");
        //     }
        //     std::cout << "Open OK!" << '\n';
        // }
        // catch (const std::exception &e)
        // {
        //     std::cerr << "Error reading file: " << filename
        //             << "\n"
        //             << e.what() << "\n";
        //     exit(1);
        // }
        // float *mat;
        // int nrow = 0, ncol = 0;
        // auto sizeTheta = 0;
        // auto sizeX = 0;
        // auto sizeY = 0;
        // auto matrixNum = 0;
        // while (fin.is_open() && !fin.eof())
        // {
        //     char buffer[4096];
        //     fin.getline(buffer, sizeof(buffer));
        //     string line(buffer);
        //     // std::cout << "splited size: " << splited.size() << "\n";
        //     if (line.data()[0] == '#')
        //     {
        //         vector<string> splited;
        //         split(line, ' ', splited);
        //         // std::cout << "splited size: " << splited.size() << "\n";
        //         // assert(str2int(splited[2], sizeTheta));
        //         // assert(str2int(splited[3], sizeX));
        //         // assert(str2int(splited[4], sizeY));
        //         str2int(splited[2], sizeTheta);
        //         str2int(splited[3], sizeX);
        //         str2int(splited[4], sizeY);
        //         mat = (float *)malloc(sizeX * sizeY * sizeof(float));
        //         Matrices.reserve(sizeTheta);
        //     }
        //     else if (line.size() == 0)
        //     {
        //         Matrices.push_back(mat);
        //         mat = (float *)malloc(sizeX * sizeY * sizeof(float));
        //         nrow = 0;
        //         ncol = 0;
        //     }
        //     else
        //     {
        //         vector<string> splited;
        //         split(line, '\t', splited);
        //         nrow = 0;
        //         for (auto &&str : splited)
        //         {
        //             float num = 0.f;
        //             // assert(str2float(str, num));
        //             str2float(str, num);
        //             mat[nrow * sizeX + ncol] = num;
        //             nrow++;
        //         }
        //         ncol++;
        //     }
        // }
        // fin.close();
        std::cout << "read finished!" << std::endl;
    }

    void readFile(const string &filename, vector<float *> &Matrices)
    {
        readFile(filename.data(), Matrices);
    }

    // void readFileUseMapping(const string &filename, vector<float *> &Matrices)
    // {
    //     HANDLE hFile = CreateFile(filename.data(),
    //         GENERIC_READ|GENERIC_WRITE,
    //         FILE_SHARED_READ,
    //         nullptr,
    //         OPEN_EXISTING,
    //         FILE_ATTRIBUTE_NORMAL,
    //         nullptr);

    //     if (!hFile)
    //     {
    //         std::cout << "File Handle creation failed!" << std::endl;
    //         return;
    //     }

    //     HANDLE hMapFile = nullptr;

    // }
}
