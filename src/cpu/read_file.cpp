#include "cpu/read_file.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#define NDEBUG
#include <assert.h>
#include <chrono>

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

void readFile(const char *filename, vector<Matrix> &Matrices)
{
    std::ifstream fin;
    try
    {
        fin.open(filename);
        if (!fin.is_open())
        {
            throw std::runtime_error("FILE OPEN ERROR: Please check the filename!");
        }
        std::cout << "Open OK!" << '\n';
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading file: " << filename
                  << "\n"
                  << e.what() << "\n";
        exit(1);
    }
    Matrix mat(Matrix::Zero(1000, 1000));
    int nrow = 0, ncol = 0;
    auto sizeTheta = 0;
    auto sizeX = 0;
    auto sizeY = 0;
    auto matrixNum = 0;
    while (fin.is_open() && !fin.eof())
    {
        char buffer[4096];
        fin.getline(buffer, sizeof(buffer));
        string line(buffer);
        // std::cout << "splited size: " << splited.size() << "\n";
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
            Matrices.reserve(sizeTheta);
            mat.resize(sizeX, sizeY);
        }
        else if (line.size() == 0)
        {
            Matrices.push_back(mat);
            mat = Matrix(Matrix::Zero(sizeX, sizeY));
            nrow = 0;
            ncol = 0;
        }
        else
        {
            vector<string> splited;
            split(line, '\t', splited);
            ncol = 0;
            for (auto &&str : splited)
            {
                float num = 0.f;
                // assert(str2float(str, num));
                str2float(str, num);
                mat(nrow, ncol) = num;
                ncol++;
            }
            nrow++;
        }
    }
    fin.close();
    std::cout << "read finished!" << std::endl;
}

void readFile(const string &filename, vector<Matrix> &Matrices)
{
    readFile(filename.data(), Matrices);
}