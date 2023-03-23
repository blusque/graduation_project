#pragma once

#include <vector>
#include <string>

namespace cuda 
{
    void readFile(const char *filename, std::vector<float *> &Matrices);

    void readFile(const std::string &filename, std::vector<float *> &Matrices);
}
