#pragma once

#include <Eigen/Core>
#include <vector>
#include <string>

using Matrix = Eigen::MatrixXf;

void readFile(const char *filename, std::vector<Matrix> &Matrices);

void readFile(const std::string &filename, std::vector<Matrix> &Matrices);