#pragma once

#include <string>
#include <tuple>
#include <vector>

std::tuple<std::vector<int>, std::vector<int>, std::vector<float>> read_file(const std::string& filename);