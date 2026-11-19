#pragma once

#include <vector>

constexpr int MEB_MAX_DIM = 10;

void delaunayCells2(std::vector<double> points, std::vector<int> &cells, std::vector<double> &centers, std::vector<double> &radii);
void delaunayCells3(std::vector<double> points, std::vector<int> &cells, std::vector<double> &centers, std::vector<double> &radii);
void delaunayCellsD(std::vector<double> points, int dim, std::vector<int> &cells, std::vector<double> &centers, std::vector<double> &radii);

void faces(std::vector<int> const &cells, int dim, std::vector<std::vector<int>> &faces_by_dim, std::vector<std::vector<std::vector<int>>> &ids_by_dim);