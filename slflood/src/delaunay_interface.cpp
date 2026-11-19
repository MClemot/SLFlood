#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "delaunay.h"

namespace nb = nanobind;

using PointsArray = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// Moves a std::vector to the heap and exposes it as a numpy array that owns it
template <typename T>
nb::ndarray<nb::numpy, T> toNumpy(std::vector<T> &&vec, std::initializer_list<size_t> shape) {
  auto *data = new std::vector<T>(std::move(vec));
  nb::capsule owner(data, [](void *p) noexcept { delete static_cast<std::vector<T> *>(p); });
  return nb::ndarray<nb::numpy, T>(data->data(), shape, owner);
}

auto delaunayCells(const PointsArray &points) {
  const size_t n = points.shape(0);
  const size_t dim = points.shape(1);
  std::vector<double> points_(points.data(), points.data() + n * dim);

  std::vector<int> cells;
  std::vector<double> centers;
  std::vector<double> radii;

  std::vector<std::vector<int>> faces_by_dim;
  std::vector<std::vector<std::vector<int>>> ids_by_dim;

  {
    nb::gil_scoped_release release;
    switch (dim) {
    case 2:
      delaunayCells2(std::move(points_), cells, centers, radii);
      break;
    case 3:
      delaunayCells3(std::move(points_), cells, centers, radii);
      break;
    default:
      if (dim < 2 || dim > MEB_MAX_DIM)
        throw nb::value_error("delaunay_cells: only point clouds of dimension up to 10 are supported");
      delaunayCellsD(std::move(points_), dim, cells, centers, radii);
    }

    faces(cells, dim, faces_by_dim, ids_by_dim);
  }

  std::vector<nb::ndarray<nb::numpy, int>> faces_arrays;
  for (unsigned d=0; d<faces_by_dim.size(); ++d) {
    const unsigned n_faces = faces_by_dim[d].size()/(d+2);
    faces_arrays.push_back(toNumpy(std::move(faces_by_dim[d]), {n_faces, d+2}));
  }    

  const unsigned n_cells = radii.size();
  return std::make_tuple(toNumpy(std::move(cells), {n_cells, dim + 1}),
                         toNumpy(std::move(radii), {n_cells}),
                         toNumpy(std::move(centers), {n_cells, dim}),
                         std::move(faces_arrays),
                         std::move(ids_by_dim));
}

NB_MODULE(slflood_impl, m) {
  m.def("delaunay_cells", &delaunayCells, nb::arg("points"));
}
