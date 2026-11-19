#include "delaunay.h"

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation.h>
#include <CGAL/Epick_d.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Min_sphere_of_points_d_traits_2.h>
#include <CGAL/Min_sphere_of_points_d_traits_3.h>
#include <CGAL/Min_sphere_of_points_d_traits_d.h>
#include <CGAL/Min_sphere_of_spheres_d.h>
#include <CGAL/Triangulation_vertex.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

#include <CGAL/spatial_sort.h>
#include <CGAL/Spatial_sort_traits_adapter_d.h>

#include <boost/unordered/unordered_map.hpp>

#include <array>
#include <numeric>
#include <stdexcept>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
// using K = CGAL::Simple_cartesian<double>;

void delaunayCells2(std::vector<double> points, std::vector<int> &cells, std::vector<double> &centers, std::vector<double> &radii) {
  using Vb = CGAL::Triangulation_vertex_base_with_info_2<int, K>;
  using TDS = CGAL::Triangulation_data_structure_2<Vb>;
  using Delaunay = CGAL::Delaunay_triangulation_2<K, TDS>;
  using Point = Delaunay::Point;
  using MEB = CGAL::Min_sphere_of_spheres_d<CGAL::Min_sphere_of_points_d_traits_2<K,double>>;

  std::vector<std::pair<Point,unsigned>> points_;
  for (unsigned i=0; i<points.size() / 2; ++i)
    points_.emplace_back(Point(points[2*i], points[2*i+1]), i);

  const Delaunay del(points_.begin(), points_.end());
  if (del.dimension() != 2)
    throw std::invalid_argument("points are not in general position (affinely degenerate)");

  for (auto const& c : del.finite_face_handles()) {
    std::array triangle = {c->vertex(0)->info(), c->vertex(1)->info(), c->vertex(2)->info()};
    std::sort(triangle.begin(), triangle.end());
    for (auto const& v : triangle)
      cells.push_back(v);
    std::array local_points = {points_[triangle[0]].first,
                               points_[triangle[1]].first,
                               points_[triangle[2]].first};
    MEB meb(local_points.begin(), local_points.end());
    radii.push_back(meb.radius());
    for (auto it = meb.center_cartesian_begin(); it != meb.center_cartesian_end(); ++it)
      centers.push_back(*it);
  }
}

void delaunayCells3(std::vector<double> points, std::vector<int> &cells, std::vector<double> &centers, std::vector<double> &radii) {
  using Vb = CGAL::Triangulation_vertex_base_with_info_3<int, K>;
  using TDS = CGAL::Triangulation_data_structure_3<Vb>;
  using Delaunay = CGAL::Delaunay_triangulation_3<K, TDS>;
  using Point = Delaunay::Point;
  using MEB = CGAL::Min_sphere_of_spheres_d<CGAL::Min_sphere_of_points_d_traits_3<K,double>>;

  std::vector<std::pair<Point,unsigned>> points_;
  for (unsigned i=0; i<points.size() / 3; ++i)
    points_.emplace_back(Point(points[3*i], points[3*i+1], points[3*i+2]), i);

  const Delaunay del(points_.begin(), points_.end());
  if (del.dimension() != 3)
    throw std::invalid_argument("points are not in general position (affinely degenerate)");

  for (auto const& c : del.finite_cell_handles()) {
    std::array tetrahedron = {c->vertex(0)->info(), c->vertex(1)->info(), c->vertex(2)->info(), c->vertex(3)->info()};
    std::sort(tetrahedron.begin(), tetrahedron.end());
    for (auto const& v : tetrahedron)
      cells.push_back(v);
    std::array local_points = {points_[tetrahedron[0]].first,
                               points_[tetrahedron[1]].first,
                               points_[tetrahedron[2]].first,
                               points_[tetrahedron[3]].first};
    MEB meb(local_points.begin(), local_points.end());
    radii.push_back(meb.radius());
    for (auto it = meb.center_cartesian_begin(); it != meb.center_cartesian_end(); ++it)
      centers.push_back(*it);

  }
}

template <int Dim>
struct MebPointTraits {
  static const int D = Dim;
  using FT = double;
  using Sphere = const double *;
  using Cartesian_const_iterator = const double *;
  using Use_square_roots = CGAL::Tag_true;
  using Algorithm = CGAL::Default_algorithm;
  static FT radius(const Sphere &) { return 0; }
  static Cartesian_const_iterator center_cartesian_begin(const Sphere &s) { return s; }
};

template <int Dim>
void mebD(std::vector<double> const &points, std::vector<int> const &cells,
          std::vector<double> &centers, std::vector<double> &radii) {
  using MEB = CGAL::Min_sphere_of_spheres_d<MebPointTraits<Dim>>;
  const size_t nv = Dim + 1;
  const size_t n_cells = cells.size() / nv;
  radii.reserve(radii.size() + n_cells);
  centers.reserve(centers.size() + n_cells * Dim);
  std::array<const double *, Dim + 1> spheres;
  for (size_t c = 0; c < n_cells; ++c) {
    for (size_t j = 0; j < nv; ++j)
      spheres[j] = points.data() + static_cast<size_t>(Dim) * cells[c * nv + j];
    MEB meb(spheres.begin(), spheres.end());
    radii.push_back(meb.radius());
    centers.insert(centers.end(), meb.center_cartesian_begin(), meb.center_cartesian_end());
  }
}

void delaunayCellsD(std::vector<double> points, const int dim, std::vector<int> &cells, std::vector<double> &centers, std::vector<double> &radii) {
  using DimTag = CGAL::Dynamic_dimension_tag;
  using KD = CGAL::Epick_d<DimTag>;
  using Vb = CGAL::Triangulation_vertex<KD, int>;
  using TDS = CGAL::Triangulation_data_structure<DimTag, Vb>;
  using Delaunay = CGAL::Delaunay_triangulation<KD, TDS>;
  using Point = Delaunay::Point;

  if (dim < 2 || dim > MEB_MAX_DIM)
    throw std::invalid_argument("delaunayCellsD: unsupported dimension");

  const size_t n = points.size() / dim;
  std::vector<Point> points_;
  points_.reserve(n);
  for (size_t i = 0; i < n; ++i)
    points_.emplace_back(dim, points.begin() + i * dim, points.begin() + (i + 1) * dim);

  // insert in spatial order (as the range insert does) but one at a time to record the input index
  std::vector<std::ptrdiff_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  CGAL::spatial_sort(order.begin(), order.end(),
                     CGAL::Spatial_sort_traits_adapter_d<KD, Point *>(points_.data()));

  Delaunay del(dim);
  Delaunay::Full_cell_handle hint;
  for (std::ptrdiff_t i : order) {
    const size_t before = del.number_of_vertices();
    auto vh = del.insert(points_[i], hint);
    if (del.number_of_vertices() > before) // duplicate points keep the index of their first occurrence in insertion order
      vh->data() = static_cast<int>(i);
    hint = vh->full_cell();
  }
  if (del.current_dimension() != dim)
    throw std::invalid_argument("points are not in general position (affinely degenerate)");

  for (auto c_it = del.finite_full_cells_begin(); c_it != del.finite_full_cells_end(); ++c_it) {
    std::array<int, MEB_MAX_DIM + 1> cell;
    for (int d = 0; d < dim + 1; ++d)
      cell[d] = c_it->vertex(d)->data();
    std::sort(cell.begin(), cell.begin() + dim + 1);
    cells.insert(cells.end(), cell.begin(), cell.begin() + dim + 1);
  }

  switch (dim) {
  case 4: mebD<4>(points, cells, centers, radii); break;
  case 5: mebD<5>(points, cells, centers, radii); break;
  case 6: mebD<6>(points, cells, centers, radii); break;
  case 7: mebD<7>(points, cells, centers, radii); break;
  case 8: mebD<8>(points, cells, centers, radii); break;
  case 9: mebD<9>(points, cells, centers, radii); break;
  case 10: mebD<10>(points, cells, centers, radii); break;
  }
}

using Simplex = std::vector<int>;

void faces(std::vector<int> const &cells,
           int dim,
           std::vector<std::vector<int>> &faces_by_dim,
           std::vector<std::vector<std::vector<int>>> &ids_by_dim) {
  dim = dim+1;
  const int n = cells.size() / dim;
  
  std::vector<boost::unordered_map<Simplex,std::vector<int>>> maps_by_dim (dim-2);

  for (unsigned i=0; i<n; ++i) {
    Simplex cell (cells.begin()+i*dim, cells.begin()+(i+1)*dim);
    for (unsigned k=1; k<(1u<<dim)-1; ++k) {
      Simplex subset;
      for (unsigned j=0; j<dim; ++j)
        if (k & (1u<<j))
          subset.push_back(cell[j]);
      if (subset.size() >= 2) {
        maps_by_dim[subset.size()-2][subset].push_back(i);
      }
    }
  }

  faces_by_dim.assign(maps_by_dim.size(), {});
  ids_by_dim.assign(maps_by_dim.size(), {});

  for (int d = 0; d<maps_by_dim.size(); ++d) {
    faces_by_dim[d].reserve(maps_by_dim[d].size() * (d+2));
    ids_by_dim[d].reserve(maps_by_dim[d].size());
    for (auto& [simplex, val] : maps_by_dim[d]) {
      faces_by_dim[d].insert(faces_by_dim[d].end(), simplex.begin(), simplex.end());
      ids_by_dim[d].push_back(std::move(val));
    }
  }
}