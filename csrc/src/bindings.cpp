#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "knn_regressor.h"

namespace py = pybind11;

py::array_t<float> knn_smooth_residuals(
    py::array_t<float, py::array::c_style | py::array::forcecast> shifts,
    int k
) {
    auto buf = shifts.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("shifts must be a 2D array (n_genes, d)");
    }

    size_t n = buf.shape[0];
    size_t d = buf.shape[1];
    const float* data_ptr = static_cast<const float*>(buf.ptr);

    gae_delta::KNNRegressor regressor(k);
    std::vector<float> residuals = regressor.compute_residuals(data_ptr, n, d);

    // Create output numpy array
    auto result = py::array_t<float>({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(d)});
    auto result_buf = result.request();
    float* result_ptr = static_cast<float*>(result_buf.ptr);
    std::copy(residuals.begin(), residuals.end(), result_ptr);

    return result;
}

PYBIND11_MODULE(_knn_ext, m) {
    m.doc() = "KNN residual correction for GAE-Δ embedding shifts";

    m.def("knn_smooth_residuals", &knn_smooth_residuals,
          py::arg("shifts"), py::arg("k") = 5,
          R"doc(
          Compute KNN-smoothed residuals for embedding shift vectors.

          For each gene, finds K nearest neighbors in shift space,
          computes the mean neighbor shift (prediction), and returns
          the residual (actual - predicted).

          Parameters
          ----------
          shifts : ndarray of shape (n_genes, d), float32
          k : number of nearest neighbors

          Returns
          -------
          residuals : ndarray of shape (n_genes, d), float32
          )doc");
}
