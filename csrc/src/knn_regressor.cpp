#include "knn_regressor.h"

namespace gae_delta {

std::vector<size_t> KNNRegressor::find_knn(
    const float* query,
    const float* all_points,
    size_t n, size_t d, size_t self_idx
) const {
    // Compute distances to all other points
    std::vector<std::pair<float, size_t>> dists;
    dists.reserve(n - 1);

    for (size_t i = 0; i < n; ++i) {
        if (i == self_idx) continue;
        float dist = sq_distance(query, all_points + i * d, d);
        dists.emplace_back(dist, i);
    }

    // Partial sort to get K nearest
    size_t k = std::min(static_cast<size_t>(k_), dists.size());
    std::partial_sort(
        dists.begin(), dists.begin() + k, dists.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; }
    );

    std::vector<size_t> indices(k);
    for (size_t i = 0; i < k; ++i) {
        indices[i] = dists[i].second;
    }
    return indices;
}

std::vector<float> KNNRegressor::compute_residuals(
    const float* shifts, size_t n, size_t d
) const {
    std::vector<float> residuals(n * d, 0.0f);

    for (size_t g = 0; g < n; ++g) {
        const float* query = shifts + g * d;
        auto neighbors = find_knn(query, shifts, n, d, g);

        // Compute mean of neighbor shifts (KNN prediction)
        std::vector<float> predicted(d, 0.0f);
        for (size_t idx : neighbors) {
            for (size_t j = 0; j < d; ++j) {
                predicted[j] += shifts[idx * d + j];
            }
        }

        float inv_k = 1.0f / static_cast<float>(neighbors.size());
        for (size_t j = 0; j < d; ++j) {
            predicted[j] *= inv_k;
        }

        // Residual = actual - predicted
        for (size_t j = 0; j < d; ++j) {
            residuals[g * d + j] = query[j] - predicted[j];
        }
    }

    return residuals;
}

}  // namespace gae_delta
