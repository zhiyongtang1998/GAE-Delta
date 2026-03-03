#pragma once

#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace gae_delta {

/**
 * KNN smoother for embedding shift vectors.
 *
 * Given a matrix of shift vectors (n_genes × d), for each gene g:
 *   1. Find K nearest neighbors in the shift space
 *   2. Compute the mean shift of those neighbors → predicted shift
 *   3. Return residual = actual_shift - predicted_shift
 *
 * Uses brute-force O(n²) search suitable for ~10-15K genes.
 */
class KNNRegressor {
public:
    KNNRegressor(int k = 15) : k_(k) {}

    /**
     * Smooth shift vectors via KNN regression and return residuals.
     *
     * @param shifts    Flat array of shape (n, d) in row-major order
     * @param n         Number of genes
     * @param d         Embedding dimension
     * @return          Residual shifts as flat vector of size n*d
     */
    std::vector<float> compute_residuals(
        const float* shifts, size_t n, size_t d
    ) const;

private:
    int k_;

    /** Compute squared Euclidean distance between two d-dimensional vectors. */
    static float sq_distance(
        const float* a, const float* b, size_t d
    ) {
        float dist = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return dist;
    }

    /** Find indices of K nearest neighbors (excluding self). */
    std::vector<size_t> find_knn(
        const float* query,
        const float* all_points,
        size_t n, size_t d, size_t self_idx
    ) const;
};

}  // namespace gae_delta
