#pragma once

#include "matrix.hpp"

#include <cereal/archives/binary.hpp>

namespace math::impl {

    template <typename Archive, typename T, size_t Dimensions>
    void save(Archive& archive, const vector<T, Dimensions>& vec) {
        archive(cereal::binary_data(vec.data(), sizeof(vec)));
    }

    template <typename Archive, typename T, size_t Dimensions>
    void load(Archive& archive, vector<T, Dimensions>& vec) {
        archive(cereal::binary_data(vec.data(), sizeof(vec)));
    }

    template <typename Archive, typename T, size_t Columns, size_t Rows>
    void save(Archive& archive, const matrix<T, Columns, Rows>& mat) {
        archive(cereal::binary_data(mat.data(), sizeof(mat)));
    }

    template <typename Archive, typename T, size_t Columns, size_t Rows>
    void load(Archive& archive, matrix<T, Columns, Rows>& mat) {
        archive(cereal::binary_data(mat.data(), sizeof(mat)));
    }

} // namespace math::impl
