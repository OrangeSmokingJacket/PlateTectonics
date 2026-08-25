#pragma once

#include <Math/math_def.hpp>

template <typename T, size_t X, size_t Y, size_t Z>
struct Array3D : public std::array<std::array<std::array<T, Z>, Y>, X> {
    static constexpr size_t size_X = X;
    static constexpr size_t size_Y = Y;
    static constexpr size_t size_Z = Z;
    static constexpr size_t cardinality = X * Y * Z;
    static constexpr size_t memory_footprint = cardinality * sizeof(T);
    using value_type = T;
    using flat_iterator = T*;
    using flat_const_iterator = const T*;

    flat_iterator flat_begin() { return std::array<std::array<std::array<T, Z>, Y>, X>::data()->data()->data(); }
    flat_iterator flat_end() { return flat_begin() + cardinality; }
    flat_const_iterator flat_cbegin() const {
        return std::array<std::array<std::array<T, Z>, Y>, X>::data()->data()->data();
    }
    flat_const_iterator flat_cend() const { return flat_cbegin() + cardinality; }
    T* flat_data() { return reinterpret_cast<T*>(this); }
    const T* flat_data() const { return reinterpret_cast<const T*>(this); }

    void SetToZero();
    void SetTo(T value);
    void CopyFrom(const Array3D& other);
    T DotProduct(const Array3D& other) const;
    T Length() const;
};

enum class InterpolationType : uint8_t { Trilinear = 0, Cubic = 1 };

template <typename T>
struct CoefficientsValues {
    T diag;
    T plusx;
    T plusy;
    T plusz;

    bool operator==(const CoefficientsValues& value) const = default;
};

using Index3D = math::vec3<size_t>;
using Index4D = math::vec4<size_t>;
using Dimensions = Index3D;

constexpr size_t Cardinality(const Index3D& index) { return index[0] * index[1] * index[2]; }
constexpr size_t Cardinality(const Index4D& index) { return index[0] * index[1] * index[2] * index[3]; }

struct MinMax {
    double min;
    double max;
};

class EulerFluidSim;

// TODO: unique_ptr dereference cost is quite noticeable now
struct SimulationDomain {
    // TODO: dimensions up to 64 or higher to minimize halo cells memory usage
    // [] is used because of constexpr and that array is initialized by the constructor
    static constexpr Dimensions real_dimensions{8, 8, 8};

    static_assert(real_dimensions[0] % 8 == 0 &&
                  "Real Domain Dimensions should be multiples of 8 for easy GPU dispatch");
    static_assert(real_dimensions[1] % 8 == 0 &&
                  "Real Domain Dimensions should be multiples of 8 for easy GPU dispatch");
    static_assert(real_dimensions[2] % 8 == 0 &&
                  "Real Domain Dimensions should be multiples of 8 for easy GPU dispatch");

    static constexpr Dimensions halo_size{1, 1, 0};
    static constexpr size_t border_size = 1;
    static constexpr Dimensions dimensions = real_dimensions + 2ull * halo_size;
    static constexpr Dimensions velocitiesXDimensions{dimensions[0] + 1, dimensions[1], dimensions[2]};
    static constexpr Dimensions velocitiesYDimensions{dimensions[0], dimensions[1] + 1, dimensions[2]};
    static constexpr Dimensions velocitiesZDimensions{dimensions[0], dimensions[1], dimensions[2] + 1};

    using GridValues = Array3D<double, dimensions[0], dimensions[1], dimensions[2]>;
    using GridCoefficientsValues = Array3D<CoefficientsValues<double>, dimensions[0], dimensions[1], dimensions[2]>;
    using GridXValues = Array3D<double, velocitiesXDimensions[0], velocitiesXDimensions[1], velocitiesXDimensions[2]>;
    using GridYValues = Array3D<double, velocitiesYDimensions[0], velocitiesYDimensions[1], velocitiesYDimensions[2]>;
    using GridZValues = Array3D<double, velocitiesZDimensions[0], velocitiesZDimensions[1], velocitiesZDimensions[2]>;

    explicit SimulationDomain(size_t face_index);
    void CalculateCoefficients();
    void CalculatePreconditioner();
    void ApplyBuoyancy(const EulerFluidSim* simulation);
    void AdvectVelocities();
    void AdvectTemperature();

    [[nodiscard]] math::vec3<float> SampleVelocity(const math::vec3<float>& nPos) const;

    [[nodiscard]] math::vec3<float> NormalizePosition(const math::vec3<float>& pos) const;    // TODO: make constexpr
    [[nodiscard]] math::vec3<float> DenormalizePosition(const math::vec3<float>& nPos) const; // TODO: make constexpr

    std::unique_ptr<GridValues> cell_pressure_;
    std::unique_ptr<GridValues> cell_temperature_, cell_temperature_temp_;
    std::unique_ptr<GridXValues> velocities_x_, velocities_x_temp_;
    std::unique_ptr<GridYValues> velocities_y_, velocities_y_temp_;
    std::unique_ptr<GridZValues> velocities_z_, velocities_z_temp_;
    std::unique_ptr<GridCoefficientsValues> coefficients_;
    std::unique_ptr<GridCoefficientsValues> preconditioner_;
    std::unique_ptr<GridValues> cell_pressure_divergence_;
    std::unique_ptr<GridValues> residual_;
    std::unique_ptr<GridValues> auxiliary_;
    std::unique_ptr<GridValues> search_;

    size_t face_index_;
};

static const math::vec3<float> ValueGridOffset{0.5, 0.5, 0.5};
static const math::vec3<float> VelocityXGridOffset{0.0, 0.5, 0.5};
static const math::vec3<float> VelocityYGridOffset{0.5, 0.0, 0.5};
static const math::vec3<float> VelocityZGridOffset{0.5, 0.5, 0.0};

template <typename T, size_t X, size_t Y, size_t Z>
void Array3D<T, X, Y, Z>::SetToZero() {
    std::memset(this->flat_data(), 0, cardinality * sizeof(T));
}

template <typename T, size_t X, size_t Y, size_t Z>
void Array3D<T, X, Y, Z>::SetTo(T value) {
    for (size_t i = 0; i < cardinality; i++) {
        *(this->flat_data() + i) = value;
    }
}

template <typename T, size_t X, size_t Y, size_t Z>
void Array3D<T, X, Y, Z>::CopyFrom(const Array3D& other) {
    std::memcpy(this->flat_data(), other.flat_data(), cardinality * sizeof(T));
}

template <typename T, size_t X, size_t Y, size_t Z>
T Array3D<T, X, Y, Z>::DotProduct(const Array3D& other) const {
    T dotproduct{};
    for (size_t i = 0; i < cardinality; i++) {
        dotproduct += *(this->flat_data() + i) * *(other.flat_data() + i);
    }
    return dotproduct;
}

template <typename T, size_t X, size_t Y, size_t Z>
T Array3D<T, X, Y, Z>::Length() const {
    return std::sqrt(this->DotProduct(*this));
}

template <typename Grid>
Grid::value_type TrilinearInterpolation(const Grid* grid, const math::vec3<float>& normalizedPos) {
    int indexX = floor(normalizedPos.x);
    int indexY = floor(normalizedPos.y);
    int indexZ = floor(normalizedPos.z);

    double fractionX = normalizedPos.x - indexX;
    double fractionY = normalizedPos.y - indexY;
    double fractionZ = normalizedPos.z - indexZ;

    // Actual interpolation
    typename Grid::value_type c00 =
        math::lerp((*grid)[indexX][indexY][indexZ], (*grid)[indexX + 1][indexY][indexZ], fractionX);
    typename Grid::value_type c01 =
        math::lerp((*grid)[indexX][indexY][indexZ + 1], (*grid)[indexX + 1][indexY][indexZ + 1], fractionX);
    typename Grid::value_type c10 =
        math::lerp((*grid)[indexX][indexY + 1][indexZ], (*grid)[indexX + 1][indexY + 1][indexZ], fractionX);
    typename Grid::value_type c11 =
        math::lerp((*grid)[indexX][indexY + 1][indexZ + 1], (*grid)[indexX + 1][indexY + 1][indexZ + 1], fractionX);

    typename Grid::value_type c0 = math::lerp(c00, c10, fractionY);
    typename Grid::value_type c1 = math::lerp(c01, c11, fractionY);

    return math::lerp(c0, c1, fractionZ);
}

template <typename Value>
Value CubicInterpolation(const Value& q0, const Value& q1, const Value& q2, const Value& q3, double x) {
    // TODO: option to disable safety measures (less reliable in advection, but more accurate)
    Value d0 = (q2 - q0) * 0.5;
    Value d1 = (q3 - q1) * 0.5;

    Value delta_q = q2 - q1;
    // safety measure to guarantee interpolation stability
    if (math::is_equal(delta_q, 0.0) || std::signbit(delta_q) != std::signbit(d0) ||
        std::signbit(delta_q) != std::signbit(d1)) {
        d0 = 0;
        d1 = 0;
    }

    Value result =
        q1 + d0 * x + (3 * delta_q - 2 * d0 - d1) * math::square(x) + (-2 * delta_q + d0 + d1) * math::cube(x);
    return result;
}

template <typename Grid>
Grid::value_type CubicInterpolation(const Grid* grid, const math::vec3<float>& normalizedPos) {
    // 2 cells border will make this check redundant
    auto gridValue = [grid](int x, int y, int z) -> typename Grid::value_type {
        if (x < 0 || y < 0 || z < 0 || x >= static_cast<int>(Grid::size_X) || y >= static_cast<int>(Grid::size_Y) ||
            z >= static_cast<int>(Grid::size_Z))
            return {};
        return (*grid)[x][y][z];
    };

    int indexX = floor(normalizedPos.x);
    int indexY = floor(normalizedPos.y);
    int indexZ = floor(normalizedPos.z);

    double fractionX = normalizedPos.x - indexX;
    double fractionY = normalizedPos.y - indexY;
    double fractionZ = normalizedPos.z - indexZ;

    typename Grid::value_type xValues[4][4];
    typename Grid::value_type yValues[4];

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            const int _y = static_cast<int>(i) - 1;
            const int _z = static_cast<int>(j) - 1;
            xValues[i][j] = CubicInterpolation(gridValue(indexX - 1, indexY + _y, indexZ + _z),
                                               gridValue(indexX, indexY + _y, indexZ + _z),
                                               gridValue(indexX + 1, indexY + _y, indexZ + _z),
                                               gridValue(indexX + 2, indexY + _y, indexZ + _z),
                                               fractionX);
        }
    }

    for (size_t i = 0; i < 4; i++) {
        yValues[i] = CubicInterpolation(xValues[0][i], xValues[1][i], xValues[2][i], xValues[3][i], fractionY);
    }

    return CubicInterpolation(yValues[0], yValues[1], yValues[2], yValues[3], fractionZ);
}

// TODO: make switching constexpr
template <typename Grid>
Grid::value_type Interpolation(const Grid* grid, const math::vec3<float>& normalizedPos, InterpolationType type) {
    switch (type) {
        case InterpolationType::Cubic:
            return CubicInterpolation(grid, normalizedPos);
        case InterpolationType::Trilinear:
        default:
            return TrilinearInterpolation(grid, normalizedPos);
    }
}
