#include "euler_fluid_sim.hpp"

#include <Core/Renderer/renderer.hpp>
#include <Math/scalar_ops.hpp>
#include <Utils/compiler_utils.hpp>

#include <algorithm>
#include <future>
#include <latch>

template <Index4D GridDimensions>
constexpr size_t FromIndex4D(size_t x, size_t y, size_t z, size_t w) {
    // [] used because of constexpr and that vec4 is initialized as array
    return w * (GridDimensions[0] * GridDimensions[1] * GridDimensions[2]) +
           x * (GridDimensions[1] * GridDimensions[2]) + y * GridDimensions[2] + z;
}

template <Index4D GridDimensions>
constexpr size_t FromIndex4D(const Index4D& index_4d) {
    // [] used because of constexpr and that vec4 is initialized as array
    return index_4d[3] * (GridDimensions[0] * GridDimensions[1] * GridDimensions[2]) +
           index_4d[0] * (GridDimensions[1] * GridDimensions[2]) + index_4d[1] * GridDimensions[2] + index_4d[2];
}

template <Index4D GridDimensions>
constexpr Index4D ToIndex4D(size_t index) {
    // wxyz order
    return Index4D{(index / (GridDimensions[1] * GridDimensions[2]) % GridDimensions[0]),
                   (index / GridDimensions[2]) % GridDimensions[1],
                   index % GridDimensions[2],
                   index / (GridDimensions[0] * GridDimensions[1] * GridDimensions[2])};
}

// wxyz; z is packed closest
template <size_t... Indices>
std::array<SimulationDomain, sizeof...(Indices)> InitializeDomainArray(std::index_sequence<Indices...>) {
    return {SimulationDomain(ToIndex4D<EulerFluidSim::domains>(Indices))...};
}

struct CoordinateSystem {
    size_t x;
    size_t y;
    size_t z;
    bool inverse;
};

constexpr math::vec3<float> ConvertToFaceCS(const CoordinateSystem& cs, const math::vec3<float>& vec) {
    // it is a reverse transformation
    math::vec3<float> result{vec[2 - cs.z], vec[2 - cs.y], vec[2 - cs.x]};
    return cs.inverse ? -result : result;
}

// Transforms coordinate system for each face
//! Warning: 3-5 faces are in reverse handedness
constexpr std::array<CoordinateSystem, 6> cube_face_cs{CoordinateSystem{.x = 0, .y = 1, .z = 2, .inverse = false},
                                                       CoordinateSystem{.x = 1, .y = 2, .z = 0, .inverse = false},
                                                       CoordinateSystem{.x = 2, .y = 0, .z = 1, .inverse = false},
                                                       CoordinateSystem{.x = 0, .y = 1, .z = 2, .inverse = true},
                                                       CoordinateSystem{.x = 1, .y = 2, .z = 0, .inverse = true},
                                                       CoordinateSystem{.x = 2, .y = 0, .z = 1, .inverse = true}};

// For W neighbours it is cheaper to look it up, than compute
struct WNeighbour {
    size_t x_plus;
    size_t x_minus;
    size_t y_plus;
    size_t y_minus;
};
// Get neighbouring face in specific direction
constexpr std::array<WNeighbour, 6> w_neighbours{WNeighbour{1, 4, 2, 5},
                                                 WNeighbour{2, 5, 0, 3},
                                                 WNeighbour{0, 3, 1, 4},
                                                 WNeighbour{4, 1, 5, 2},
                                                 WNeighbour{5, 2, 3, 0},
                                                 WNeighbour{3, 0, 4, 1}};

// TODO: separate array for orthogonal directions
static constexpr std::array directions{
    math::vec3<int>{-1, -1, -1},
    math::vec3<int>{-1, -1, 0},
    math::vec3<int>{-1, -1, 1},
    math::vec3<int>{-1, 0, -1},
    math::vec3<int>{-1, 0, 0},
    math::vec3<int>{-1, 0, 1},
    math::vec3<int>{-1, 1, -1},
    math::vec3<int>{-1, 1, 0},
    math::vec3<int>{-1, 1, 1},
    math::vec3<int>{0, -1, -1},
    math::vec3<int>{0, -1, 0},
    math::vec3<int>{0, -1, 1},
    math::vec3<int>{0, 0, -1},
    //math::vec3<int>{0, 0, 0}, skip itself
    math::vec3<int>{0, 0, 1},
    math::vec3<int>{0, 1, -1},
    math::vec3<int>{0, 1, 0},
    math::vec3<int>{0, 1, 1},
    math::vec3<int>{1, -1, -1},
    math::vec3<int>{1, -1, 0},
    math::vec3<int>{1, -1, 1},
    math::vec3<int>{1, 0, -1},
    math::vec3<int>{1, 0, 0},
    math::vec3<int>{1, 0, 1},
    math::vec3<int>{1, 1, -1},
    math::vec3<int>{1, 1, 0},
    math::vec3<int>{1, 1, 1},
};

// Is not influenced by .w
template <math::vec3<int> Stagger>
static constexpr std::pair<size_t, size_t> DestinationRange(int dir, size_t axis_index) {
    // if stagger is negative, apply size increase in different direction
    if (Stagger[axis_index] >= 0) {
        switch (dir) {
            case -1:
                return {0, SimulationDomain::halo_size[axis_index]};
            case 0:
                return {SimulationDomain::halo_size[axis_index],
                        SimulationDomain::real_dimensions[axis_index] + Stagger[axis_index]};
            case 1:
                return {SimulationDomain::dimensions[axis_index] - SimulationDomain::halo_size[axis_index],
                        SimulationDomain::halo_size[axis_index] + Stagger[axis_index]};
            default:
                utils::unreachable();
        }
    } else {
        switch (dir) {
            case -1:
                return {0, SimulationDomain::halo_size[axis_index] - Stagger[axis_index]};
            case 0:
                return {SimulationDomain::halo_size[axis_index],
                        SimulationDomain::real_dimensions[axis_index] - Stagger[axis_index]};
            case 1:
                return {SimulationDomain::dimensions[axis_index], SimulationDomain::halo_size[axis_index]};
            default:
                utils::unreachable();
        }
    }
}

namespace SameW {

    template <math::vec3<int> Stagger = math::vec3<int>{0, 0, 0}>
    static constexpr size_t SourceAdjustment(int dir, size_t axis_index) {
        switch (dir) {
            case -1:
                return SimulationDomain::real_dimensions[axis_index];
            case 0:
                return SimulationDomain::halo_size[axis_index];
            case 1:
                if constexpr (math::component_sum(Stagger) >= 0) {
                    return SimulationDomain::halo_size[axis_index];
                } else {
                    return SimulationDomain::halo_size[axis_index] - Stagger[axis_index];
                }
            default:
                utils::unreachable();
        }
    }
} // namespace SameW

namespace DifferentW {

    enum class ScanDir { Forward, Backward };

    struct SourceAdj {
        size_t x_start;
        size_t y_start;
        ScanDir x_scan_dir;
        ScanDir y_scan_dir;
    };

    static constexpr SourceAdj SourceAdjustmentX(int dir) {
        if (dir == -1) {
            return {.x_start = SimulationDomain::dimensions[0] - SimulationDomain::halo_size[0] - 1,
                    .y_start = SimulationDomain::halo_size[1] * 2 - 1,
                    .x_scan_dir = ScanDir::Backward,
                    .y_scan_dir = ScanDir::Backward};
        }
        if (dir == 1) {
            return {.x_start = SimulationDomain::halo_size[0],
                    .y_start = SimulationDomain::dimensions[1] - SimulationDomain::halo_size[1] - 1,
                    .x_scan_dir = ScanDir::Forward,
                    .y_scan_dir = ScanDir::Backward};
        }
        utils::unreachable();
    }
    static constexpr SourceAdj SourceAdjustmentY(int dir) {
        if (dir == -1) {
            return {.x_start = SimulationDomain::halo_size[0] * 2 - 1,
                    .y_start = SimulationDomain::dimensions[1] - SimulationDomain::halo_size[1] - 1,
                    .x_scan_dir = ScanDir::Backward,
                    .y_scan_dir = ScanDir::Backward};
        }
        if (dir == 1) {
            return {.x_start = SimulationDomain::dimensions[0] - SimulationDomain::halo_size[0] - 1,
                    .y_start = SimulationDomain::halo_size[1],
                    .x_scan_dir = ScanDir::Backward,
                    .y_scan_dir = ScanDir::Forward};
        }
        utils::unreachable();
    }

    // X or Y have to be outside domains bounds
    template <Index4D GridDimensions>
    constexpr size_t NeighbourDomainIndex(int64_t x, int64_t y, size_t z, size_t w) {
        static_assert(GridDimensions[0] == GridDimensions[1]);

        if (x < 0) {
            return FromIndex4D<GridDimensions>(GridDimensions.x - y - 1, 0, z, w);
        } else if (x == static_cast<int64_t>(GridDimensions.x)) {
            return FromIndex4D<GridDimensions>(y, GridDimensions.y - 1, z, w);
        } else if (y < 0) {
            return FromIndex4D<GridDimensions>(0, GridDimensions.y - x - 1, z, w);
        } else if (y == static_cast<int64_t>(GridDimensions.y)) {
            return FromIndex4D<GridDimensions>(GridDimensions.x - 1, x, z, w);
        }
        utils::unreachable();
    }

} // namespace DifferentW

// Cube-corner halo cells do not have corresponding real cells
// They are filled with averages
// It is not correct, but good enough for now
template <typename Grid, math::vec3<int> Stagger = math::vec3<int>{0, 0, 0}>
void FillCornerHalos(Grid& grid, const Index4D& domain_index) {
    static constexpr std::array<std::pair<int, int>, 4> corner_dirs{{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}}};
    for (auto [dx, dy] : corner_dirs) {
        // Averaging is applied only for cube face corners
        bool in_x = static_cast<int>(domain_index.x) + dx >= 0 &&
                    static_cast<int>(domain_index.x) + dx < static_cast<int>(EulerFluidSim::domains.x);
        bool in_y = static_cast<int>(domain_index.y) + dy >= 0 &&
                    static_cast<int>(domain_index.y) + dy < static_cast<int>(EulerFluidSim::domains.y);
        if (in_x || in_y) {
            continue;
        }

        auto [hx, cx] = DestinationRange<Stagger>(dx, 0);
        auto [hy, cy] = DestinationRange<Stagger>(dy, 1);
        auto [hx0, cx0] = DestinationRange<Stagger>(0, 0);
        auto [hy0, cy0] = DestinationRange<Stagger>(0, 1);

        for (size_t ii = 0; ii < cx; ++ii) {
            size_t i = dx > 0 ? hx + ii : hx + cx - 1 - ii; // inward first
            if (i >= hx0 && i < hx0 + cx0) {
                continue;
            }
            for (size_t jj = 0; jj < cy; ++jj) {
                size_t j = dy > 0 ? hy + jj : hy + cy - 1 - jj;
                if (j >= hy0 && j < hy0 + cy0) {
                    continue;
                }
                for (size_t k = 0; k < Grid::size_Z; ++k) {
                    grid[i][j][k] = 0.5 * (grid[i - dx][j][k] + grid[i][j - dy][k]);
                }
            }
        }
    }
}

// Mixed corner: only one axis crosses the cube-face edge
// and values have to be copied
template <typename Grid, math::vec3<int> Stagger = math::vec3<int>{0, 0, 0}>
void FillMixedCornerHalos(std::array<SimulationDomain, Cardinality(EulerFluidSim::domains)>& domains,
                          std::unique_ptr<Grid> SimulationDomain::* grid,
                          const Index4D& di) {
    static constexpr std::array<std::pair<int, int>, 4> corner_dirs{{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}}};
    for (auto [dx, dy] : corner_dirs) {
        const bool in_x = static_cast<int>(di.x) + dx >= 0 &&
                          static_cast<int>(di.x) + dx < static_cast<int>(EulerFluidSim::domains.x);
        const bool in_y = static_cast<int>(di.y) + dy >= 0 &&
                          static_cast<int>(di.y) + dy < static_cast<int>(EulerFluidSim::domains.y);
        if (in_x == in_y) {
            continue;
        }
        auto& dst = *(domains[FromIndex4D<EulerFluidSim::domains>(di.x, di.y, di.z, di.w)].*grid);
        auto& src =
            *(domains[FromIndex4D<EulerFluidSim::domains>(di.x + (in_x ? dx : 0), di.y + (in_y ? dy : 0), di.z, di.w)].*
              grid);
        auto [hx, cx] = DestinationRange<Stagger>(dx, 0);
        auto [hy, cy] = DestinationRange<Stagger>(dy, 1);
        int64_t shift_x = in_x ? dx * static_cast<int64_t>(SimulationDomain::real_dimensions[0]) : 0;
        int64_t shift_y = in_y ? dy * static_cast<int64_t>(SimulationDomain::real_dimensions[1]) : 0;
        for (size_t i = 0; i < cx; ++i) {
            for (size_t j = 0; j < cy; ++j) {
                for (size_t k = 0; k < Grid::size_Z; ++k) {
                    dst[hx + i][hy + j][k] = src[hx + i - shift_x][hy + j - shift_y][k];
                }
            }
        }
    }
}

// Works for regular grid and MAC grid over Z direction
// TODO: select dir array based on sync scope
template <typename Grid, math::vec3<int> Stagger = math::vec3<int>{0, 0, 0}>
requires(Stagger[0] == 0 && Stagger[1] == 0)
void SyncGridValues(std::array<SimulationDomain, Cardinality(EulerFluidSim::domains)>& domains,
                    std::unique_ptr<Grid> SimulationDomain::* grid,
                    SyncScope scope = SyncScope::Full) {
    for (size_t dw = 0; dw < EulerFluidSim::domains.w; ++dw) {
        for (size_t dx = 0; dx < EulerFluidSim::domains.x; ++dx) {
            for (size_t dy = 0; dy < EulerFluidSim::domains.y; ++dy) {
                for (size_t dz = 0; dz < EulerFluidSim::domains.z; ++dz) {
                    for (const auto& dir : directions) {
                        // Solver grids never read diagonal halos - skip all non-face work
                        if (scope == SyncScope::Orthogonal && (dir.x != 0) + (dir.y != 0) + (dir.z != 0) != 1) {
                            continue;
                        }
                        // Those are border domains, nothing to sync there
                        if (static_cast<int>(dz) + dir.z < static_cast<int>(0) ||
                            static_cast<int>(dz) + dir.z >= static_cast<int>(EulerFluidSim::domains.z)) {
                            continue;
                        }
                        auto domain_index_to = FromIndex4D<EulerFluidSim::domains>(dx, dy, dz, dw);

                        if (static_cast<int>(dx) + dir.x < static_cast<int>(0) ||
                            static_cast<int>(dy) + dir.y < static_cast<int>(0) ||
                            static_cast<int>(dx) + dir.x >= static_cast<int>(EulerFluidSim::domains.x) ||
                            static_cast<int>(dy) + dir.y >= static_cast<int>(EulerFluidSim::domains.y)) {
                            // x/y borders have to be synced with other faces
                            // neighbour domain index will be different
                            // cell's grids are also rotated differently
                            // For velocities X and Y have to be swapped

                            auto [hx, cx] = DestinationRange<Stagger>(dir.x, 0);
                            auto [hy, cy] = DestinationRange<Stagger>(dir.y, 1);
                            auto [hz, cz] = DestinationRange<Stagger>(dir.z, 2);

                            if (dir.x * dir.y != 0) {
                                auto sx = SameW::SourceAdjustment(-dir.x, 0);
                                auto sy = SameW::SourceAdjustment(-dir.y, 1);
                                auto sz = SameW::SourceAdjustment(-dir.z, 2);

                                auto domain_index = FromIndex4D<EulerFluidSim::domains>(dx, dy, dz, dw);
                                for (size_t i = 0; i < cx; ++i) {
                                    for (size_t j = 0; j < cy; ++j) {
                                        for (size_t k = 0; k < cz; ++k) {
                                            (*(domains[domain_index].*grid))[hx + i][hy + j][hz + k] =
                                                (*(domains[domain_index].*grid))[sx + i][sy + j][sz + k];
                                        }
                                    }
                                }
                            } else {
                                auto sz = SameW::SourceAdjustment(dir.z, 2);

                                auto adj = dir.x == 0 ? DifferentW::SourceAdjustmentY(dir.y)
                                                      : DifferentW::SourceAdjustmentX(dir.x);
                                size_t domain_index_from;
                                if (dir.y == 0) {
                                    if (dir.x > 0) {
                                        domain_index_from = DifferentW::NeighbourDomainIndex<EulerFluidSim::domains>(
                                            dx + dir.x,
                                            dy,
                                            dz + dir.z,
                                            w_neighbours[dw].x_plus);
                                    } else {
                                        domain_index_from = DifferentW::NeighbourDomainIndex<EulerFluidSim::domains>(
                                            dx + dir.x,
                                            dy,
                                            dz + dir.z,
                                            w_neighbours[dw].x_minus);
                                    }
                                } else {
                                    if (dir.y > 0) {
                                        domain_index_from = DifferentW::NeighbourDomainIndex<EulerFluidSim::domains>(
                                            dx,
                                            dy + dir.y,
                                            dz + dir.z,
                                            w_neighbours[dw].y_plus);
                                    } else {
                                        domain_index_from = DifferentW::NeighbourDomainIndex<EulerFluidSim::domains>(
                                            dx,
                                            dy + dir.y,
                                            dz + dir.z,
                                            w_neighbours[dw].y_minus);
                                    }
                                }
                                for (int64_t i = 0; i < cx; ++i) {
                                    for (int64_t j = 0; j < cy; ++j) {
                                        for (size_t k = 0; k < cz; ++k) {
                                            int64_t from_i = adj.x_scan_dir == DifferentW::ScanDir::Forward ? j : -j;
                                            int64_t from_j = adj.y_scan_dir == DifferentW::ScanDir::Forward ? i : -i;
                                            (*(domains[domain_index_to].*grid))[hx + i][hy + j][hz + k] =
                                                (*(domains[domain_index_from].*
                                                   grid))[adj.x_start + from_i][adj.y_start + from_j][sz + k];
                                        }
                                    }
                                }
                            }
                        } else {
                            // Within same face logic is straight forward

                            auto [hx, cx] = DestinationRange<Stagger>(dir.x, 0);
                            auto [hy, cy] = DestinationRange<Stagger>(dir.y, 1);
                            auto [hz, cz] = DestinationRange<Stagger>(dir.z, 2);

                            auto sx = SameW::SourceAdjustment(dir.x, 0);
                            auto sy = SameW::SourceAdjustment(dir.y, 1);
                            auto sz = SameW::SourceAdjustment(dir.z, 2);

                            auto domain_index_from =
                                FromIndex4D<EulerFluidSim::domains>(dx + dir.x, dy + dir.y, dz + dir.z, dw);
                            for (size_t i = 0; i < cx; ++i) {
                                for (size_t j = 0; j < cy; ++j) {
                                    for (size_t k = 0; k < cz; ++k) {
                                        (*(domains[domain_index_to].*grid))[hx + i][hy + j][hz + k] =
                                            (*(domains[domain_index_from].*grid))[sx + i][sy + j][sz + k];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Because of the Staggered grid, some values are lying on the split seem
// That is solved by copying extra row of values
// Always copying from the same direction won't work for current cube faces arrangement
// X and Y have to be copied from opposing directions
// Copy is done from X+ and Y- axes
// velocity_z and StaggerZ could be done either way
void SyncGridVelocitiesXY(std::array<SimulationDomain, Cardinality(EulerFluidSim::domains)>& domains) {
    static constexpr math::vec3<int> StaggerX{1, 0, 0};
    static constexpr math::vec3<int> StaggerY{0, -1, 0};

    for (size_t dw = 0; dw < EulerFluidSim::domains.w; ++dw) {
        for (size_t dx = 0; dx < EulerFluidSim::domains.x; ++dx) {
            for (size_t dy = 0; dy < EulerFluidSim::domains.y; ++dy) {
                for (size_t dz = 0; dz < EulerFluidSim::domains.z; ++dz) {
                    for (const auto& dir : directions) {
                        // Those are border domains, nothing to sync there
                        if (static_cast<int>(dz) + dir.z < static_cast<int>(0) ||
                            static_cast<int>(dz) + dir.z >= static_cast<int>(EulerFluidSim::domains.z)) {
                            continue;
                        }
                        auto domain_index_to = FromIndex4D<EulerFluidSim::domains>(dx, dy, dz, dw);

                        if (static_cast<int>(dx) + dir.x < static_cast<int>(0) ||
                            static_cast<int>(dy) + dir.y < static_cast<int>(0) ||
                            static_cast<int>(dx) + dir.x >= static_cast<int>(EulerFluidSim::domains.x) ||
                            static_cast<int>(dy) + dir.y >= static_cast<int>(EulerFluidSim::domains.y)) {
                            // x/y borders have to be synced with other faces
                            // neighbour domain index will be different
                            // cell's grids are also rotated differently
                            // For velocities X and Y have to be swapped

                            auto [x_hx, x_cx] = DestinationRange<StaggerX>(dir.x, 0);
                            auto [x_hy, x_cy] = DestinationRange<StaggerX>(dir.y, 1);
                            auto [x_hz, x_cz] = DestinationRange<StaggerX>(dir.z, 2);
                            auto [y_hx, y_cx] = DestinationRange<StaggerY>(dir.x, 0);
                            auto [y_hy, y_cy] = DestinationRange<StaggerY>(dir.y, 1);
                            auto [y_hz, y_cz] = DestinationRange<StaggerY>(dir.z, 2);

                            if (dir.x * dir.y != 0) {
                                continue;
                            } else {
                                auto sz = SameW::SourceAdjustment(dir.z, 2);

                                auto adj = dir.x == 0 ? DifferentW::SourceAdjustmentY(dir.y)
                                                      : DifferentW::SourceAdjustmentX(dir.x);
                                size_t domain_index_from;
                                if (dir.y == 0) {
                                    if (dir.x > 0) {
                                        domain_index_from = DifferentW::NeighbourDomainIndex<EulerFluidSim::domains>(
                                            dx + dir.x,
                                            dy,
                                            dz + dir.z,
                                            w_neighbours[dw].x_plus);
                                    } else {
                                        domain_index_from = DifferentW::NeighbourDomainIndex<EulerFluidSim::domains>(
                                            dx + dir.x,
                                            dy,
                                            dz + dir.z,
                                            w_neighbours[dw].x_minus);
                                    }
                                } else {
                                    if (dir.y > 0) {
                                        domain_index_from = DifferentW::NeighbourDomainIndex<EulerFluidSim::domains>(
                                            dx,
                                            dy + dir.y,
                                            dz + dir.z,
                                            w_neighbours[dw].y_plus);
                                    } else {
                                        domain_index_from = DifferentW::NeighbourDomainIndex<EulerFluidSim::domains>(
                                            dx,
                                            dy + dir.y,
                                            dz + dir.z,
                                            w_neighbours[dw].y_minus);
                                    }
                                }
                                for (int64_t i = 0; i < static_cast<int64_t>(x_cx); ++i) {
                                    for (int64_t j = 0; j < static_cast<int64_t>(x_cy); ++j) {
                                        for (size_t k = 0; k < static_cast<int64_t>(x_cz); ++k) {
                                            auto mult = adj.y_scan_dir == DifferentW::ScanDir::Forward ? 1 : -1;
                                            int64_t from_i = adj.x_scan_dir == DifferentW::ScanDir::Forward ? j : -j;
                                            int64_t from_j = adj.y_scan_dir == DifferentW::ScanDir::Forward
                                                                 ? i
                                                                 : (-i - dir.y + (dir.x != 0));
                                            (*domains[domain_index_to].velocities_x_)[x_hx + i][x_hy + j][x_hz + k] =
                                                mult * (*domains[domain_index_from]
                                                             .velocities_y_)[adj.x_start + from_i][adj.y_start + from_j]
                                                                            [sz + k];
                                        }
                                    }
                                }
                                for (int64_t i = 0; i < static_cast<int64_t>(y_cx); ++i) {
                                    for (int64_t j = 0; j < static_cast<int64_t>(y_cy); ++j) {
                                        for (size_t k = 0; k < static_cast<int64_t>(y_cz); ++k) {
                                            auto mult = adj.x_scan_dir == DifferentW::ScanDir::Forward ? 1 : -1;
                                            int64_t from_i = adj.x_scan_dir == DifferentW::ScanDir::Forward
                                                                 ? j
                                                                 : (-j - dir.x + (dir.y < 0));
                                            int64_t from_j = adj.y_scan_dir == DifferentW::ScanDir::Forward ? i : -i;
                                            (*domains[domain_index_to].velocities_y_)[y_hx + i][y_hy + j][y_hz + k] =
                                                mult * (*domains[domain_index_from]
                                                             .velocities_x_)[adj.x_start + from_i][adj.y_start + from_j]
                                                                            [sz + k];
                                        }
                                    }
                                }
                            }
                        } else {
                            {
                                auto sx = SameW::SourceAdjustment<StaggerX>(dir.x, 0);
                                auto sy = SameW::SourceAdjustment<StaggerX>(dir.y, 1);
                                auto sz = SameW::SourceAdjustment<StaggerX>(dir.z, 2);
                                auto [hx, cx] = DestinationRange<StaggerX>(dir.x, 0);
                                auto [hy, cy] = DestinationRange<StaggerX>(dir.y, 1);
                                auto [hz, cz] = DestinationRange<StaggerX>(dir.z, 2);

                                auto domain_index_from =
                                    FromIndex4D<EulerFluidSim::domains>(dx + dir.x, dy + dir.y, dz + dir.z, dw);
                                for (size_t i = 0; i < cx; ++i) {
                                    for (size_t j = 0; j < cy; ++j) {
                                        for (size_t k = 0; k < cz; ++k) {
                                            (*domains[domain_index_to].velocities_x_)[hx + i][hy + j][hz + k] =
                                                (*domains[domain_index_from].velocities_x_)[sx + i][sy + j][sz + k];
                                        }
                                    }
                                }
                            }
                            {
                                auto sx = SameW::SourceAdjustment<StaggerY>(dir.x, 0);
                                auto sy = SameW::SourceAdjustment<StaggerY>(dir.y, 1);
                                auto sz = SameW::SourceAdjustment<StaggerY>(dir.z, 2);
                                auto [hx, cx] = DestinationRange<StaggerY>(dir.x, 0);
                                auto [hy, cy] = DestinationRange<StaggerY>(dir.y, 1);
                                auto [hz, cz] = DestinationRange<StaggerY>(dir.z, 2);

                                auto domain_index_from =
                                    FromIndex4D<EulerFluidSim::domains>(dx + dir.x, dy + dir.y, dz + dir.z, dw);
                                for (size_t i = 0; i < cx; ++i) {
                                    for (size_t j = 0; j < cy; ++j) {
                                        for (size_t k = 0; k < cz; ++k) {
                                            (*domains[domain_index_to].velocities_y_)[hx + i][hy + j][hz + k] =
                                                (*domains[domain_index_from].velocities_y_)[sx + i][sy + j][sz + k];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto& domain : domains) {
        FillMixedCornerHalos<SimulationDomain::GridXValues, StaggerX>(domains,
                                                                      &SimulationDomain::velocities_x_,
                                                                      domain.domain_index_);
        FillMixedCornerHalos<SimulationDomain::GridYValues, StaggerY>(domains,
                                                                      &SimulationDomain::velocities_y_,
                                                                      domain.domain_index_);
        FillCornerHalos<SimulationDomain::GridXValues, StaggerX>(*domain.velocities_x_, domain.domain_index_);
        FillCornerHalos<SimulationDomain::GridYValues, StaggerY>(*domain.velocities_y_, domain.domain_index_);
    }
}

// TODO: main thread is idle, while worker threads crunch through domains.
// Coroutines could help with than
// Ideally main thread should also participate in computations
// for 8 worker thread utilization is 8/9 or ~88.8%, which is acceptable for now
EulerFluidSim::EulerFluidSim(Core::RenderPipeline* renderer, utils::DefaultThreadPool* thread_pool)
    : thread_pool_(thread_pool)
    , domains_(std::make_unique<std::array<SimulationDomain, Cardinality(domains)>>(
          InitializeDomainArray(std::make_index_sequence<Cardinality(domains)>()))) {
    for (auto& texture : render_textures_) {
        texture = Core::Engine::Assets::AssetManager::Get()->RegisterAsset(
            std::make_unique<Core::Engine::Assets::Texture3Df>(full_dimensions.x,
                                                               full_dimensions.y,
                                                               full_dimensions.z));
    }

    DebugRandomizeBorderVelocities();
    SetInitialTemperature();

    // TODO: look into coefficients and preconditioner sync
    SyncGridValues(&SimulationDomain::preconditioner_);

    if (drawGrid) {
        DrawGrid(renderer, true);
    }
}

void EulerFluidSim::Update(float delta_time) {
    // TODO: Combine temperature advection and velocity advection to reduce sampling and sync calls
    // Advection
    {
        std::latch sync_latch{Cardinality(domains)};

        for (size_t i = 0; i < Cardinality(domains); ++i) {
            std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
            thread_pool_->Enqueue([domain_ref, &sync_latch]() {
                domain_ref.get().AdvectVelocities();
                sync_latch.count_down();
            });
        }

        sync_latch.wait();
    }
    SyncVelocities();
    {
        std::latch sync_latch{Cardinality(domains)};

        for (size_t i = 0; i < Cardinality(domains); ++i) {
            std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
            thread_pool_->Enqueue([domain_ref, &sync_latch]() {
                domain_ref.get().AdvectTemperature();
                sync_latch.count_down();
            });
        }

        sync_latch.wait();
    }
    SyncGridValues(&SimulationDomain::cell_temperature_);
    // Forces
    {
        std::latch sync_latch{Cardinality(domains)};

        for (size_t i = 0; i < Cardinality(domains); ++i) {
            std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
            thread_pool_->Enqueue([domain_ref, &sync_latch, this]() {
                domain_ref.get().ApplyBuoyancy(this);
                sync_latch.count_down();
            });
        }

        sync_latch.wait();
    }
    SyncVelocities();
    // Projection
    ProjectVelocities();

    SetupRenderTexture();
}

void EulerFluidSim::RenderDebug(Core::RenderPipeline* renderer) const {
    // Draw calls
    if (draw_boundary_velocities) {
        DrawBoundaryVelocities(renderer, false);
    }
    if (draw_interpolated_velocities) {
        DrawInterpolatedVelocities(renderer, false);
    }
}

void EulerFluidSim::SetupRenderTexture() {
    if (draw_slice == DrawSlice::None) {
        return;
    }
    // combining multiple slices results in a visual mess, so only one is rendered
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();

    std::unique_ptr<SimulationDomain::GridValues> SimulationDomain::* grid = nullptr;
    Core::Engine::Assets::Color* min_color = nullptr;
    Core::Engine::Assets::Color* max_color = nullptr;

    switch (draw_slice) {
        case DrawSlice::Temperature:
            grid = &SimulationDomain::cell_temperature_;
            min_color = &min_temperature_color;
            max_color = &max_temperature_color;
            break;
        case DrawSlice::Pressure:
            grid = &SimulationDomain::cell_pressure_;
            min_color = &min_pressure_color;
            max_color = &max_pressure_color;
            break;
        case DrawSlice::PressureDivergence:
            grid = &SimulationDomain::cell_pressure_divergence_;
            min_color = &min_divergence_color;
            max_color = &max_divergence_color;
            break;
        default:
            assert(false && "draw_slice variant is not handled om SetupRenderTexure");
    }

    // find min/max
    for (const auto& domain : *domains_) {
        for (size_t x = 0; x < SimulationDomain::dimensions.x; ++x) {
            for (size_t y = 0; y < SimulationDomain::dimensions.y; ++y) {
                for (size_t z = 0; z < SimulationDomain::dimensions.z; ++z) {
                    min = std::min(min, (*(domain.*grid))[x][y][z]);
                    max = std::max(max, (*(domain.*grid))[x][y][z]);
                }
            }
        }
    }

    for (const auto& domain : *domains_) {
        auto face_index = domain.domain_index_.w;
        for (size_t x = 0; x < SimulationDomain::dimensions.x; ++x) {
            for (size_t y = 0; y < SimulationDomain::dimensions.y; ++y) {
                for (size_t z = 0; z < SimulationDomain::dimensions.z; ++z) {
                    auto world_index = WorldIndex(static_cast<Index3D>(domain.domain_index_), Index3D{x, y, z});
                    float t = ((*(domain.*grid))[x][y][z] - min) / (max - min);
                    // TODO: remove const cast when mutable assets will be implemented
                    const_cast<Core::Engine::Assets::Texture3Df*>(render_textures_[face_index].GetAsset())
                        ->At(world_index.x, world_index.y, world_index.z) = math::lerp(*min_color, *max_color, t);
                }
            }
        }
    }
}

const std::array<Core::Engine::Assets::AssetRef<Core::Engine::Assets::Texture3Df>, 6>&
EulerFluidSim::RenderTextures() const {
    return render_textures_;
}

void EulerFluidSim::DebugRandomizeBorderVelocities() {
    // TODO: Use smooth noise function to get more realistic results
    // Could be optimized, but it is a debug function anyway

    std::latch sync_latch{Cardinality(domains)};

    for (size_t i = 0; i < Cardinality(domains); ++i) {
        std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
        thread_pool_->Enqueue([domain_ref, &sync_latch]() {
            auto& domain = domain_ref.get();
            for (auto it = domain.velocities_x_->flat_begin(); it != domain.velocities_x_->flat_end(); ++it) {
                *it = (math::random<double>() * 2 - 1.0) * SimulationConstants::scale;
            }
            for (auto it = domain.velocities_y_->flat_begin(); it != domain.velocities_y_->flat_end(); ++it) {
                *it = (math::random<double>() * 2 - 1.0) * SimulationConstants::scale;
            }
            for (auto it = domain.velocities_z_->flat_begin(); it != domain.velocities_z_->flat_end(); ++it) {
                *it = (math::random<double>() * 2 - 1.0) * SimulationConstants::scale;
            }

            // Set all solid/solid and solid/fluid borders to 0
            for (size_t x = 0; x < SimulationDomain::dimensions.x; ++x) {
                for (size_t y = 0; y < SimulationDomain::dimensions.y; ++y) {
                    for (size_t z = 0; z < SimulationDomain::dimensions.z; ++z) {
                        if (IsSolidCell(WorldIndex<2>(domain.domain_index_.z, z))) {
                            (*domain.velocities_x_)[x][y][z] = 0.0;
                            (*domain.velocities_x_)[x + 1][y][z] = 0.0;
                            (*domain.velocities_y_)[x][y][z] = 0.0;
                            (*domain.velocities_y_)[x][y + 1][z] = 0.0;
                            (*domain.velocities_z_)[x][y][z] = 0.0;
                            (*domain.velocities_z_)[x][y][z + 1] = 0.0;
                        }
                    }
                }
            }

            sync_latch.count_down();
        });
    }

    sync_latch.wait();

    SyncVelocities();
    for (auto& domain : *domains_) {
        domain.velocities_x_temp_->CopyFrom(*domain.velocities_x_);
        domain.velocities_y_temp_->CopyFrom(*domain.velocities_y_);
        domain.velocities_z_temp_->CopyFrom(*domain.velocities_z_);
    }
}

void EulerFluidSim::SyncVelocitiesZ() {
    ::SyncGridValues<SimulationDomain::GridZValues, math::vec3<int>{0, 0, 1}>(*domains_,
                                                                              &SimulationDomain::velocities_z_);
    for (auto& domain : *domains_) {
        FillMixedCornerHalos<SimulationDomain::GridZValues, math::vec3<int>{0, 0, 1}>(*domains_,
                                                                                      &SimulationDomain::velocities_z_,
                                                                                      domain.domain_index_);
        FillCornerHalos<SimulationDomain::GridZValues, math::vec3<int>{0, 0, 1}>(*domain.velocities_z_,
                                                                                 domain.domain_index_);
    }
}

void EulerFluidSim::SyncVelocities() {
    SyncGridVelocitiesXY(*domains_);
    SyncVelocitiesZ();
}

void EulerFluidSim::SyncGridValues(std::unique_ptr<SimulationDomain::GridValues> SimulationDomain::* grid,
                                   SyncScope scope) {
    ::SyncGridValues<SimulationDomain::GridValues>(*domains_, grid, scope);
    if (scope == SyncScope::Orthogonal) {
        return;
    }
    for (auto& domain : *domains_) {
        FillMixedCornerHalos<SimulationDomain::GridValues>(*domains_, grid, domain.domain_index_);
        FillCornerHalos<SimulationDomain::GridValues>(*(domain.*grid), domain.domain_index_);
    }
}

void EulerFluidSim::SetInitialTemperature() {
    std::latch sync_latch{Cardinality(domains)};

    for (size_t i = 0; i < Cardinality(domains); ++i) {
        std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
        thread_pool_->Enqueue([domain_ref,
                               &sync_latch,
                               core_temp = core_temperature_,
                               ambient_temp = ambient_temperature_,
                               crust_temp = crust_temperature_]() {
            auto& domain = domain_ref.get();
            // set all to ambient, except lower boundary
            domain.cell_temperature_->SetTo(ambient_temp);

            if (domain.domain_index_.z == 0) {
                for (size_t x = 0; x < SimulationDomain::dimensions.x; ++x) {
                    for (size_t y = 0; y < SimulationDomain::dimensions.y; ++y) {
                        (*domain.cell_temperature_)[x][y][0] = core_temp;
                    }
                }
            }
            if (domain.domain_index_.z == domains.z - 1) {
                for (size_t x = 0; x < SimulationDomain::dimensions.x; ++x) {
                    for (size_t y = 0; y < SimulationDomain::dimensions.y; ++y) {
                        (*domain.cell_temperature_)[x][y][SimulationDomain::dimensions.z - 1] = crust_temp;
                    }
                }
            }

            domain.cell_temperature_temp_->CopyFrom(*domain.cell_temperature_);

            sync_latch.count_down();
        });
    }

    sync_latch.wait();
}

void EulerFluidSim::PCG() {
    // TODO: there is pressure divergence accumulation on domain borders, that has to be resolved
    // Having precondition and coefficients local to domain increases iterations by a factor of 2
    // If divergence is low enough zero pressure will be close enough
    // It might be faster in some cases to skip this check
    double maxDivergence = std::numeric_limits<double>::lowest();
    double minDivergence = std::numeric_limits<double>::max();
    {
        std::vector<std::future<MinMax>> min_max_results;
        min_max_results.reserve(Cardinality(domains));

        for (size_t i = 0; i < Cardinality(domains); ++i) {
            std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
            auto promise = std::make_shared<std::promise<MinMax>>();
            min_max_results.push_back(promise->get_future());
            thread_pool_->Enqueue([domain_ref, promise]() {
                auto& domain = domain_ref.get();
                // initial guess, past values may be also viable here
                domain.cell_pressure_->SetToZero();
                promise->set_value(domain.DivergenceMinMax());
            });
        }

        for (auto& res : min_max_results) {
            // TODO: order of results are undefined, and there could be a value ready, but main thread is waiting for a specific one
            auto min_max = res.get();
            minDivergence = std::min(minDivergence, min_max.min);
            maxDivergence = std::max(maxDivergence, min_max.max);
        }
    }

    if (fabs(maxDivergence) < std::numeric_limits<double>::epsilon() &&
        fabs(minDivergence) < std::numeric_limits<double>::epsilon())
        return;

    {
        std::latch sync_latch{Cardinality(domains)};

        for (size_t i = 0; i < Cardinality(domains); ++i) {
            std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
            thread_pool_->Enqueue([domain_ref, &sync_latch]() {
                auto& domain = domain_ref.get();
                domain.residual_->CopyFrom(*domain.cell_pressure_divergence_);
                domain.ApplyPreconditioner();
                sync_latch.count_down();
            });
        }

        sync_latch.wait();
    }

    SyncGridValues(&SimulationDomain::auxiliary_, SyncScope::Orthogonal);
    double sigma = 0;
    {
        std::vector<std::future<double>> partial_dot_products;
        partial_dot_products.reserve(Cardinality(domains));

        for (size_t i = 0; i < Cardinality(domains); ++i) {
            std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
            auto promise = std::make_shared<std::promise<double>>();
            partial_dot_products.push_back(promise->get_future());
            thread_pool_->Enqueue([domain_ref, promise]() {
                auto& domain = domain_ref.get();
                domain.search_->CopyFrom(*domain.auxiliary_);
                promise->set_value(
                    domain.RealCellsDotProduct(&SimulationDomain::auxiliary_, &SimulationDomain::residual_));
            });
        }

        for (auto& res : partial_dot_products) {
            sigma += res.get();
        }
    }

    for (last_PCG_iteration_count = 0; last_PCG_iteration_count < SimulationConstants::max_solver_iterations;
         last_PCG_iteration_count++) {
        assert(!std::isinf(sigma));
        assert(!std::isnan(sigma));
        {
            std::latch sync_latch{Cardinality(domains)};

            for (size_t i = 0; i < Cardinality(domains); ++i) {
                std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
                thread_pool_->Enqueue([domain_ref, &sync_latch]() {
                    auto& domain = domain_ref.get();
                    domain.ApplyCoefficients();
                    sync_latch.count_down();
                });
            }

            sync_latch.wait();
        }

        SyncGridValues(&SimulationDomain::auxiliary_, SyncScope::Orthogonal);
        double dot_product = 0;
        {
            std::vector<std::future<double>> partial_dot_products;
            partial_dot_products.reserve(Cardinality(domains));

            for (size_t i = 0; i < Cardinality(domains); ++i) {
                std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
                auto promise = std::make_shared<std::promise<double>>();
                partial_dot_products.push_back(promise->get_future());
                thread_pool_->Enqueue([domain_ref, promise]() {
                    auto& domain = domain_ref.get();
                    promise->set_value(
                        domain.RealCellsDotProduct(&SimulationDomain::auxiliary_, &SimulationDomain::search_));
                });
            }

            for (auto& res : partial_dot_products) {
                dot_product += res.get();
            }
        }

        double alpha = sigma / dot_product;

        {
            std::vector<std::future<double>> partial_length;
            partial_length.reserve(Cardinality(domains));

            for (size_t i = 0; i < Cardinality(domains); ++i) {
                std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
                auto promise = std::make_shared<std::promise<double>>();
                partial_length.push_back(promise->get_future());
                thread_pool_->Enqueue([domain_ref, promise, alpha]() {
                    auto& domain = domain_ref.get();
                    for (size_t j = 0; j < domain.cell_pressure_->cardinality; ++j) {
                        *(domain.cell_pressure_->flat_data() + j) += *(domain.search_->flat_data() + j) * alpha;
                        *(domain.residual_->flat_data() + j) -= *(domain.auxiliary_->flat_data() + j) * alpha;
                    }
                    promise->set_value(domain.RealCellsLength(&SimulationDomain::residual_));
                });
            }

            double total_error = 0.0;
            for (auto& res : partial_length) {
                total_error += res.get();
            }

            if (total_error < SimulationConstants::solver_tolerance)
                break;
        }

        {
            std::latch sync_latch{Cardinality(domains)};

            for (size_t i = 0; i < Cardinality(domains); ++i) {
                std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
                thread_pool_->Enqueue([domain_ref, &sync_latch]() {
                    auto& domain = domain_ref.get();
                    domain.ApplyPreconditioner();
                    sync_latch.count_down();
                });
            }

            sync_latch.wait();
        }
        SyncGridValues(&SimulationDomain::auxiliary_, SyncScope::Orthogonal);

        double sigma_new = 0;
        {
            std::vector<std::future<double>> partial_dot_products;
            partial_dot_products.reserve(Cardinality(domains));

            for (size_t i = 0; i < Cardinality(domains); ++i) {
                std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
                auto promise = std::make_shared<std::promise<double>>();
                partial_dot_products.push_back(promise->get_future());
                thread_pool_->Enqueue([domain_ref, promise]() {
                    auto& domain = domain_ref.get();
                    promise->set_value(
                        domain.RealCellsDotProduct(&SimulationDomain::auxiliary_, &SimulationDomain::residual_));
                });
            }

            for (auto& res : partial_dot_products) {
                sigma_new += res.get();
            }
        }
        double beta = sigma_new / sigma;
        {
            std::latch sync_latch{Cardinality(domains)};

            for (size_t i = 0; i < Cardinality(domains); ++i) {
                std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
                thread_pool_->Enqueue([domain_ref, &sync_latch, beta]() {
                    auto& domain = domain_ref.get();
                    for (size_t j = 0; j < domain.search_->cardinality; ++j) {
                        *(domain.search_->flat_data() + j) =
                            *(domain.auxiliary_->flat_data() + j) + *(domain.search_->flat_data() + j) * beta;
                    }
                    sync_latch.count_down();
                });
            }

            sync_latch.wait();
        }

        //assert(sigma_new < sigma);
        sigma = sigma_new;
    }
    SyncGridValues(&SimulationDomain::cell_pressure_);
    // TODO: Log used iterations properly
}

void EulerFluidSim::ProjectVelocities() {
    {
        std::latch sync_latch{Cardinality(domains)};

        for (size_t i = 0; i < Cardinality(domains); ++i) {
            std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
            thread_pool_->Enqueue([domain_ref, &sync_latch]() {
                domain_ref.get().CalculateDivergence();
                sync_latch.count_down();
            });
        }

        sync_latch.wait();
    }
    SyncGridValues(&SimulationDomain::cell_pressure_divergence_);
    PCG();
    {
        std::latch sync_latch{Cardinality(domains)};

        for (size_t i = 0; i < Cardinality(domains); ++i) {
            std::reference_wrapper<SimulationDomain> domain_ref = (*domains_)[i];
            thread_pool_->Enqueue([domain_ref, &sync_latch]() {
                domain_ref.get().UpdateVelocities();
                sync_latch.count_down();
            });
        }

        sync_latch.wait();
    }
    SyncVelocities();
}

// Debug draw methods
void EulerFluidSim::DrawGrid(Core::RenderPipeline* renderer, bool persistent) const {
    // There is some overdraw on face seems, but it is negligible
    // And it is a debug drawing method anyway
    for (size_t i = 0; i < domains.w; ++i) {
        // Z aligned lines are batched into a single line
        for (size_t x = 0; x <= real_face_dimensions.x; ++x) {
            for (size_t y = 0; y <= real_face_dimensions.y; ++y) {
                auto line_start =
                    math::vec3<float>{world_origin_offset.x + static_cast<float>(x) * SimulationConstants::scale,
                                      world_origin_offset.y + static_cast<float>(y) * SimulationConstants::scale,
                                      0.0f};
                auto line_end =
                    math::vec3<float>{world_origin_offset.x + static_cast<float>(x) * SimulationConstants::scale,
                                      world_origin_offset.y + static_cast<float>(y) * SimulationConstants::scale,
                                      static_cast<float>(real_face_dimensions.z) * SimulationConstants::scale};
                renderer->DebugDrawLine(ToWorldCoordinates(i, line_start),
                                        ToWorldCoordinates(i, line_end),
                                        grid_color_,
                                        persistent);
            }
        }
        // XY lines do not have that luxury
        // To avoid overdraw, only X+ and Y+ are drawn
        for (size_t z = 0; z <= real_face_dimensions.z; ++z) {
            for (size_t y = 0; y <= real_face_dimensions.y; ++y) {
                for (size_t x = 0; x <= real_face_dimensions.x; ++x) {
                    auto line_start =
                        math::vec3<float>{world_origin_offset.x + static_cast<float>(x) * SimulationConstants::scale,
                                          world_origin_offset.y + static_cast<float>(y) * SimulationConstants::scale,
                                          static_cast<float>(z) * SimulationConstants::scale};

                    if (x != real_face_dimensions.x) {
                        auto line_end = math::vec3<float>{
                            world_origin_offset.x + static_cast<float>(x + 1) * SimulationConstants::scale,
                            world_origin_offset.y + static_cast<float>(y) * SimulationConstants::scale,
                            static_cast<float>(z) * SimulationConstants::scale};
                        renderer->DebugDrawLine(ToWorldCoordinates(i, line_start),
                                                ToWorldCoordinates(i, line_end),
                                                grid_color_,
                                                persistent);
                    }
                    if (y != real_face_dimensions.y) {
                        auto line_end = math::vec3<float>{
                            world_origin_offset.x + static_cast<float>(x) * SimulationConstants::scale,
                            world_origin_offset.y + static_cast<float>(y + 1) * SimulationConstants::scale,
                            static_cast<float>(z) * SimulationConstants::scale};
                        renderer->DebugDrawLine(ToWorldCoordinates(i, line_start),
                                                ToWorldCoordinates(i, line_end),
                                                grid_color_,
                                                persistent);
                    }
                }
            }
        }
    }
}

// TODO: There is a display bug, where XY border velocities have 30 degree misalignment
void EulerFluidSim::DrawBoundaryVelocities(Core::RenderPipeline* renderer, bool persistent) const {
    for (const auto& domain : *domains_) {
        auto face_index = domain.domain_index_.w;
        auto offset = math::vec3<float>{world_origin_offset.x, world_origin_offset.y, 0.0f};
        offset -= static_cast<math::vec3<float>>(SimulationDomain::halo_size) * SimulationConstants::scale;
        offset += static_cast<math::vec3<float>>(static_cast<Index3D>(domain.domain_index_) *
                                                 SimulationDomain::real_dimensions) *
                  SimulationConstants::scale;
        for (size_t x = SimulationDomain::halo_size.x;
             x <= SimulationDomain::dimensions.x - SimulationDomain::halo_size.x;
             ++x) {
            for (size_t y = SimulationDomain::halo_size.y;
                 y <= SimulationDomain::dimensions.y - SimulationDomain::halo_size.y;
                 ++y) {
                for (size_t z = SimulationDomain::halo_size.z;
                     z <= SimulationDomain::dimensions.z - SimulationDomain::halo_size.z;
                     ++z) {
                    if (y != SimulationDomain::dimensions.y - SimulationDomain::halo_size.y &&
                        z != SimulationDomain::dimensions.z - SimulationDomain::halo_size.z) {
                        auto start = offset + (math::vec3<float>{x, y, z} + math::vec3<float>{0.0f, 0.5f, 0.5f}) *
                                                  SimulationConstants::scale;
                        double arrow_value = (*domain.velocities_x_)[x][y][z];
                        math::vec3<float> arrow{arrow_value * 0.5f, 0.0f, 0.0f};
                        renderer->DebugDrawArrow(ToWorldCoordinates(face_index, start),
                                                 ToWorldCoordinates(face_index, start + arrow),
                                                 fabsf(arrow_value) * 0.25f,
                                                 boundary_velocity_color_x_,
                                                 persistent);
                    }
                    if (x != SimulationDomain::dimensions.x - SimulationDomain::halo_size.x &&
                        z != SimulationDomain::dimensions.z - SimulationDomain::halo_size.z) {
                        auto start = offset + (math::vec3<float>{x, y, z} + math::vec3<float>{0.5f, 0.0f, 0.5f}) *
                                                  SimulationConstants::scale;
                        double arrow_value = (*domain.velocities_y_)[x][y][z];
                        math::vec3<float> arrow{0.0f, arrow_value * 0.5f, 0.0f};
                        renderer->DebugDrawArrow(ToWorldCoordinates(face_index, start),
                                                 ToWorldCoordinates(face_index, start + arrow),
                                                 fabsf(arrow_value) * 0.25f,
                                                 boundary_velocity_color_y_,
                                                 persistent);
                    }
                    if (x != SimulationDomain::dimensions.x - SimulationDomain::halo_size.x &&
                        y != SimulationDomain::dimensions.y - SimulationDomain::halo_size.y) {
                        auto start = offset + (math::vec3<float>{x, y, z} + math::vec3<float>{0.5f, 0.5f, 0.0f}) *
                                                  SimulationConstants::scale;
                        double arrow_value = (*domain.velocities_z_)[x][y][z];
                        math::vec3<float> arrow{0.0f, 0.0f, arrow_value * 0.5f};
                        renderer->DebugDrawArrow(ToWorldCoordinates(face_index, start),
                                                 ToWorldCoordinates(face_index, start + arrow),
                                                 fabsf(arrow_value) * 0.25f,
                                                 boundary_velocity_color_z_,
                                                 persistent);
                    }
                }
            }
        }
    }
}

void EulerFluidSim::DrawInterpolatedVelocities(Core::RenderPipeline* renderer, bool persistent) const {
    // 3 lines per arrow, but EulerFluidSim shouldn't know about that
    auto arrow_count = math::cube(cell_sample_count) * Cardinality(real_face_dimensions) * 6 * 3;
    renderer->ReserveLineSpace(interpolated_velocity_color_, arrow_count, persistent);

    float sample_stride = 1.0f / static_cast<float>(cell_sample_count);
    float sample_offset = sample_stride * 0.5f;

    for (const auto& domain : *domains_) {
        auto face_index = domain.domain_index_.w;
        auto offset = math::vec3<float>{world_origin_offset.x, world_origin_offset.y, 0.0f};
        offset -= static_cast<math::vec3<float>>(SimulationDomain::halo_size) * SimulationConstants::scale;
        offset += static_cast<math::vec3<float>>(static_cast<Index3D>(domain.domain_index_) *
                                                 SimulationDomain::real_dimensions) *
                  SimulationConstants::scale;
        for (size_t x = 0; x < SimulationDomain::real_dimensions.x * cell_sample_count; ++x) {
            for (size_t y = 0; y < SimulationDomain::real_dimensions.y * cell_sample_count; ++y) {
                for (size_t z = 0; z < SimulationDomain::real_dimensions.z * cell_sample_count; ++z) {
                    math::vec3<float> nPos =
                        static_cast<math::vec3<float>>(SimulationDomain::halo_size) + sample_offset;
                    nPos += math::vec3<float>{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)} *
                            sample_stride;
                    math::vec3<float> interpolationPos = domain.DenormalizePosition(nPos);
                    math::vec3<float> velocity = domain.SampleVelocity(nPos) / static_cast<float>(cell_sample_count);
                    interpolationPos += offset;
                    renderer->DebugDrawArrow(ToWorldCoordinates(face_index, interpolationPos),
                                             ToWorldCoordinates(face_index, interpolationPos + velocity),
                                             velocity.magnitude() * 0.25f,
                                             interpolated_velocity_color_,
                                             persistent);
                }
            }
        }
    }
}

math::vec3<float> EulerFluidSim::ToWorldCoordinates(size_t face_index, const math::vec3<float>& pos) const {
    math::vec3<float> result{pos.x, pos.y, world_origin_offset.z};
    result /= result.magnitude() / (inner_sphere_radius + pos.z);
    return ConvertToFaceCS(cube_face_cs[face_index], result);
}
