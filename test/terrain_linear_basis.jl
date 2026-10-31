include(joinpath(@__DIR__, "setup.jl"))

# The materialized `LinearDecay` stores its decay basis b(r) = 1 − r/z_top at the Center and Face levels and
# b′ = −1/z_top (`LinearBasis`), filled from the expression itself on the grid's device; znode, σ and the slopes load
# them instead of dividing. Every metric and the stepped state are bitwise those of the expression (the formulation
# without a basis), on every architecture.

using Breeze
using Breeze.TerrainFollowingDiscretization: LinearDecay, LinearBasis
using Oceananigans
using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ, ∂x_zᶠᶜᶜ, ∂y_zᶜᶠᶜ
using KernelAbstractions: @kernel, @index
using Test

terrain(λ, φ) = 1500 * exp(-((λ - 2)^2 + (φ - 30)^2) / 4)
const r_faces = collect(range(0, 20e3, length = 13))

llg(FT, formulation) = LatitudeLongitudeGrid(default_arch, FT; size = (12, 10, 12), halo = (5, 5, 5),
                                             longitude = (-4, 8), latitude = (25, 35),
                                             z = TerrainFollowingVerticalDiscretization(r_faces; formulation),
                                             topology = (Bounded, Bounded, Bounded))

function terrain_grid(FT)
    grid = llg(FT, LinearDecay())
    materialize_terrain!(grid, terrain)
    return grid
end

# The same terrain without the basis (the expression path): a formulation that already holds its data is preserved
# by the grid constructor, basis `nothing` included.
function without_basis(grid)
    f = grid.z.formulation
    return llg(eltype(grid), LinearDecay(f.z_top, f.h, f.∂x_h, f.∂y_h))
end

@kernel function _metrics!(out, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        out[i, j, k, 1] = znode(i, j, k, grid, Center(), Center(), Center())
        out[i, j, k, 2] = znode(i, j, k, grid, Center(), Center(), Face())
        out[i, j, k, 3] = Δzᶜᶜᶜ(i, j, k, grid)
        out[i, j, k, 4] = Δzᶜᶜᶠ(i, j, k, grid)
        out[i, j, k, 5] = ∂x_zᶠᶜᶜ(i, j, k, grid)
        out[i, j, k, 6] = ∂y_zᶜᶠᶜ(i, j, k, grid)
    end
end

function metrics(grid)
    arch = Oceananigans.Architectures.architecture(grid)
    out = Oceananigans.Architectures.on_architecture(arch, zeros(eltype(grid), size(grid)..., 6))
    _metrics!(Oceananigans.Architectures.device(arch), (4, 4, 4))(out, grid; ndrange = size(grid))
    return Array(out)
end

@testset "LinearDecay basis tables [$FT]" for FT in (Float32, Float64)
    grid = terrain_grid(FT)
    @test grid.z.formulation.basis isa LinearBasis
    plain = without_basis(grid)
    @test plain.z.formulation.basis === nothing
    @test isequal(metrics(grid), metrics(plain))
end
