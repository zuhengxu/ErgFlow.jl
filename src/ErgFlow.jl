module ErgFlow

using LinearAlgebra, Distributions, Random, StatsBase, SpecialFunctions, ProgressMeter
using Base.Threads: @threads


abstract type ErgodicFlow end
abstract type flow_params end

struct HF_params <: flow_params
    leapfrog_stepsize::Vector{Float64}  
    μ::Vector{Float64}
    D::Vector{Float64}
end


export ErgodicFlow, HamFlow, HF_params
export cdf_laplace_std, invcdf_laplace_std, cdf_normal, invcdf_normal
export pseudo_refresh, pseudo_refresh_coord, inv_refresh, inv_refresh_coord
export flow_sampler, Sampler, Sampler!, flow_fwd, flow_fwd_trace


##################################3
# ErgodicFlow via Hamiltonian dynamics 
####################################
include("util.jl") 
include("flow.jl") # general function used by Ergodic Flow (leapfrog, flow_fwd, flow_bwd, sampler) 
include("elbo.jl") # ELBOs for ErgFlow
include("ham_1d.jl") # ErgFlow only for 1d target
include("ham.jl") # standard Ergodic Flow using coordinate CDF/inv_cdf refresh
include("error_checking.jl") # functions checking numerical error

end
