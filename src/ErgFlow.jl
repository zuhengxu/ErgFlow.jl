module ErgFlow

using ForwardDiff, LinearAlgebra, Distributions, Random, StatsBase, SpecialFunctions, ProgressMeter
using Base.Threads: @threads


abstract type ErgodicFlow end
abstract type flow_params end

struct HF_params <: flow_params
    leapfrog_stepsize::Vector{Float64}  
    μ::Vector{Float64}
    D::Vector{Float64}
end


export ErgodicFlow, HamFlowRot, HamFlow, HF_params
export cdf_exp, invcdf_exp, cdf_laplace_std, invcdf_laplace_std
export pseudo_refresh, pseudo_refresh_coord, inv_refresh, inv_refresh_coord


##################################3
# ErgodicFlow via Hamiltonian dynamics 
####################################
include("util.jl") 
include("flow.jl") # general function used by Ergodic Flow (leapfrog, flow_fwd, flow_bwd, sampler) 
include("elbo.jl") # ELBOs for ErgFlow
include("ham.jl") # standard Ergodic Flow using coordinate CDF/inv_cdf refresh
include("ham_rot.jl") # ErgFlow with Gaussian momentum using pairwise Gaussian refresh (rotation + rescale norm)
include("error_checking.jl")

end
