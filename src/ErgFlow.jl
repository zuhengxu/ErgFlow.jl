module ErgFlow

using ForwardDiff, LinearAlgebra, Distributions, Random, StatsBase, SpecialFunctions, ProgressMeter
using Base.Threads: @threads


abstract type ErgodicFlow end
abstract type flow_params end

export ErgodicFlow, HamFlow_1d, HF1d_params, HamFlowRot, HamFlow, HF_params
export cdf_exp, invcdf_exp, cdf_laplace_std, invcdf_laplace_std

##################################3
# ErgodicFlow via Hamiltonian dynamics 
####################################
include("train.jl") # SGD using Zygote
include("util.jl") 
include("leapfrog.jl")  
include("ham_rot.jl")
include("ham.jl")
include("error_checking.jl")


end
