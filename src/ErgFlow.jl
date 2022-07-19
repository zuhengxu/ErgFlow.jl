module ErgFlow

using ForwardDiff, LinearAlgebra, Distributions, Random, StatsBase, SpecialFunctions

abstract type ErgodicFlow end
abstract type flow_params end

export HamFlow_1d, HF1d_params, HamFlowRot, HamFlow, HF_params
##################################3
# ErgodicFlow via Hamiltonian dynamics 
####################################
include("train.jl") # SGD using Zygote
include("old_util.jl")
include("util.jl")
include("ham_rot.jl")
include("ham.jl")


end
